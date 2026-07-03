"""Muon optimizer (orthogonalized-momentum) + a hybrid builder for HF Trainer.

Muon (Keller Jordan, 2024) replaces the raw momentum update of a 2D weight
matrix with its nearest orthogonal (semi-)matrix, computed by a few
Newton-Schulz iterations. The intuition for THIS task (research direction 1):
orthogonalized steps move every singular direction of a weight matrix by a
comparable amount instead of letting a few high-curvature directions dominate,
so the update installs the matching-game associations while drifting the
pretrained weights less along the directions that carry the model's general
preference structure. Less drift -> more decisiveness retained.

Muon only makes sense on >=2D matrices. Embeddings, the lm_head, LayerNorm
scales, and biases are optimized with a normal AdamW group ("aux adam"). This
mirrors the hybrid the researcher seeded ("Muon on 2D weight matrices, AdamW on
embeddings/scalars").
"""
from __future__ import annotations

import importlib
from typing import Any

torch = importlib.import_module("torch")


def zeropower_via_newtonschulz5(G: Any, steps: int = 5, eps: float = 1e-7) -> Any:
    """Newton-Schulz iteration approximating the orthogonal factor of G.

    Uses the quintic iteration with the standard (a,b,c) tuned so the singular
    values converge into [~0.7, ~1.3] — good enough to act as an orthogonal
    "sign" of the update without an exact SVD. Computed in bf16 for speed.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2D params + a fused AdamW group for everything else.

    Param groups are tagged with `use_muon` (bool). Muon groups use
    orthogonalized momentum; the aux groups use standard decoupled AdamW.
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        *,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        # NB: every group carries its OWN "lr" so HF's LR scheduler (which
        # multiplies each param_group["lr"] by a shared schedule lambda) decays
        # the Muon and aux-AdamW groups proportionally from their own base LRs.
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps,
            weight_decay=weight_decay,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            use_muon = group.get("use_muon", False)
            if use_muon:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            update = g.add(buf, alpha=momentum) if nesterov else buf
            if update.ndim > 2:
                update = update.view(update.size(0), -1)
            ortho = zeropower_via_newtonschulz5(update, steps=ns_steps).to(g.dtype)
            # Scale so the effective step size is comparable across shapes.
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            if wd > 0:
                p.mul_(1 - lr * wd)
            p.add_(ortho.view_as(p), alpha=-lr * scale)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)
            if wd > 0:
                p.mul_(1 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / bias1)


def build_muon_optimizer(
    model: Any,
    *,
    muon_lr: float,
    adamw_lr: float,
    momentum: float = 0.95,
    ns_steps: int = 5,
    weight_decay: float = 0.0,
) -> Muon:
    """Split trainable params: >=2D non-embedding weights -> Muon, rest -> AdamW.

    Embeddings / lm_head are routed to the aux AdamW group by name (they are
    usually frozen under freeze_embeddings anyway, in which case they carry no
    grad and are skipped).
    """
    muon_params: list[Any] = []
    adamw_params: list[Any] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = ("embed" in name.lower()) or ("lm_head" in name.lower())
        if p.ndim >= 2 and not is_embed:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    param_groups = [
        {"params": muon_params, "use_muon": True, "lr": muon_lr,
         "momentum": momentum, "nesterov": True, "ns_steps": ns_steps,
         "adamw_betas": (0.9, 0.95), "adamw_eps": 1e-8, "weight_decay": weight_decay},
        {"params": adamw_params, "use_muon": False, "lr": adamw_lr,
         "momentum": momentum, "nesterov": True, "ns_steps": ns_steps,
         "adamw_betas": (0.9, 0.95), "adamw_eps": 1e-8, "weight_decay": weight_decay},
    ]
    return Muon(param_groups, lr=muon_lr, momentum=momentum,
                ns_steps=ns_steps, weight_decay=weight_decay)
