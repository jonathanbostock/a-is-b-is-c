"""Muon optimizer (orthogonalized-momentum updates) for the matching-game FT.

Seeded research direction #1. Muon takes the momentum-averaged gradient of each
2D weight matrix and replaces it with the nearest orthogonal (semi-orthogonal)
matrix via a few Newton-Schulz iterations, then applies that as the update. The
update therefore has (approximately) uniform singular values — it rotates the
weight matrix rather than stretching it along a few dominant directions. The
hypothesis for THIS task: such geometry-aware steps install the matching-game
associations while drifting less along the specific directions that encode the
model's general preference structure, so decisiveness is better preserved than
under AdamW's coordinate-wise steps.

We apply Muon to 2D parameters (for a LoRA run, the adapter A/B matrices) and fall
back to AdamW for any 1D/embedding/scalar trainable params, per the standard
hybrid recipe.

Reference: Jordan et al., "Muon: An optimizer for the hidden layers of neural
networks" (single-GPU form; no distributed all-gather here).
"""
from __future__ import annotations

import importlib
from typing import Any


def _make_muon_class(torch: Any):
    @torch.no_grad()
    def zeropower_via_newtonschulz5(G: Any, steps: int) -> Any:
        """Newton-Schulz iteration to compute the orthogonal polar factor of G.

        Runs in bfloat16 for speed/stability. Quintic coefficients (a,b,c) chosen
        so the iteration converges for singular values in a wide range after ~5
        steps (standard Muon constants)."""
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        transposed = False
        if G.size(0) > G.size(1):
            X = X.T
            transposed = True
        # Normalize so the spectral norm is <= 1 going in.
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if transposed:
            X = X.T
        return X

    class Muon(torch.optim.Optimizer):
        def __init__(self, params, lr=2e-3, momentum=0.95, nesterov=True,
                     ns_steps=5, weight_decay=0.0):
            defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                            ns_steps=ns_steps, weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group["lr"]; mom = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "m" not in state:
                        state["m"] = torch.zeros_like(g)
                    buf = state["m"]
                    buf.mul_(mom).add_(g)
                    u = g.add(buf, alpha=mom) if group["nesterov"] else buf
                    u = zeropower_via_newtonschulz5(u, group["ns_steps"])
                    # Scale so the update RMS is comparable across matrix shapes.
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if group["weight_decay"] > 0:
                        p.mul_(1 - lr * group["weight_decay"])
                    p.add_(u.to(p.dtype), alpha=-lr * scale)
            return loss

    return Muon


def make_muon_trainer(*, base_trainer_cls: Any, muon_lr: float, adamw_lr: float,
                      momentum: float = 0.95, ns_steps: int = 5) -> Any:
    """Trainer subclass whose create_optimizer builds a hybrid Muon+AdamW optimizer.

    2D trainable params -> Muon; everything else -> AdamW. A single combined
    optimizer is returned so HF's scheduler wiring works unchanged."""
    torch = importlib.import_module("torch")
    Muon = _make_muon_class(torch)

    class _Combined(torch.optim.Optimizer):
        """Holds two optimizers and steps/zeros both; exposes a merged
        param_groups so HF's LR scheduler can scale both."""
        def __init__(self, opts: list[Any]):
            self._opts = opts
            self.param_groups = [g for o in opts for g in o.param_groups]
            self.defaults = opts[0].defaults
            self.state = {}

        def zero_grad(self, set_to_none: bool = True):
            for o in self._opts:
                o.zero_grad(set_to_none=set_to_none)

        def step(self, closure=None):
            for o in self._opts:
                o.step()

        def state_dict(self):
            return {"opts": [o.state_dict() for o in self._opts]}

        def load_state_dict(self, sd):
            for o, s in zip(self._opts, sd["opts"]):
                o.load_state_dict(s)

    class MuonTrainer(base_trainer_cls):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            decay_2d, other = [], []
            for _, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                (decay_2d if p.ndim == 2 else other).append(p)
            opts = []
            if decay_2d:
                opts.append(Muon(decay_2d, lr=muon_lr, momentum=momentum,
                                 ns_steps=ns_steps))
            if other:
                opts.append(torch.optim.AdamW(other, lr=adamw_lr, betas=(0.9, 0.95),
                                              weight_decay=0.0))
            self.optimizer = _Combined(opts) if len(opts) > 1 else opts[0]
            return self.optimizer

    return MuonTrainer
