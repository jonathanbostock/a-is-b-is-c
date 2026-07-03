"""Custom-optimizer Trainer factories. Currently: Muon (orthogonalized momentum).

Muon (Jordan et al., 2024): momentum-SGD whose update is orthogonalized via
Newton-Schulz iteration before being applied. Applies ONLY to 2D hidden weight
matrices; embeddings / lm_head / norms / biases (and any 1D params) are handled
by an auxiliary AdamW. We use the reference package's single-device classes
(`pip install git+https://github.com/KellerJordan/Muon`) — the plain `Muon`
class there assumes torch.distributed is initialised and will crash single-GPU.

Why here: Muon's geometry-constrained steps plausibly install new associations
with less drift from the pretrained weights than AdamW — the crystallize-
without-cooking bet, and a natural SDF knob.

Composes with regularizers.make_l2sp_trainer (that overrides training_step;
this overrides create_optimizer — disjoint hooks, chain freely).
"""
from __future__ import annotations

from typing import Any

import torch


class CompositeOptimizer(torch.optim.Optimizer):
    """Steps several child optimizers as one. Shares the CHILD group dicts in
    param_groups (by reference), so HF/torch LR schedulers that mutate
    group["lr"] drive every child correctly."""

    def __init__(self, opts: list):
        self._opts = opts
        groups = [g for o in opts for g in o.param_groups]
        first_lr = groups[0].get("lr", 1e-3)
        # Re-registering the SAME dict objects keeps them shared by reference.
        super().__init__(groups, dict(lr=first_lr))

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for o in self._opts:
            o.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for o in self._opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"children": [o.state_dict() for o in self._opts]}

    def load_state_dict(self, sd):
        for o, csd in zip(self._opts, sd.get("children", [])):
            try: o.load_state_dict(csd)
            except Exception: pass


def make_muon_trainer(*, base_trainer_cls: Any, muon_lr: float, adam_lr: float,
                      weight_decay: float = 0.0, momentum: float = 0.95) -> Any:
    """Return a Trainer subclass whose optimizer is SingleDeviceMuonWithAuxAdam.

    Param routing (reference-recommended):
      - Muon group:  requires_grad, ndim >= 2, NOT embedding/lm_head
      - AdamW group: everything else trainable (norms, biases, embed if unfrozen)
    """
    try:
        from muon import SingleDeviceMuonWithAuxAdam  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Muon requested but the package is missing. Install with: "
            "pip install git+https://github.com/KellerJordan/Muon"
        ) from exc

    class MuonTrainer(base_trainer_cls):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            model = self.model
            # Identify embedding/lm_head tensors by module type & name.
            embed_param_ids = set()
            for name, module in model.named_modules():
                cls_name = type(module).__name__.lower()
                if "embedding" in cls_name or name.endswith("lm_head") or name.endswith("embed_tokens"):
                    for p in module.parameters(recurse=False):
                        embed_param_ids.add(id(p))
            # LoRA factors must NOT get naive Muon: orthogonalizing the A/B
            # factor updates is not orthogonalizing the update to dW = BA, and
            # the factorization is parametrization-ambiguous — see "LoRA meets
            # Riemannion" (arXiv:2507.12142, ICLR 2026). DEFAULT: LoRA pairs are
            # optimized with our verified Riemannion port (pretrained_llms/
            # riemannion.py — Alg. 4 on the fixed-rank manifold).
            from .riemannion import RiemannionLoRA, pair_lora_params
            lora_pairs = pair_lora_params(model)
            lora_param_ids = {id(p) for n, p in model.named_parameters() if "lora_" in n}
            muon_params, aux_params = [], []
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if id(p) in lora_param_ids:
                    continue  # exclusively Riemannion's — never duplicated into muon/aux
                if p.ndim >= 2 and id(p) not in embed_param_ids:
                    muon_params.append(p)
                else:
                    aux_params.append(p)
            if not muon_params and not lora_pairs:
                raise RuntimeError("Muon: no eligible >=2D hidden params found")

            children = []
            if lora_pairs:
                children.append(RiemannionLoRA(lora_pairs, lr=muon_lr, momentum=momentum, weight_decay=weight_decay))
                print(f"[riemannion] {len(lora_pairs)} LoRA adapters on Riemannion lr={muon_lr} "
                      f"(fixed-rank-manifold Muon, arXiv:2507.12142)")
            if muon_params or aux_params:
                groups = []
                if muon_params:
                    groups.append(dict(params=muon_params, use_muon=True,
                                       lr=muon_lr, momentum=momentum, weight_decay=weight_decay))
                if aux_params:
                    groups.append(dict(params=aux_params, use_muon=False,
                                       lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay))
                children.append(SingleDeviceMuonWithAuxAdam(groups))
                n_m = sum(p.numel() for p in muon_params) / 1e6
                n_a = sum(p.numel() for p in aux_params) / 1e6
                print(f"[muon] {len(muon_params)} matrices ({n_m:.0f}M) on Muon lr={muon_lr}; "
                      f"{len(aux_params)} tensors ({n_a:.0f}M) on aux AdamW lr={adam_lr}")
            self.optimizer = children[0] if len(children) == 1 else CompositeOptimizer(children)
            return self.optimizer

    return MuonTrainer
