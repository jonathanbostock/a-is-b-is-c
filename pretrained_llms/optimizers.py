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
            muon_params, aux_params = [], []
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.ndim >= 2 and id(p) not in embed_param_ids:
                    muon_params.append(p)
                else:
                    aux_params.append(p)
            if not muon_params:
                raise RuntimeError("Muon: no eligible >=2D hidden params found")
            groups = [
                dict(params=muon_params, use_muon=True,
                     lr=muon_lr, momentum=momentum, weight_decay=weight_decay),
                dict(params=aux_params, use_muon=False,
                     lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay),
            ]
            self.optimizer = SingleDeviceMuonWithAuxAdam(groups)
            n_m = sum(p.numel() for p in muon_params) / 1e6
            n_a = sum(p.numel() for p in aux_params) / 1e6
            print(f"[muon] {len(muon_params)} matrices ({n_m:.0f}M params) on Muon lr={muon_lr}; "
                  f"{len(aux_params)} tensors ({n_a:.0f}M) on aux AdamW lr={adam_lr}")
            return self.optimizer

    return MuonTrainer
