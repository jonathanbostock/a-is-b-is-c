"""Full-rank regularizers for fine-tuning. Currently: L2-to-init (a.k.a. L2-SP).

L2-SP penalty = (l2_sp_lambda / 2) * sum_p ||p - p_init||_2^2 over trainable
parameters. Snapshots p_init at construction time.

Implementation note — why we add the gradient directly instead of adding a
penalty term to the loss:

  The obvious implementation, `loss += 0.5*lambda*sum((p-p_init)**2)`, makes
  autograd build a graph node for every trainable parameter and hold an fp32
  `(p-p_init)` intermediate for ALL of them simultaneously until backward.
  For a 14B full-parameter fine-tune that's ~50 GB of fp32 temporaries on top
  of the model/grads/optimizer state — it fits on a 180 GB B200 but OOMs a
  140 GB H200.

  The gradient of the penalty is analytic: d/dp [0.5*lambda*(p-p_init)^2] =
  lambda*(p-p_init). So we skip the autograd graph entirely and add
  lambda*(p-p_init) straight into p.grad after the main backward, one
  parameter at a time under no_grad. Extra memory = just the (bf16) init
  snapshot; the per-parameter temporary is transient.

Gradient-accumulation correctness: HF's training_step scales the task loss by
1/grad_accum before backward so that the grad_accum micro-step gradients sum to
the mean gradient for one optimizer step. To match, we add the L2-SP gradient
contribution scaled by 1/grad_accum on every micro-step, so it too sums to
exactly lambda*(p-p_init) per optimizer step.
"""
from __future__ import annotations

import importlib
from typing import Any


def make_l2sp_trainer(*, base_trainer_cls: Any, l2_sp_lambda: float) -> Any:
    """Return a Trainer subclass that adds an L2-to-init gradient penalty.

    Works for both full-fine-tune and LoRA — it only penalises currently
    trainable parameters (skips frozen base weights when LoRA is on, and skips
    frozen embeddings under freeze_embeddings).
    """
    torch = importlib.import_module("torch")

    class L2SPTrainer(base_trainer_cls):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Snapshot trainable parameter values at the start of training.
            # Kept in the parameter's own dtype (bf16 for our runs) to halve
            # the snapshot's footprint vs fp32.
            self._init_snapshot: dict[str, Any] = {}
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self._init_snapshot[name] = p.detach().clone()
            self._l2sp_lambda = l2_sp_lambda

        def training_step(self, model, inputs, num_items_in_batch=None):
            # Run the normal task-loss forward+backward (populates p.grad,
            # already scaled by 1/grad_accum inside HF).
            loss = super().training_step(model, inputs, num_items_in_batch)
            if self._l2sp_lambda > 0:
                accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
                coeff = self._l2sp_lambda / accum
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        init = self._init_snapshot.get(name)
                        if init is None:
                            continue
                        if init.device != p.device:
                            init = init.to(p.device)
                            self._init_snapshot[name] = init
                        # grad += lambda * (p - p_init), scaled for grad-accum.
                        p.grad.add_((p - init), alpha=coeff)
            return loss

    return L2SPTrainer
