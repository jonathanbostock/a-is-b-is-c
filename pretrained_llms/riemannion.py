"""Riemannion for standard peft LoRA — Muon generalized to the fixed-rank manifold.

Implements "LoRA meets Riemannion" (Bogachev et al., arXiv:2507.12142, ICLR 2026),
Algorithm 4, natively over peft's parametrization ΔW = s·B@A (s = alpha/r,
B: m×r init 0, A: r×n init kaiming) — no doubled-rank adapter rebuild needed.

Why not naive Muon on A/B: the factorization is parametrization-ambiguous
((B,A) ~ (BQ, Q⁻¹A) give identical ΔW), and orthogonalizing factor updates is
not orthogonalizing the update to ΔW. Riemannion works intrinsically on the
rank-r manifold: Riemannian gradient -> transported heavy-ball momentum ->
tangent-space orthogonalization (the Muon step) -> re-projection -> retraction.

Never materializes any m×n matrix. Per step per adapter: O((m+n)r² + r³).

Representation used throughout
------------------------------
Manifold point  X = U Sc V^T   (U ∈ St(m,r), V ∈ St(n,r), Sc r×r, from QR of B/A)
Tangent vector  ξ = U M V^T + U_p V^T + U V_p^T,  U^T U_p = 0, V^T V_p = 0
              (stored as the triple (M, U_p, V_p) w.r.t. the current (U, V))

Ambient-gradient products from peft factor grads (G = dL/dX never formed):
  G_B = dL/dB = s·G A^T,  G_A = dL/dA = s·B^T G
  with B = U R_B (QR), C := s·R_B A, C^T = Q_C R_C (QR) => V = Q_C, Sc = R_C^T:
    A = (1/s) R_B^{-1} Sc V^T
    G V   = G_B R_B^T Sc^{-1}          [m×r]
    U^T G = (1/s) R_B^{-T} G_A         [r×n]   (=> G^T U and U^T G V follow)

Deviation from the paper, documented: we skip the LOI initialization (needs a
dedicated backward at init); since peft's B=0 gives a rank-0 point (manifold-
invalid), Riemannion re-inits B to a small orthonormal frame (sigma * U0) at
optimizer construction. ΔW starts ~1e-3-scale instead of exactly 0.
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer

__all__ = ["RiemannionLoRA", "pair_lora_params"]


# --------------------------------------------------------------------------- #
# small linear-algebra helpers (all thin-matrix ops)
# --------------------------------------------------------------------------- #

def _qr(x: torch.Tensor):
    return torch.linalg.qr(x.float(), mode="reduced")


def _point_from_factors(B: torch.Tensor, A: torch.Tensor, s: float):
    """(B, A, s) -> (U, V, Sc, R_B) with X = s·B@A = U Sc V^T."""
    U, R_B = _qr(B)                       # B = U R_B
    C = s * (R_B @ A.float())             # r×n
    Qc, Rc = _qr(C.T)                     # C^T = Qc Rc
    V = Qc                                # n×r
    Sc = Rc.T                             # r×r  (X = U Sc V^T)
    return U, V, Sc, R_B


def _tangent_from_grads(U, V, Sc, R_B, G_A, G_B, s: float):
    """Riemannian gradient as (M, U_p, V_p) from peft factor grads."""
    # G_B = s·G A^T and A^T = (1/s) V Sc^T R_B^{-T}  =>  G V = G_B R_B^T Sc^{-T}.
    # Right-multiply by Sc^{-T} via a solve: X Sc^{-T} = solve(Sc, X^T)^T.
    GBR = G_B.float() @ R_B.T                               # m×r
    GV = torch.linalg.solve(Sc, GBR.T).T                    # m×r  = G V
    UtG = torch.linalg.solve(R_B.T, G_A.float()) / s        # r×n  = U^T G
    M = UtG @ V                                             # r×r  = U^T G V
    U_p = GV - U @ (U.T @ GV)                               # (I-UU^T) G V
    V_p = UtG.T - V @ (V.T @ UtG.T)                         # (I-VV^T) G^T U
    return M, U_p, V_p


def _tangent_to_LR(U, V, M, U_p, V_p):
    """ξ = L R^T with L = [U M + U_p, U] (m×2r), R = [V, V_p] (n×2r)."""
    L = torch.cat([U @ M + U_p, U], dim=1)
    R = torch.cat([V, V_p], dim=1)
    return L, R


def _project_LR(U, V, L, R):
    """Project an ambient rank-≤2r matrix Z = L R^T onto T_X M_r -> (M, U_p, V_p)."""
    RtV = R.T @ V                                           # 2r×r
    ZV = L @ RtV                                            # m×r  = Z V
    LtU = L.T @ U                                           # 2r×r
    ZtU = R @ LtU                                           # n×r  = Z^T U
    M = U.T @ ZV                                            # r×r
    U_p = ZV - U @ M
    V_p = ZtU - V @ M.T
    return M, U_p, V_p


def _ortho_LR(L, R, eps: float = 1e-8):
    """OrthoLR (Alg. 1): msign of ξ = L R^T without an m×n SVD.
    QR both factors, SVD the 2r×2r core, set nonzero singular values to 1."""
    Ql, Tl = _qr(L)
    Qr, Tr = _qr(R)
    core = Tl @ Tr.T                                        # 2r×2r
    Uc, Sc, Vct = torch.linalg.svd(core)
    ones = (Sc > eps * Sc.max().clamp_min(eps)).to(core.dtype)
    core_o = (Uc * ones.unsqueeze(0)) @ Vct                 # singular values -> {0,1}
    # ξ_orth = Ql core_o Qr^T, in LR form:
    return Ql @ core_o, Qr


def _retract(U, V, Sc, M, U_p, V_p, eta: float, r: int):
    """X_+ = SVD_r(X - eta·ξ) in factored form (never m×n).
    X - eta·ξ = [U, U_p] K [V, V_p]^T with K = [[Sc - eta·M, -eta·I],[-eta·I, 0]]."""
    dev, dt = Sc.device, Sc.dtype
    L = torch.cat([U, U_p], dim=1)                          # m×2r
    R = torch.cat([V, V_p], dim=1)                          # n×2r
    I = torch.eye(r, device=dev, dtype=dt)
    Z = torch.zeros(r, r, device=dev, dtype=dt)
    K = torch.cat([
        torch.cat([Sc - eta * M, -eta * I], dim=1),
        torch.cat([-eta * I, Z], dim=1),
    ], dim=0)                                               # 2r×2r
    Ql, Tl = _qr(L)
    Qr, Tr = _qr(R)
    core = Tl @ K @ Tr.T
    Uc, S, Vct = torch.linalg.svd(core)
    U_new = Ql @ Uc[:, :r]
    V_new = Qr @ Vct[:r, :].T
    S_new = S[:r]
    return U_new, V_new, S_new


# --------------------------------------------------------------------------- #
# param pairing
# --------------------------------------------------------------------------- #

def pair_lora_params(model) -> list[dict]:
    """Find (lora_B, lora_A, scaling) triples on a peft-wrapped model."""
    pairs = []
    for name, module in model.named_modules():
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if lora_A is None or lora_B is None:
            continue
        try:
            keys = list(lora_A.keys())
        except Exception:
            continue
        for key in keys:
            a_w = lora_A[key].weight
            b_w = lora_B[key].weight
            if not (a_w.requires_grad and b_w.requires_grad):
                continue
            s = float(module.scaling[key]) if hasattr(module, "scaling") else 1.0
            pairs.append({"name": f"{name}.{key}", "A": a_w, "B": b_w, "s": s})
    return pairs


# --------------------------------------------------------------------------- #
# the optimizer
# --------------------------------------------------------------------------- #

class RiemannionLoRA(Optimizer):
    """Riemannion (Alg. 4) over peft LoRA pairs.

    params: iterable of dicts from pair_lora_params(model). lr is the step on
    the ORTHOGONALIZED tangent direction (Muon-like scale-free step). momentum
    is Polyak heavy-ball with tangent transport. B=0 is re-initialized to a
    small orthonormal frame (init_sigma) to give a valid rank-r point.
    """

    def __init__(self, pairs: list[dict], lr: float = 1e-3, momentum: float = 0.9,
                 init_sigma: float = 1e-3, eps: float = 1e-8, weight_decay: float = 0.0):
        self.pairs = pairs
        params = []
        for p in pairs:
            params.extend([p["A"], p["B"]])
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self._lr = lr
        self._beta = momentum
        self._eps = eps
        # Decoupled, parametrization-invariant decay: shrinks the SINGULAR VALUES
        # of the delta X = s*B@A after retraction (X <- (1-lr*wd) X). Since LoRA
        # init is X=0, this is exactly L2-SP toward the pretrained weights.
        # Naive decay on the B/A factors would be gauge-dependent ((B,A)~(BQ,Q^-1 A)).
        self._wd = weight_decay
        self._pair_state: dict[str, dict] = {}
        # Valid manifold point at start: B must have rank r.
        with torch.no_grad():
            for pr in pairs:
                B = pr["B"]
                if float(B.abs().max()) == 0.0:
                    m, r = B.shape
                    Q, _ = _qr(torch.randn(m, r, device=B.device))
                    B.copy_((init_sigma * Q).to(B.dtype))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        loss = closure() if closure is not None else None
        # Read lr from the param group so HF/torch LR schedulers (which mutate
        # group["lr"]) actually apply — a frozen self._lr would silently ignore
        # warmup + cosine decay, and unit-spectral steps NEED a decaying schedule
        # to converge rather than orbit (see tests/test_riemannion.py case 4).
        self._lr = float(self.param_groups[0]["lr"])
        for pr in self.pairs:
            A, B, s = pr["A"], pr["B"], pr["s"]
            if A.grad is None or B.grad is None:
                continue
            r = B.shape[1]
            st = self._pair_state.setdefault(pr["name"], {})

            U, V, Sc, R_B = _point_from_factors(B.detach(), A.detach(), s)
            M, U_p, V_p = _tangent_from_grads(U, V, Sc, R_B, A.grad, B.grad, s)

            # -- transported heavy-ball momentum (Alg. 4 steps 2-3) --
            if "mom_L" in st:
                Mm, U_pm, V_pm = _project_LR(U, V, st["mom_L"], st["mom_R"])
                M = self._beta * Mm + M
                U_p = self._beta * U_pm + U_p
                V_p = self._beta * V_pm + V_p
            mom_L, mom_R = _tangent_to_LR(U, V, M, U_p, V_p)
            st["mom_L"], st["mom_R"] = mom_L, mom_R

            # -- tangent orthogonalization + re-projection (Alg. 4 step 4) --
            oL, oR = _ortho_LR(mom_L, mom_R, self._eps)
            M_o, U_po, V_po = _project_LR(U, V, oL, oR)

            # -- retraction (Alg. 4 step 5) --
            U_new, V_new, S_new = _retract(U, V, Sc, M_o, U_po, V_po, self._lr, r)

            # -- decoupled manifold weight decay (post-retraction, on singular values) --
            if self._wd > 0.0:
                S_new = S_new * (1.0 - self._lr * self._wd)

            # -- write back to peft factors: balanced split, absorb s --
            sqrt_s = s ** 0.5
            S_half = torch.sqrt(torch.clamp(S_new, min=0.0))
            B_new = (U_new * S_half.unsqueeze(0)) / sqrt_s
            A_new = (S_half.unsqueeze(1) * V_new.T) / sqrt_s
            B.copy_(B_new.to(B.dtype))
            A.copy_(A_new.to(A.dtype))
        return loss
