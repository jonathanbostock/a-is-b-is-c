"""Verification for pretrained_llms/riemannion.py — run: uv run python tests/test_riemannion.py

1. tangent_from_grads matches the DENSE tangent projection computed from a
   materialized ambient gradient (catches any transpose/inverse slip).
2. Parametrization invariance (the paper's headline property): (B,A) and
   (BQ, Q^-1 A) — same dW, same data — produce the SAME dW after Riemannion
   steps (incl. momentum), while naive per-factor SGD diverges between them.
3. Orthogonalization: nonzero singular values of the ortho'd tangent are ~1.
4. Descent + validity on a toy fixed-rank regression.
"""
import sys, torch
sys.path.insert(0, ".")
from pretrained_llms.riemannion import (
    RiemannionLoRA, _point_from_factors, _tangent_from_grads,
    _tangent_to_LR, _ortho_LR, _project_LR)

torch.manual_seed(0)
m, n, r, s = 23, 17, 4, 2.0

def make_pair(seed=1):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(m, r, generator=g) * 0.3
    A = torch.randn(r, n, generator=g) * 0.3
    return B.clone().requires_grad_(True), A.clone().requires_grad_(True)

def loss_fn(X, T):
    return ((X - T) ** 2).sum() + 0.3 * (X * T.roll(1, 0)).sum()

T = torch.randn(m, n)

# ---- 1. tangent projection vs dense ----
B, A = make_pair()
X = s * B @ A
loss = loss_fn(X, T)
loss.backward()
Xl = (s * B.detach() @ A.detach()).clone().requires_grad_(True)
loss_fn(Xl, T).backward()
G = Xl.grad                                   # dense ambient gradient
U, V, Sc, R_B = _point_from_factors(B.detach(), A.detach(), s)
M, U_p, V_p = _tangent_from_grads(U, V, Sc, R_B, A.grad, B.grad, s)
# dense projection
GV, UtG = G @ V, U.T @ G
M_d = UtG @ V
U_p_d = GV - U @ (U.T @ GV)
V_p_d = UtG.T - V @ (V.T @ UtG.T)
for got, want, nm in [(M, M_d, "M"), (U_p, U_p_d, "U_p"), (V_p, V_p_d, "V_p")]:
    err = (got - want).abs().max().item()
    assert err < 1e-4, f"tangent {nm} mismatch: {err}"
print("1. tangent projection matches dense       PASS")

# ---- 2. parametrization invariance over 3 steps (with momentum) ----
def run_steps(B, A, Q=None, steps=3, riem=True):
    if Q is not None:
        with torch.no_grad():
            B2 = (B @ Q).clone().requires_grad_(True)
            A2 = (torch.linalg.solve(Q, A)).clone().requires_grad_(True)
        B, A = B2, A2
    pair = [{"name": "t", "A": A, "B": B, "s": s}]
    opt = (RiemannionLoRA(pair, lr=0.05, momentum=0.9) if riem
           else torch.optim.SGD([A, B], lr=0.05, momentum=0.9))
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(s * B @ A, T).backward()
        opt.step()
    return (s * B @ A).detach()

Q = torch.linalg.qr(torch.randn(r, r))[0] @ torch.diag(torch.tensor([1.7, 0.4, 2.3, 0.9]))
B, A = make_pair(); X1 = run_steps(B, A)
B, A = make_pair(); X2 = run_steps(B, A, Q=Q)
riem_gap = (X1 - X2).abs().max().item()
B, A = make_pair(); Y1 = run_steps(B, A, riem=False)
B, A = make_pair(); Y2 = run_steps(B, A, Q=Q, riem=False)
naive_gap = (Y1 - Y2).abs().max().item()
print(f"   riemannion gap={riem_gap:.2e}  naive-SGD gap={naive_gap:.2e}")
assert riem_gap < 1e-3, f"Riemannion not parametrization-invariant: {riem_gap}"
assert naive_gap > 100 * max(riem_gap, 1e-9), "naive should differ across parametrizations"
print("2. parametrization invariance             PASS")

# ---- 3. orthogonalization gives unit singular values ----
L, R = _tangent_to_LR(U, V, M, U_p, V_p)
oL, oR = _ortho_LR(L, R)
sv = torch.linalg.svdvals(oL @ oR.T)
nz = sv[sv > 1e-6]
assert ((nz - 1).abs() < 1e-4).all(), f"ortho sv: {nz}"
print("3. msign singular values ~1               PASS")

# ---- 4. descent + validity on fixed-rank regression ----
Ttarget_low = (torch.randn(m, r) @ torch.randn(r, n))
B, A = make_pair(seed=7)
pair = [{"name": "d", "A": A, "B": B, "s": s}]
opt = RiemannionLoRA(pair, lr=0.15, momentum=0.9)
first = last = None
N = 300
for i in range(N):
    opt.zero_grad()
    l = ((s * B @ A - Ttarget_low) ** 2).mean()
    l.backward()
    opt.param_groups[0]['lr'] = 0.15 * (1 - i / N)   # scheduler-style decay via the group, as HF does
    opt.step()
    if i == 0: first = l.item()
    last = l.item()
assert last < 0.15 * first, f"no descent: {first} -> {last}"
assert torch.isfinite(B).all() and torch.isfinite(A).all()
assert torch.linalg.matrix_rank(s * B.detach() @ A.detach()) == r
print(f"4. descent {first:.3f} -> {last:.3f}, rank preserved   PASS")
print("ALL RIEMANNION TESTS PASS")
