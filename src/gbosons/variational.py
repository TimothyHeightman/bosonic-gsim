"""
gbosons.variational -- differentiable g-sim.

The adjoint formulation is differentiable: an observable evaluated through the
transfer matrix can be differentiated w.r.t. all Hamiltonian parameters at the
cost of one extra matrix-exponential Frechet evaluation -- the reverse-mode
(backprop) property -- so the whole simulable family supports exact-gradient
variational optimisation, classically.

Here we demonstrate it in the cleanest setting: a passive interferometer
S(theta) = exp(-i H(theta)), H(theta) = sum_p theta_p G_p with Hermitian
generators G_p, fed a single photon in mode ``src``. The output intensity
profile is I_i = |S[i, src]|^2 and the loss L = ||I - target||^2. The same
construction differentiates the bounded-N / non-Gaussian observables -- the
forward pass is the only thing that changes.

``loss_and_grad`` returns the exact gradient (reverse mode); ``finite_diff_grad``
is the slow ground-truth check, in the spirit of ``fock_ref``.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm, expm_frechet


def hermitian_basis(n: int) -> list[np.ndarray]:
    """Real-parameter basis of n x n Hermitian generators (dimension n^2):
    n diagonal, n(n-1)/2 real off-diagonal, n(n-1)/2 imaginary off-diagonal."""
    gens = []
    for k in range(n):
        G = np.zeros((n, n), dtype=complex); G[k, k] = 1.0
        gens.append(G)
    for k in range(n):
        for l in range(k + 1, n):
            G = np.zeros((n, n), dtype=complex); G[k, l] = 1.0; G[l, k] = 1.0
            gens.append(G)
            G = np.zeros((n, n), dtype=complex); G[k, l] = 1j; G[l, k] = -1j
            gens.append(G)
    return gens


def transfer(theta: np.ndarray, gens) -> np.ndarray:
    """Passive transfer S(theta) = exp(-i sum_p theta_p G_p) in U(n)."""
    H = sum(t * G for t, G in zip(theta, gens))
    return expm(-1j * H)


def intensities(S: np.ndarray, src: int) -> np.ndarray:
    """Single-photon output intensity profile I_i = |S[i, src]|^2 (sums to 1)."""
    return np.abs(S[:, src]) ** 2


def loss_and_grad(theta, gens, src, target):
    """Exact loss L = ||I - target||^2 and its reverse-mode gradient dL/dtheta.

    The gradient over ALL parameters costs one ``expm`` plus one ``expm_frechet``
    (the adjoint of the exponential), independent of the number of parameters.
    """
    theta = np.asarray(theta, dtype=float)
    H = sum(t * G for t, G in zip(theta, gens))
    A = -1j * H
    S = expm(A)
    I = np.abs(S[:, src]) ** 2
    diff = I - np.asarray(target, dtype=float)
    L = float(np.sum(diff ** 2))

    # cotangent dL/d conj(S): only column `src` enters the loss
    C_S = np.zeros_like(S)
    C_S[:, src] = 2.0 * diff * S[:, src]
    # reverse-mode through expm: adjoint of the Frechet derivative is the
    # Frechet derivative at A^H  ->  C_A = dL/d conj(A)
    C_A = expm_frechet(A.conj().T, C_S, compute_expm=False)
    # A = -i sum theta_p G_p  ->  dA/dtheta_p = -i G_p; theta_p real
    grad = np.array([2.0 * np.real(np.vdot(C_A, -1j * G)) for G in gens])
    return L, grad, I


def finite_diff_grad(theta, gens, src, target, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient -- slow ground truth for the check."""
    theta = np.asarray(theta, dtype=float)
    target = np.asarray(target, dtype=float)
    g = np.zeros(len(theta))
    for p in range(len(theta)):
        tp = theta.copy(); tp[p] += eps
        tm = theta.copy(); tm[p] -= eps
        Lp = np.sum((intensities(transfer(tp, gens), src) - target) ** 2)
        Lm = np.sum((intensities(transfer(tm, gens), src) - target) ** 2)
        g[p] = (Lp - Lm) / (2 * eps)
    return g


def evolve_loss_and_grad(theta, gens, psi0, weights, target, t):
    """Variational gradient through *interacting* sector dynamics.

    H(theta) = sum_p theta_p G_p with dense Hermitian sector generators G_p (e.g.
    hopping + on-site Kerr); the state is propagated psi_t = exp(-i t H) psi0 and
    read out through diagonal observables o_i = <psi_t| diag(weights[i]) |psi_t>.
    Loss = ||o - target||^2. The gradient over all parameters is exact and costs
    one ``expm`` plus one ``expm_frechet`` (the adjoint of the exponential) --
    identical machinery to the passive case, only the forward pass is interacting.
    Returns (L, grad, o).
    """
    theta = np.asarray(theta, dtype=float)
    H = sum(th * G for th, G in zip(theta, gens))
    A = -1j * t * H
    S = expm(A)
    psit = S @ psi0
    prob = np.abs(psit) ** 2
    o = weights @ prob
    diff = o - np.asarray(target, dtype=float)
    L = float(np.sum(diff ** 2))
    # reverse mode: psi_t -> S -> A -> theta
    psibar = 2.0 * (diff @ weights) * psit          # dL/d conj(psi_t)
    S_bar = np.outer(psibar, psi0.conj())           # dL/d conj(S)
    A_bar = expm_frechet(A.conj().T, S_bar, compute_expm=False)
    grad = np.array([2.0 * np.real(np.vdot(A_bar, -1j * t * G)) for G in gens])
    return L, grad, o


def finite_diff_evolve(theta, gens, psi0, weights, target, t, eps: float = 1e-6):
    """Central finite-difference gradient of the interacting loss -- ground truth."""
    theta = np.asarray(theta, dtype=float)

    def loss(th):
        S = expm(-1j * t * sum(x * G for x, G in zip(th, gens)))
        psit = S @ psi0
        return float(np.sum((weights @ np.abs(psit) ** 2 - target) ** 2))

    g = np.zeros(len(theta))
    for p in range(len(theta)):
        tp = theta.copy(); tp[p] += eps
        tm = theta.copy(); tm[p] -= eps
        g[p] = (loss(tp) - loss(tm)) / (2 * eps)
    return g


def layered_loss_and_grad(theta, blocks, psi0, weights, target):
    """Reverse-mode gradient through a *layered* ansatz
        psi = U_B ... U_2 U_1 psi0,   U_b = exp(-i sum_p theta^(b)_p G^(b)_p),
    where ``blocks`` is a list of generator-lists (one per block) and ``theta`` is
    the flat concatenation of the per-block parameters. Diagonal read-out
    o_i = <psi| diag(weights[i]) |psi>, loss = ||o - target||^2. Backprops through
    the product of exponentials; each block costs one expm + one expm_frechet.
    Returns (L, grad_flat, o)."""
    theta = np.asarray(theta, dtype=float)
    psis, Ss, As, sizes = [psi0], [], [], []
    idx = 0
    for gens in blocks:
        P = len(gens); th = theta[idx:idx + P]; idx += P; sizes.append(P)
        A = -1j * sum(x * G for x, G in zip(th, gens))
        S = expm(A)
        As.append(A); Ss.append(S); psis.append(S @ psis[-1])
    psit = psis[-1]
    o = weights @ np.abs(psit) ** 2
    diff = o - np.asarray(target, dtype=float)
    L = float(np.sum(diff ** 2))

    psibar = 2.0 * (diff @ weights) * psit          # dL/d conj(psi_final)
    grad = np.zeros(len(theta))
    idx = len(theta)
    for b in range(len(blocks) - 1, -1, -1):
        P = sizes[b]; idx -= P
        S_bar = np.outer(psibar, psis[b].conj())     # input to block b is psis[b]
        A_bar = expm_frechet(As[b].conj().T, S_bar, compute_expm=False)
        grad[idx:idx + P] = [2.0 * np.real(np.vdot(A_bar, -1j * G)) for G in blocks[b]]
        psibar = Ss[b].conj().T @ psibar             # propagate cotangent backwards
    return L, grad, o


def sparse_layered_loss_and_grad(theta, layer_gens, psi0, obs_diag, target, diags=None):
    """Exact reverse-mode (costate / adjoint) gradient through a layered ansatz of
    *sparse* single-generator exponentials -- the version that runs at large n,
    where dense ``expm`` is impossible. Each layer is U_l = exp(-i theta_l G_l)
    applied by ``expm_multiply`` (no dense matrix); diagonal observables
    o_m = sum_a obs_diag[m, a] |psi_a|^2; loss = ||o - target||^2.

    The whole gradient costs one forward sparse evolution + one backward sparse
    evolution (~2x the forward pass), independent of the number of parameters --
    the adjoint-state method. ``layer_gens`` are sparse Hermitian operators;
    ``obs_diag`` is (m, d) with the diagonal of each observable.
    """
    from scipy.sparse.linalg import expm_multiply
    theta = np.asarray(theta, dtype=float)
    L = len(layer_gens)
    # ``diags[l]`` is the diagonal of layer l if it is a number-diagonal generator
    # (Kerr / on-site potential), else None -> then exp acts elementwise (cheap) and
    # only the genuinely off-diagonal layers (hopping) pay for ``expm_multiply``.
    def fwd(l, v):
        if diags is not None and diags[l] is not None:
            return np.exp(-1j * theta[l] * diags[l]) * v
        return expm_multiply(-1j * theta[l] * layer_gens[l], v)

    def bwd(l, v):
        if diags is not None and diags[l] is not None:
            return np.exp(1j * theta[l] * diags[l]) * v
        return expm_multiply(1j * theta[l] * layer_gens[l], v)

    psis = [np.asarray(psi0, dtype=complex)]
    for l in range(L):
        psis.append(fwd(l, psis[-1]))
    psiL = psis[-1]
    o = obs_diag @ np.abs(psiL) ** 2
    diff = o - np.asarray(target, dtype=float)
    Lval = float(np.sum(diff ** 2))

    lam = 2.0 * (diff @ obs_diag) * psiL                # costate dL/d conj(psi_L)
    grad = np.zeros(L)
    for l in range(L - 1, -1, -1):
        if diags is not None and diags[l] is not None:
            Gpsi = diags[l] * psis[l + 1]
        else:
            Gpsi = layer_gens[l] @ psis[l + 1]
        grad[l] = 2.0 * np.imag(np.vdot(lam, Gpsi))     # 2 Im <lam|G_l|psi_l>
        lam = bwd(l, lam)
    return Lval, grad, o


def adam(theta0, lossgrad, steps: int, lr: float = 0.05, mask=None,
         b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    """Adam on ``lossgrad(theta) -> (L, grad, obs)``. ``mask`` (bool array) freezes
    parameters (their gradient is zeroed, so they stay at their initial value).
    Returns (theta, loss_history, final_obs)."""
    th = np.asarray(theta0, dtype=float).copy()
    mask = np.ones_like(th, dtype=bool) if mask is None else np.asarray(mask, bool)
    m = np.zeros_like(th); v = np.zeros_like(th); hist = []
    o = None
    for s in range(1, steps + 1):
        L, g, o = lossgrad(th)
        hist.append(L)
        g = g * mask
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        th = th - lr * (m / (1 - b1 ** s)) / (np.sqrt(v / (1 - b2 ** s)) + eps)
    L, _, o = lossgrad(th); hist.append(L)
    return th, np.array(hist), o


def optimize(theta0, gens, src, target, lr: float = 0.5, steps: int = 200):
    """Plain gradient descent using the exact reverse-mode gradient.
    Returns (theta, loss_history, final_intensity)."""
    theta = np.asarray(theta0, dtype=float).copy()
    hist = []
    I = None
    for _ in range(steps):
        L, grad, I = loss_and_grad(theta, gens, src, target)
        hist.append(L)
        theta = theta - lr * grad
    L, _, I = loss_and_grad(theta, gens, src, target)
    hist.append(L)
    return theta, np.array(hist), I
