"""
gbosons.fock_ref -- brute-force truncated-Fock reference simulator.

This is the SLOW, ground-truth method: it builds the multimode Fock space
explicitly (dimension ~ cutoff^n) and computes expectation values directly.
Used only to VALIDATE the poly-time g-sim engine in core.py on small n.
It is exactly the method that becomes impossible at large n / large photon
number -- which is the whole point.
"""
from __future__ import annotations
import numpy as np
from functools import reduce
from scipy.linalg import expm


def _adag_a(cutoff: int):
    """Single-mode a, a^dag as (cutoff x cutoff) truncated matrices."""
    m = np.arange(1, cutoff)                                      # ladder weights sqrt(m)
    a = np.zeros((cutoff, cutoff))
    a[np.arange(cutoff - 1), np.arange(1, cutoff)] = np.sqrt(m)   # a|m> = sqrt(m)|m-1>
    return a, a.T


def _kron_list(mats):
    return reduce(np.kron, mats)


def multimode_ops(n_modes: int, cutoff: int):
    """Return lists [a_1..a_n], [a_1^dag..a_n^dag] on the full Fock space."""
    a1, ad1 = _adag_a(cutoff)
    I = np.eye(cutoff)
    a_ops, ad_ops = [], []
    for k in range(n_modes):
        mats_a = [a1 if j == k else I for j in range(n_modes)]
        mats_ad = [ad1 if j == k else I for j in range(n_modes)]
        a_ops.append(_kron_list(mats_a))
        ad_ops.append(_kron_list(mats_ad))
    return a_ops, ad_ops


def number_state(occ, cutoff: int) -> np.ndarray:
    """State vector for product Fock |occ_1,...,occ_n> in the truncated space."""
    vecs = []
    for m in occ:
        v = np.zeros(cutoff)
        v[m] = 1.0
        vecs.append(v)
    return _kron_list(vecs)


def passive_evolve_fock(W: np.ndarray, occ, cutoff: int):
    """Apply the passive interferometer W in the Heisenberg picture by
    transforming the OUTPUT mode operators b_i = sum_k W_ik a_k, and return
    the input state vector together with output number operators n_i^out.

    Heisenberg picture: <n_i^out> = <psi_in| b_i^dag b_i |psi_in>.
    """
    a_ops, ad_ops = multimode_ops(len(occ), cutoff)
    n = len(occ)
    b = [sum(W[i, k] * a_ops[k] for k in range(n)) for i in range(n)]
    bd = [sum(np.conj(W[i, k]) * ad_ops[k] for k in range(n)) for i in range(n)]
    nout = [bd[i] @ b[i] for i in range(n)]
    psi = number_state(occ, cutoff)
    return psi, nout


def nij_fock(W: np.ndarray, occ, cutoff: int | None = None) -> np.ndarray:
    """Brute-force <n_i n_j>(out) matrix on truncated Fock space.

    The truncation must hold the largest possible occupation. Photons can bunch
    into a single output mode, so a valid reference needs cutoff >= sum(occ)+1;
    if cutoff is None it is set to this minimum.
    """
    total = int(np.sum(occ))
    if cutoff is None:
        cutoff = total + 1
    if cutoff <= total:
        raise ValueError(f"cutoff={cutoff} too small for {total} photons; "
                         f"need cutoff >= {total + 1}")
    psi, nout = passive_evolve_fock(W, occ, cutoff)
    n = len(occ)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = np.real(psi.conj() @ (nout[i] @ (nout[j] @ psi)))
    return out


def kerr_circuit_fock(h: np.ndarray, chi: np.ndarray, occ, t: float, cutoff: int):
    """Brute-force full-Fock evolution of  H = sum_kl h_kl a_k^dag a_l
    + sum_{k<=l} chi_kl n_k n_l  on input |occ>, returning the <n_i n_j> matrix
    after time t.  Ground truth for the bounded-N sector simulator.

    Kerr conserves total photon number, so cutoff >= sum(occ)+1 is exact.
    """
    total = int(np.sum(occ))
    if cutoff <= total:
        raise ValueError(f"cutoff={cutoff} too small for {total} photons; "
                         f"need cutoff >= {total + 1}")
    n = len(occ)
    a_ops, ad_ops = multimode_ops(n, cutoff)
    num = [ad_ops[k] @ a_ops[k] for k in range(n)]
    dim = cutoff ** n
    H = np.zeros((dim, dim), dtype=complex)
    for k in range(n):
        for l in range(n):
            if abs(h[k, l]) > 0:
                H += h[k, l] * (ad_ops[k] @ a_ops[l])
    for k in range(n):
        for l in range(k, n):
            if abs(chi[k, l]) > 0:
                H += chi[k, l] * (num[k] @ num[l])
    U = expm(-1j * H * t)
    psi = U @ number_state(occ, cutoff)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = np.real(psi.conj() @ (num[i] @ (num[j] @ psi)))
    return out


def otoc_fock(h: np.ndarray, chi: np.ndarray, psi: np.ndarray, c: int,
              ts, cutoff: int) -> np.ndarray:
    """Brute-force full-Fock OTOC light-cone for the Bose-Hubbard-type Hamiltonian
    H = sum_kl h_kl a_k^dag a_l + sum_{k<=l} chi_kl n_k n_l.

    Returns C[t, i] = || [n_i(t), n_c] |psi> ||^2 with n_i(t) = e^{iHt} n_i e^{-iHt}
    -- the SAME squared-commutator the bounded-N sector simulator computes, but in
    the explicit cutoff**n Fock space. Ground truth for validating the g-sim OTOC.
    ``psi`` is the initial state in the full truncated Fock space (dim cutoff**n).
    """
    n = h.shape[0]
    a_ops, ad_ops = multimode_ops(n, cutoff)
    num = [ad_ops[k] @ a_ops[k] for k in range(n)]
    dim = cutoff ** n
    H = np.zeros((dim, dim), dtype=complex)
    for k in range(n):
        for l in range(n):
            if abs(h[k, l]) > 0:
                H += h[k, l] * (ad_ops[k] @ a_ops[l])
    for k in range(n):
        for l in range(k, n):
            if abs(chi[k, l]) > 0:
                H += chi[k, l] * (num[k] @ num[l])
    nc = num[c]
    ncpsi = nc @ psi
    ts = np.asarray(ts, dtype=float)
    C = np.zeros((len(ts), n))
    for ti, t in enumerate(ts):
        if t == 0:
            continue
        U = expm(-1j * H * t); Ud = U.conj().T
        Upsi, Uncpsi = U @ psi, U @ ncpsi
        for i in range(n):
            # [n_i(t), n_c]|psi> = U^dag n_i U (n_c|psi>) - n_c U^dag n_i U |psi>
            v = Ud @ (num[i] @ Uncpsi) - nc @ (Ud @ (num[i] @ Upsi))
            C[ti, i] = float(np.real(np.vdot(v, v)))
    return C


def squeeze_mean_n_fock(xi: complex, t: float, cutoff: int = 40) -> float:
    """Brute-force <n(t)> for single-mode squeezing on vacuum.
    H = (i/2)(conj(xi) a^2 - xi a^dag^2)."""
    a, ad = _adag_a(cutoff)
    H = 0.5j * (np.conj(xi) * (a @ a) - xi * (ad @ ad))
    U = expm(-1j * H * t)
    vac = np.zeros(cutoff); vac[0] = 1.0
    psi = U @ vac
    nop = ad @ a
    return float(np.real(psi.conj() @ (nop @ psi)))
