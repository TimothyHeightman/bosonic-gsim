"""
gbosons.banded -- photon-number-banded Fock simulator.

Realises finite projected calculations for bounded-N dynamics perturbed by
squeezing.  Since a physical squeezer changes total photon number by two, the
order-k reachable carrier contains only the parity-compatible sectors reached
in at most k steps.  A sufficiently large projected carrier can provide a
numerically converged reference, but is not labelled exact without a separate
error certificate.
"""
from itertools import combinations_with_replacement
import numpy as np
import scipy.sparse as sp
from . import bounded_n as bn


def banded_basis(n: int, Nmin: int, Nmax: int):
    """Occupation vectors on n modes with total photon number in [Nmin, Nmax]."""
    basis = []
    for N in range(max(0, Nmin), Nmax + 1):
        for stars in combinations_with_replacement(range(n), N):
            occ = [0] * n
            for s in stars:
                occ[s] += 1
            basis.append(tuple(occ))
    basis.sort(reverse=True)
    return basis, {b: i for i, b in enumerate(basis)}


def reachable_sectors(N: int, k: int):
    """Photon sectors reachable from ``H_N`` by at most ``k`` pair changes."""
    if N < 0 or k < 0:
        raise ValueError("N and k must be nonnegative")
    lower = max(0, N - 2 * k)
    if (lower - N) % 2:
        lower += 1
    return tuple(range(lower, N + 2 * k + 1, 2))


def reachable_basis(n: int, N: int, k: int):
    """Parity-aware order-k carrier, including an explicit sector-block map."""
    return bn.sector_union_basis(n, reachable_sectors(N, k))


def numdiag(k, basis):
    return np.array([b[k] for b in basis], dtype=float)


def hopping(h, basis, index):
    """sum_kl h_kl a_k^dag a_l on the banded basis (h Hermitian -> Hermitian)."""
    n = len(basis[0]); d = len(basis)
    R, C, V = [], [], []
    for col, b in enumerate(basis):
        for k in range(n):
            for l in range(n):
                if h[k, l] == 0:
                    continue
                if k == l:
                    if b[k]:
                        # diagonal of a Hermitian H is real; take .real so a
                        # complex-diagonal input cannot make the result non-Hermitian
                        R.append(col); C.append(col); V.append(np.real(h[k, k]) * b[k])
                elif b[l]:
                    nb = list(b); amp = np.sqrt(b[l] * (b[k] + 1)); nb[l] -= 1; nb[k] += 1
                    t = index.get(tuple(nb))
                    if t is not None:
                        R.append(t); C.append(col); V.append(h[k, l] * amp)
    return sp.csr_matrix((V, (R, C)), shape=(d, d), dtype=complex)


def cross_kerr(chi, basis):
    """sum_{k<=l} chi_kl n_k n_l (number-diagonal). chi may be given in either
    triangle: off-diagonal entries are summed (chi_kl + chi_lk)."""
    n = len(basis[0])
    occ = np.array(basis, dtype=float)            # (d, n) occupation matrix
    diag = np.zeros(len(basis))
    for k in range(n):
        if chi[k, k]:
            diag += chi[k, k] * occ[:, k] ** 2
        for l in range(k + 1, n):
            c = chi[k, l] + chi[l, k]
            if c:
                diag += c * occ[:, k] * occ[:, l]
    return sp.diags(diag).tocsr()


def squeeze(i, j, r, basis, index):
    """(r/2)(a_i a_j + a_i^dag a_j^dag), Hermitian; i=j gives single-mode
    (r/2)(a_i^2 + a_i^{dag 2}). Shifts total photon number by +/-2; matrix
    elements to out-of-band targets are dropped (the band truncation).

    The projected operator stays Hermitian because every retained transition
    has both endpoints in the supplied basis, and the reverse transition is
    emitted when the loop visits the other endpoint.
    """
    d = len(basis); R, C, V = [], [], []
    for col, b in enumerate(basis):
        # lowering a_i a_j
        if i == j:
            ok = b[i] >= 2; amp = np.sqrt(b[i] * (b[i] - 1)) if ok else 0.0
        else:
            ok = b[i] >= 1 and b[j] >= 1; amp = np.sqrt(b[i] * b[j]) if ok else 0.0
        if ok:
            nb = list(b); nb[i] -= 1; nb[j] -= 1
            t = index.get(tuple(nb))
            if t is not None:
                R.append(t); C.append(col); V.append(0.5 * r * amp)
        # raising a_i^dag a_j^dag
        if i == j:
            amp = np.sqrt((b[i] + 1) * (b[i] + 2))
        else:
            amp = np.sqrt((b[i] + 1) * (b[j] + 1))
        nb = list(b); nb[i] += 1; nb[j] += 1
        t = index.get(tuple(nb))
        if t is not None:
            R.append(t); C.append(col); V.append(0.5 * r * amp)
    return sp.csr_matrix((V, (R, C)), shape=(d, d), dtype=complex)


def basis_state(occ, index, d):
    v = np.zeros(d, dtype=complex); v[index[tuple(occ)]] = 1.0
    return v


def nn_expect(i, j, basis, psi):
    return float(np.real(np.sum(numdiag(i, basis) * numdiag(j, basis) * np.abs(psi) ** 2)))


def band_dim(n, N, k):
    """Legacy full-interval dimension for ``[N-2k,N+2k]``.

    Use :func:`reachable_band_dim` for a squeezing-reachable parity band.
    """
    from math import comb
    return sum(comb(M + n - 1, M) for M in range(max(0, N - 2 * k), N + 2 * k + 1))


def reachable_band_dim(n, N, k):
    """Dimension of the parity-aware order-k squeezing carrier."""
    return bn.sector_union_dim(n, reachable_sectors(N, k))
