"""
gbosons.bounded_n -- fixed and bounded unions of photon-number sectors, the
bosonic counterpart of bounded-Hamming-weight U(1)-equivariant carriers.

Key point: number-conserving dynamics -- including GENUINELY NON-GAUSSIAN
nonlinearities (self-Kerr (a^dag a)^2, cross-Kerr n_k n_l) -- collapses to a
FINITE matrix on a fixed-total-photon sector

    H_N = span{ |n_1..n_n> : sum_k n_k = N },   dim d_{n,N} = C(N+n-1, N).

For bounded N this is poly(n) (Theta(n^N)), so the whole family is efficiently
simulable, and -- unlike the Gaussian/sp(2n) family -- it admits Kerr.

We expose two views:
  * Schrodinger on the sector (the efficient simulator: sparse expm_multiply
    on a poly-dim state vector -- no exponential Fock space).
  * the MGGM operator basis of u(d_{n,N}): matrix units E^{ab}=|a><b| with the
    O(1)-sparse structure constants [E^{ab},E^{cd}]=d_{bc}E^{ad}-d_{da}E^{cb},
    giving the g-sim adjoint primitive and (later) gradients.

The corresponding numerical account is in ``notes/numerical_demonstrations_revised.tex``.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
from itertools import combinations_with_replacement
from math import comb


def _dense(M):
    """Return M as a dense ndarray, whether it is sparse or already dense."""
    return M.toarray() if sp.issparse(M) else np.asarray(M)


# ---------------------------------------------------------------------------
# Fixed-photon-number sector basis.
# ---------------------------------------------------------------------------
def sector_basis(n: int, N: int):
    """All occupation vectors (n_1,...,n_n), n_k>=0, sum = N.

    Returns (basis, index, dim) where basis is a list of int tuples, index maps
    tuple -> row, dim = C(N+n-1, N).
    """
    basis = []
    # distribute N indistinguishable photons into n modes (stars and bars)
    for stars in combinations_with_replacement(range(n), N):
        occ = [0] * n
        for s in stars:
            occ[s] += 1
        basis.append(tuple(occ))
    # Descending lexicographic order, so |N,0,...,0> is row 0. The order is a
    # fixed contract: all downstream lookups go through `index`, so it may be
    # changed only if `index` is rebuilt consistently.
    basis.sort(reverse=True)
    index = {b: i for i, b in enumerate(basis)}
    return basis, index, len(basis)


def sector_dim(n: int, N: int) -> int:
    return comb(N + n - 1, N)


def sector_union_basis(n: int, sectors):
    """Basis for a finite direct sum of total-photon-number sectors.

    Returns ``(basis, index, blocks)``.  ``blocks[N]`` is the slice occupied by
    sector ``N`` in the concatenated basis.  The explicit block map makes the
    distinction between a carrier direct sum and its operator-space blocks
    available to experiments without relying on basis ordering conventions.
    """
    sectors = tuple(sorted({int(N) for N in sectors}))
    if not sectors or sectors[0] < 0:
        raise ValueError("sectors must be a nonempty collection of nonnegative integers")
    basis = []
    blocks = {}
    for N in sectors:
        block, _, _ = sector_basis(n, N)
        start = len(basis)
        basis.extend(block)
        blocks[N] = slice(start, len(basis))
    index = {b: i for i, b in enumerate(basis)}
    return basis, index, blocks


def sector_union_dim(n: int, sectors) -> int:
    """Carrier dimension of ``direct_sum_{N in sectors} H_N``."""
    return sum(sector_dim(n, int(N)) for N in sorted(set(sectors)))


# ---------------------------------------------------------------------------
# Number-conserving generators as sparse d x d matrices on the sector.
# ---------------------------------------------------------------------------
def hop_matrix(k: int, l: int, basis, index) -> sp.csr_matrix:
    """a_k^dag a_l restricted to the sector.

    a_k^dag a_l |..n_k..n_l..> = sqrt(n_l (n_k+1)) |..n_k+1..n_l-1..>  (k != l)
    a_k^dag a_k |b> = n_k |b>.
    """
    d = len(basis)
    rows, cols, vals = [], [], []
    for j, b in enumerate(basis):
        if k == l:
            if b[k] != 0:
                rows.append(j); cols.append(j); vals.append(float(b[k]))
            continue
        if b[l] == 0:
            continue
        nb = list(b)
        amp = np.sqrt(b[l] * (b[k] + 1))
        nb[l] -= 1; nb[k] += 1
        i = index[tuple(nb)]
        rows.append(i); cols.append(j); vals.append(amp)
    return sp.csr_matrix((vals, (rows, cols)), shape=(d, d))


def number_diag(k: int, basis) -> np.ndarray:
    """Diagonal of the number operator n_k on the sector."""
    return np.array([b[k] for b in basis], dtype=float)


def passive_hamiltonian(h: np.ndarray, basis, index) -> sp.csr_matrix:
    """H_LO = sum_kl h_kl a_k^dag a_l on the sector (h Hermitian -> H Hermitian).

    Built from one COO accumulation over all bonds, not a sum of per-bond CSR
    matrices, so the cost is O(nnz) once rather than O(nnz) per nonzero bond.
    """
    n = h.shape[0]
    d = len(basis)
    rows, cols, vals = [], [], []
    nonzero = np.argwhere(h != 0)
    for j, b in enumerate(basis):
        for k, l in nonzero:
            if k == l:
                if b[k]:
                    rows.append(j); cols.append(j); vals.append(h[k, k] * b[k])
            elif b[l]:
                nb = list(b); nb[l] -= 1; nb[k] += 1
                rows.append(index[tuple(nb)]); cols.append(j)
                vals.append(h[k, l] * np.sqrt(b[l] * (b[k] + 1)))
    return sp.csr_matrix((vals, (rows, cols)), shape=(d, d), dtype=complex)


def cross_kerr_hamiltonian(chi: np.ndarray, basis) -> sp.csr_matrix:
    """Number-diagonal NON-GAUSSIAN layer  H = sum_{k<=l} chi_kl n_k n_l.

    chi_kk is the coefficient of the self-Kerr n_k^2; for k != l, chi_kl is the
    coefficient of the cross-Kerr n_k n_l.  The matrix may be supplied in either
    triangle: off-diagonal entries are summed (chi_kl + chi_lk), so a lower- or
    full-matrix input gives the same operator as the upper-triangular one. These
    terms are quartic in a, a^dag -- outside the Gaussian family -- but diagonal
    in the Fock basis, hence finite on the sector.
    """
    n = chi.shape[0]
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


def number_conserving_hamiltonian(h: np.ndarray, chi: np.ndarray, basis, index):
    """Quadratic hopping plus number-diagonal Kerr terms on any sector union."""
    return passive_hamiltonian(h, basis, index) + cross_kerr_hamiltonian(chi, basis)


def annihilation_matrix(k: int, basis, index) -> sp.csr_matrix:
    """Restriction of ``a_k`` to a supplied finite union of sectors.

    Matrix elements whose target sector is absent are omitted.  The adjoint of
    the returned matrix is therefore the consistently projected creation map.
    """
    d = len(basis)
    rows, cols, vals = [], [], []
    for col, b in enumerate(basis):
        if b[k] == 0:
            continue
        nb = list(b); nb[k] -= 1
        row = index.get(tuple(nb))
        if row is not None:
            rows.append(row); cols.append(col); vals.append(np.sqrt(b[k]))
    return sp.csr_matrix((vals, (rows, cols)), shape=(d, d), dtype=complex)


def quadrature_matrix(k: int, basis, index) -> sp.csr_matrix:
    """Projected quadrature ``x_k=(a_k+a_k^dag)/sqrt(2)``."""
    a = annihilation_matrix(k, basis, index)
    return (a + a.getH()) / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Schrodinger evolution on the sector (the efficient simulator).
# ---------------------------------------------------------------------------
def basis_state(occ, index) -> np.ndarray:
    """Unit vector for |occ> in the sector."""
    v = np.zeros(len(index))
    v[index[tuple(occ)]] = 1.0
    return v.astype(complex)


def evolve(H: sp.spmatrix, t: float, psi0: np.ndarray) -> np.ndarray:
    """exp(-i H t) psi0 via Krylov action (no dense d x d exponential)."""
    return expm_multiply(-1j * H * t, psi0)


def n_expect(k: int, basis, psi: np.ndarray) -> float:
    return float(np.real(np.sum(number_diag(k, basis) * np.abs(psi) ** 2)))


def nn_expect(k: int, l: int, basis, psi: np.ndarray) -> float:
    nk = number_diag(k, basis); nl = number_diag(l, basis)
    return float(np.real(np.sum(nk * nl * np.abs(psi) ** 2)))


# ---------------------------------------------------------------------------
# MGGM operator layer: structure constants & adjoint primitive (g-sim view).
# ---------------------------------------------------------------------------
def adjoint_action(Hmat, Omat):
    """Infinitesimal adjoint  ad_H(O) = i[H, O]  on d x d operators (dense).

    This is Phi^ad(H) applied to the operator O in the matrix-unit basis of
    u(d): trivially i(H O - O H).  The structure constants of the matrix-unit
    basis are the elementary [E^ab,E^cd]=d_bc E^ad - d_da E^cb (O(1)-sparse),
    so this is the g-sim adjoint primitive; we apply it densely here only to
    cross-check the Schrodinger path on small sectors.
    """
    H, O = _dense(Hmat), _dense(Omat)
    return 1j * (H @ O - O @ H)


def heisenberg_observable(H, O, t: float):
    """Heisenberg-evolved observable O(t) = e^{iHt} O e^{-iHt} on the sector,
    via a dense matrix exponential of the (polynomial-size) sector Hamiltonian.

    Returns the d x d operator. Used to check that adjoint-space propagation
    reproduces the Schrodinger expectation values.
    """
    U = expm(-1j * _dense(H) * t)
    return U.conj().T @ _dense(O) @ U
