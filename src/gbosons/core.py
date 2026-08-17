"""
gbosons.core -- bosonic g-sim engine (Heisenberg-Weyl / sp(2n) families).

Mirrors the spirit of Adelina's g-sim backends: the only inputs are the
structure constants (the CCR, encoded by the symplectic form Omega) and the
adjoint action (the symplectic transfer matrix S(t) = exp(t * Omega G)).
Expectation values of correlators are then multilinear contractions of S with
the input state's moment tensor -- the master formula

    <O(t)> = < w , S(t)^{otimes m} E^in > .

The corresponding validation is in ``notes/numerical_appendices_revised.tex``.

No Fock-space truncation anywhere in this module: everything is O(poly(n)) in
the number of modes n, independent of photon number cutoff.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------------------
# Structure constants = the CCR.  Holomorphic basis r = (a_1..a_n, a1d..and).
# ---------------------------------------------------------------------------
def omega_holo(n: int) -> np.ndarray:
    """Symplectic form for [r_i, r_j] = (Omega_a)_ij, r=(a..., a^dag...).

    [a_k, a_l^dag] = delta_kl  ->  top-right block +I, bottom-left -I.
    This matrix IS the table of structure constants of the HW algebra.
    """
    I = np.eye(n)
    Z = np.zeros((n, n))
    return np.block([[Z, I], [-I, Z]])


# ---------------------------------------------------------------------------
# Adjoint representation = symplectic transfer matrix.
# ---------------------------------------------------------------------------
def passive_transfer(h: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Heisenberg transfer for a passive (number-conserving) quadratic
    H = sum_kl h_kl a_k^dag a_l, h Hermitian.

    Heisenberg eq: da/dt = -i h a, so a(t) = W a with W = expm(-i h t) in U(n).
    Returns the n x n mode-transfer W (acts on the annihilation operators).
    """
    h = np.asarray(h, dtype=complex)
    return expm(-1j * h * t)


def bogoliubov_transfer(A: np.ndarray, B: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Heisenberg transfer S(t) = exp(t * M) on the holomorphic vector
    r=(a..., a^dag...), for the quadratic H = a^dag A a + (1/2)(a^dag B a^dag + h.c.)
    with A Hermitian and B symmetric.  The transfer generator is

        M = -i [[ A,  B], [-B*, -A*]]         (Heisenberg eq dr/dt = M r).

    The leading -i is essential: it turns a hopping block into a U(1) rotation
    and a pairing block into a hyperbolic squeeze.  Returns the 2n x 2n
    symplectic transfer S(t).
    """
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    M = -1j * np.block([[A, B], [-B.conj(), -A.conj()]])
    return expm(t * M)


def single_mode_squeeze_generator(xi: complex) -> tuple[np.ndarray, np.ndarray]:
    """Bogoliubov (A,B) blocks for single-mode squeezing
    H = (i/2)(conj(xi) a^2 - xi a^dag^2), i.e. (1/2)(B a^dag^2 + h.c.) with B = -i xi.
    Yields a(t)=cosh(|xi| t) a - e^{i arg xi} sinh(|xi| t) a^dag, so <n(t)>_vac=sinh^2(|xi| t).
    """
    return np.array([[0.0 + 0j]]), np.array([[-1j * xi]])


# ---------------------------------------------------------------------------
# Correlators on product number-state inputs (Fock |n_k> per mode).
# These are exact, O(poly(n)), and come from the cumulant collapse of the
# master formula (notes sec:wick).  No Fock cutoff.
# ---------------------------------------------------------------------------
def mean_photon_numbers(W: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """<n_i>(out) for input prod|occ_k>, output modes b_i = sum_k W_ik a_k.

    <n_i> = sum_k occ_k |W_ik|^2.
    """
    occ = np.asarray(occ, dtype=float)
    P = np.abs(W) ** 2                       # P[i,k] = |W_ik|^2
    return P @ occ


def number_number_correlator(W: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """Full matrix <n_i n_j>(out) for product number-state input prod|occ_k>,
    output modes b_i = sum_k W_ik a_k.  Exact for all i,j; the dense full-matrix
    implementation below costs O(n^3) through three matrix products.

    General closed form (derived from the ordered 4-point moment of a product
    number state; see notes eq:ninj):

        <n_i n_j> = delta_ij I_i
                    + sum_k occ_k(occ_k-1)|W_ik|^2|W_jk|^2
                    + I_i I_j + |C_ij|^2
                    - 2 sum_k occ_k^2 |W_ik|^2|W_jk|^2,

        I_i  = sum_k occ_k|W_ik|^2,      C_ij = sum_k occ_k W_ik^* W_jk.

    Off-diagonal (occ=1) reduces to  I_i I_j + |C_ij|^2 - 2 sum_k|W_ik|^2|W_jk|^2,
    the Hong-Ou-Mandel coincidence (=0 for a 50/50 beamsplitter).
    """
    occ = np.asarray(occ, dtype=float)
    P = np.abs(W) ** 2                          # |W_ik|^2
    I = P @ occ                                 # intensities <n_i>
    M = (W * occ[None, :]) @ W.conj().T         # M[i,j] = conj(C_ij), |M|^2=|C_ij|^2
    coh = np.abs(M) ** 2
    Q = (P * (occ ** 2)[None, :]) @ P.T         # sum_k occ_k^2 |W_ik|^2|W_jk|^2
    S2 = (P * (occ * (occ - 1.0))[None, :]) @ P.T  # bosonic bunching term
    nn = np.outer(I, I) + coh - 2.0 * Q + S2
    nn[np.diag_indices_from(nn)] += I           # delta_ij I_i
    return nn


# ---------------------------------------------------------------------------
# Gaussian-family two-point function via second moments (vacuum reference).
# ---------------------------------------------------------------------------
def mean_n_single_mode_vacuum(S: np.ndarray) -> float:
    """<n(t)> = <a^dag(t) a(t)> for SINGLE-MODE vacuum, from the 2x2 transfer S
    acting on (a, a^dag): a(t) = S[0,0] a + S[0,1] a^dag.

    Vacuum moments give <a^dag(t)a(t)> = |S[0,1]|^2. This is the single-mode
    formula only; a multimode 2n x 2n transfer needs the full V-block row
    sum_k |S[i, n+k]|^2, so the shape is checked here to prevent misuse.
    """
    if np.shape(S) != (2, 2):
        raise ValueError("mean_n_single_mode_vacuum expects a 2x2 transfer; "
                         "for n>1 sum |S[i, n+k]|^2 over the pairing block")
    return float(np.abs(S[0, 1]) ** 2)


def haar_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random U(n) (passive interferometer transfer)."""
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    ph = np.diagonal(r) / np.abs(np.diagonal(r))
    return q * ph[None, :]
