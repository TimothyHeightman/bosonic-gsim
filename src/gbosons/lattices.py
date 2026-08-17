"""
gbosons.lattices -- lattice single-particle hopping matrices for the bounded
photon-number engine.  These feed bounded_n.passive_hamiltonian (which accepts
a complex Hermitian hopping matrix) and bounded_n.cross_kerr_hamiltonian.

The two-dimensional Hofstadter helpers support finite-size studies of interacting
bound-pair spectra and boundary transport.  Spectral edge localisation and
flux-reversed motion alone are not treated here as a computation of a topological
invariant; related Chern-number analyses appear in arXiv:2310.09565 and
arXiv:2502.05847.
"""
from __future__ import annotations
import numpy as np


def hofstadter(Lx: int, Ly: int, flux: float = 0.25, t: float = 1.0,
               disorder: float = 0.0, seed: int = 0):
    """Square-lattice Hofstadter hopping matrix, open boundaries (edges exist).

    Landau gauge A_y = 2*pi*flux*x: x-hops are real -t, y-hops carry the Peierls
    phase exp(i 2*pi*flux*x).  Returns (h, xy) where h is the (M x M) complex
    Hermitian single-particle matrix, M = Lx*Ly, and xy[s] = (x, y).
    Optional on-site disorder of half-width `disorder` (uniform) is placed on the
    diagonal.

    Convention: with this gauge and traversal sense the accumulated phase around
    a plaquette is -2*pi*flux, so the effective flux per plaquette is -flux
    (mod 1). The magnitude is `flux` and the chirality reverses with its sign; if
    an external Chern-number sign convention matters, fix it against this.
    """
    M = Lx * Ly
    idx = lambda x, y: x * Ly + y
    xy = np.array([(s // Ly, s % Ly) for s in range(M)])
    h = np.zeros((M, M), dtype=complex)
    for x in range(Lx):
        for y in range(Ly):
            s = idx(x, y)
            if x + 1 < Lx:                                   # x-bond: real
                h[s, idx(x + 1, y)] += -t
                h[idx(x + 1, y), s] += -t
            if y + 1 < Ly:                                   # y-bond: Peierls phase
                ph = np.exp(1j * 2 * np.pi * flux * x)
                h[s, idx(x, y + 1)] += -t * ph
                h[idx(x, y + 1), s] += -t * np.conj(ph)
    if disorder > 0:
        rng = np.random.default_rng(seed)
        h[np.diag_indices(M)] += rng.uniform(-disorder, disorder, M)
    return h, xy


def hofstadter_torus(Lx: int, Ly: int, flux: float = 0.25, t: float = 1.0,
                     theta_x: float = 0.0, theta_y: float = 0.0):
    """Hofstadter hopping on a torus with boundary twists.

    The Landau-gauge convention matches :func:`hofstadter`.  ``Lx * flux``
    must be integral so that the Peierls phases are compatible with the
    periodic x boundary.  Boundary crossings acquire ``exp(i theta_x)`` or
    ``exp(i theta_y)`` in the positive coordinate direction.
    """
    if not np.isclose(Lx * flux, round(Lx * flux), atol=1e-12):
        raise ValueError("Lx * flux must be integral on the Landau-gauge torus")
    M = Lx * Ly
    idx = lambda x, y: (x % Lx) * Ly + (y % Ly)
    xy = np.array([(s // Ly, s % Ly) for s in range(M)])
    h = np.zeros((M, M), dtype=complex)
    for x in range(Lx):
        for y in range(Ly):
            s = idx(x, y)
            xp = idx(x + 1, y)
            yp = idx(x, y + 1)
            phase_x = np.exp(1j * theta_x) if x == Lx - 1 else 1.0
            phase_y = np.exp(1j * theta_y) if y == Ly - 1 else 1.0
            amp_x = -t * phase_x
            amp_y = -t * np.exp(1j * 2 * np.pi * flux * x) * phase_y
            h[s, xp] += amp_x
            h[xp, s] += np.conj(amp_x)
            h[s, yp] += amp_y
            h[yp, s] += np.conj(amp_y)
    return h, xy


def ssh(n: int, t1: float = 0.4, t2: float = 1.0,
        disorder: float = 0.0, seed: int = 0):
    """1D Su-Schrieffer-Heeger chain: alternating hoppings t1 (intracell, even
    bonds) and t2 (intercell, odd bonds), open boundaries.  t2>t1 is the
    topological phase (edge states), t1>t2 the trivial phase.  Returns (h, x)
    with x the site index 0..n-1.  Optional on-site disorder on the diagonal.
    With on-site Kerr, the bound pair (doublon) inherits an effective SSH model
    and a topological doublon edge state (arXiv:2502.05847).
    """
    h = np.zeros((n, n), dtype=complex)
    for i in range(n - 1):
        amp = -(t1 if i % 2 == 0 else t2)
        h[i, i + 1] += amp
        h[i + 1, i] += amp
    if disorder > 0:
        rng = np.random.default_rng(seed)
        h[np.diag_indices(n)] += rng.uniform(-disorder, disorder, n)
    return h, np.arange(n)


def edge_sites(xy, Lx: int, Ly: int):
    """Boolean mask of boundary sites of the open lattice."""
    return np.array([(x == 0 or x == Lx - 1 or y == 0 or y == Ly - 1)
                     for x, y in xy])


def chiral_winding(coms: np.ndarray, center: np.ndarray) -> float:
    """Signed angle (in turns) swept by a sequence of centre-of-mass points
    `coms` (shape (T,2)) about `center` -- a scalar chirality/winding measure."""
    v = coms - center[None, :]
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    return (ang[-1] - ang[0]) / (2 * np.pi)
