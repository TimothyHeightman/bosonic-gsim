"""Gauge-invariant lattice Chern diagnostics for isolated multiplets."""
from __future__ import annotations

import numpy as np


def _unit_link(left: np.ndarray, right: np.ndarray) -> complex:
    """Unit-modulus determinant link between two orthonormal frames."""
    sign, _ = np.linalg.slogdet(left.conj().T @ right)
    if abs(sign) == 0:
        raise ValueError("neighboring subspaces have singular overlap")
    return complex(sign / abs(sign))


def multiplet_chern(frames: np.ndarray) -> tuple[float, np.ndarray]:
    """Non-Abelian Fukui--Hatsugai--Suzuki Chern number.

    ``frames[ix, iy]`` is a ``(carrier_dimension, rank)`` orthonormal basis for
    the same isolated spectral multiplet at every point of a periodic twist
    grid.  The determinant link makes the result invariant under arbitrary
    unitary rotations within each frame.
    """
    if frames.ndim != 4:
        raise ValueError("frames must have shape (nx, ny, dimension, rank)")
    nx, ny = frames.shape[:2]
    ux = np.empty((nx, ny), dtype=complex)
    uy = np.empty((nx, ny), dtype=complex)
    for ix in range(nx):
        for iy in range(ny):
            ux[ix, iy] = _unit_link(frames[ix, iy], frames[(ix + 1) % nx, iy])
            uy[ix, iy] = _unit_link(frames[ix, iy], frames[ix, (iy + 1) % ny])
    curvature = np.empty((nx, ny), dtype=float)
    for ix in range(nx):
        for iy in range(ny):
            plaquette = (ux[ix, iy] * uy[(ix + 1) % nx, iy]
                          / (ux[ix, (iy + 1) % ny] * uy[ix, iy]))
            curvature[ix, iy] = np.angle(plaquette)
    return float(curvature.sum() / (2 * np.pi)), curvature
