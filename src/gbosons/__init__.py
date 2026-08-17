"""Bosonic g-sim: Lie-algebraic classical simulation on the Heisenberg-Weyl
algebra. See the paper in ``notes/`` for the underlying mathematics.

Submodules
----------
core        Gaussian / symplectic-transfer engine.
bounded_n   bounded-photon-number sector (number-conserving, including Kerr).
banded      photon-number-banded Fock (bounded-N perturbed by squeezing).
nilpotent_phase  exact bounded-degree phase-shear propagation.
lattices    Hofstadter and SSH single-particle hopping matrices.
fock_ref    brute-force truncated-Fock reference (ground truth).
plotting    shared figure style and experiment helpers.
"""
from . import (banded, benchmarking, bounded_n, core, fock_ref, lattices,
              nilpotent_phase, plotting)  # noqa: F401
