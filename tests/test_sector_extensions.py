import unittest
from math import comb

import numpy as np

from gbosons import banded as bd
from gbosons import bounded_n as bn
from gbosons import nilpotent_phase as nph


class SectorUnionTests(unittest.TestCase):
    def test_bounded_union_dimension(self):
        for n in (1, 3, 6):
            for cutoff in (0, 1, 3):
                sectors = range(cutoff + 1)
                self.assertEqual(
                    bn.sector_union_dim(n, sectors),
                    comb(n + cutoff, cutoff),
                )

    def test_number_conserving_hamiltonian_is_block_diagonal(self):
        n = 4
        basis, index, blocks = bn.sector_union_basis(n, [0, 1, 2])
        rng = np.random.default_rng(7)
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        h = (A + A.conj().T) / 2
        chi = np.diag(np.linspace(0.1, 0.4, n))
        H = bn.number_conserving_hamiltonian(h, chi, basis, index).toarray()
        for N, block_N in blocks.items():
            for M, block_M in blocks.items():
                if N != M:
                    self.assertLess(np.max(np.abs(H[block_N, block_M])), 1e-14)

    def test_quadrature_connects_adjacent_sectors(self):
        basis, index, _ = bn.sector_union_basis(3, [0, 1, 2])
        X = bn.quadrature_matrix(1, basis, index).toarray()
        self.assertLess(np.max(np.abs(X - X.conj().T)), 1e-14)
        nz = np.argwhere(np.abs(X) > 0)
        for row, col in nz:
            self.assertEqual(abs(sum(basis[row]) - sum(basis[col])), 1)

    def test_sparse_hamiltonian_matches_direct_construction(self):
        n, N = 4, 2
        basis, index, _ = bn.sector_basis(n, N)
        rng = np.random.default_rng(3)
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        h = (A + A.conj().T) / 2
        expected = np.zeros((len(basis), len(basis)), dtype=complex)
        for k in range(n):
            for l in range(n):
                expected += h[k, l] * bn.hop_matrix(k, l, basis, index).toarray()
        actual = bn.passive_hamiltonian(h, basis, index).toarray()
        np.testing.assert_allclose(actual, expected, atol=1e-13)


class SqueezingBandTests(unittest.TestCase):
    def test_reachable_sectors_preserve_parity(self):
        self.assertEqual(bd.reachable_sectors(2, 2), (0, 2, 4, 6))
        self.assertEqual(bd.reachable_sectors(1, 2), (1, 3, 5))
        self.assertEqual(bd.reachable_band_dim(3, 2, 1),
                         bn.sector_dim(3, 0) + bn.sector_dim(3, 2)
                         + bn.sector_dim(3, 4))

    def test_projected_squeezer_is_hermitian(self):
        basis, index, _ = bd.reachable_basis(3, 2, 2)
        S = bd.squeeze(0, 1, 0.3, basis, index).toarray()
        np.testing.assert_allclose(S, S.conj().T, atol=1e-14)


class NilpotentPhaseTests(unittest.TestCase):
    def test_dimension_formula(self):
        self.assertEqual(nph.algebra_dimension(1, 3), 5)
        self.assertEqual(nph.algebra_dimension(2, 3), 12)
        self.assertEqual(nph.algebra_dimension(4, 3), 39)

    def test_translation_substitution(self):
        polynomial = nph.add(nph.monomial((2,), 2.0), nph.monomial((1,), -1.0))
        shifted = nph.shift(polynomial, (0.5,))
        expected = nph.add(nph.monomial((2,), 2.0), nph.monomial((1,), 1.0))
        for exponent in set(shifted) | set(expected):
            self.assertAlmostEqual(shifted.get(exponent, 0.0),
                                   expected.get(exponent, 0.0))

    def test_effective_phase_respects_gate_order(self):
        cubic = nph.monomial((3,), 0.2)
        gates = [
            {"kind": "translation", "shift": (0.4,)},
            {"kind": "phase", "potential": cubic},
        ]
        np.testing.assert_equal(nph.effective_phase(1, gates), nph.shift(cubic, (0.4,)))

    def test_vacuum_momentum_moments(self):
        self.assertAlmostEqual(nph.vacuum_momentum_moment({}, (0, 0)).real, 0.5)
        self.assertAlmostEqual(nph.vacuum_momentum_moment({}, (0, 0, 0, 0)).real,
                               0.75)
        self.assertAlmostEqual(nph.momentum_cumulant4({}).real, 0.0)
        cubic = nph.monomial((3,), 0.1)
        self.assertAlmostEqual(nph.vacuum_momentum_moment(cubic, (0,)).real, -0.15)
        self.assertGreater(abs(nph.momentum_cumulant4(cubic).real), 1e-3)

    def test_dense_cubic_translation(self):
        rng = np.random.default_rng(9)
        n = 3
        linear = rng.standard_normal(n)
        quadratic = rng.standard_normal((n, n))
        quadratic = (quadratic + quadratic.T) / 2
        cubic = rng.standard_normal((n, n, n))
        cubic = sum(cubic.transpose(order) for order in
                    ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                     (1, 2, 0), (2, 0, 1), (2, 1, 0))) / 6
        coefficients = (0.7, linear, quadratic, cubic)
        shift = rng.standard_normal(n)
        point = rng.standard_normal(n)
        translated = nph.translate_cubic_tensors(coefficients, shift)
        self.assertAlmostEqual(
            nph.evaluate_cubic_tensors(translated, point),
            nph.evaluate_cubic_tensors(coefficients, point + shift),
        )


if __name__ == "__main__":
    unittest.main()
