import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def result(name):
    with open(ROOT / "experiments" / name / "results.json") as handle:
        return json.load(handle)


class NumericalResultTests(unittest.TestCase):
    def test_exact_reference_errors(self):
        self.assertLess(result("fig1_gaussian")["squeezing_max_fock_error"], 1e-10)
        self.assertLess(result("fig2_beyond_gaussian")["random_interferometer_max_fock_error"],
                        1e-10)
        self.assertLess(result("fig3_bounded_n")["fixed_sector_max_fock_error"], 1e-10)
        self.assertLess(result("fig4_doublon")["validation_max_fock_error"], 1e-10)
        self.assertLess(result("fig6_otoc")["validation_max_fock_error"], 1e-10)
        cubic = result("fig5_nilpotent_phase")
        self.assertLess(cubic["single_mode"]["max_module_grid_error"], 1e-8)
        self.assertLess(cubic["two_mode"]["max_module_grid_error"], 1e-8)
        self.assertLess(cubic["single_mode"]["max_grid_refinement_gap"], 1e-8)
        self.assertLess(cubic["two_mode"]["max_grid_refinement_gap"], 1e-8)

    def test_mixed_sector_readout_distinction(self):
        mixed = result("fig3_bounded_n")["mixed_sector"]
        self.assertLess(mixed["max_number_difference"], 1e-10)
        self.assertGreater(mixed["max_quadrature_separation"], 0.1)

    def test_doublon_strong_coupling_power(self):
        exponent = result("fig4_doublon")["strong_coupling"]["fit_exponent"]
        self.assertLess(abs(exponent + 1.0), 0.15)

    def test_squeezing_reference_and_orders(self):
        data = result("fig9_squeezing")
        self.assertLess(data["max_successive_reference_gap"], 1e-10)
        for fit in data["error_scaling"]["fits"]:
            self.assertLess(abs(fit["fitted_slope"] - fit["expected_slope"]), 0.1)

    def test_nilpotent_phase_signals_are_nontrivial(self):
        data = result("fig5_nilpotent_phase")
        self.assertGreater(abs(data["single_mode"]["final_cumulant"]), 1e-3)
        self.assertGreater(abs(data["two_mode"]["final_connected_correlator"]), 1e-3)

    def test_gradient_and_qualified_boundary_claims(self):
        self.assertLess(result("fig12_kerr_control_2d")["gradient_max_absolute_error"],
                        1e-8)
        self.assertFalse(result("fig7_doublon_topology")["computed_invariant"])
        self.assertFalse(result("fig8_chiral_transport")["computed_invariant"])


if __name__ == "__main__":
    unittest.main()
