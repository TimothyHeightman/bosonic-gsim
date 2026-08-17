import json
from pathlib import Path
import unittest

import numpy as np

from experiments.extended_results.common import (
    DATA_DIR, OUTPUT_DIR, ROOT, output_path, sha256)
from gbosons.plotting import DARK_PALETTE, LIGHT_PALETTE, PAPER_COLORS


class ExtendedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA_DIR / "manifest.json").read_text())

    def test_production_bundle_is_validated(self):
        self.assertTrue(self.manifest["validation_passed"])
        self.assertEqual(self.manifest["profile"], "production")
        self.assertEqual(
            set(self.manifest["task_counts"]),
            {"bounded", "doublon", "otoc", "topology",
             "control", "cubic", "squeezing"})
        self.assertTrue(all(check["passed"]
                            for check in self.manifest["validation_checks"]))

    def test_protected_sources_are_unchanged(self):
        for relative, expected in self.manifest["protected_source_checksums"].items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)

    def test_output_guard(self):
        self.assertEqual(output_path("example.pdf").parent, OUTPUT_DIR.resolve())
        for invalid in ("../fig3_bounded_n.pdf", "fig3_bounded_n.png", "/tmp/a.pdf"):
            with self.assertRaises(ValueError):
                output_path(invalid)

    def test_publication_palette(self):
        self.assertEqual(
            DARK_PALETTE,
            ("#4863A0", "#829F82", "#7F525D", "#C19A6B", "#CD5C5C"))
        self.assertEqual(
            LIGHT_PALETTE,
            ("#659EC7", "#9CB071", "#C08081", "#DEB887", "#E77471"))
        self.assertEqual(PAPER_COLORS["method"], DARK_PALETTE[0])
        self.assertEqual(PAPER_COLORS["baseline"], DARK_PALETTE[1])
        self.assertEqual(PAPER_COLORS["nonlinear"], DARK_PALETTE[2])

    def test_selected_otoc_fields(self):
        summary = json.loads((DATA_DIR / "otoc.json").read_text())
        self.assertEqual(summary["task_count"], 72)
        for task_id, interaction in ((54, 0.0), (63, 8.0)):
            record = json.loads(
                (DATA_DIR / f"task-{task_id:05d}.json").read_text())
            self.assertEqual(record["task"]["modes"], 400)
            self.assertEqual(record["task"]["U"], interaction)
            with np.load(DATA_DIR / record["arrays"]) as arrays:
                self.assertEqual(arrays["otoc"].shape, (48, 400))
                self.assertTrue(np.all(np.isfinite(arrays["otoc"])))

    def test_topology_certification_count(self):
        summary = json.loads((DATA_DIR / "topology.json").read_text())
        multiplets = summary["multiplets"]
        self.assertEqual(len(multiplets), 22)
        certified = [
            row for row in multiplets
            if row["minimum_gap"] > 1e-8 and row["integer_distance"] < 0.05
        ]
        self.assertEqual(len(certified), 21)
        failed = [row for row in multiplets if row not in certified]
        self.assertEqual(
            (failed[0]["L"], failed[0]["U"], failed[0]["flux"], failed[0]["grid"]),
            (12, 8.0, 0.25, 8))
        for row in certified:
            self.assertAlmostEqual(row["chern"], 4 * np.sign(row["flux"]), places=8)
        displayed = [
            row for row in multiplets
            if row["U"] == 10.0 and row["grid"] == 8
            and row["L"] in {8, 12, 16}
            and row["flux"] in {0.25, -0.25}
        ]
        self.assertEqual(len(displayed), 6)
        self.assertTrue(all(row["minimum_gap"] > 0.088 for row in displayed))
        self.assertTrue(all(
            abs(row["chern"] - 4 * np.sign(row["flux"])) < 1e-8
            for row in displayed))

    def test_control_and_squeezing_coverage(self):
        control = json.loads((DATA_DIR / "control.json").read_text())
        statistics = [
            row for row in control["optimization_statistics"]
            if row["geometry"] == 0
        ]
        self.assertTrue(statistics)
        self.assertTrue(all(row["count"] == 30 for row in statistics))
        transition = next(
            row for row in statistics
            if row["L"] == 5 and row["depth"] == 3 and row["model"] == "kerr")
        self.assertAlmostEqual(
            transition["fraction_above_passive_bound"], 12 / 30)
        successful = [
            row["target_probability"] for row in control["tasks"]
            if row["geometry"] == 0 and row["L"] == 5
            and row["depth"] == 3 and row["model"] == "kerr"
            and row["target_probability"] > 0.5
        ]
        self.assertEqual(len(successful), 12)
        self.assertAlmostEqual(min(successful), 0.5032153801754323)
        self.assertAlmostEqual(max(successful), 0.7934500645533568)

        squeezing = json.loads((DATA_DIR / "squeezing.json").read_text())
        for order in range(4):
            rows = [row for row in squeezing["fits"] if row["order"] == order]
            self.assertEqual({row["modes"] for row in rows}, {2, 4, 6, 8, 12})
            self.assertTrue(all(row["leakage_points"] >= 3 for row in rows))
            self.assertTrue(all(row["observable_points"] >= 3 for row in rows))
            self.assertTrue(all(
                abs(row["leakage_slope"] - (order + 1)) < 0.02
                for row in rows))
            self.assertTrue(all(
                abs(row["observable_slope"] - 2 * (order + 1)) < 0.03
                for row in rows))
        unresolved = [row for row in squeezing["fits"] if row["order"] == 4]
        self.assertTrue(all(row["observable_points"] < 3 for row in unresolved))

    def test_extended_figures_exist_separately(self):
        expected = {
            "fig1_gaussian_extended.pdf",
            "fig2_beyond_gaussian_extended.pdf",
            "fig3_bounded_n_extended.pdf",
            "fig4_doublon_extended.pdf",
            "fig5_nilpotent_phase_extended.pdf",
            "fig6_otoc_alternative.pdf",
            "fig6_otoc_extended.pdf",
            "fig7b_doublon_chern_extended.pdf",
            "fig10_topological_doublon_extended.pdf",
            "fig9_squeezing_extended.pdf",
            "fig12_kerr_control_extended.pdf",
            "fig8_chiral_transport_extended.pdf",
        }
        self.assertEqual(
            {path.name for path in OUTPUT_DIR.glob("*.pdf")},
            expected)
        self.assertTrue(all((OUTPUT_DIR / name).stat().st_size > 10_000
                            for name in expected))


if __name__ == "__main__":
    unittest.main()
