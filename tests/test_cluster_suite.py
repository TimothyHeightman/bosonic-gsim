import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.cluster import cli
from experiments.cluster.manifest import build
from experiments.cluster.tasks import _sector_otoc_chunked, bose_hubbard
from gbosons import bounded_n as bn
from gbosons import lattices
from gbosons import nilpotent_phase as nph
from gbosons.topology import multiplet_chern


class ClusterScriptTests(unittest.TestCase):
    def test_submitted_jobs_do_not_resolve_helpers_from_slurm_spool(self):
        root = Path(__file__).resolve().parents[1]
        submitted = [
            "bounded_scaling.sh", "doublon_scaling.sh", "otoc_scaling.sh",
            "doublon_chern.sh", "kerr_control.sh", "nilpotent_phase.sh",
            "squeezing_bands.sh", "run_experiment.sh", "aggregate.sh", "validate.sh",
        ]
        for name in submitted:
            script = (root / "cluster" / "jobs" / name).read_text()
            self.assertIn("GBOSONS_REPO_ROOT", script, name)
            self.assertNotIn('dirname "$0"', script, name)

        submit_script = (root / "cluster" / "submit_all.sh").read_text()
        self.assertEqual(submit_script.count("GBOSONS_REPO_ROOT=$REPO_ROOT"), 3)

        common_script = (root / "cluster" / "common.sh").read_text()
        self.assertIn('GBOSONS_ENV_PREFIX/bin/python', common_script)
        self.assertNotIn("srun --ntasks=1 micromamba", common_script)

        aggregate_script = (root / "cluster" / "jobs" / "aggregate.sh").read_text()
        self.assertNotIn("-v exp=", aggregate_script)


class ManifestTests(unittest.TestCase):
    def test_control_comparisons_share_initial_seeds(self):
        tasks = [t for t in build("pilot", "control") if t["kind"] == "optimize"]
        paired = {}
        for task in tasks:
            key = (task["L"], task["geometry"], task["depth"],
                   task["restart_start"], task["restart_stop"])
            paired.setdefault(key, set()).add(task["seed"])
        self.assertTrue(paired)
        self.assertTrue(all(len(seeds) == 1 for seeds in paired.values()))

    def test_manifests_are_deterministic_and_resource_binned(self):
        first = build("pilot", "topology")
        self.assertEqual(first, build("pilot", "topology"))
        self.assertEqual([task["task_id"] for task in first], list(range(len(first))))
        self.assertTrue(all(task["resource"] in {"small", "medium", "large", "xlarge"}
                            for task in first))

    def test_completed_task_is_restartable(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(profile="pilot", experiment="bounded", resource=None,
                                   array_index=0, run_dir=directory)
            cli.run_task_command(args)
            path = Path(directory) / "raw" / "bounded" / "task-00000.json"
            before = path.stat().st_mtime_ns
            cli.run_task_command(args)
            self.assertEqual(before, path.stat().st_mtime_ns)

    def test_promotion_is_explicit_and_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root / "run"
            (run / "summaries").mkdir(parents=True); (run / "figures").mkdir()
            (root / "notes" / "figures").mkdir(parents=True)
            canonical = root / "notes" / "figures" / cli.FIGURE_MAP["bounded"]
            canonical.write_bytes(b"canonical paper figure")
            result_path = root / "results.json"
            result_path.write_text('{"existing": 1}\n')
            (run / "validation_report.json").write_text('{"passed": true}\n')
            (run / "summaries" / "bounded.json").write_text('{"task_count": 3}\n')
            (run / "figures" / cli.FIGURE_MAP["bounded"]).write_bytes(b"candidate")
            args = SimpleNamespace(run=str(run), experiments=["bounded"], force=False)
            with patch.object(cli, "ROOT", root), patch.dict(cli.RESULT_MAP,
                                                              {"bounded": result_path}, clear=True):
                cli.promote_command(args)
            promoted = json.loads(result_path.read_text())
            self.assertEqual(promoted["existing"], 1)
            self.assertEqual(promoted["cluster_extension"]["task_count"], 3)
            self.assertEqual(canonical.read_bytes(), b"canonical paper figure")
            stem = Path(cli.FIGURE_MAP["bounded"]).stem
            candidate = root / "notes" / "figures" / f"{stem}_cluster.pdf"
            self.assertEqual(candidate.read_bytes(), b"candidate")


class TopologyTests(unittest.TestCase):
    def test_twisted_hofstadter_is_hermitian(self):
        h, _ = lattices.hofstadter_torus(4, 4, flux=0.25,
                                         theta_x=0.31, theta_y=-0.27)
        np.testing.assert_allclose(h, h.conj().T, atol=1e-14)
        with self.assertRaises(ValueError):
            lattices.hofstadter_torus(5, 4, flux=0.25)

    def test_fhs_is_invariant_under_local_frame_gauge(self):
        grid = 13
        frames = np.empty((grid, grid, 2, 1), complex)
        for ix in range(grid):
            kx = 2 * np.pi * ix / grid
            for iy in range(grid):
                ky = 2 * np.pi * iy / grid
                d = np.array([np.sin(kx), np.sin(ky),
                              -1 + np.cos(kx) + np.cos(ky)])
                H = d[0] * np.array([[0, 1], [1, 0]], dtype=complex)
                H += d[1] * np.array([[0, -1j], [1j, 0]])
                H += d[2] * np.diag([1, -1])
                _, vectors = np.linalg.eigh(H)
                frames[ix, iy, :, 0] = vectors[:, 0]
        chern, _ = multiplet_chern(frames)
        rng = np.random.default_rng(4)
        gauge = np.exp(1j * rng.uniform(0, 2 * np.pi, (grid, grid)))
        transformed = frames * gauge[:, :, None, None]
        gauged, _ = multiplet_chern(transformed)
        self.assertAlmostEqual(abs(chern), 1.0, places=8)
        self.assertAlmostEqual(chern, gauged, places=12)


class PolynomialAndOtocTests(unittest.TestCase):
    def test_packed_translation_and_composition(self):
        phase = nph.add(nph.monomial((4, 0), 0.2), nph.monomial((2, 3), -0.1))
        shift = (0.3, -0.2)
        translated = nph.translate_packed(phase, shift)
        point = (0.7, 0.4)
        self.assertAlmostEqual(nph.evaluate(translated, point),
                               nph.evaluate(phase, np.add(point, shift)))
        composed = nph.effective_packed_phases([phase, nph.scale(phase, 0.5)],
                                                [shift, (-0.1, 0.2)])
        self.assertTrue(composed)

    def test_chunked_otoc_is_chunk_size_independent(self):
        H, basis, _, d = bose_hubbard(5, 2, 1.0, 2.0)
        rng = np.random.default_rng(2)
        psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        psi /= np.linalg.norm(psi)
        times = np.array([0.0, 0.2])
        sites_small, small = _sector_otoc_chunked(H, basis, psi, 2, times, 2)
        sites_full, full = _sector_otoc_chunked(H, basis, psi, 2, times, 5)
        np.testing.assert_array_equal(sites_small, np.arange(5))
        np.testing.assert_array_equal(sites_small, sites_full)
        np.testing.assert_allclose(small, full, atol=1e-12)

    def test_strided_otoc_matches_full_calculation_on_kept_sites(self):
        H, basis, _, d = bose_hubbard(5, 2, 1.0, 2.0)
        rng = np.random.default_rng(2)
        psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        psi /= np.linalg.norm(psi)
        times = np.array([0.0, 0.2])
        _, full = _sector_otoc_chunked(H, basis, psi, 2, times, 5)
        sites, strided = _sector_otoc_chunked(H, basis, psi, 2, times, 2, stride=2)
        np.testing.assert_array_equal(sites, np.arange(0, 5, 2))
        np.testing.assert_allclose(strided, full[:, ::2], atol=1e-12)

    def test_windowed_power_fit_masks_floor_and_higher_orders(self):
        r = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
        y = r ** 3
        floored = np.where(y < 1e-5, 1e-5, y)
        unmasked, _ = cli._fit_log_power(r, floored)
        self.assertLess(unmasked, 2.8)
        slope, _, points = cli._fit_log_power_window(r, floored, floor=2e-5)
        self.assertEqual(points, 4)
        self.assertAlmostEqual(slope, 3.0, places=8)
        slope, _, points = cli._fit_log_power_window(r, y, floor=1e-9, cap=1e-3)
        self.assertEqual(points, 3)
        self.assertAlmostEqual(slope, 3.0, places=8)
        slope, _, points = cli._fit_log_power_window(r[:2], y[:2])
        self.assertEqual(points, 2)
        self.assertTrue(np.isnan(slope))


if __name__ == "__main__":
    unittest.main()
