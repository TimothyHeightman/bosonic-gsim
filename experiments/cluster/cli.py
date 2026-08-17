"""Command-line lifecycle for local and Slurm cluster experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from gbosons import topology as topo
from .manifest import EXPERIMENTS, build, load_profile
from . import tasks as kernels

ROOT = Path(__file__).resolve().parents[2]
FIGURE_MAP = {"bounded": "fig2_bounded_n.pdf", "doublon": "fig3_doublon.pdf",
              "otoc": "fig5_otoc.pdf", "topology": "fig6_doublon_topology.pdf",
              "control": "fig9_kerr_control_2d.pdf",
              "cubic": "fig10_nilpotent_phase.pdf", "squeezing": "fig4_squeezing.pdf"}
RESULT_MAP = {name: ROOT / "experiments" / folder / "results.json" for name, folder in {
    "bounded": "fig2_bounded_n", "doublon": "fig3_doublon", "otoc": "fig5_otoc",
    "topology": "fig6_doublon_topology", "control": "fig9_kerr_control_2d",
    "cubic": "fig10_nilpotent_phase", "squeezing": "fig4_squeezing"}.items()}


def _jsonable(value):
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hash(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:16]


def _git_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                          capture_output=True, check=True).stdout.strip()


def manifest_command(args):
    tasks = build(args.profile, args.experiment)
    payload = {"profile": args.profile, "experiment": args.experiment, "tasks": tasks}
    if args.output:
        _write_json(Path(args.output), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def count_command(args):
    tasks = build(args.profile, args.experiment)
    if args.resource:
        tasks = [task for task in tasks if task["resource"] == args.resource]
    print(len(tasks))


def run_task_command(args):
    all_tasks = build(args.profile, args.experiment)
    selected = [task for task in all_tasks if not args.resource or task["resource"] == args.resource]
    task = selected[args.array_index]
    run_dir = Path(args.run_dir)
    raw = run_dir / "raw" / args.experiment
    json_path = raw / f"task-{task['task_id']:05d}.json"
    digest = _hash(task)
    if json_path.exists():
        prior = json.loads(json_path.read_text())
        if prior.get("status") == "complete" and prior.get("config_hash") == digest:
            print(f"task {task['task_id']} already complete")
            return
    started = perf_counter()
    metadata, arrays = kernels.run(args.experiment, task)
    metadata["wall_seconds"] = perf_counter() - started
    npz_name = None
    if arrays:
        npz_name = f"task-{task['task_id']:05d}.npz"
        destination = raw / npz_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
        temporary = scratch / f"gbosons-{os.getpid()}-{npz_name}"
        np.savez_compressed(temporary, **arrays)
        partial = destination.with_suffix(".npz.partial")
        shutil.copy2(temporary, partial); partial.replace(destination); temporary.unlink()
    payload = {"status": "complete", "profile": args.profile,
               "experiment": args.experiment, "config_hash": digest, "task": task,
               "result": metadata, "arrays": npz_name, "git_sha": _git_sha(),
               "environment": {"python": platform.python_version(),
                               "numpy": np.__version__, "scipy": scipy.__version__,
                               "matplotlib": matplotlib.__version__,
                               "platform": platform.platform()},
               "slurm": {key: os.environ.get(key) for key in
                         ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                          "SLURMD_NODENAME")}}
    _write_json(json_path, payload)
    print(f"completed {args.experiment} task {task['task_id']}")


def _load_complete(run_dir, profile, experiment):
    expected = build(profile, experiment)
    records = []
    for task in expected:
        path = run_dir / "raw" / experiment / f"task-{task['task_id']:05d}.json"
        if not path.exists():
            raise RuntimeError(f"missing {path}")
        record = json.loads(path.read_text())
        if record.get("status") != "complete" or record.get("config_hash") != _hash(task):
            raise RuntimeError(f"invalid or stale result {path}")
        record["_path"] = str(path)
        records.append(record)
    return records


def _fit_log_power(x, y):
    x, y = np.asarray(x), np.asarray(y)
    mask = (x > 0) & (y > 0) & np.isfinite(y)
    if np.count_nonzero(mask) < 3: return float("nan"), float("nan")
    fit, covariance = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1, cov=True)
    return float(fit[0]), float(np.sqrt(covariance[0, 0]))


def _fit_log_power_window(x, y, x_min=0.0, x_max=float("inf"),
                          floor=0.0, cap=float("inf")):
    """Log-log power fit restricted to the resolvable window.

    Points below `floor` sit at the numerical noise floor of the propagation
    and points above `cap` are contaminated by higher orders; both flatten the
    fitted slope if included (this mirrors fit_r_min/fit_r_max/fit_error_floor
    of the original fig4_squeezing experiment).  Returns (slope, stderr,
    points_used).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ((x >= x_min) & (x <= x_max) & (x > 0) & np.isfinite(y)
            & (y > floor) & (y < cap))
    points = int(np.count_nonzero(mask))
    if points < 3:
        return float("nan"), float("nan"), points
    fit, covariance = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1, cov=True)
    return float(fit[0]), float(np.sqrt(covariance[0, 0])), points


def _aggregate_topology(run_dir, records):
    grouped = defaultdict(list)
    for record in records:
        t = record["task"]
        grouped[(t["L"], t["U"], t["flux"], t["twist_grid"])].append(record)
    results = []
    for (L, U, flux, grid), group in grouped.items():
        group.sort(key=lambda r: (r["task"]["ix"], r["task"]["iy"]))
        frames = [[None for _ in range(grid)] for _ in range(grid)]
        gaps = []
        for record in group:
            data = np.load(Path(record["_path"]).with_name(record["arrays"]))
            t = record["task"]
            frames[t["ix"]][t["iy"]] = data["frame"]
            gaps.append(record["result"]["subband_gap"])
        frame_array = np.asarray(frames)
        chern, curvature = topo.multiplet_chern(frame_array)
        results.append({"L": L, "U": U, "flux": flux, "grid": grid,
                        "chern": chern, "integer_distance": abs(chern - round(chern)),
                        "minimum_gap": min(gaps), "curvature": curvature.tolist()})
    return results


def _make_summary(experiment, records, run_dir, profile=None):
    flat = []
    for record in records:
        task, result = record["task"], record["result"]
        if experiment in {"control", "squeezing"} and "records" in result:
            shared = {**task, **{k: v for k, v in result.items() if k != "records"}}
            flat.extend({**shared, **entry} for entry in result["records"])
        else:
            flat.append({**task, **result})
    summary = {"experiment": experiment, "task_count": len(records), "tasks": flat}
    if experiment == "topology":
        summary["multiplets"] = _aggregate_topology(run_dir, records)
    if experiment == "doublon":
        spectrum = [x for x in flat if x["kind"] == "spectrum"]
        by_n = defaultdict(list)
        for row in spectrum: by_n[row["modes"]].append(row)
        summary["strong_coupling_fits"] = []
        for n, rows in by_n.items():
            rows.sort(key=lambda x: x["U"])
            slope, stderr = _fit_log_power([x["U"] for x in rows if x["U"] >= 8],
                                           [x["J_eff"] for x in rows if x["U"] >= 8])
            strong = [x for x in rows if x["U"] >= 8]
            inv_u2 = np.asarray([1 / x["U"] ** 2 for x in strong])
            scaled = np.asarray([x["J_eff"] * x["U"] for x in strong])
            correction, asymptote = np.polyfit(inv_u2, scaled, 1)
            summary["strong_coupling_fits"].append({
                "modes": n, "slope": slope, "stderr": stderr,
                "asymptotic_U_times_J_eff": float(asymptote),
                "leading_inverse_U2_correction": float(correction)})
    if experiment == "control":
        optimizations = [x for x in flat if x["kind"] == "optimize"]
        grouped = defaultdict(list)
        for row in optimizations:
            grouped[(row["L"], row["geometry"], row["depth"], row["model"])].append(row)
        rng = np.random.default_rng(20260714)
        statistics = []
        for (L, geometry, depth, model), rows in grouped.items():
            values = np.asarray([x["target_probability"] for x in rows])
            boot = np.median(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
            statistics.append({"L": L, "geometry": geometry, "depth": depth, "model": model,
                               "count": len(values), "median": float(np.median(values)),
                               "q25": float(np.quantile(values, 0.25)),
                               "q75": float(np.quantile(values, 0.75)),
                               "bootstrap_median_95": [float(np.quantile(boot, 0.025)),
                                                       float(np.quantile(boot, 0.975))],
                               "fraction_above_passive_bound": float(np.mean(values > 0.5))})
        paired = defaultdict(dict)
        for row in optimizations:
            key = row["L"], row["geometry"], row["depth"], row["restart"]
            paired[key][row["model"]] = row["target_probability"]
        differences = [values["kerr"] - values["passive"] for values in paired.values()
                       if {"passive", "kerr"} <= values.keys()]
        summary["optimization_statistics"] = statistics
        summary["paired_kerr_minus_passive"] = {
            "count": len(differences),
            "median": float(np.median(differences)) if differences else float("nan"),
            "q25": float(np.quantile(differences, 0.25)) if differences else float("nan"),
            "q75": float(np.quantile(differences, 0.75)) if differences else float("nan")}
    if experiment == "squeezing":
        fit_cfg = {}
        if profile:
            fit_cfg = load_profile(profile).get("squeezing", {}).get("fit", {})
        window = {"x_min": fit_cfg.get("r_min", 0.0),
                  "x_max": fit_cfg.get("r_max", float("inf"))}
        obs_bounds = {"floor": fit_cfg.get("observable_floor", 1e-12),
                      "cap": fit_cfg.get("observable_cap", 5e-2)}
        leak_bounds = {"floor": fit_cfg.get("leakage_floor", 1e-6),
                       "cap": fit_cfg.get("leakage_cap", 0.5)}
        fits = []
        grouped = defaultdict(list)
        for row in flat: grouped[(row["modes"], row["order"])].append(row)
        for (n, order), rows in grouped.items():
            rows.sort(key=lambda x: x["r"])
            r = [x["r"] for x in rows]
            leak, leak_err, leak_points = _fit_log_power_window(
                r, [x["leakage_norm"] for x in rows], **window, **leak_bounds)
            obs, obs_err, obs_points = _fit_log_power_window(
                r, [x["observable_error"] for x in rows], **window, **obs_bounds)
            fits.append({"modes": n, "order": order, "leakage_slope": leak,
                         "leakage_stderr": leak_err, "leakage_points": leak_points,
                         "observable_slope": obs, "observable_stderr": obs_err,
                         "observable_points": obs_points,
                         "expected_leakage": order + 1,
                         "expected_observable": 2 * (order + 1)})
        summary["fits"] = fits
    return summary


def _plot_summary(experiment, summary, path):
    rows = summary["tasks"]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    if experiment == "bounded":
        for cutoff in sorted(set(x["Nmax"] for x in rows)):
            data = sorted((x for x in rows if x["Nmax"] == cutoff), key=lambda x: x["modes"])
            ax.loglog([x["modes"] for x in data], [x["median_seconds"] for x in data], "o-",
                      label=fr"$N_{{\max}}={cutoff}$")
        ax.set(xlabel="modes $n$", ylabel="propagation time (s)"); ax.legend(frameon=False)
    elif experiment == "doublon":
        for n in sorted(set(x["modes"] for x in rows if x["kind"] == "spectrum")):
            data = sorted((x for x in rows if x["kind"] == "spectrum" and x["modes"] == n),
                          key=lambda x: x["U"])
            ax.loglog([x["U"] for x in data], [x["J_eff"] for x in data], "o-", label=f"n={n}")
        ax.set(xlabel="$U/J$", ylabel=r"$J_{\rm eff}/J$"); ax.legend(frameon=False)
    elif experiment == "otoc":
        threshold = str(rows[0]["thresholds"][0])
        for n in sorted(set(x["modes"] for x in rows)):
            data = [x for x in rows if x["modes"] == n]
            ax.scatter([x["U"] for x in data], [x["front_velocities"][threshold] for x in data],
                       label=f"n={n}", s=18)
        ax.set(xlabel="$U/J$", ylabel="front velocity"); ax.legend(frameon=False)
    elif experiment == "topology":
        data = summary["multiplets"]
        labels = [f"L={x['L']}, g={x['grid']}" for x in data]
        ax.scatter(range(len(data)), [x["chern"] for x in data], c=[x["flux"] for x in data])
        ax.set_xticks(range(len(data)), labels, rotation=70, ha="right", fontsize=6)
        ax.set_ylabel("multiplet Chern number")
    elif experiment == "control":
        rows = [x for x in rows if x["kind"] == "optimize"]
        for model in ("passive", "kerr"):
            depths = sorted(set(x["depth"] for x in rows))
            med = [np.median([x["target_probability"] for x in rows
                              if x["model"] == model and x["depth"] == d]) for d in depths]
            ax.plot(depths, med, "o-", label=model)
        ax.axhline(0.5, color="0.5", ls="--"); ax.set(xlabel="depth", ylabel="median target probability")
        ax.legend(frameon=False)
    elif experiment == "cubic":
        for degree in sorted(set(x["degree"] for x in rows)):
            data = sorted((x for x in rows if x["degree"] == degree), key=lambda x: x["modes"])
            ax.loglog([x["modes"] for x in data], [x["median_seconds"] for x in data], "o-",
                      label=f"degree {degree}")
        ax.set(xlabel="modes $n$", ylabel="propagation time (s)"); ax.legend(frameon=False)
    else:
        for fit in summary["fits"]:
            ax.scatter(fit["expected_leakage"], fit["leakage_slope"], marker="s", color="#1f4e79")
            ax.scatter(fit["expected_observable"], fit["observable_slope"], marker="o", color="#c0392b")
        max_order = max(x["expected_observable"] for x in summary["fits"])
        ax.plot([0, max_order], [0, max_order], color="0.6", ls="--")
        ax.set(xlabel="predicted power", ylabel="fitted power")
    ax.set_title(f"Cluster extension: {experiment}")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path); plt.close(fig)


def aggregate_command(args):
    run_dir = Path(args.run_dir)
    records = _load_complete(run_dir, args.profile, args.experiment)
    summary = _make_summary(args.experiment, records, run_dir, args.profile)
    _write_json(run_dir / "summaries" / f"{args.experiment}.json", summary)
    _plot_summary(args.experiment, summary, run_dir / "figures" / FIGURE_MAP[args.experiment])
    print(f"aggregated {args.experiment}: {len(records)} tasks")


def validate_command(args):
    run_dir = Path(args.run_dir); checks = []
    for experiment in EXPERIMENTS:
        path = run_dir / "summaries" / f"{experiment}.json"
        if not path.exists():
            checks.append({"experiment": experiment, "passed": False, "reason": "missing summary"})
            continue
        summary = json.loads(path.read_text()); passed, reason = True, "complete"
        if experiment == "topology" and args.profile == "production":
            rows = summary.get("multiplets", [])
            # a closed gap makes the multiplet Chern number undefined there:
            # report it instead of failing the certified parameter sets
            certified = [x for x in rows if x["minimum_gap"] > 1e-7]
            uncertified = [x for x in rows if x["minimum_gap"] <= 1e-7]
            passed = bool(certified) and all(x["integer_distance"] < 0.05
                                             for x in certified)
            stable = defaultdict(list)
            for row in certified:
                stable[(row["U"], row["flux"])].append(row["chern"])
            groups = {(row["U"], row["flux"]) for row in rows}
            passed = passed and all(key in stable for key in groups)
            passed = passed and all(max(values) - min(values) < 0.05
                                    for values in stable.values() if len(values) > 1)
            detail = f"{len(certified)}/{len(rows)} multiplets certified"
            if uncertified:
                detail += "; gap closure at " + ", ".join(
                    f"L={x['L']} U={x['U']:g} flux={x['flux']:+g} grid={x['grid']}"
                    for x in uncertified)
            reason = (f"gapped, integer, size/grid-stable multiplets ({detail})"
                      if passed else f"Chern/gap/stability certification failed ({detail})")
        if experiment == "doublon" and args.profile == "production":
            fits = summary.get("strong_coupling_fits", [])
            J = summary["tasks"][0]["J"] if summary.get("tasks") else 1.0
            passed = bool(fits) and all(
                abs(fit["asymptotic_U_times_J_eff"] - 2 * J * J) < 0.2 * J * J
                and abs(fit["slope"] + 1.0) < 0.1 for fit in fits)
            reason = ("strong-coupling asymptote 2J^2/U certified" if passed
                      else "strong-coupling certification failed")
        if experiment == "otoc" and args.profile == "production":
            rows = summary.get("tasks", [])
            passed = bool(rows) and all(
                any(np.isfinite(v) and v > 0 for v in row["front_velocities"].values())
                for row in rows)
            reason = ("ballistic fronts extracted for every task" if passed
                      else "front-velocity extraction failed")
        if experiment == "squeezing":
            gaps = [x["reference_observable_gap"] for x in summary["tasks"]]
            fits = summary.get("fits", [])
            # only fits with >= 3 points inside the resolvable window carry
            # slope information; below-floor orders are reported, not gated
            def _slope_ok(fit, slope_key, points_key, expected_key):
                if fit[points_key] < 3:
                    return True
                tolerance = max(0.15, 2 * fit[slope_key.replace("slope", "stderr")])
                return abs(fit[slope_key] - fit[expected_key]) < tolerance
            slope_ok = all(_slope_ok(x, "observable_slope", "observable_points",
                                     "expected_observable")
                           and _slope_ok(x, "leakage_slope", "leakage_points",
                                         "expected_leakage") for x in fits)
            if args.profile == "production":
                low_orders = [x for x in fits if x["order"] <= 2]
                slope_ok = slope_ok and low_orders and all(
                    x["observable_points"] >= 3 and x["leakage_points"] >= 3
                    for x in low_orders)
            unresolved = sum(1 for x in fits if x["observable_points"] < 3)
            passed = max(gaps, default=1) < (1e-8 if args.profile == "production" else 1e-5) and slope_ok
            reason = (f"references and perturbative orders certified "
                      f"({unresolved} below-floor fits excluded)" if passed
                      else "reference/order certification failed")
        if experiment == "control":
            gradient = [x["gradient_max_absolute_error"] for x in summary["tasks"]
                        if x["kind"] == "gradient"]
            passed = len(gradient) == 1 and gradient[0] < 1e-8
            reason = "finite-difference gradient certified" if passed else "gradient certification failed"
        checks.append({"experiment": experiment, "passed": passed, "reason": reason})
    report = {"profile": args.profile, "passed": all(x["passed"] for x in checks),
              "checks": checks, "git_sha": _git_sha()}
    _write_json(run_dir / "validation_report.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]: raise SystemExit(1)


def promote_command(args):
    run_dir = Path(args.run)
    report = json.loads((run_dir / "validation_report.json").read_text())
    if not report.get("passed") and not args.force:
        raise RuntimeError("run validation did not pass; use --force only after manual review")
    for experiment in args.experiments:
        summary = json.loads((run_dir / "summaries" / f"{experiment}.json").read_text())
        # cluster candidates are diagnostic plots; never clobber the canonical
        # hand-styled paper figure of the same name
        stem = Path(FIGURE_MAP[experiment]).stem
        shutil.copy2(run_dir / "figures" / FIGURE_MAP[experiment],
                     ROOT / "notes" / "figures" / f"{stem}_cluster.pdf")
        target = RESULT_MAP[experiment]
        existing = json.loads(target.read_text()) if target.exists() else {}
        existing["cluster_extension"] = summary
        _write_json(target, existing)
        print(f"promoted {experiment}")


def parser():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("manifest"); q.add_argument("--profile", required=True); q.add_argument("--experiment", choices=EXPERIMENTS, required=True); q.add_argument("--output"); q.set_defaults(func=manifest_command)
    q = sub.add_parser("count"); q.add_argument("--profile", required=True); q.add_argument("--experiment", choices=EXPERIMENTS, required=True); q.add_argument("--resource"); q.set_defaults(func=count_command)
    q = sub.add_parser("run-task"); q.add_argument("--profile", required=True); q.add_argument("--experiment", choices=EXPERIMENTS, required=True); q.add_argument("--resource"); q.add_argument("--array-index", type=int, required=True); q.add_argument("--run-dir", required=True); q.set_defaults(func=run_task_command)
    q = sub.add_parser("aggregate"); q.add_argument("--profile", required=True); q.add_argument("--experiment", choices=EXPERIMENTS, required=True); q.add_argument("--run-dir", required=True); q.set_defaults(func=aggregate_command)
    q = sub.add_parser("validate"); q.add_argument("--profile", required=True); q.add_argument("--run-dir", required=True); q.set_defaults(func=validate_command)
    q = sub.add_parser("promote"); q.add_argument("--run", required=True); q.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, required=True); q.add_argument("--force", action="store_true"); q.set_defaults(func=promote_command)
    return p


def main(argv=None):
    args = parser().parse_args(argv); args.func(args)


if __name__ == "__main__": main()
