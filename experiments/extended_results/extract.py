"""Extract a compact, validated publication bundle from the production run."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .common import DATA_DIR, ROOT, sha256, write_json


RUN_ID = "20260714T212532Z-production-08c652e"
SOURCE = ROOT / "cluster_results" / RUN_ID
EXPERIMENTS = ("bounded", "doublon", "otoc", "topology",
               "control", "cubic", "squeezing")
OTOC_TASKS = (54, 63)


def _read(path: Path):
    return json.loads(path.read_text())


def _task_git_revisions() -> list[str]:
    revisions = set()
    for experiment in EXPERIMENTS:
        for path in (SOURCE / "raw" / experiment).glob("task-*.json"):
            revision = _read(path).get("git_sha")
            if revision:
                revisions.add(revision)
    return sorted(revisions)


def main() -> None:
    validation_path = SOURCE / "validation_report.json"
    if not validation_path.exists():
        raise FileNotFoundError(f"missing production validation: {validation_path}")
    validation = _read(validation_path)
    if validation.get("profile") != "production" or not validation.get("passed"):
        raise RuntimeError("publication extraction requires a passed production run")
    checks = {entry["experiment"]: entry["passed"] for entry in validation["checks"]}
    if checks != {name: True for name in EXPERIMENTS}:
        raise RuntimeError(f"incomplete validation checks: {checks}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    task_counts = {}
    for experiment in EXPERIMENTS:
        source = SOURCE / "summaries" / f"{experiment}.json"
        summary = _read(source)
        task_counts[experiment] = summary["task_count"]
        destination = DATA_DIR / f"{experiment}.json"
        shutil.copy2(source, destination)
        files[destination.name] = sha256(destination)

    selected_otoc = []
    for task_id in OTOC_TASKS:
        stem = f"task-{task_id:05d}"
        record_source = SOURCE / "raw" / "otoc" / f"{stem}.json"
        record = _read(record_source)
        task = record["task"]
        expected_u = 0.0 if task_id == 54 else 8.0
        if task["modes"] != 400 or task["seed"] != 0 or task["U"] != expected_u:
            raise RuntimeError(f"unexpected OTOC task mapping for {stem}: {task}")
        array_source = record_source.with_name(record["arrays"])
        with np.load(array_source) as arrays:
            if arrays["otoc"].shape != (48, 400):
                raise RuntimeError(f"unexpected OTOC shape in {array_source}")
        for source in (record_source, array_source):
            destination = DATA_DIR / source.name
            shutil.copy2(source, destination)
            files[destination.name] = sha256(destination)
        selected_otoc.append({"task_id": task_id, "U": expected_u,
                              "record": record_source.name,
                              "arrays": array_source.name})

    protected = sorted((ROOT / "notes" / "figures").glob("fig*.pdf"))
    protected.append(ROOT / "notes" / "main.tex")
    source_checksums = {
        str(path.relative_to(ROOT)): sha256(path) for path in protected
    }
    manifest = {
        "source_run_id": RUN_ID,
        "source_path": str(SOURCE.relative_to(ROOT)),
        "profile": validation["profile"],
        "validation_passed": validation["passed"],
        "validation_git_sha": validation["git_sha"],
        "task_git_shas": _task_git_revisions(),
        "validation_checks": validation["checks"],
        "task_counts": task_counts,
        "selected_otoc": selected_otoc,
        "files": files,
        "protected_source_checksums": source_checksums,
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    print(f"wrote compact publication bundle to {DATA_DIR}")


if __name__ == "__main__":
    main()
