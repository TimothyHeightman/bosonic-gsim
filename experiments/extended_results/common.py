"""Shared paths, data loading, and output guards for extended figures."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "production-20260714"
OUTPUT_DIR = ROOT / "notes" / "figures" / "extended_results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_json(name: str):
    return json.loads((DATA_DIR / name).read_text())


def output_path(name: str) -> Path:
    if Path(name).name != name or not name.endswith(".pdf"):
        raise ValueError(f"invalid extended figure name: {name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = (OUTPUT_DIR / name).resolve()
    if destination.parent != OUTPUT_DIR.resolve():
        raise ValueError("extended figures must remain in the isolated output directory")
    return destination


def load_canonical(folder: str):
    path = ROOT / "experiments" / folder / "run.py"
    spec = importlib.util.spec_from_file_location(f"_canonical_{folder}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def finite(values, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"non-finite values in {label}")
    return array
