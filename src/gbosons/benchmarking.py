"""Small reproducibility helpers shared by the numerical experiments."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy


def timed_call(fn, repeats=7, warmups=1, target_seconds=0.05):
    """Return median/IQR per-call timings with timer overhead amortized."""
    for _ in range(warmups):
        fn()
    t0 = perf_counter(); fn(); estimate = perf_counter() - t0
    iterations = max(1, int(target_seconds / max(estimate, 1e-9)))
    samples = []
    for _ in range(repeats):
        t0 = perf_counter()
        for _ in range(iterations):
            fn()
        samples.append((perf_counter() - t0) / iterations)
    values = np.asarray(samples)
    return {
        "median_seconds": float(np.median(values)),
        "q25_seconds": float(np.quantile(values, 0.25)),
        "q75_seconds": float(np.quantile(values, 0.75)),
        "samples_seconds": values.tolist(),
        "iterations_per_sample": iterations,
        "warmups": warmups,
        "repeats": repeats,
    }


def environment_metadata():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unreported",
        "float_precision": "float64/complex128",
    }


def write_results(run_file, payload):
    """Write deterministic experiment metadata beside its configuration."""
    path = Path(run_file).resolve().parent / "results.json"
    data = {"environment": environment_metadata(), **payload}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path
