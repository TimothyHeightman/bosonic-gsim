"""Expand compact YAML profiles into deterministic task manifests."""
from __future__ import annotations

from itertools import product
from math import comb
from pathlib import Path

import yaml

EXPERIMENTS = ("bounded", "doublon", "otoc", "topology", "control", "cubic",
               "squeezing")


def load_profile(name):
    path = Path(__file__).with_name("profiles") / f"{name}.yaml"
    with open(path) as handle:
        return yaml.safe_load(handle)


def _otoc_resource(task):
    """Walltime-aware bins for reshaped OTOC tasks (those carrying site_stride).

    The dominant cost is n_time x (2 + 2*ceil(sites/chunk)) backward
    propagations whose Krylov step count grows with t*||H||.  Calibrated on
    LiCCA production array 9872245: n=400, U=2, T=48, n_time=48, chunk=12,
    stride=1 completed in ~9 h single-thread, matching this model at an
    effective 1e9 flop/s.  A 3x safety margin keeps tasks inside their bins.
    """
    d = comb(task["modes"] + 1, 2)
    sites = -(-task["modes"] // task["site_stride"])
    solves = task["n_time"] * (2 + 2 * (-(-sites // task["site_chunk"])))
    spmvs_per_solve = 1.2 * 0.5 * task["T"] * (2 * task["U"] + 4 * task["J"] + 2)
    flops = solves * spmvs_per_solve * 5 * d * task["site_chunk"] * 8
    hours = 3 * flops / (1e9 * 3600)
    if hours > 40:
        return "xlarge"
    if hours > 9:
        return "large"
    if hours > 1.5:
        return "medium"
    return "small"


def _resource(experiment, task):
    if experiment == "topology":
        return "large" if task["L"] >= 16 else ("medium" if task["L"] >= 12 else "small")
    if experiment == "control":
        return "large" if task["L"] >= 9 else ("medium" if task["L"] >= 7 else "small")
    if experiment == "otoc" and "site_stride" in task:
        return _otoc_resource(task)
    if experiment == "bounded":
        d = comb(task["modes"] + task["Nmax"], task["Nmax"])
    elif experiment in {"doublon", "otoc"}:
        d = comb(task["modes"] + 1, 2)
    elif experiment in {"topology", "control"}:
        d = comb(task["L"] ** 2 + 1, 2)
    elif experiment == "cubic":
        if task["degree"] == 3:
            estimated_bytes = (task["depth"] + 2) * task["modes"] ** 3 * 8
        else:
            estimated_bytes = comb(task["modes"] + task["degree"], task["degree"]) * 192
        return "large" if estimated_bytes > 16e9 else ("medium" if estimated_bytes > 4e9 else "small")
    else:
        d = comb(task["modes"] + task["N"] + 2 * task["certificate_order"] - 1,
                 task["N"] + 2 * task["certificate_order"])
    if d < 50_000:
        return "small"
    if d < 500_000:
        return "medium"
    if d < 5_000_000:
        return "large"
    return "xlarge"


def _bounded(cfg):
    common = {k: cfg[k] for k in ("J", "U", "time", "seed", "warmups", "repeats")}
    return [{**common, "Nmax": int(cutoff), "modes": n}
            for cutoff, modes in cfg["modes_by_cutoff"].items() for n in modes]


def _doublon(cfg):
    common = {"J": cfg["J"], "eig_tol": cfg["eig_tol"]}
    tasks = [{**common, "kind": "dynamics", "modes": n, "U": U,
              "T": cfg["T_by_modes"][str(n)], "n_time": cfg["n_time"]}
             for n, U in product(cfg["dynamics_modes"], cfg["U_values"])]
    tasks += [{**common, "kind": "spectrum", "modes": n, "U": U}
              for n, U in product(cfg["spectrum_modes"], cfg["spectrum_U_values"])]
    return tasks


def _otoc(cfg):
    # U_values_by_modes / seeds_by_modes / site_stride are optional so that
    # profiles without them keep producing byte-identical task dicts (and
    # therefore content hashes) for already-running submissions.
    tasks = []
    for n in cfg["modes"]:
        U_list = cfg.get("U_values_by_modes", {}).get(str(n), cfg["U_values"])
        seed_list = cfg.get("seeds_by_modes", {}).get(str(n), cfg["seeds"])
        for U, seed in product(U_list, seed_list):
            task = {"modes": n, "U": U, "seed": seed, "J": cfg["J"],
                    "T": cfg["T_by_modes"][str(n)], "n_time": cfg["n_time"],
                    "site_chunk": cfg["site_chunk"], "thresholds": cfg["thresholds"]}
            if "site_stride" in cfg:
                task["site_stride"] = int(cfg["site_stride"])
            tasks.append(task)
    return tasks


def _topology(cfg):
    parameter_sets = [(L, U, flux, cfg["main_grid"])
                      for L, U, flux in product(cfg["L_values"], cfg["U_values"],
                                                cfg["flux_values"])]
    parameter_sets += [(cfg["stability_L"], cfg["stability_U"], flux, grid)
                       for flux, grid in product(cfg["flux_values"],
                                                 cfg["stability_grids"])]
    parameter_sets = list(dict.fromkeys(parameter_sets))
    tasks = []
    for L, U, flux, grid in parameter_sets:
        for ix, iy in product(range(grid), repeat=2):
            tasks.append({"L": L, "U": U, "flux": flux, "twist_grid": grid,
                          "ix": ix, "iy": iy, "oversample": cfg["oversample"],
                          "eig_tol": cfg["eig_tol"]})
    return tasks


def _control(cfg):
    tasks = []
    chunk_size = cfg["restart_chunk_size"]
    for L in cfg["L_values"]:
        depths = cfg["depths_by_L"][str(L)]
        geometries = cfg["geometries_by_L"][str(L)]
        chunks = range(0, cfg["restarts"], chunk_size)
        for geometry, depth, restart_start, model in product(
                geometries, depths, chunks, ("passive", "kerr")):
            seed = cfg["seed"] + 100_000 * L + 10_000 * geometry + 100 * depth
            tasks.append({"L": L, "geometry": geometry, "depth": depth,
                          "kind": "optimize", "restart_start": restart_start,
                          "restart_stop": min(restart_start + chunk_size, cfg["restarts"]),
                          "model": model, "seed": seed,
                          "flux": cfg["flux"], "steps": cfg["steps"], "lr": cfg["lr"]})
    check = cfg["gradient_check"]
    tasks.append({"kind": "gradient", "L": check["L"], "geometry": 0,
                  "depth": check["depth"], "restart": -1, "model": "kerr",
                  "seed": cfg["seed"], "flux": cfg["flux"], "steps": 0,
                  "lr": cfg["lr"], "finite_difference_step": check["step"]})
    return tasks


def _cubic(cfg):
    common = {k: cfg[k] for k in ("depth", "coefficient_scale", "shift_scale",
                                   "warmups", "repeats")}
    return [{**common, "degree": int(degree), "modes": n}
            for degree, modes in cfg["modes_by_degree"].items() for n in modes]


def _squeezing(cfg):
    common = {k: cfg[k] for k in ("N", "U", "J", "time", "reference_order",
                                   "certificate_order")}
    return [{**common, "modes": n, "orders": cfg["orders"], "r": float(r)}
            for n, r in product(cfg["modes"], cfg["r_values"])]


BUILDERS = {"bounded": _bounded, "doublon": _doublon, "otoc": _otoc,
            "topology": _topology, "control": _control, "cubic": _cubic,
            "squeezing": _squeezing}


def build(profile, experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment}")
    cfg = load_profile(profile)[experiment]
    tasks = BUILDERS[experiment](cfg)
    for task_id, task in enumerate(tasks):
        task["task_id"] = task_id
        task["resource"] = _resource(experiment, task)
    return tasks
