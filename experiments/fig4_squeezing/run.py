"""Parity-aware convergence of bounded carriers under weak squeezing."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply

from gbosons import banded as bd
from gbosons import benchmarking as bm
from gbosons.plotting import (
    DARK_PALETTE, LIGHT_PALETTE, paper_style, figures_dir, load_config)

C = list(DARK_PALETTE[:4])
FILL = list(LIGHT_PALETTE[:4])


def hamiltonian(n, N, k, U, J, r):
    basis, index, blocks = bd.reachable_basis(n, N, k)
    h = np.zeros((n, n), complex)
    for j in range(n - 1):
        h[j, j + 1] = h[j + 1, j] = -J
    chi = np.zeros((n, n)); chi[0, 1] = U
    H = (bd.hopping(h, basis, index) + bd.cross_kerr(chi, basis)
         + bd.squeeze(0, 1, r, basis, index))
    return H, basis, index, blocks


def trajectory(n, N, k, U, J, occ, r, ts):
    H, basis, index, _ = hamiltonian(n, N, k, U, J, r)
    psi0 = bd.basis_state(occ, index, len(basis))
    states = expm_multiply(-1j * H, psi0, start=0, stop=ts[-1],
                           num=len(ts), endpoint=True)
    return np.array([bd.nn_expect(0, 1, basis, state) for state in states])


def fit_power(rs, errors, r_min, r_max, floor):
    mask = (rs >= r_min) & (rs <= r_max) & (errors > floor)
    coeff, covariance = np.polyfit(np.log(rs[mask]), np.log(errors[mask]), 1, cov=True)
    return float(coeff[0]), float(np.sqrt(covariance[0, 0])), mask


def benchmark_curve(cfg, system, k):
    entries = []
    for n in cfg["modes_by_order"][str(k)]:
        occ = [0] * n; occ[0] = occ[1] = 1
        H, basis, index, _ = hamiltonian(n, system["N"], k, system["U"],
                                         system["J"], cfg["r"])
        psi0 = bd.basis_state(occ, index, len(basis))
        timing = bm.timed_call(lambda: expm_multiply(-1j * cfg["time"] * H, psi0),
                               repeats=cfg["repeats"], warmups=cfg["warmups"],
                               target_seconds=cfg["target_seconds"])
        entries.append({"order": k, "modes": n, "carrier_dimension": len(basis),
                        "hamiltonian_nnz": int(H.nnz), **timing})
    return entries


def main():
    cfg = load_config(__file__)
    paper_style()
    system = cfg["system"]
    n, N, U, J = system["modes"], system["N"], system["U"], system["J"]
    occ = system["init"]
    fig, ax = plt.subplots(1, 3, figsize=(8.5, 2.65))

    cv = cfg["convergence"]
    ts = np.linspace(0, cv["t_max"], cv["n_time"])
    reference = trajectory(n, N, cv["reference_order"], U, J, occ, cv["r"], ts)
    certificate = trajectory(n, N, cv["certificate_order"], U, J, occ, cv["r"], ts)
    reference_gap = float(np.max(np.abs(reference - certificate)))
    ax[0].plot(ts, reference, color="k", lw=1.8,
               label=fr"reference $k={cv['reference_order']}$")
    for k in cv["orders"]:
        ax[0].plot(ts, trajectory(n, N, k, U, J, occ, cv["r"], ts),
                   color=C[k], ls="--", lw=1.2, label=fr"$k={k}$")
    ax[0].set_xlabel("time $t$  ($1/J$)")
    ax[0].set_ylabel(r"$\langle \hat n_1\hat n_2\rangle$")
    ax[0].set_title(fr"(a) convergence, $r={cv['r']}$")
    ax[0].legend(frameon=False, fontsize=6.5, ncol=2)

    er = cfg["error"]
    rs = np.logspace(er["log10_r_min"], er["log10_r_max"], er["n_r"])
    error_ts = np.linspace(0, er["t_max"], er["n_time"])
    error_results = []
    references = [trajectory(n, N, er["reference_order"], U, J, occ, r, error_ts)
                  for r in rs]
    certificates = [trajectory(n, N, er["certificate_order"], U, J, occ, r, error_ts)
                    for r in rs]
    reference_gap = max(reference_gap, max(float(np.max(np.abs(a - b)))
                                            for a, b in zip(references, certificates)))
    for k in er["orders"]:
        errors = np.array([np.max(np.abs(trajectory(n, N, k, U, J, occ, r, error_ts) - ref))
                           for r, ref in zip(rs, references)])
        slope, stderr, mask = fit_power(rs, errors, er["fit_r_min"], er["fit_r_max"],
                                        er["fit_error_floor"])
        ax[1].loglog(rs, errors, "o-", color=C[k], ms=3,
                     label=fr"$k={k}$: ${slope:.2f}\pm{stderr:.2f}$")
        ax[1].loglog(rs[mask], errors[mask], "o", color=C[k], ms=4, mfc="white")
        error_results.append({"order": k, "expected_slope": 2 * (k + 1),
                              "fitted_slope": slope, "slope_stderr": stderr,
                              "errors": errors.tolist(), "fit_mask": mask.tolist()})
    ax[1].set_xlabel("squeezing amplitude $r$")
    ax[1].set_ylabel(r"$\epsilon_k=\max_t|F_k-F_{\rm ref}|$")
    ax[1].set_title("(b) perturbative error order")
    ax[1].legend(frameon=False, fontsize=6.5, loc="lower right")

    timing_results = []
    for k in cfg["cost"]["orders"]:
        entries = benchmark_curve(cfg["cost"], system, k)
        timing_results.extend(entries)
        ns = np.array([entry["modes"] for entry in entries])
        med = np.array([entry["median_seconds"] for entry in entries])
        lo = np.array([entry["q25_seconds"] for entry in entries])
        hi = np.array([entry["q75_seconds"] for entry in entries])
        ax[2].loglog(ns, med, "o-", color=C[k], ms=3,
                     label=fr"$k={k}$, $D=O(n^{{{N + 2 * k}}})$")
        ax[2].fill_between(ns, lo, hi, color=FILL[k], alpha=0.24)
    ax[2].set_xlabel("modes $n$")
    ax[2].set_ylabel("propagation time (s)")
    ax[2].set_title("(c) fixed-order carrier cost")
    ax[2].legend(frameon=False, fontsize=6.5)

    fig.tight_layout()
    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    result = bm.write_results(__file__, {
        "figure": cfg["output"],
        "reference_orders": [cv["reference_order"], cv["certificate_order"]],
        "max_successive_reference_gap": reference_gap,
        "error_scaling": {"r_values": rs.tolist(), "fits": error_results},
        "cost": timing_results,
    })
    print(f"wrote {out} and {result} (reference gap {reference_gap:.2e})")


if __name__ == "__main__":
    main()
