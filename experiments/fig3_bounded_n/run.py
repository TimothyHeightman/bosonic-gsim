"""Bounded-sector dynamics, finite sector unions, and measured carrier cost."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply

from gbosons import benchmarking as bm
from gbosons import bounded_n as bn
from gbosons import core, fock_ref
from gbosons.plotting import (
    DARK_PALETTE, LIGHT_PALETTE, PAPER_COLORS, PAPER_FILLS,
    paper_style, figures_dir, load_config)

C = {
    "quadratic": PAPER_COLORS["baseline"],
    "kerr": PAPER_COLORS["nonlinear"],
    "fock": PAPER_COLORS["reference"],
    "coherent": PAPER_COLORS["method"],
    "dephased": PAPER_COLORS["theory"],
}


def chain_hamiltonian(n, J):
    h = np.zeros((n, n), dtype=complex)
    for k in range(n - 1):
        h[k, k + 1] = h[k + 1, k] = -J
    return h


def panel_fixed_sector(cfg, ax):
    n, N = cfg["modes"], cfg["N"]
    rng = np.random.default_rng(cfg["seed"])
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * cfg["hop_scale"]
    h = (A + A.conj().T) / 2
    chi = np.zeros((n, n))
    for k, l, value in cfg["chi"]:
        chi[k, l] = value
    basis, index, _ = bn.sector_basis(n, N)
    H = bn.number_conserving_hamiltonian(h, chi, basis, index)
    psi0 = bn.basis_state(cfg["init"], index)
    ts = np.linspace(0, cfg["t_max"], cfg["n_time"])
    quadratic = [core.number_number_correlator(core.passive_transfer(h, t),
                                               np.asarray(cfg["init"]))[0, 1] for t in ts]
    kerr = [bn.nn_expect(0, 1, basis, bn.evolve(H, t, psi0)) for t in ts]
    tf = np.linspace(0.2, cfg["t_max"], cfg["fock_points"])
    fock = [fock_ref.kerr_circuit_fock(h, chi, cfg["init"], t,
                                      cutoff=cfg["fock_cutoff"])[0, 1] for t in tf]
    sector_at_tf = [bn.nn_expect(0, 1, basis, bn.evolve(H, t, psi0)) for t in tf]

    ax.plot(ts, quadratic, color=C["quadratic"], label="quadratic baseline", linestyle='--')
    ax.plot(ts, kerr, color=C["kerr"], label="cross-Kerr")
    ax.plot(tf, fock, "o", ms=3.5, color=C["fock"], label="Fock reference", zorder=5)
    ax.fill_between(ts, quadratic, kerr, color=PAPER_FILLS["nonlinear"], alpha=0.20)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\langle \hat n_0\hat n_1\rangle$")
    ax.set_title(fr"(a) fixed sector, $N={N}$")
    ax.legend(frameon=False, fontsize=7)
    return float(np.max(np.abs(np.asarray(sector_at_tf) - np.asarray(fock))))


def panel_mixed_sectors(cfg, ax):
    n, c, j = cfg["modes"], cfg["source"], cfg["readout"]
    basis, index, blocks = bn.sector_union_basis(n, [0, 1, 2])
    h = chain_hamiltonian(n, cfg["J"])
    chi = np.zeros((n, n)); np.fill_diagonal(chi, cfg["U"] / 2)
    H = bn.number_conserving_hamiltonian(h, chi, basis, index)

    occs = [[0] * n for _ in range(3)]
    occs[1][c] = 1; occs[2][c] = 2
    components = [bn.basis_state(occ, index) for occ in occs]
    coherent0 = sum(components) / np.sqrt(3.0)
    ts = np.linspace(0, cfg["t_max"], cfg["n_time"])
    coherent = expm_multiply(-1j * H, coherent0, start=0, stop=ts[-1],
                             num=len(ts), endpoint=True)
    evolved_components = [expm_multiply(-1j * H, psi, start=0, stop=ts[-1],
                                        num=len(ts), endpoint=True) for psi in components]
    ndiag = bn.number_diag(j, basis)
    X = bn.quadrature_matrix(j, basis, index)

    n_coh = np.array([np.sum(ndiag * np.abs(psi) ** 2).real for psi in coherent])
    n_mix = np.mean([[np.sum(ndiag * np.abs(psi) ** 2).real for psi in states]
                     for states in evolved_components], axis=0)
    x_coh = np.array([np.vdot(psi, X @ psi).real for psi in coherent])
    x_mix = np.mean([[np.vdot(psi, X @ psi).real for psi in states]
                     for states in evolved_components], axis=0)

    ax.plot(ts, n_coh, color=C["coherent"], ls="--", label=r"$\langle n_c\rangle$, coherent")
    ax.plot(ts[::8], n_mix[::8], "o", ms=2.8, mfc="white", color=C["dephased"],
            label=r"$\langle n_c\rangle$, dephased")
    ax.plot(ts, x_coh, color=C["coherent"], label=r"$\langle x_c\rangle$, coherent")
    ax.plot(ts, x_mix, color=C["dephased"], label=r"$\langle x_c\rangle$, dephased")
    ax.axhline(0, color="0.75", lw=0.6)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("mean value")
    ax.set_title(r"(b) $\mathcal{H}_0\oplus\mathcal{H}_1\oplus\mathcal{H}_2$")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    return {
        "carrier_dimension": len(basis),
        "sector_dimensions": {str(N): blocks[N].stop - blocks[N].start for N in blocks},
        "max_number_difference": float(np.max(np.abs(n_coh - n_mix))),
        "max_quadrature_separation": float(np.max(np.abs(x_coh - x_mix))),
    }


def panel_scaling(cfg, ax):
    entries = []
    for Nmax, color, fill in zip(cfg["cutoffs"], DARK_PALETTE, LIGHT_PALETTE):
        medians, q25, q75 = [], [], []
        for n in cfg["modes"]:
            basis, index, _ = bn.sector_union_basis(n, range(Nmax + 1))
            h = chain_hamiltonian(n, cfg["J"])
            chi = np.zeros((n, n)); np.fill_diagonal(chi, cfg["U"] / 2)
            H = bn.number_conserving_hamiltonian(h, chi, basis, index)
            rng = np.random.default_rng(cfg["seed"] + 100 * Nmax + n)
            psi = rng.standard_normal(len(basis)) + 1j * rng.standard_normal(len(basis))
            psi /= np.linalg.norm(psi)
            timing = bm.timed_call(lambda: expm_multiply(-1j * cfg["time"] * H, psi),
                                   repeats=cfg["repeats"], warmups=cfg["warmups"],
                                   target_seconds=cfg["target_seconds"])
            medians.append(timing["median_seconds"])
            q25.append(timing["q25_seconds"]); q75.append(timing["q75_seconds"])
            entries.append({"Nmax": Nmax, "modes": n, "carrier_dimension": len(basis),
                            "hamiltonian_nnz": int(H.nnz), **timing})
        ns = np.asarray(cfg["modes"])
        medians = np.asarray(medians); q25 = np.asarray(q25); q75 = np.asarray(q75)
        ax.loglog(ns, medians, "o-", ms=3.2, color=color,
                  label=fr"$N_{{\max}}={Nmax}$, $D=\binom{{n+{Nmax}}}{{{Nmax}}}$")
        ax.fill_between(ns, q25, q75, color=fill, alpha=0.24)
    ax.set_xlabel("modes $n$")
    ax.set_ylabel("propagation time (s)")
    ax.set_title("(c) carrier-state propagation")
    ax.legend(frameon=False, fontsize=6.5)
    return entries


def main():
    cfg = load_config(__file__)
    paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.75))
    fock_error = panel_fixed_sector(cfg["dynamics"], axes[0])
    mixed = panel_mixed_sectors(cfg["mixed_sectors"], axes[1])
    timings = panel_scaling(cfg["scaling"], axes[2])
    fig.tight_layout()
    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    result = bm.write_results(__file__, {
        "figure": cfg["output"],
        "fixed_sector_max_fock_error": fock_error,
        "mixed_sector": mixed,
        "scaling": timings,
    })
    print(f"wrote {out} and {result} (Fock error {fock_error:.2e})")


if __name__ == "__main__":
    main()
