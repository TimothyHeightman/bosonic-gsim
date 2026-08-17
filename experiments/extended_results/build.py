"""Build publication-ready extended figures without touching canonical outputs."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import tempfile
import matplotlib.ticker as ticker

os.environ.setdefault("MPLBACKEND", "Agg")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.sparse.linalg import expm_multiply

from gbosons import bounded_n as bn
from gbosons.plotting import (
    DARK_PALETTE, LIGHT_COLORS, LIGHT_PALETTE, PAPER_COLORS, PAPER_FILLS,
    load_config, paper_style)

from .common import DATA_DIR, OUTPUT_DIR, ROOT, finite, load_canonical, load_json, output_path


METHOD = PAPER_COLORS["method"]
BASELINE = PAPER_COLORS["baseline"]
NONLINEAR = PAPER_COLORS["nonlinear"]
REFERENCE = PAPER_COLORS["reference"]
THEORY = PAPER_COLORS["theory"]
SERIES_COLORS = DARK_PALETTE
SERIES_FILLS = LIGHT_PALETTE


def _save(fig, name):
    destination = output_path(name)
    fig.savefig(destination)
    plt.close(fig)
    print(f"wrote {destination}")


def gaussian_figure():
    source = load_canonical("fig7_gaussian")
    cfg = load_config(source.__file__)
    results = json.loads(
        (ROOT / "experiments" / "fig7_gaussian" / "results.json").read_text())
    paper_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.6, 2.6))

    squeezing = cfg["squeezing"]
    rate = squeezing["xi"]
    times = np.linspace(0, squeezing["t_max"], squeezing["n_time"])
    generator_a, generator_b = source.core.single_mode_squeeze_generator(rate)
    propagated = [
        source.core.mean_n_single_mode_vacuum(
            source.core.bogoliubov_transfer(generator_a, generator_b, t=time))
        for time in times
    ]
    ax_left.plot(
        times, np.sinh(rate * times) ** 2, color=PAPER_COLORS["theory"],
        lw=2.0, label=r"analytic $\sinh^2(rt)$")
    ax_left.plot(
        times, propagated, color=PAPER_COLORS["method"], ls="--",
        label="Bogoliubov transfer")
    fock_times = np.linspace(
        0.1, squeezing["t_max"], squeezing["fock_points"])
    fock_values = [
        source.fock_ref.squeeze_mean_n_fock(
            rate, time, cutoff=squeezing["fock_cutoff"])
        for time in fock_times
    ]
    ax_left.plot(
        fock_times, fock_values, "o", ms=4,
        color=PAPER_COLORS["reference"],
        label="Fock reference", zorder=5)
    ax_left.set(
        xlabel="time $t$", ylabel=r"$\langle \hat n(t)\rangle$",
        title="(a) single-mode squeezing")
    ax_left.legend(frameon=False, loc="upper left", fontsize=7)

    timings = sorted(
        results["coincidence_timings"], key=lambda row: row["modes"])
    modes = finite([row["modes"] for row in timings], "Gaussian modes")
    medians = finite(
        [row["median_seconds"] for row in timings], "Gaussian timings")
    lower = finite(
        [row["q25_seconds"] for row in timings], "Gaussian q25")
    upper = finite(
        [row["q75_seconds"] for row in timings], "Gaussian q75")
    ax_right.loglog(
        modes, medians, "o-", color=PAPER_COLORS["method"], ms=4,
        label="moment propagation")
    guide = medians[-1] * (modes / modes[-1]) ** 3
    ax_right.loglog(
        modes, guide, "--", color=PAPER_COLORS["theory"],
        label=r"$\mathcal{O}(n^3)$ reference")
    ax_right.fill_between(
        modes, lower, upper, color=PAPER_FILLS["method"], alpha=0.24)
    ax_right.set(
        xlabel="modes $n$",
        ylabel="evaluation time (s)",
        title=r"(b) $n\times n$ number-correlation matrix")
    ax_right.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig7_gaussian_extended.pdf")


def beyond_gaussian_figure():
    source = load_canonical("fig8_beyond_gaussian")
    cfg = load_config(source.__file__)
    paper_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.6, 2.6))

    hom = cfg["hom"]
    angles = np.linspace(0, np.pi / 2, hom["n_theta"])
    propagated = np.asarray([
        source.core.number_number_correlator(
            source.beamsplitter(angle), [1, 1])[0, 1]
        for angle in angles
    ])
    ax_left.plot(
        angles, np.cos(2 * angles) ** 2, color=PAPER_COLORS["theory"],
        lw=2.0, label=r"analytic $\cos^2(2\theta)$")
    ax_left.plot(
        angles, propagated, color=PAPER_COLORS["method"], ls="--",
        label="moment propagation")
    fock_angles = np.asarray(
        [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
    fock_values = np.asarray([
        source.fock_ref.nij_fock(
            source.beamsplitter(angle), [1, 1],
            cutoff=hom["fock_cutoff"])[0, 1]
        for angle in fock_angles
    ])
    ax_left.plot(
        fock_angles, fock_values, "o", ms=4,
        color=PAPER_COLORS["reference"], label="Fock reference")
    ax_left.axvline(np.pi / 4, color="0.65", lw=0.6, ls=":")
    ax_left.set(
        xlabel=r"beamsplitter angle $\theta$",
        ylabel=r"$\langle \hat n_1\hat n_2\rangle$",
        title=r"(a) Hong--Ou--Mandel interference")
    ax_left.legend(frameon=False, fontsize=7)

    validation = cfg["validation"]
    rng = np.random.default_rng(validation["seed"])
    predicted, reference = [], []
    for _ in range(validation["instances"]):
        unitary = source.core.haar_unitary(validation["modes"], rng)
        predicted.extend(
            source.core.number_number_correlator(
                unitary, validation["occupation"]).ravel())
        reference.extend(
            source.fock_ref.nij_fock(
                unitary, validation["occupation"],
                cutoff=validation["fock_cutoff"]).ravel())
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    lower = min(predicted.min(), reference.min())
    upper = max(predicted.max(), reference.max())
    padding = 0.03 * (upper - lower)
    limits = (max(0.0, lower - padding), upper + padding)
    ax_right.plot(
        limits, limits,
        color=PAPER_COLORS["reference"], lw=1,
        label="exact agreement")
    ax_right.scatter(
        reference, predicted, s=9, alpha=0.7,
        color=PAPER_COLORS["method"], label="moment propagation")
    ax_right.set(
        xlabel="Fock reference", ylabel="moment propagation",
        title="(b) random-interferometer validation",
        xlim=limits, ylim=limits)
    ax_right.set_aspect("equal", adjustable="box")
    #ax_right.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig8_beyond_gaussian_extended.pdf")


@lru_cache(maxsize=1)
def _chiral_transport_data():
    source = load_canonical("fig6_chiral_transport")
    cfg = load_config(source.__file__)
    system = cfg["system"]
    size, interaction = system["L"], system["U"]
    duration, time_count = system["T"], system["n_time"]
    energy_window = system["energy_window"]
    band_size = system["kband"]
    center = np.asarray([(size - 1) / 2, (size - 1) / 2])
    times = np.linspace(0, duration, time_count)
    panel_cfg = cfg["panels"]

    saved = {}
    for flux in [panel_cfg["flux_plus"], panel_cfg["flux_minus"]]:
        packet, energies, vectors, doublon_rows, coordinates, _, _, _ = (
            source.edge_wavepacket(
                size, flux, interaction, energy_window, band_size))
        coefficients = vectors.conj().T @ packet
        states = np.asarray([
            vectors @ (np.exp(-1j * energies * time) * coefficients)
            for time in times
        ])
        com, fraction, _ = source.track(
            states, doublon_rows, coordinates, center)
        saved[flux] = {
            "times": times,
            "com": com,
            "fraction": fraction,
            "density": np.abs(states[:, doublon_rows]) ** 2,
        }
    return source, cfg, center, saved


def chiral_transport_figure():
    source, cfg, center, saved = _chiral_transport_data()
    system = cfg["system"]
    size = system["L"]
    panel_cfg = cfg["panels"]

    plus = saved[panel_cfg["flux_plus"]]
    minus = saved[panel_cfg["flux_minus"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    image = None
    for ax, values, title, path_color in [
            (axes[0], plus, r"(a) flux $+\phi$",
             LIGHT_COLORS["blue_koi"]),
            (axes[1], minus, r"(b) flux $-\phi$",
             LIGHT_COLORS["pink_coral"])]:
        grid = values["density"][-1].reshape(size, size).T
        image = ax.imshow(
            grid, origin="lower", cmap="magma",
            extent=[0, size - 1, 0, size - 1])
        com = values["com"]
        ax.plot(com[:, 0], com[:, 1], color=path_color, lw=1.3)
        ax.plot(
            com[0, 0], com[0, 1], "o", color="white",
            ms=4, mec="k", mew=0.4)
        ax.set_title(title)
        ax.set_xlabel("site $x$")
        ax.set_xticks([0, size - 1])
        ax.set_yticks([0, size - 1])
    axes[0].set_ylabel("site $y$")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("doublon density", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    axes[2].plot(
        plus["times"], source.winding(plus["com"], center),
        color=PAPER_COLORS["method"], label=r"$+\phi$")
    axes[2].plot(
        minus["times"], source.winding(minus["com"], center),
        color=PAPER_COLORS["theory"], ls="--", label=r"$-\phi$")
    axes[2].axhline(0, color="0.65", lw=0.5)
    axes[2].set(
        xlabel="time $t$  ($1/J$)", ylabel="chiral winding (turns)",
        title="(c) direction reverses with flux")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig6_chiral_transport_extended.pdf")


def bounded_figure():
    source = load_canonical("fig2_bounded_n")
    cfg = load_config(source.__file__)
    summary = load_json("bounded.json")
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.8))
    source.panel_fixed_sector(cfg["dynamics"], axes[0])
    source.panel_mixed_sectors(cfg["mixed_sectors"], axes[1])

    grouped = defaultdict(list)
    for row in summary["tasks"]:
        grouped[row["Nmax"]].append(row)
    for cutoff, color, fill in zip(
            sorted(grouped), SERIES_COLORS, SERIES_FILLS):
        rows = sorted(grouped[cutoff], key=lambda row: row["modes"])
        modes = finite([row["modes"] for row in rows], "bounded modes")
        med = finite([row["median_seconds"] for row in rows], "bounded medians")
        lo = finite([row["q25_seconds"] for row in rows], "bounded q25")
        hi = finite([row["q75_seconds"] for row in rows], "bounded q75")
        axes[2].loglog(modes, med, "o-", ms=3.3, color=color,
                       label=fr"$N_{{\max}}={cutoff}$")
        axes[2].fill_between(modes, lo, hi, color=fill, alpha=0.24)
        scaling_guide = med[0] * (modes / modes[0]) ** cutoff
        axes[2].loglog(
            modes, scaling_guide, "--", color=color, lw=0.9, zorder=1)
        #axes[2].annotate(fr"$n={int(modes[-1])}$",
        #                 (modes[-1], med[-1]), xytext=(15, -8),
        #                 textcoords="offset points", ha="right", fontsize=6,
        #                 color=color)
    axes[2].set_xlabel("modes $n$")
    axes[2].set_ylim(10**(-3.9),20)
    axes[2].set_xlim(10, 10**3.2)
    axes[2].set_ylabel("propagation time (s)")
    axes[2].set_title(r"(c) bounded sector unions")
    axes[2].legend(frameon=False, fontsize=6.2, ncol=3, loc="upper left")
    axes[2].text(
        0.04, 0.04, r"dashed: expected $O(n^{N_{\max}})$",
        transform=axes[2].transAxes, ha="left", va="bottom",
        fontsize=6, color="0.35")
    #axes[2].text(0.97, 0.04, "median and IQR; carrier states", transform=axes[2].transAxes, ha="right", fontsize=6, color="0.4")
    fig.tight_layout()
    _save(fig, "fig2_bounded_n_extended.pdf")


def doublon_figure():
    source = load_canonical("fig3_doublon")
    cfg = load_config(source.__file__)
    summary = load_json("doublon.json")
    system = cfg["system"]
    n, J, T, nt = system["modes"], system["J"], system["T"], system["n_time"]
    basis, index, _ = bn.sector_basis(n, 2)
    occ = [0] * n
    occ[n // 2] = 2
    psi0 = bn.basis_state(occ, index)
    free = cfg["lightcones"]["U_free"]
    bound = cfg["lightcones"]["U_bound"]
    prof_free, _ = source.density_profile(
        source.bose_hubbard(n, J, free, basis, index), psi0, basis, n, T, nt)
    prof_bound, _ = source.density_profile(
        source.bose_hubbard(n, J, bound, basis, index), psi0, basis, n, T, nt)

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.65))
    vmax = max(prof_free.max(), prof_bound.max())
    for ax, profile, title in [
        (axes[0], prof_free, fr"(a) free pair $U={free:g}J$"),
        (axes[1], prof_bound, fr"(b) bound pair $U={bound:g}J$"),
    ]:
        image = ax.imshow(profile, origin="lower", aspect="auto",
                          extent=[0, n - 1, 0, T], cmap="magma", vmin=0, vmax=vmax)
        ax.set_xlabel("site $j$")
        ax.set_title(title)
    axes[0].set_ylabel("time $t$  ($1/J$)")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=-0.15)
    colorbar.ax.set_title(r"   $\langle \hat n_j(t)\rangle$", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    times = np.linspace(0, T, nt)
    for interaction, color in cfg["binding"]["curves"]:
        states = expm_multiply(
            -1j * source.bose_hubbard(n, J, interaction, basis, index), psi0,
            start=0, stop=T, num=nt, endpoint=True)
        axes[2].plot(times, source.doublon_fraction(states, basis, n),
                     color=color, label=fr"$U={interaction:g}J$")
    axes[2].set(xlabel="time $t$  ($1/J$)", ylabel="doublon fraction",
                title="(c) pair binding", ylim=(-0.03, 1.03))
    axes[2].legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(0.95, 0.85))

    spectrum = [row for row in summary["tasks"] if row["kind"] == "spectrum"]
    markers = {101: "o", 201: "s", 401: "^"}
    for modes in sorted(markers):
        rows = sorted((row for row in spectrum if row["modes"] == modes),
                      key=lambda row: row["U"])
        interactions = finite([row["U"] for row in rows], "doublon U")
        effective = finite([row["J_eff"] for row in rows], "doublon J_eff")
        axes[3].loglog(interactions, effective, marker=markers[modes], ls="none",
                       ms=3.8, mfc="white", mec=METHOD, label=fr"$n={modes}$")
    interactions = np.asarray(sorted({row["U"] for row in spectrum}), dtype=float)
    axes[3].loglog(interactions, 2 / interactions, "--", color=THEORY,
                   label=r"strong-coupling $2J^2/U$")
    largest = next(item for item in summary["strong_coupling_fits"]
                   if item["modes"] == 401)
    axes[3].set(xlabel="interaction $U/J$", ylabel=r"$J_{\mathrm{eff}}/J$",
                title="(d) strong-coupling scaling")
    axes[3].legend(frameon=False, fontsize=6.2, loc="lower left")

    # Local Bose-Hubbard chain schematic.
    inset = axes[3].inset_axes([0.5, 0.74, 0.42, 0.24])
    inset.set_xlim(-0.9, 5.9)
    inset.set_ylim(-0.45, 0.62)
    inset.axis("off")

    spacing = 1.25
    sites = spacing * np.arange(5)
    center = sites[2]

    inset.plot(sites, np.zeros_like(sites), color="0.35", lw=1.0, zorder=1)
    inset.scatter(
        sites, np.zeros_like(sites), s=42,
        facecolors="white", edgecolors="0.3", linewidths=0.8, zorder=2
    )

    # Central site containing two distinguishable photons.
    inset.scatter(
        [center], [0], s=72, facecolors="white",
        edgecolors=NONLINEAR, linewidths=1.2, zorder=3
    )
    inset.scatter(
        [center - 0.19, center + 0.19], [0, 0],
        s=8, color=NONLINEAR, zorder=4
    )

    # Continuation of the chain.
    inset.text(-0.63, 0, r"$\cdots$", ha="center", va="center",
               fontsize=6, clip_on=False)
    inset.text(5.66, 0, r"$\cdots$", ha="center", va="center",
               fontsize=6, clip_on=False)

    # Coupling and interaction labels.
    inset.text(
        0.5 * (sites[0] + sites[1]), 0.11, r"$J$",
        ha="center", va="bottom", fontsize=6.5
    )
    inset.text(
        center, 0.18, r"$U$",
        ha="center", va="bottom", fontsize=6.5, color=NONLINEAR
    )
    inset.text(
        center, -0.27, r"$|2_c\rangle$",
        ha="center", va="top", fontsize=6.5
    )

    axes[3].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=2))
    axes[3].xaxis.set_minor_formatter(ticker.NullFormatter())
    #axes[3].text(0.97, 0.96,
    #    fr"$\alpha={largest['slope']:.3f}\pm{largest['stderr']:.3f}$"
    #    "\n" + fr"$UJ_{{\rm eff}}\to{largest['asymptotic_U_times_J_eff']:.3f}J^2$",
    #    transform=axes[3].transAxes, ha="right", va="top", fontsize=6.3,
    #    color="0.35")
    fig.tight_layout()
    pos_b = axes[1].get_position()
    axes[1].set_position([pos_b.x0 - 0.05, pos_b.y0, pos_b.width, pos_b.height])
    _save(fig, "fig3_doublon_extended.pdf")


def cubic_figure():
    source = load_canonical("fig10_nilpotent_phase")
    cfg = load_config(source.__file__)
    summary = load_json("cubic.json")
    local = json.loads(
        (ROOT / "experiments" / "fig10_nilpotent_phase" / "results.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.75))
    source.panel_single_mode(cfg["single_mode"], axes[0])
    source.panel_two_mode(cfg["two_mode"], axes[1])

    grouped = defaultdict(list)
    for row in summary["tasks"]:
        grouped[row["degree"]].append(row)

    local_rows = sorted(local["scaling"]["entries"], key=lambda row: row["modes"])
    local_modes = finite([row["modes"] for row in local_rows], "local cubic modes")
    local_med = finite([row["median_seconds"] for row in local_rows], "local cubic medians")
    local_lo = finite([row["q25_seconds"] for row in local_rows], "local cubic q25")
    local_hi = finite([row["q75_seconds"] for row in local_rows], "local cubic q75")
    axes[2].loglog(local_modes, local_med, "o-", ms=3.4, color=METHOD,
                   label=r"$m=3$, local")
    axes[2].fill_between(local_modes, local_lo, local_hi,
                         color=PAPER_FILLS["method"], alpha=0.24)

    cluster_cubic = sorted(grouped[3], key=lambda row: row["modes"])
    cluster_modes = finite([row["modes"] for row in cluster_cubic], "cluster cubic modes")
    cluster_med = finite([row["median_seconds"] for row in cluster_cubic],
                         "cluster cubic medians")
    cluster_lo = finite([row["q25_seconds"] for row in cluster_cubic], "cluster cubic q25")
    cluster_hi = finite([row["q75_seconds"] for row in cluster_cubic], "cluster cubic q75")
    axes[2].errorbar(
        cluster_modes, cluster_med,
        yerr=[cluster_med - cluster_lo, cluster_hi - cluster_med],
        marker="o", ls="none", ms=3.6, mfc="white", mec=METHOD, ecolor=METHOD,
        elinewidth=0.7, capsize=1.5, label=r"$m=3$, HPC")

    guide_modes = np.geomspace(local_modes[0], cluster_modes[-1], 100)
    guide = local_med[-1] * (guide_modes / local_modes[-1]) ** 3
    axes[2].loglog(guide_modes, guide, "--", color=METHOD, lw=0.9)

    for degree, color, fill, marker in [
            (4, BASELINE, PAPER_FILLS["baseline"], "s"),
            (5, NONLINEAR, PAPER_FILLS["nonlinear"], "^")]:
        rows = sorted(grouped[degree], key=lambda row: row["modes"])
        modes = finite([row["modes"] for row in rows], "cubic modes")
        med = finite([row["median_seconds"] for row in rows], "cubic medians")
        lo = finite([row["q25_seconds"] for row in rows], "cubic q25")
        hi = finite([row["q75_seconds"] for row in rows], "cubic q75")
        axes[2].loglog(modes, med, marker=marker, ls="-", ms=3.4, color=color,
                       label=fr"$m={degree}$, HPC")
        axes[2].fill_between(modes, lo, hi, color=fill, alpha=0.24)
        scaling_guide = med[0] * (modes / modes[0]) ** degree
        axes[2].loglog(modes, scaling_guide, "--", color=color, lw=0.9)
        axes[2].annotate(fr"$n={int(modes[-1])}$",
                         (modes[-1], med[-1]), xytext=(-3, 5),
                         textcoords="offset points", ha="right", fontsize=6,
                         color=color)
    axes[2].annotate(fr"$n={int(cluster_modes[-1])}$",
                     (cluster_modes[-1], cluster_med[-1]), xytext=(-3, 5),
                     textcoords="offset points", ha="right", fontsize=6,
                     color=METHOD)
    axes[2].set(xlabel="modes $n$", ylabel="propagation time (s)",
                title="(c) fixed-degree polynomial phases")
    axes[2].legend(frameon=False, fontsize=5.8, loc="lower right")
    axes[2].text(0.97, 0.93, r"expected $O(n^m)$ scaling",
                 transform=axes[2].transAxes, ha="right", fontsize=6, color="0.4")
    axes[2].set_ylim(10**(-4.5),10**3)
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    _save(fig, "fig10_nilpotent_phase_extended.pdf")


def _otoc_record(task_id):
    record = json.loads((DATA_DIR / f"task-{task_id:05d}.json").read_text())
    arrays = np.load(DATA_DIR / record["arrays"])
    return record, arrays


def otoc_figure():
    source = load_canonical("fig5_otoc")
    cfg = load_config(source.__file__)
    summary = load_json("otoc.json")
    fig = plt.figure(figsize=(11.6, 2.9))
    grid = fig.add_gridspec(
        1, 4, width_ratios=[1.05, 1.05, 1.28, 1.05], wspace=0.32)
    axes = [fig.add_subplot(grid[0, index]) for index in [0, 1, 2, 3]]
    #color_axis = fig.add_subplot(grid[0, 3])

    times, exact_error, baseline_error, modes_v, cutoff = source.validation(cfg["validation"])
    axes[0].semilogy(times, baseline_error, "o-", ms=3, color=BASELINE, linestyle='--',
                     label=r"$U=0J$ baseline")
    axes[0].semilogy(times, exact_error, "s-", ms=3, color=METHOD,
                     label=r"$U=8J$ sector result")
    axes[0].set(xlabel="time $t$  ($1/J$)", ylabel=r"$\max_i|C-C_{\mathrm{Fock}}|$",
                title=fr"(a) validation and interaction effect, $n={modes_v}$", ylim=(1e-16, 5))
    axes[0].legend(frameon=False, fontsize=6.3, loc="center right")
    #axes[0].text(0.54, 0.05, fr"Fock dim ${cutoff}^{modes_v}={cutoff ** modes_v}$",
    #             transform=axes[0].transAxes, fontsize=6.2, color="0.4")

    def add_otoc_chain_inset(ax: plt.Axes) -> None:
        """Sketch the interacting chain and the two operators entering the OTOC."""
        inset = ax.inset_axes([0.50, 0.65, 0.41, 0.21])
        inset.set_xlim(-0.75, 5.85)
        inset.set_ylim(-0.56, 0.55)
        inset.set_facecolor((1, 1, 1, 0.92))
        inset.axis("off")

        spacing = 1.25
        sites = spacing * np.arange(5)
        center = sites[2]
        probe = sites[4]

        inset.plot(sites, np.zeros_like(sites), color="0.35", lw=1.0, zorder=1)
        inset.scatter(
            sites, np.zeros_like(sites), s=38, facecolors="white",
            edgecolors="0.3", linewidths=0.8, zorder=2)
        inset.scatter(
            [center], [0], s=58, facecolors="white", edgecolors=NONLINEAR,
            linewidths=1.2, zorder=3)
        inset.scatter(
            [probe], [0], s=58, facecolors="white", edgecolors=METHOD,
            linewidths=1.2, zorder=3)

        inset.text(
            -0.54, 0, r"$\cdots$", ha="center", va="center",
            fontsize=6, clip_on=False)
        inset.text(
            5.63, 0, r"$\cdots$", ha="center", va="center",
            fontsize=6, clip_on=False)
        inset.text(
            0.5 * (sites[0] + sites[1]), 0.11, r"$J$",
            ha="center", va="bottom", fontsize=6.2)
        inset.text(
            center, 0.17, r"$U$",
            ha="center", va="bottom", fontsize=6.2, color=NONLINEAR)
        inset.text(
            center, -0.22, r"$\hat n_c$",
            ha="center", va="top", fontsize=6.2, color=NONLINEAR)
        inset.text(
            probe, -0.22, r"$\hat n_i(t)$",
            ha="center", va="top", fontsize=6.2, color=METHOD)
        inset.annotate(
            "", xy=(probe - 0.18, 0.42), xytext=(center + 0.18, 0.42),
            arrowprops={
                "arrowstyle": "->",
                "color": "0.4",
                "lw": 0.7,
                "connectionstyle": "arc3,rad=-0.22",
            })

    add_otoc_chain_inset(axes[0])

    image = None
    for ax, task_id, interaction, panel in zip(
            axes[1:3], [54, 63], [0, 8], ["(b)", "(c)"]):
        record, arrays = _otoc_record(task_id)
        values = arrays["otoc"]
        times_cluster = arrays["times"]
        n = record["task"]["modes"]
        sites = np.arange(n) - n // 2
        image = ax.imshow(np.log10(values + 1e-12), origin="lower", aspect="auto",
                          extent=[sites[0], sites[-1], times_cluster[0], times_cluster[-1]],
                          cmap="inferno", vmin=-8, vmax=-2)
        ax.set_xlabel(r"displacement $i-c$")
        ax.set_title(fr"{panel} $U={interaction}J$, $n={n}$")
        ax.axvline(
            0, color=LIGHT_COLORS["blue_koi"], lw=0.7, ls=":")
    axes[1].set_ylabel("time $t$  ($1/J$)")
    #axes[1].text(0.04, 0.08, r"$d_{400,2}=80200$",
    #             transform=axes[1].transAxes, va="top", fontsize=6.5, color="white")
    axes[1].set_xlim(-150,150)
    axes[2].set_xlim(-150,150)
    cbar = fig.colorbar(image)
    cbar.set_ticks([-8, -6, -4, -2])
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.ax.set_title(r"$\log_{10}C_\psi(i,t)+10^{-12}$", fontsize=6.5)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")

    grouped = defaultdict(list)
    for row in summary["tasks"]:
        grouped[(row["modes"], row["U"])].append(
            row["front_velocities"]["1e-05"])
    for modes, color, fill, marker in zip(
            [120, 200, 300, 400], SERIES_COLORS, SERIES_FILLS,
            ["o", "s", "^", "D"]):
        interactions = sorted({interaction for size, interaction in grouped if size == modes})
        samples = [finite(grouped[(modes, interaction)], "OTOC velocities")
                   for interaction in interactions]
        medians = np.asarray([np.median(values) for values in samples])
        lower = np.asarray([np.min(values) for values in samples])
        upper = np.asarray([np.max(values) for values in samples])
        axes[3].plot(interactions, medians, marker=marker, color=color, ms=3.5,
                     label=fr"$n={modes}$")
        axes[3].fill_between(
            interactions, lower, upper, color=fill, alpha=0.24)
    axes[3].set(xlabel="interaction $U/J$", ylabel=r"threshold front $v_{10^{-5}}$",
                title="(d) threshold-front slopes")
    #axes[3].yaxis.set_label_position("right")
    axes[3].legend(frameon=False, fontsize=6.2, ncol=2, loc='center right', bbox_to_anchor=(0.95, 0.75))
    #axes[3].text(0.96, 0.1, "range over three seeds",
    #            transform=axes[3].transAxes, ha="right", fontsize=6, color="0.4")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.20, top=0.88)
    _save(fig, "fig5_otoc_extended.pdf")


def topology_figure():
    summary = load_json("topology.json")
    multiplets = summary["multiplets"]
    representative = next(row for row in multiplets
                          if row["L"] == 8 and row["U"] == 10
                          and row["flux"] == 0.25 and row["grid"] == 8)
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.75))
    curvature = finite(representative["curvature"], "Berry curvature")
    image = axes[0].imshow(curvature.T, origin="lower", cmap="RdBu_r",
                           extent=[0, 1, 0, 1], aspect="equal")
    axes[0].set(xlabel=r"$\theta_x/2\pi$", ylabel=r"$\theta_y/2\pi$",
                title=r"(a) multiplet Berry curvature")
    axes[0].text(0.04, 0.94, r"$L=8,\ U=10J,\ C=+4$",
                 transform=axes[0].transAxes, va="top", fontsize=6.5,
                 color="white",
                 bbox={"facecolor": "black", "alpha": 0.45,
                       "edgecolor": "none", "pad": 1.5})
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("plaquette curvature", fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    flux_markers = {0.25: "^", -0.25: "v"}
    u_colors = {8.0: METHOD, 10.0: BASELINE, 16.0: NONLINEAR}
    u_offsets = {8.0: -0.18, 10.0: 0.0, 16.0: 0.18}
    certified = []
    uncertified = []
    for row in multiplets:
        target = certified if row["minimum_gap"] > 1e-8 and row["integer_distance"] < 0.05 else uncertified
        target.append(row)
    for row in certified:
        axes[1].scatter(row["L"] + u_offsets[row["U"]], row["chern"],
                        marker=flux_markers[row["flux"]],
                        s=27, facecolor=u_colors[row["U"]], edgecolor="white",
                        linewidth=0.35, zorder=3)
    for row in uncertified:
        axes[1].scatter(row["L"] + u_offsets[row["U"]], row["chern"],
                        marker="x", s=38, color="black", zorder=4)
    axes[1].axhline(4, color="0.8", lw=0.7)
    axes[1].axhline(-4, color="0.8", lw=0.7)
    axes[1].set(xlabel="linear size $L$", ylabel="multiplet Chern number",
                title="(b) quantization and flux reversal",
                xticks=[8, 12, 16], ylim=(-5.2, 5.2))
    legend = [
        Line2D([], [], marker="^", ls="none", color="0.3", label=r"$\phi=+1/4$"),
        Line2D([], [], marker="v", ls="none", color="0.3", label=r"$\phi=-1/4$"),
        Line2D([], [], marker="x", ls="none", color="black", label="gap closure"),
        Line2D([], [], marker="o", ls="none", color=METHOD, label=r"$U=8J$"),
        Line2D([], [], marker="o", ls="none", color=BASELINE, label=r"$U=10J$"),
        Line2D([], [], marker="o", ls="none", color=NONLINEAR, label=r"$U=16J$"),
    ]
    axes[1].legend(handles=legend, frameon=False, fontsize=5.8, ncol=2,
                   loc="center right")

    for size, color in zip([8, 12, 16], [METHOD, BASELINE, NONLINEAR]):
        for flux, linestyle in [(0.25, "-"), (-0.25, "--")]:
            rows = sorted((row for row in multiplets
                           if row["L"] == size and row["flux"] == flux
                           and row["grid"] == 8), key=lambda row: row["U"])
            axes[2].semilogy([row["U"] for row in rows],
                             [max(row["minimum_gap"], 1e-12) for row in rows],
                             marker=flux_markers[flux], ms=3.5, color=color,
                             ls=linestyle, label=fr"$L={size}$" if flux > 0 else None)
    axes[2].scatter([8], [1e-12], marker="x", color="black", s=34, zorder=5)
    axes[2].set(xlabel="interaction $U/J$", ylabel="minimum subband gap",
                title="(c) gap and finite-size check", xticks=[8, 10, 16])
    axes[2].legend(frameon=False, fontsize=6.5)
    axes[2].text(0.96, 0.06, "solid/dashed: $\\phi=\\pm1/4$",
                 transform=axes[2].transAxes, ha="right", fontsize=6, color="0.4")
    fig.tight_layout()
    _save(fig, "fig6b_doublon_chern_extended.pdf")


@lru_cache(maxsize=1)
def _edge_spectrum_data():
    source = load_canonical("fig6_doublon_topology")
    cfg = load_config(source.__file__)
    system = cfg["system"]
    spectra = cfg["spectra"]
    selection = cfg["selection"]
    size, interaction = system["L"], system["U"]

    # ARPACK otherwise chooses a random starting vector. Restoring the NumPy
    # state keeps this publication build deterministic without affecting the
    # random streams of other experiments.
    random_state = np.random.get_state()
    np.random.seed(1729)
    try:
        records = {}
        for flux in (spectra["flux_nonzero"], spectra["flux_zero"]):
            energies, _, fractions, edge_weights, _, _, _, _ = (
                source.sector_spectrum(
                    size, flux, interaction, spectra["kband"]))
            selected = (
                (fractions > selection["doublon_frac_thr"])
                & (energies > 0.5 * interaction)
            )
            records[flux] = {
                "energies": energies[selected],
                "edge_weights": edge_weights[selected],
            }
    finally:
        np.random.set_state(random_state)
    return cfg, records


def _add_path_arrows(ax, path, color):
    displacement = np.linalg.norm(path - path[0], axis=1)
    candidates = np.flatnonzero(displacement >= 0.9)
    index = int(candidates[0]) if len(candidates) else min(len(path) - 1, 3)
    index = max(3, index)
    ax.annotate(
        "",
        xy=path[index],
        xytext=path[index - 3],
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.0,
            "mutation_scale": 8.5,
        },
        zorder=6,
    )


def topological_doublon_figure():
    spectrum_cfg, spectrum_records = _edge_spectrum_data()
    transport_source, transport_cfg, center, transport = (
        _chiral_transport_data())
    multiplets = load_json("topology.json")["multiplets"]

    system = spectrum_cfg["system"]
    spectra = spectrum_cfg["spectra"]
    selection = spectrum_cfg["selection"]
    size, interaction = system["L"], system["U"]
    plus_flux = transport_cfg["panels"]["flux_plus"]
    minus_flux = transport_cfg["panels"]["flux_minus"]
    plus = transport[plus_flux]
    minus = transport[minus_flux]

    fig = plt.figure(figsize=(11.5, 3.0))
    grid = fig.add_gridspec(
        2, 4,
        width_ratios=[1.05, 1.0, 1.24, 1.0],
        height_ratios=[1.0, 0.075],
        hspace=0.58,
        wspace=0.62,
    )
    axes = np.asarray([
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[:, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[:, 3]),
    ])
    spectrum_color_axis = fig.add_subplot(grid[1, 0])
    flux_color_axis = fig.add_subplot(grid[1, 2])

    energy_offsets = np.concatenate([
        record["energies"] - 2 * interaction
        for record in spectrum_records.values()
    ])
    energy_pad = 0.05 * np.ptp(energy_offsets)
    spectrum_image = None
    grouped_fluxes = (spectra["flux_zero"], spectra["flux_nonzero"])
    for group, flux in enumerate(grouped_fluxes):
        record = spectrum_records[flux]
        order = np.argsort(record["energies"])
        ranks = np.linspace(group - 0.34, group + 0.34, len(order))
        spectrum_image = axes[0].scatter(
            ranks,
            record["energies"][order] - 2 * interaction,
            c=record["edge_weights"][order],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=20,
            edgecolor="none",
        )
    axes[0].axvline(0.5, color="0.82", lw=0.6)
    axes[0].set(
        ylabel=r"$(E_\alpha-2U)/J$",
        title="(a) edge-localized spectrum",
        xlim=(-0.46, 1.46),
        ylim=(float(energy_offsets.min() - energy_pad),
              float(energy_offsets.max() + energy_pad)),
        xticks=[0, 1],
        xticklabels=[r"$\phi=0$", r"$\phi=1/4$"],
    )
    spectrum_colorbar = fig.colorbar(
        spectrum_image, cax=spectrum_color_axis, orientation="horizontal")
    spectrum_colorbar.set_label(r"boundary weight $\eta_\alpha$", fontsize=8)
    spectrum_colorbar.set_ticks([0, selection["edge_loc_thr"], 1])
    spectrum_colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    spectrum_colorbar.ax.xaxis.labelpad = 5

    chern_rows = sorted(
        (
            row for row in multiplets
            if row["U"] == interaction
            and row["grid"] == 8
            and row["flux"] in (plus_flux, minus_flux)
            and row["L"] in (8, 12, 16)
        ),
        key=lambda row: (row["flux"], row["L"]),
    )
    for flux, color, marker, label in (
        (plus_flux, METHOD, "^", r"$+\phi$"),
        (minus_flux, THEORY, "v", r"$-\phi$"),
    ):
        rows = [row for row in chern_rows if row["flux"] == flux]
        axes[1].plot(
            [row["L"] for row in rows],
            [row["chern"] for row in rows],
            marker=marker,
            color=color,
            ms=5.5,
            label=label,
        )
    axes[1].axhline(4, color="0.82", lw=0.7, linestyle='dotted')
    axes[1].axhline(-4, color="0.82", lw=0.7, linestyle='dotted')
    minimum_gap = min(row["minimum_gap"] for row in chern_rows)
    axes[1].set(
        xlabel="linear size $L$",
        ylabel="multiplet Chern number",
        title=fr"(b) Chern quantization, $U={interaction:g}J$",
        xticks=[8, 12, 16],
        yticks=[-4, 0, 4],
        ylim=(-5.1, 5.1),
    )
    #axes[1].text(
    #    0.5, 0.50, fr"$\Delta_{{\min}}\geq {minimum_gap:.3f}J$",
    #    transform=axes[1].transAxes, ha="center", va="center",
    #    fontsize=6.2, color="0.38",)
    axes[1].legend(frameon=False, fontsize=8, loc="center right")

    plus_grid = plus["density"][-1].reshape(size, size).T
    minus_grid = minus["density"][-1].reshape(size, size).T
    difference = plus_grid - minus_grid
    limit = float(np.max(np.abs(difference)))
    #flux_cmap = LinearSegmentedColormap.from_list(
    #    "flux_difference", [THEORY, "white", METHOD])
    image = axes[2].imshow(
        difference,
        origin="lower",
        cmap='RdYlGn',
        vmin=-limit,
        vmax=limit,
        extent=[0, size - 1, 0, size - 1],
    )
    axes[2].plot(
        plus["com"][:, 0], plus["com"][:, 1],
        color=METHOD, lw=1.25, label=r"$+\phi$")
    axes[2].plot(
        minus["com"][:, 0], minus["com"][:, 1],
        color=THEORY, lw=1.25, label=r"$-\phi$")
    _add_path_arrows(axes[2], plus["com"], METHOD)
    _add_path_arrows(axes[2], minus["com"], THEORY)
    axes[2].plot(
        plus["com"][0, 0], plus["com"][0, 1], "o",
        color="white", ms=4.2, mec="black", mew=0.45, zorder=7)
    axes[2].set(
        xlabel="site $x$",
        ylabel="site $y$",
        title="(c) flux-reversed edge motion",
        xticks=[0, size - 1],
        yticks=[0, size - 1],
    )
    axes[2].set_anchor("N")
    axes[2].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(0.9, 0.9))
    colorbar = fig.colorbar(
        image, cax=flux_color_axis, orientation="horizontal")
    colorbar.set_label(r"$\Delta\rho_d(T)$", fontsize=8)
    colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    colorbar.ax.xaxis.labelpad = 5

    axes[3].plot(
        plus["times"],
        transport_source.winding(plus["com"], center),
        color=METHOD,
        label=r"$+\phi$",
    )
    axes[3].plot(
        minus["times"],
        transport_source.winding(minus["com"], center),
        color=THEORY,
        label=r"$-\phi$",
    )
    axes[3].axhline(0, color="0.65", lw=0.5)
    axes[3].set(
        xlabel="time $t$  ($1/J$)",
        ylabel="chiral winding (turns)",
        title="(d) chiral winding",
    )
    axes[3].legend(frameon=False, fontsize=8)

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.13, top=0.90, wspace=0.01)
    _save(fig, "fig6_topological_doublon_extended.pdf")


def squeezing_figure():
    source = load_canonical("fig4_squeezing")
    cfg = load_config(source.__file__)
    summary = load_json("squeezing.json")
    local = json.loads(
        (ROOT / "experiments" / "fig4_squeezing" / "results.json").read_text())
    system = cfg["system"]
    n, number, interaction, hopping = (
        system["modes"], system["N"], system["U"], system["J"])
    initial = system["init"]
    fig, axes = plt.subplots(1, 4, figsize=(11.1, 2.75))

    convergence = cfg["convergence"]
    times = np.linspace(0, convergence["t_max"], convergence["n_time"])
    scaled_times = hopping * times
    reference = source.trajectory(
        n, number, convergence["reference_order"], interaction, hopping,
        initial, convergence["r"], times)
    axes[0].plot(scaled_times, reference, color="black", lw=1.8,
                 label=fr"reference $k={convergence['reference_order']}$")
    for order in convergence["orders"]:
        axes[0].plot(
            scaled_times,
            source.trajectory(n, number, order, interaction, hopping,
                              initial, convergence["r"], times),
            color=source.C[order], ls="--", lw=1.2, label=fr"$k={order}$")
    axes[0].set(
        xlabel=r"dimensionless time $Jt$",
        ylabel=r"$\langle\hat n_0\hat n_1\rangle$",
        title=fr"(a) convergence, $r/J={convergence['r'] / hopping:g}$")
    axes[0].legend(
        frameon=False, fontsize=6.2, ncol=1, loc="upper left",
        bbox_to_anchor=(0.1, 0.98))
    axes[0].set_ylim(0.22, 1.35)

    error_cfg = cfg["error"]
    amplitudes = np.logspace(error_cfg["log10_r_min"], error_cfg["log10_r_max"],
                             error_cfg["n_r"])
    scaled_amplitudes = amplitudes / hopping
    error_times = np.linspace(0, error_cfg["t_max"], error_cfg["n_time"])
    references = [
        source.trajectory(n, number, error_cfg["reference_order"], interaction,
                          hopping, initial, amplitude, error_times)
        for amplitude in amplitudes
    ]
    for order in error_cfg["orders"]:
        errors = np.asarray([
            np.max(np.abs(source.trajectory(
                n, number, order, interaction, hopping, initial, amplitude,
                error_times) - ref))
            for amplitude, ref in zip(amplitudes, references)
        ])
        slope, stderr, mask = source.fit_power(
            amplitudes, errors, error_cfg["fit_r_min"], error_cfg["fit_r_max"],
            error_cfg["fit_error_floor"])
        axes[1].loglog(
            scaled_amplitudes, errors, "o-", color=source.C[order], ms=3,
            label=fr"$k={order}$: ${slope:.3f}\pm{stderr:.3f}$")
        axes[1].loglog(scaled_amplitudes[mask], errors[mask], "o",
                       color=source.C[order], ms=4, mfc="white")
    axes[1].set(
        xlabel=r"squeezing strength $r/J$",
        ylabel=r"$\epsilon_k=\max_t|F_k-F_{\rm ref}|$",
        title="(b) perturbative error order")
    axes[1].legend(frameon=False, fontsize=6.1, loc="lower right")

    for order in cfg["cost"]["orders"]:
        entries = sorted(
            (entry for entry in local["cost"] if entry["order"] == order),
            key=lambda entry: entry["modes"])
        modes = finite([entry["modes"] for entry in entries], "squeezing modes")
        med = finite([entry["median_seconds"] for entry in entries], "squeezing timings")
        lo = finite([entry["q25_seconds"] for entry in entries], "squeezing q25")
        hi = finite([entry["q75_seconds"] for entry in entries], "squeezing q75")
        axes[2].loglog(modes, med, "o-", color=source.C[order], ms=3,
                       label=fr"$k={order}$, $d_k=\mathcal{{O}}(n^{{{number + 2 * order}}})$")
        axes[2].fill_between(modes, lo, hi, color=source.C[order], alpha=0.15)
    axes[2].set(
        xlabel="modes $n$", ylabel="propagation time (s)",
        title="(c) projected-sector propagation")
    axes[2].legend(frameon=False, fontsize=6.1, loc="lower right")
    axes[2].set_xlim(2.5, 200)
    #axes[2].text(
    #    0.04, 0.95,
    #    fr"$r/J={cfg['cost']['r'] / hopping:g},\ Jt={cfg['cost']['time'] * hopping:g}$",
    #    transform=axes[2].transAxes, va="top", fontsize=6, color="0.4")

    fits = [row for row in summary["fits"] if row["order"] <= 3]
    mode_offsets = {2: -0.16, 4: -0.08, 6: 0.0, 8: 0.08, 12: 0.16}
    mode_strengths = {2: 0.42, 4: 0.56, 6: 0.70, 8: 0.84, 12: 1.0}

    def faded(color, strength):
        rgb = np.asarray(to_rgb(color))
        return tuple(1.0 - strength * (1.0 - rgb))

    for row in fits:
        x = row["order"] + mode_offsets[row["modes"]]
        strength = mode_strengths[row["modes"]]
        if row["leakage_points"] >= 3:
            axes[3].errorbar(
                x, row["leakage_slope"], yerr=row["leakage_stderr"],
                marker="^", ms=3.2, color=faded(BASELINE, strength),
                ls="none", capsize=1.5)
        if row["observable_points"] >= 3:
            axes[3].errorbar(
                x, row["observable_slope"], yerr=row["observable_stderr"],
                marker="o", ms=3.2, color=faded(METHOD, strength),
                ls="none", capsize=1.5)
    orders = np.arange(4)
    axes[3].plot(
        orders, orders + 1, color=BASELINE, ls=":", lw=1)
    axes[3].plot(
        orders, 2 * (orders + 1), color=METHOD, ls="--", lw=1)
    axes[3].set(
        xlabel="band depth $k$", ylabel="fitted exponent",
        title="(d) fitted exponents", xticks=orders,
        ylim=(0.6, 8.5))
    quantity_handles = [
        Line2D(
            [], [], marker="^", color=BASELINE, ls=":",
            label=r"state leakage: $k+1$"),
        Line2D(
            [], [], marker="o", color=METHOD, ls="--",
            label=r"readout error: $2(k+1)$"),
    ]
    quantity_legend = axes[3].legend(
        handles=quantity_handles, frameon=False, fontsize=5.8,
        loc="upper left")
    axes[3].add_artist(quantity_legend)
    size_handles = [
        Line2D(
            [], [], color=faded("0.15", mode_strengths[mode]), lw=2.5,
            label=fr"$n={mode}$")
        for mode in mode_strengths
    ]
    axes[3].legend(
        handles=size_handles, frameon=False, fontsize=5.5,
        loc="center left", bbox_to_anchor=(0.01, 0.56),
        handlelength=1.5, handletextpad=0.5, labelspacing=0.25)
    fig.tight_layout()
    _save(fig, "fig4_squeezing_extended.pdf")


def control_figure():
    source = load_canonical("fig9_kerr_control_2d")
    cfg = load_config(source.__file__)
    summary = load_json("control.json")
    paper_style()
    L, modes, carrier, coordinates, hopping, kerr, ndiag, pots, weights, psi0 = source.build(cfg)
    target_site = cfg["system"]["target_site"]
    target = np.eye(modes)[target_site]
    depth = cfg["optimize"]["map_layers"]

    generators, diagonals, _ = source.layers(depth, modes, hopping, kerr, ndiag, pots)
    rng = np.random.default_rng(cfg["seed"])
    theta = 0.15 * rng.standard_normal(len(generators))
    from gbosons import variational as var
    _, gradient, _ = var.sparse_layered_loss_and_grad(
        theta, generators, psi0, weights, target, diagonals)
    finite_difference = np.zeros_like(gradient)
    for index in range(len(gradient)):
        shift = np.zeros_like(gradient)
        shift[index] = 1e-6
        plus = var.sparse_layered_loss_and_grad(
            theta + shift, generators, psi0, weights, target, diagonals)[0]
        minus = var.sparse_layered_loss_and_grad(
            theta - shift, generators, psi0, weights, target, diagonals)[0]
        finite_difference[index] = (plus - minus) / 2e-6
    gradient_error = np.max(np.abs(gradient - finite_difference))

    _, best = source.optimise(cfg, depth, True, modes, hopping, kerr, ndiag,
                              pots, weights, psi0, target, target_site)
    fig = plt.figure(figsize=(9.8, 3.0))
    ax_a = fig.add_axes([0.07, 0.20, 0.24, 0.66])
    ax_b = fig.add_axes([0.39, 0.39, 0.31, 0.47])
    ax_b_success = fig.add_axes([0.39, 0.15, 0.31, 0.15], sharex=ax_b)
    ax_c = fig.add_axes([0.75, 0.20, 0.21, 0.66])
    limit = 1.05 * np.max(np.abs(finite_difference))
    ax_a.plot([-limit, limit], [-limit, limit], color="0.7", lw=1)
    ax_a.scatter(finite_difference, gradient, s=10, color=METHOD)
    ax_a.set(xlabel="finite-difference grad", ylabel="reverse-mode grad",
             title="(a) gradient validation")

    statistics = [row for row in summary["optimization_statistics"]
                  if row["geometry"] == 0]
    ax_b.axhline(0.5, color="0.55", ls="--", lw=1)
    size_colors = {5: METHOD, 7: BASELINE, 9: NONLINEAR}
    size_fills = {
        5: PAPER_FILLS["method"],
        7: PAPER_FILLS["baseline"],
        9: PAPER_FILLS["nonlinear"],
    }
    for size in [5, 7, 9]:
        rows = sorted((row for row in statistics
                       if row["L"] == size and row["model"] == "kerr"),
                      key=lambda row: row["depth"])
        depths = finite([row["depth"] for row in rows], "control depths")
        med = finite([row["median"] for row in rows], "control medians")
        lo = finite([row["q25"] for row in rows], "control q25")
        hi = finite([row["q75"] for row in rows], "control q75")
        success = finite(
            [row["fraction_above_passive_bound"] for row in rows],
            "control success fractions")
        ax_b.plot(depths, med, "o-", ms=3.7,
                  color=size_colors[size], label=fr"Kerr, $L={size}$")
        ax_b.fill_between(depths, lo, hi,
                          color=size_fills[size], alpha=0.24)
        ax_b_success.plot(depths, success, "o-", ms=2.8, lw=0.9,
                          color=size_colors[size])
    passive_rows = sorted((row for row in statistics
                           if row["L"] == 5 and row["model"] == "passive"),
                          key=lambda row: row["depth"])
    ax_b.plot([row["depth"] for row in passive_rows],
              [row["median"] for row in passive_rows],
              "s-", color="0.45", ms=3, label=r"passive, $L=5$")
    ax_b.set(ylabel=r"doublon probability $p_{\rm target}$",
             title="(b) optimization versus depth", ylim=(-0.03, 1.05),
             xlim=(0.8, 8.2), xticks=np.arange(1, 9))
    ax_b.tick_params(axis="x", labelbottom=False)
    ax_b.legend(frameon=False, fontsize=6.3, loc="lower right")
    ax_b_success.axhline(0.5, color="0.8", ls=":", lw=0.8)
    ax_b_success.set(
        xlabel="ansatz depth", ylabel="fraction\nabove $1/2$",
        xlim=(0.8, 8.2), ylim=(-0.04, 1.04),
        xticks=np.arange(1, 9), yticks=[0, 0.5, 1])
    ax_b_success.tick_params(labelsize=6)

    density = np.zeros((L, L))
    for site, (x, y) in enumerate(coordinates):
        density[int(x), int(y)] = best["prof"][site]
    image = ax_c.imshow(density.T, origin="lower", cmap="inferno",
                        vmin=0, vmax=density.max())
    target_x, target_y = coordinates[target_site]
    ax_c.plot([target_x], [target_y], "o", mfc="none",
              mec=PAPER_COLORS["nonlinear"], mew=2, ms=11)
    for site in cfg["system"]["input_sites"]:
        x, y = coordinates[site]
        ax_c.plot([x], [y], "x",
                  color=LIGHT_COLORS["blue_koi"], ms=7, mew=2)
    ax_c.set_title("(c) optimized doublon profile")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    marker_handles = [
        Line2D([], [], marker="x", color="white", ls="none", ms=6,
               mew=1.6, label="input"),
        Line2D([], [], marker="o", color="white", markerfacecolor="none",
               ls="none", ms=7, mew=1.6, label="target"),
    ]
    ax_c.legend(handles=marker_handles, frameon=False, fontsize=7,
                labelcolor="white", loc="upper right", handletextpad=0.4,
                borderaxespad=0.4)
    cbar = fig.colorbar(image, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r"pair density $p_j$", fontsize=7)
    _save(fig, "fig9_kerr_control_extended.pdf")


def contact_sheet():
    canonical = [
        ROOT / "notes" / "figures" / "fig7_gaussian.pdf",
        ROOT / "notes" / "figures" / "fig8_beyond_gaussian.pdf",
        ROOT / "notes" / "figures" / "fig2_bounded_n.pdf",
        ROOT / "notes" / "figures" / "fig3_doublon.pdf",
        ROOT / "notes" / "figures" / "fig10_nilpotent_phase.pdf",
        ROOT / "notes" / "figures" / "fig5_otoc.pdf",
        ROOT / "notes" / "figures" / "fig6_doublon_topology.pdf",
        ROOT / "notes" / "figures" / "fig4_squeezing.pdf",
        ROOT / "notes" / "figures" / "fig9_kerr_control_2d.pdf",
        ROOT / "notes" / "figures" / "fig6_chiral_transport.pdf",
    ]
    extended = [
        OUTPUT_DIR / "fig7_gaussian_extended.pdf",
        OUTPUT_DIR / "fig8_beyond_gaussian_extended.pdf",
        OUTPUT_DIR / "fig2_bounded_n_extended.pdf",
        OUTPUT_DIR / "fig3_doublon_extended.pdf",
        OUTPUT_DIR / "fig10_nilpotent_phase_extended.pdf",
        OUTPUT_DIR / "fig5_otoc_extended.pdf",
        OUTPUT_DIR / "fig6b_doublon_chern_extended.pdf",
        OUTPUT_DIR / "fig4_squeezing_extended.pdf",
        OUTPUT_DIR / "fig9_kerr_control_extended.pdf",
        OUTPUT_DIR / "fig6_chiral_transport_extended.pdf",
    ]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        images = []
        for index, path in enumerate(canonical + extended):
            prefix = directory / f"figure-{index:02d}"
            subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-png",
                            "-r", "90", str(path), str(prefix)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            images.append(plt.imread(prefix.with_suffix(".png")))
        fig, axes = plt.subplots(len(canonical), 2, figsize=(12, 24))
        for row in range(len(canonical)):
            for column, title in [(0, "canonical"), (1, "extended")]:
                ax = axes[row, column]
                ax.imshow(images[row + column * len(canonical)])
                ax.axis("off")
                ax.set_title(title if row == 0 else "", fontsize=11)
        fig.tight_layout()
        destination = OUTPUT_DIR / "before_after_contact_sheet.png"
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        print(f"wrote {destination}")


def main():
    manifest = load_json("manifest.json")
    if not manifest["validation_passed"]:
        raise RuntimeError("extended figures require validated production data")
    paper_style()
    gaussian_figure()
    beyond_gaussian_figure()
    bounded_figure()
    doublon_figure()
    cubic_figure()
    otoc_figure()
    topology_figure()
    topological_doublon_figure()
    squeezing_figure()
    control_figure()
    chiral_transport_figure()
    contact_sheet()


if __name__ == "__main__":
    main()
