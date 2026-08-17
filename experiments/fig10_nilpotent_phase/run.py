"""Exact nilpotent cubic-phase dynamics and fixed-degree scaling."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from gbosons import benchmarking as bm
from gbosons import nilpotent_phase as nph
from gbosons.plotting import (
    PAPER_COLORS, PAPER_FILLS, figures_dir, load_config, paper_style)


COLORS = {
    "module": PAPER_COLORS["method"],
    "grid": PAPER_COLORS["reference"],
    "dimension": PAPER_COLORS["method"],
}


def layer_gates(shift, potential):
    """Chronological translation followed by a position phase gate."""
    return [
        {"kind": "translation", "shift": tuple(shift)},
        {"kind": "phase", "potential": potential},
    ]


def single_mode_layers(cfg):
    layers = []
    for shift, strength in zip(cfg["shifts"], cfg["cubic_strengths"]):
        layers.append(layer_gates((shift,), nph.monomial((3,), strength)))
    return layers


def two_mode_layers(cfg):
    weights = cfg["monomial_weights"]
    base = nph.add(
        nph.monomial((2, 1), weights["x0^2x1"]),
        nph.monomial((1, 2), weights["x0x1^2"]),
        nph.monomial((3, 0), weights["x0^3"]),
        nph.monomial((0, 3), weights["x1^3"]),
    )
    return [layer_gates(shift, nph.scale(base, strength))
            for shift, strength in zip(cfg["shifts"], cfg["cubic_strengths"])]


def module_trajectory(layers, readout):
    values = [readout({})]
    gates = []
    for layer in layers:
        gates.extend(layer)
        values.append(readout(nph.effective_phase(len(layer[0]["shift"]), gates)))
    return np.real_if_close(np.asarray(values)).real


def grid_coordinates(n, points, extent):
    x = np.linspace(-extent, extent, points, endpoint=False)
    dx = float(x[1] - x[0])
    if n == 1:
        return x, (x,), dx
    meshes = np.meshgrid(*([x] * n), indexing="ij", sparse=True)
    return x, tuple(meshes), dx


def vacuum_state(coordinates, dx):
    exponent = 0.0
    for coordinate in coordinates:
        exponent = exponent + coordinate ** 2
    psi = np.exp(-0.5 * exponent).astype(complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx ** len(coordinates))
    return psi


def apply_translation(psi, shift, dx):
    out = psi
    for axis, amount in enumerate(shift):
        wave_numbers = 2 * np.pi * np.fft.fftfreq(out.shape[axis], d=dx)
        shape = [1] * out.ndim
        shape[axis] = out.shape[axis]
        phase = np.exp(-1j * amount * wave_numbers).reshape(shape)
        transformed = np.fft.fft(out, axis=axis, norm="ortho")
        out = np.fft.ifft(phase * transformed, axis=axis, norm="ortho")
    return out


def apply_momentum(psi, variable, dx):
    wave_numbers = 2 * np.pi * np.fft.fftfreq(psi.shape[variable], d=dx)
    shape = [1] * psi.ndim
    shape[variable] = psi.shape[variable]
    transformed = np.fft.fft(psi, axis=variable, norm="ortho")
    return np.fft.ifft(wave_numbers.reshape(shape) * transformed,
                       axis=variable, norm="ortho")


def grid_momentum_moment(psi, indices, dx):
    acted = psi
    for variable in reversed(tuple(indices)):
        acted = apply_momentum(acted, variable, dx)
    return np.vdot(psi.ravel(), acted.ravel()) * dx ** psi.ndim


def grid_cumulant4(psi, variable, dx):
    moments = [1.0 + 0.0j]
    moments.extend(grid_momentum_moment(psi, (variable,) * order, dx)
                   for order in range(1, 5))
    mean = moments[1]
    variance = moments[2] - mean ** 2
    central4 = (moments[4] - 4 * mean * moments[3]
                + 6 * mean ** 2 * moments[2] - 3 * mean ** 4)
    return central4 - 3 * variance ** 2


def grid_connected_correlator(psi, first, second, dx):
    joint = grid_momentum_moment(psi, (first, second), dx)
    return (joint - grid_momentum_moment(psi, (first,), dx)
            * grid_momentum_moment(psi, (second,), dx))


def grid_trajectory(layers, points, extent, readout):
    _, coordinates, dx = grid_coordinates(len(layers[0][0]["shift"]), points, extent)
    psi = vacuum_state(coordinates, dx)
    values = [readout(psi, dx)]
    for layer in layers:
        for gate in layer:
            if gate["kind"] == "translation":
                psi = apply_translation(psi, gate["shift"], dx)
            else:
                potential = nph.evaluate(gate["potential"], coordinates)
                psi = np.exp(-1j * potential) * psi
        values.append(readout(psi, dx))
    return np.real_if_close(np.asarray(values)).real


def panel_single_mode(cfg, ax):
    layers = single_mode_layers(cfg)
    module = module_trajectory(layers, lambda phase: nph.momentum_cumulant4(phase, 0))
    grid = grid_trajectory(layers, cfg["grid_points"], cfg["extent"],
                           lambda psi, dx: grid_cumulant4(psi, 0, dx))
    certificate = grid_trajectory(layers, cfg["certificate_points"], cfg["extent"],
                                  lambda psi, dx: grid_cumulant4(psi, 0, dx))
    depths = np.arange(len(module))
    ax.plot(depths, module, color=COLORS["module"], label="coefficient propagation")
    ax.plot(depths, grid, "o", ms=3.6, mfc="white", color=COLORS["grid"],
            label="FFT-grid reference")
    ax.axhline(0, color="0.75", lw=0.7)
    ax.set_xlabel("circuit depth $L$")
    ax.set_ylabel(r"fourth cumulant $\kappa_4(\hat p)$")
    ax.set_title("(a) non-Gaussian shear")
    ax.legend(frameon=False, fontsize=7)
    return {
        "module_values": module.tolist(),
        "grid_values": grid.tolist(),
        "max_module_grid_error": float(np.max(np.abs(module - grid))),
        "max_grid_refinement_gap": float(np.max(np.abs(grid - certificate))),
        "final_cumulant": float(module[-1]),
    }


def panel_two_mode(cfg, ax):
    layers = two_mode_layers(cfg)
    module = module_trajectory(
        layers, lambda phase: nph.connected_momentum_correlator(phase, 0, 1))
    grid = grid_trajectory(
        layers, cfg["grid_points"], cfg["extent"],
        lambda psi, dx: grid_connected_correlator(psi, 0, 1, dx))
    certificate = grid_trajectory(
        layers, cfg["certificate_points"], cfg["extent"],
        lambda psi, dx: grid_connected_correlator(psi, 0, 1, dx))
    depths = np.arange(len(module))
    ax.plot(depths, module, color=COLORS["module"], label="coefficient propagation")
    ax.plot(depths, grid, "s", ms=3.4, mfc="white", color=COLORS["grid"],
            label="FFT-grid reference")
    ax.axhline(0, color="0.75", lw=0.7)
    ax.set_xlabel("circuit depth $L$")
    ax.set_ylabel(r"$\langle \hat p_0 \hat p_1\rangle_c$")
    ax.set_title("(b) cross-cubic phase")
    ax.legend(frameon=False, fontsize=7)
    return {
        "module_values": module.tolist(),
        "grid_values": grid.tolist(),
        "max_module_grid_error": float(np.max(np.abs(module - grid))),
        "max_grid_refinement_gap": float(np.max(np.abs(grid - certificate))),
        "final_connected_correlator": float(module[-1]),
    }


def scaling_layers(n, cfg):
    indices = np.arange(n, dtype=float)
    total_index = (indices[:, None, None] + indices[None, :, None]
                   + indices[None, None, :])
    base = cfg["coefficient_scale"] * np.cos(0.37 * (total_index + 3.0)) / np.sqrt(n)
    phases, shifts = [], []
    for layer in range(cfg["depth"]):
        shifts.append(cfg["shift_scale"]
                      * np.cos((layer + 1) * (indices + 1)) / np.sqrt(n))
        phases.append(base / (layer + 1))
    return phases, shifts


def panel_scaling(cfg, ax):
    entries = []
    for n in cfg["modes"]:
        phases, shifts = scaling_layers(n, cfg)
        timing = bm.timed_call(lambda: nph.effective_cubic_tensors(phases, shifts),
                               repeats=cfg["repeats"], warmups=cfg["warmups"],
                               target_seconds=cfg["target_seconds"])
        coefficients = nph.effective_cubic_tensors(phases, shifts)
        entries.append({
            "modes": n,
            "degree": cfg["degree"],
            "depth": cfg["depth"],
            "algebra_dimension": nph.algebra_dimension(n, cfg["degree"]),
            "dense_coefficient_count": int(sum(np.size(array) for array in coefficients)),
            "dense_coefficient_bytes": int(sum(np.asarray(array).nbytes
                                                for array in coefficients)),
            **timing,
        })
    modes = np.asarray([entry["modes"] for entry in entries])
    medians = np.asarray([entry["median_seconds"] for entry in entries])
    lower = np.asarray([entry["q25_seconds"] for entry in entries])
    upper = np.asarray([entry["q75_seconds"] for entry in entries])
    ax.loglog(modes, medians, "o-", ms=3.5, color=COLORS["dimension"],
              label=fr"measured, $L={cfg['depth']}$")
    ax.fill_between(modes, lower, upper, color=PAPER_FILLS["method"], alpha=0.24)
    guide = medians[-1] * (modes / modes[-1]) ** cfg["degree"]
    ax.loglog(modes, guide, "--", color="0.35", lw=1.0,
              label=fr"$d=n+\binom{{n+3}}{{3}}=O(n^{cfg['degree']})$")
    selected_ticks = [modes[0], modes[2], modes[4], modes[-1]]
    ax.set_xticks(selected_ticks, labels=[str(value) for value in selected_ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("modes $n$")
    ax.set_ylabel("coefficient-propagation time (s)")
    ax.set_title("(c) fixed-degree scaling")
    ax.legend(frameon=False, fontsize=7)
    return {"entries": entries}


def main():
    cfg = load_config(__file__)
    paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.7))
    single = panel_single_mode(cfg["single_mode"], axes[0])
    two_mode = panel_two_mode(cfg["two_mode"], axes[1])
    scaling = panel_scaling(cfg["scaling"], axes[2])
    fig.tight_layout()
    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    result = bm.write_results(__file__, {
        "figure": cfg["output"],
        "single_mode": single,
        "two_mode": two_mode,
        "scaling": scaling,
    })
    print(f"wrote {out} and {result}")
    print("single-mode module/grid error "
          f"{single['max_module_grid_error']:.2e}; two-mode "
          f"{two_mode['max_module_grid_error']:.2e}")


if __name__ == "__main__":
    main()
