"""Gradient validation and matched-depth control in the two-photon carrier."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from gbosons import benchmarking as bm
from gbosons import bounded_n as bn, lattices, variational as var
from gbosons.plotting import (
    LIGHT_COLORS, PAPER_COLORS, PAPER_FILLS,
    paper_style, figures_dir, load_config)

C = {"kerr": PAPER_COLORS["nonlinear"], "passive": PAPER_COLORS["baseline"]}


def build(cfg):
    s = cfg["system"]
    L = s["L"]; M = L * L
    h, xy = lattices.hofstadter(L, L, flux=s["flux"])
    basis, index, d = bn.sector_basis(M, 2)
    Hhop = bn.passive_hamiltonian(h, basis, index)
    chi = np.zeros((M, M)); np.fill_diagonal(chi, 1.0)
    Hker = bn.cross_kerr_hamiltonian(chi, basis)
    ndiag = [bn.number_diag(k, basis) for k in range(M)]
    pots = [sp.diags(ndiag[k]).tocsr() for k in range(M)]
    occ = np.array(basis, float)
    weights = (0.5 * occ * (occ - 1)).T                       # (M, d) on-site pair prob
    psi0 = bn.basis_state([1 if i in s["input_sites"] else 0 for i in range(M)], index)
    return L, M, d, xy, Hhop, Hker, ndiag, pots, weights, psi0


def layers(p, M, Hhop, Hker, ndiag, pots):
    gens, diags, kmask = [], [], []
    for _ in range(p):
        gens.append(Hhop); diags.append(None); kmask.append(False)
        gens.append(Hker); diags.append(np.asarray(Hker.diagonal())); kmask.append(True)
        for k in range(M):
            gens.append(pots[k]); diags.append(ndiag[k]); kmask.append(False)
    return gens, diags, np.array(kmask)


def optimise(cfg, p, kerr_on, M, Hhop, Hker, ndiag, pots, weights, psi0, target, tgt):
    gens, diags, kmask = layers(p, M, Hhop, Hker, ndiag, pots)
    op = cfg["optimize"]
    mask = np.ones(len(kmask), bool) if kerr_on else ~kmask

    def lg(t):
        return var.sparse_layered_loss_and_grad(t, gens, psi0, weights, target, diags)

    records = []
    for r in range(op["restarts"]):
        rng = np.random.default_rng(op["restart_seed"] + r)
        t0 = 0.15 * rng.standard_normal(len(kmask))
        if not kerr_on:
            t0[kmask] = 0.0
        _, hist, o = var.adam(t0, lg, steps=op["steps"], lr=op["lr"], mask=mask)
        records.append({"restart": r, "loss": float(hist[-1]),
                        "ptgt": float(o[tgt]), "prof": o})
    best = max(records, key=lambda record: record["ptgt"])
    return records, best


def main():
    cfg = load_config(__file__)
    paper_style()
    L, M, d, xy, Hhop, Hker, ndiag, pots, weights, psi0 = build(cfg)
    tgt = cfg["system"]["target_site"]
    target = np.eye(M)[tgt]
    op = cfg["optimize"]

    # (a) trust gate: gradient vs finite difference at map depth
    gens, diags, _ = layers(op["map_layers"], M, Hhop, Hker, ndiag, pots)
    rng = np.random.default_rng(cfg["seed"])
    th = 0.15 * rng.standard_normal(len(gens))
    _, g, _ = var.sparse_layered_loss_and_grad(th, gens, psi0, weights, target, diags)
    gfd = np.zeros_like(g)
    for j in range(len(g)):
        e = np.zeros_like(g); e[j] = 1e-6
        lp = var.sparse_layered_loss_and_grad(th + e, gens, psi0, weights, target, diags)[0]
        lm = var.sparse_layered_loss_and_grad(th - e, gens, psi0, weights, target, diags)[0]
        gfd[j] = (lp - lm) / 2e-6
    gerr = np.max(np.abs(g - gfd))

    # (b) matched-depth sweep: median and spread across deterministic restarts
    depths = op["layers"]
    summaries = {"passive": [], "kerr": []}; kbest = None
    for p in depths:
        passive_records, _ = optimise(cfg, p, False, M, Hhop, Hker, ndiag,
                                      pots, weights, psi0, target, tgt)
        kerr_records, best_kerr = optimise(cfg, p, True, M, Hhop, Hker, ndiag,
                                           pots, weights, psi0, target, tgt)
        for name, records in [("passive", passive_records), ("kerr", kerr_records)]:
            values = np.array([record["ptgt"] for record in records])
            summaries[name].append({
                "depth": p, "values": values.tolist(),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            })
        if p == op["map_layers"]:
            kbest = best_kerr

    fig = plt.figure(figsize=(9.6, 3.0))
    axA = fig.add_axes([0.07, 0.20, 0.24, 0.66])
    axB = fig.add_axes([0.40, 0.20, 0.25, 0.66])
    axC = fig.add_axes([0.74, 0.20, 0.20, 0.66])

    # (a) trust gate scatter
    lim = 1.05 * np.max(np.abs(gfd))
    axA.plot([-lim, lim], [-lim, lim], color="0.7", lw=1.0, zorder=1)
    axA.scatter(gfd, g, s=10, color=C["kerr"], zorder=2)
    axA.set_xlabel("finite-difference grad"); axA.set_ylabel("reverse-mode grad")
    axA.set_title("(a) trust gate", fontsize=10)
    axA.text(0.05, 0.9, fr"max $|\Delta|={gerr:.0e}$", transform=axA.transAxes, fontsize=8)

    # (b) passive ceiling and nonlinear control
    axB.axhline(0.5, color="0.6", ls="--", lw=1.0)
    for name, marker, label in [("passive", "s", "passive"), ("kerr", "o", "with Kerr")]:
        med = np.array([entry["median"] for entry in summaries[name]])
        lo = np.array([entry["q25"] for entry in summaries[name]])
        hi = np.array([entry["q75"] for entry in summaries[name]])
        axB.plot(depths, med, marker + "-", color=C[name], ms=4.5, label=label)
        axB.fill_between(depths, lo, hi, color=PAPER_FILLS[
            "nonlinear" if name == "kerr" else "baseline"], alpha=0.24)
    axB.set_xlabel("ansatz depth"); axB.set_ylabel(r"doublon prob. $p_{\mathrm{target}}$")
    axB.set_title("(b) past the passive bound", fontsize=10)
    axB.set_ylim(0, 1.05); axB.set_xticks(depths)
    axB.text(depths[0], 0.53, "passive bound $1/2$", fontsize=7, color="0.4", va="bottom")
    axB.legend(frameon=False, fontsize=7.5, loc="lower right")

    # (c) 2D learned pair-density map (Kerr)
    grid = np.zeros((L, L))
    for k, (x, y) in enumerate(xy):
        grid[int(x), int(y)] = kbest["prof"][k]
    im = axC.imshow(grid.T, origin="lower", cmap="inferno", vmin=0, vmax=grid.max())
    tx, ty = xy[tgt]
    axC.plot([tx], [ty], "o", mfc="none",
             mec=PAPER_COLORS["nonlinear"], mew=2.0, ms=12)
    for s_ in cfg["system"]["input_sites"]:
        ix, iy = xy[s_]
        axC.plot([ix], [iy], "x", color=LIGHT_COLORS["blue_koi"], ms=7, mew=2.0)
    axC.set_title(f"(c) 2D doublon density\n$p_{{\\rm tgt}}={kbest['ptgt']:.2f}$", fontsize=9)
    axC.set_xticks([]); axC.set_yticks([])
    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)

    fig.text(0.5, 0.005, fr"two-photon carrier $d_{{{M},2}}={d}$; "
             r"$\circ$ target, $\times$ input", ha="center", fontsize=7, color="0.4")

    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    serializable = {name: summaries[name] for name in summaries}
    result = bm.write_results(__file__, {
        "figure": cfg["output"], "carrier_dimension": d,
        "gradient_max_absolute_error": float(gerr),
        "restarts": op["restarts"], "restart_seed": op["restart_seed"],
        "depth_sweep": serializable,
        "best_kerr_target_probability": float(kbest["ptgt"]),
        "passive_bound": 0.5,
    })
    print(f"wrote {out} and {result} (trust gate {gerr:.1e})")


if __name__ == "__main__":
    main()
