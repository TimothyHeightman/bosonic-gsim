"""Finite-size edge-localised doublon modes with and without magnetic flux."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from gbosons import benchmarking as bm
from gbosons import bounded_n as bn, lattices as lat
from gbosons.plotting import paper_style, figures_dir, load_config


def sector_spectrum(L, flux, U, kband):
    M = L * L
    h, xy = lat.hofstadter(L, L, flux=flux)
    basis, index, d = bn.sector_basis(M, 2)
    chi = np.zeros((M, M)); np.fill_diagonal(chi, U / 2)
    H = (bn.passive_hamiltonian(h, basis, index)
         + bn.cross_kerr_hamiltonian(chi, basis))
    E, V = eigsh(H, k=kband, sigma=2 * U, which="LM")
    drows = np.array([index[tuple([2 if s == k else 0 for s in range(M)])] for k in range(M)])
    edge = lat.edge_sites(xy, L, L)
    Wt = np.abs(V[drows, :]) ** 2          # (M, d) doublon weight per site
    frac = Wt.sum(0)                       # doublon fraction of each eigenstate
    eloc = Wt[edge].sum(0) / np.maximum(frac, 1e-12)
    return E, V, frac, eloc, Wt, xy, edge, drows


def main():
    cfg = load_config(__file__)
    paper_style()
    sysc = cfg["system"]
    L, U = sysc["L"], sysc["U"]
    d = bn.sector_dim(L * L, 2)
    sel = cfg["selection"]
    fthr, ethr, ylim = sel["doublon_frac_thr"], sel["edge_loc_thr"], sel["band_ylim"]

    sp = cfg["spectra"]
    res = {}
    for flux in [sp["flux_nonzero"], sp["flux_zero"]]:
        E, V, frac, eloc, Wt, xy, edge, drows = sector_spectrum(L, flux, U, sp["kband"])
        dmask = (frac > fthr) & (E > 0.5 * U)
        edge_states = dmask & (eloc > ethr)
        res[flux] = dict(E=E[dmask], eloc=eloc[dmask], Wt=Wt, xy=xy,
                         full_E=E, full_eloc=eloc, edge_states=edge_states)

    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
    panels = [(ax[0], sp["flux_nonzero"], r"(a) nonzero flux $\phi=1/4$"),
              (ax[1], sp["flux_zero"], r"(b) zero flux $\phi=0$")]
    sc = None
    for axi, flux, ttl in panels:
        E = res[flux]["E"]; el = res[flux]["eloc"]; o = np.argsort(E)
        sc = axi.scatter(np.arange(len(E)), E[o], c=el[o], cmap="viridis",
                         s=9, vmin=0, vmax=1)
        axi.set_xlabel("doublon state (sorted)"); axi.set_title(ttl)
        axi.set_ylim(2 * U - ylim, 2 * U + ylim)
    ax[0].set_ylabel("energy $E$")
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.04)
    cb.set_label("edge localisation", fontsize=7); cb.ax.tick_params(labelsize=6)

    # (c) representative edge-localised state in the sampled doublon window
    R = res[cfg["edge_state"]["flux"]]
    es = R["edge_states"]
    a = np.where(es)[0][np.argmax(R["full_eloc"][es])]
    dens = R["Wt"][:, a].reshape(L, L).T
    pct = int(round(100 * R["full_eloc"][a]))
    im = ax[2].imshow(dens, origin="lower", cmap="magma", extent=[0, L - 1, 0, L - 1])
    ax[2].set_title(fr"(c) edge-localised mode, ${pct}\%$ on boundary")
    ax[2].set_xlabel("site $x$"); ax[2].set_ylabel("site $y$")
    ax[2].set_xticks([0, L - 1]); ax[2].set_yticks([0, L - 1])
    cb2 = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    cb2.set_label("doublon density", fontsize=7); cb2.ax.tick_params(labelsize=6)

    fig.tight_layout()
    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    counts = {str(flux): int(np.count_nonzero(res[flux]["edge_states"])) for flux in res}
    result = bm.write_results(__file__, {
        "figure": cfg["output"], "carrier_dimension": d,
        "computed_invariant": False, "sampled_eigenpairs": sp["kband"],
        "edge_candidate_counts": counts,
        "representative_boundary_weight": float(R["full_eloc"][a]),
        "representative_energy": float(R["full_E"][a]),
    })
    print(f"wrote {out} and {result} ({pct}% representative boundary weight)")


if __name__ == "__main__":
    main()
