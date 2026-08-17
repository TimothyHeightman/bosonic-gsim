"""Flux-reversed directional transport of an edge-localised doublon packet.

This finite-size dynamical signature is reported without assigning a topological
invariant to the sampled spectral subspace.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from gbosons import benchmarking as bm
from gbosons import bounded_n as bn, lattices as lat
from gbosons.plotting import LIGHT_COLORS, PAPER_COLORS, paper_style, figures_dir, load_config


def sector(L, flux, U):
    M = L * L
    h, xy = lat.hofstadter(L, L, flux=flux)
    basis, index, d = bn.sector_basis(M, 2)
    chi = np.zeros((M, M)); np.fill_diagonal(chi, U / 2)
    Hs = bn.passive_hamiltonian(h, basis, index) + bn.cross_kerr_hamiltonian(chi, basis)
    drows = np.array([index[tuple([2 if s == k else 0 for s in range(M)])] for k in range(M)])
    return Hs, basis, index, xy, M, drows


def edge_wavepacket(L, flux, U, ewin, kband):
    """Energy-selected edge-localised wavepacket from a shift-invert window."""
    Hs, basis, index, xy, M, drows = sector(L, flux, U)
    edge = lat.edge_sites(xy, L, L)
    E, V = eigsh(Hs, k=kband, sigma=2 * U, which="LM")        # doublon band near E=2U
    Wt = np.abs(V[drows, :]) ** 2
    frac = Wt.sum(0); eloc = Wt[edge].sum(0) / np.maximum(frac, 1e-12)
    band = (frac > 0.5) & (eloc > 0.7)                       # bound, edge-localised
    emed = np.median(E[band]) if band.any() else 2 * U
    win = band & (np.abs(E - emed) < ewin)                   # narrow window = one chirality
    idx = np.where(win)[0]
    s0 = 0 * L + L // 2                                       # mid-left-edge seed
    c = np.conj(V[drows[s0], idx])
    wp = V[:, idx] @ c; wp /= np.linalg.norm(wp)
    eb = np.sort(E[band]); gap = np.diff(eb).max() if len(eb) > 2 else 0.0
    return wp, E[idx], V[:, idx], drows, xy, M, len(idx), gap


def track(states, drows, xy, center):
    dens = np.abs(states[:, drows]) ** 2                      # (T, M) doublon density
    frac = dens.sum(1)
    com = (dens @ xy) / frac[:, None]
    return com, frac, lat.chiral_winding(com, center)


def winding(com, center):
    v = com - center[None, :]
    a = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    return (a - a[0]) / (2 * np.pi)


def main():
    cfg = load_config(__file__)
    paper_style()
    sysc = cfg["system"]
    L, U, T, nt = sysc["L"], sysc["U"], sysc["T"], sysc["n_time"]
    ewin, kband = sysc["energy_window"], sysc["kband"]
    center = np.array([(L - 1) / 2, (L - 1) / 2])
    ts = np.linspace(0, T, nt)

    pn = cfg["panels"]
    saved = {}
    for flux in [pn["flux_plus"], pn["flux_minus"]]:
        wp, Eb, Vb, drows, xy, M, nb, gap = edge_wavepacket(L, flux, U, ewin, kband)
        c0 = Vb.conj().T @ wp
        states = np.array([Vb @ (np.exp(-1j * Eb * t) * c0) for t in ts])   # (T, d)
        com, frac, w = track(states, drows, xy, center)
        saved[flux] = dict(ts=ts, com=com, frac=frac,
                           dens=np.abs(states[:, drows]) ** 2, gap=gap)

    P = saved[pn["flux_plus"]]; Mn = saved[pn["flux_minus"]]

    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
    im = None
    for axi, Z, ttl, trajectory_color in [
            (ax[0], P, r"(a) flux $+\phi$", LIGHT_COLORS["blue_koi"]),
            (ax[1], Mn, r"(b) flux $-\phi$", LIGHT_COLORS["pink_coral"])]:
        grid = Z["dens"][-1].reshape(L, L).T
        im = axi.imshow(grid, origin="lower", cmap="magma", extent=[0, L - 1, 0, L - 1])
        com = Z["com"]
        axi.plot(com[:, 0], com[:, 1], color=trajectory_color, lw=1.3)
        axi.plot(com[0, 0], com[0, 1], "o", color="white", ms=4, mec="k", mew=0.4)
        axi.set_title(ttl); axi.set_xlabel("site $x$")
        axi.set_xticks([0, L - 1]); axi.set_yticks([0, L - 1])
    ax[0].set_ylabel("site $y$")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cb.set_label("doublon density", fontsize=7); cb.ax.tick_params(labelsize=6)

    ax[2].plot(P["ts"], winding(P["com"], center),
               color=PAPER_COLORS["method"], label=r"$+\phi$")
    ax[2].plot(Mn["ts"], winding(Mn["com"], center),
               color=PAPER_COLORS["theory"], ls="--", label=r"$-\phi$")
    ax[2].axhline(0, color="grey", lw=0.5)
    ax[2].set_xlabel("time $t$  ($1/J$)"); ax[2].set_ylabel("chiral winding (turns)")
    ax[2].set_title("(c) direction reverses with flux"); ax[2].legend(frameon=False)

    fig.tight_layout()
    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    wp_end = winding(P["com"], center)[-1]; wm_end = winding(Mn["com"], center)[-1]
    result = bm.write_results(__file__, {
        "figure": cfg["output"], "computed_invariant": False,
        "winding_plus_flux": float(wp_end), "winding_minus_flux": float(wm_end),
        "minimum_doublon_fraction_plus": float(P["frac"].min()),
        "minimum_doublon_fraction_minus": float(Mn["frac"].min()),
        "sampled_eigenpairs": kband,
    })
    print(f"wrote {out} and {result} (winding {wp_end:+.2f}, {wm_end:+.2f})")


if __name__ == "__main__":
    main()
