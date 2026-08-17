"""State-dependent squared commutator in the two-photon carrier.

The operator-level quantity is evaluated exactly through repeated sparse
Schrodinger propagations on the fixed sector.  The implementation therefore
does not claim to materialise the full vectorised operator module.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
from gbosons import benchmarking as bm
from gbosons import bounded_n as bn, fock_ref
from gbosons.plotting import (
    LIGHT_COLORS, PAPER_COLORS, paper_style, figures_dir, load_config)

CC = {"gsim": PAPER_COLORS["method"], "gauss": PAPER_COLORS["baseline"]}


def chain(n, J=1.0):
    h = np.zeros((n, n), dtype=complex)
    for k in range(n - 1):
        h[k, k + 1] = -J; h[k + 1, k] = -J
    return h


def build_sector_H(n, J, U):
    basis, index, d = bn.sector_basis(n, 2)
    h = chain(n, J)
    chi = np.zeros((n, n)); np.fill_diagonal(chi, U / 2.0)
    H = bn.passive_hamiltonian(h, basis, index) + bn.cross_kerr_hamiltonian(chi, basis)
    return H, basis, d, h, chi


def sector_otoc(H, basis, n, psi, c, ts):
    """Return ``C[t,i]=||[n_i(t),n_c] psi||^2`` in the fixed sector."""
    ndiag = np.array([bn.number_diag(i, basis) for i in range(n)])    # (n, d)
    nc = ndiag[c]
    C = np.zeros((len(ts), n))
    for ti, t in enumerate(ts):
        if t == 0:
            continue
        psit = expm_multiply(-1j * H * t, psi)
        At = expm_multiply(-1j * H * t, nc * psi)
        EA = expm_multiply(1j * H * t, (ndiag * At[None, :]).T)
        Ep = expm_multiply(1j * H * t, (ndiag * psit[None, :]).T)
        comm = EA - nc[:, None] * Ep
        C[ti] = np.sum(np.abs(comm) ** 2, axis=0)
    return C


def lightcone(n, J, U, T, nt, seed):
    H, basis, d, _, _ = build_sector_H(n, J, U)
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)
    ts = np.linspace(0.0, T, nt)
    return ts, sector_otoc(H, basis, n, psi, n // 2, ts), d


def validation(cfg_v):
    """max-site OTOC error vs full-Fock, for g-sim (Kerr kept) and Gaussian (Kerr dropped)."""
    n, cutoff = cfg_v["modes"], cfg_v["cutoff"]
    U, T, nt, seed = cfg_v["U"], cfg_v["T"], cfg_v["n_time"], cfg_v["seed"]
    c = n // 2
    ts = np.linspace(0.0, T, nt)

    H8, basis, d, h, chi = build_sector_H(n, 1.0, U)          # full (Kerr) sector H
    H0, _, _, _, _ = build_sector_H(n, 1.0, 0.0)             # Gaussian: quadratic part only
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)

    C_gsim = sector_otoc(H8, basis, n, psi, c, ts)            # keeps Kerr -> exact
    C_gauss = sector_otoc(H0, basis, n, psi, c, ts)           # drops Kerr -> Gaussian

    psiF = np.zeros(cutoff ** n, dtype=complex)               # embed same state into Fock
    for b, occ in enumerate(basis):
        psiF += psi[b] * fock_ref.number_state(occ, cutoff)
    C_fock = fock_ref.otoc_fock(h, chi, psiF, c, ts, cutoff)  # ground truth (Kerr)

    err_g = np.maximum(np.max(np.abs(C_gsim - C_fock), axis=1), 1e-16)
    err_q = np.maximum(np.max(np.abs(C_gauss - C_fock), axis=1), 1e-16)
    return ts, err_g, err_q, n, cutoff


def main():
    cfg = load_config(__file__)
    paper_style()
    sys = cfg["system"]
    n, J, T, nt, seed = sys["modes"], sys["J"], sys["T"], sys["n_time"], sys["seed"]
    pl = cfg["plot"]

    fig, ax = plt.subplots(1, 3, figsize=(9.8, 2.8))

    # (a) trust gate: interacting carrier and quadratic baseline vs full Fock
    ts_v, err_g, err_q, nv, cut = validation(cfg["validation"])
    ax[0].semilogy(ts_v, err_q, "o-", ms=3, color=CC["gauss"],
                   label="quadratic baseline ($U=0$)")
    ax[0].semilogy(ts_v, err_g, "s-", ms=3, color=CC["gsim"],
                   label="fixed-sector result")
    ax[0].set_xlabel("time $t$"); ax[0].set_ylabel(r"$\max_i\,|C-C_{\mathrm{Fock}}|$")
    ax[0].set_title(fr"(a) error vs Fock, $n={nv}$")
    ax[0].set_ylim(1e-16, 5.0)
    ax[0].legend(frameon=False, fontsize=7, loc="center right")
    ax[0].text(0.04, 0.06, fr"Fock dim ${cut}^{{{nv}}}={cut ** nv}$", transform=ax[0].transAxes,
               fontsize=7, color="grey", va="bottom")

    # (b,c) exact light-cones on the polynomial two-photon carrier
    im = None
    for a, U in zip(ax[1:], cfg["panels"]["U_values"]):
        ts, C, d = lightcone(n, J, U, T, nt, seed)
        im = a.imshow(np.log10(C + pl["log_floor"]), origin="lower", aspect="auto",
                      cmap="inferno", extent=[0, n - 1, 0, ts[-1]],
                      vmin=pl["vmin"], vmax=pl["vmax"])
        a.set_xlabel("site $i$")
        a.axvline(
            n // 2, color=LIGHT_COLORS["blue_koi"], lw=0.7, ls=":")
    ax[1].set_ylabel("time $t$")
    ax[1].set_title(fr"(b) $U=0J$,  $n={n}$")
    ax[2].set_title(fr"(c) $U=8J$,  $n={n}$")
    ax[1].text(0.04, 0.96, fr"two-photon carrier\n$d_{{{n},2}}={d}$",
               transform=ax[1].transAxes, ha="left", va="top", fontsize=6.5, color="white")
    fig.colorbar(im, ax=ax[2], fraction=0.046, label=r"$\log_{10}C(i,t)$")

    fig.tight_layout()
    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    result = bm.write_results(__file__, {
        "figure": cfg["output"],
        "observable": "state-dependent squared commutator",
        "algorithm": "repeated sparse fixed-sector Schrodinger propagation",
        "main_carrier_dimension": d,
        "validation_max_fock_error": float(err_g.max()),
        "quadratic_baseline_max_difference": float(err_q.max()),
        "seed": seed,
    })
    print(f"wrote {out} and {result} (sector error {err_g.max():.1e}, "
          f"quadratic-baseline difference {err_q.max():.1e})")


if __name__ == "__main__":
    main()
