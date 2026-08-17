"""Repulsively bound pairs in the exact two-photon carrier."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, expm_multiply

from gbosons import benchmarking as bm
from gbosons import bounded_n as bn
from gbosons import fock_ref
from gbosons.plotting import PAPER_COLORS, paper_style, figures_dir, load_config


def hopping_matrix(n, J, periodic=False):
    h = np.zeros((n, n), dtype=complex)
    for k in range(n - 1):
        h[k, k + 1] = h[k + 1, k] = -J
    if periodic:
        h[0, -1] = h[-1, 0] = -J
    return h


def bose_hubbard(n, J, U, basis, index, periodic=False):
    h = hopping_matrix(n, J, periodic=periodic)
    chi = np.zeros((n, n)); np.fill_diagonal(chi, U / 2.0)
    return bn.number_conserving_hamiltonian(h, chi, basis, index)


def density_profile(H, psi0, basis, n, T, nt):
    states = expm_multiply(-1j * H, psi0, start=0.0, stop=T, num=nt, endpoint=True)
    nk = np.array([bn.number_diag(k, basis) for k in range(n)])
    return np.array([nk @ np.abs(state) ** 2 for state in states]), states


def doublon_fraction(states, basis, n):
    nk = np.array([bn.number_diag(k, basis) for k in range(n)])
    same = 0.5 * np.sum(nk * (nk - 1.0), axis=0)
    return np.array([np.sum(same * np.abs(state) ** 2).real for state in states])


def validate_against_fock(cfg):
    n, N, cutoff = cfg["modes"], 2, cfg["cutoff"]
    basis, index, _ = bn.sector_basis(n, N)
    occ = [0] * n; occ[n // 2] = 2
    psi0 = bn.basis_state(occ, index)
    h = hopping_matrix(n, cfg["J"])
    chi = np.zeros((n, n)); np.fill_diagonal(chi, cfg["U"] / 2)
    H = bn.number_conserving_hamiltonian(h, chi, basis, index)
    errors = []
    for t in np.linspace(0, cfg["T"], cfg["n_time"]):
        psi = bn.evolve(H, t, psi0)
        sector = np.array([[bn.nn_expect(i, j, basis, psi) for j in range(n)]
                           for i in range(n)])
        fock = fock_ref.kerr_circuit_fock(h, chi, occ, t, cutoff)
        errors.append(np.max(np.abs(sector - fock)))
    return float(np.max(errors))


def bound_band_scaling(cfg):
    n, J = cfg["modes"], cfg["J"]
    basis, index, _ = bn.sector_basis(n, 2)
    values = []
    for U in cfg["U_values"]:
        H = bose_hubbard(n, J, U, basis, index, periodic=True)
        energies = eigsh(H, k=n, which="LA", return_eigenvectors=False,
                         tol=cfg["eig_tol"])
        width = float(np.max(energies) - np.min(energies))
        values.append((U, width / 4.0))
    Us = np.array([x[0] for x in values]); Jeff = np.array([x[1] for x in values])
    fit_mask = Us >= cfg["fit_U_min"]
    exponent, log_prefactor = np.polyfit(np.log(Us[fit_mask]), np.log(Jeff[fit_mask]), 1)
    return Us, Jeff, float(exponent), float(np.exp(log_prefactor))


def main():
    cfg = load_config(__file__)
    paper_style()
    sys = cfg["system"]
    n, J, T, nt = sys["modes"], sys["J"], sys["T"], sys["n_time"]
    basis, index, d = bn.sector_basis(n, 2)
    c = n // 2
    occ = [0] * n; occ[c] = 2
    psi0 = bn.basis_state(occ, index)

    U_free, U_bound = cfg["lightcones"]["U_free"], cfg["lightcones"]["U_bound"]
    prof_free, _ = density_profile(bose_hubbard(n, J, U_free, basis, index),
                                   psi0, basis, n, T, nt)
    prof_bound, _ = density_profile(bose_hubbard(n, J, U_bound, basis, index),
                                    psi0, basis, n, T, nt)

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.55))
    ext = [0, n - 1, 0, T]
    vmax = max(prof_free.max(), prof_bound.max())
    for ax, prof, title in [(axes[0], prof_free, fr"(a) free pair $U={U_free:g}$"),
                            (axes[1], prof_bound, fr"(b) bound pair $U={U_bound:g}J$")]:
        im = ax.imshow(prof, origin="lower", aspect="auto", extent=ext,
                       cmap="magma", vmin=0, vmax=vmax)
        ax.set_xlabel("site $j$"); ax.set_title(title)
    axes[0].set_ylabel("time $t$  ($1/J$)")
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label(r"$\langle \hat n_j(t)\rangle$", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ts = np.linspace(0, T, nt)
    for U, color in cfg["binding"]["curves"]:
        states = expm_multiply(-1j * bose_hubbard(n, J, U, basis, index), psi0,
                               start=0.0, stop=T, num=nt, endpoint=True)
        axes[2].plot(ts, doublon_fraction(states, basis, n), color=color,
                     label=fr"$U={U:g}J$")
    axes[2].set_xlabel("time $t$  ($1/J$)")
    axes[2].set_ylabel("doublon fraction")
    axes[2].set_title("(c) pair binding")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].legend(frameon=False, fontsize=7)

    Us, Jeff, exponent, prefactor = bound_band_scaling(cfg["strong_coupling"])
    axes[3].loglog(Us, Jeff, "o", color=PAPER_COLORS["method"],
                   label="bound-band width $/4$")
    axes[3].loglog(Us, 2 * J ** 2 / Us, "--", color=PAPER_COLORS["theory"],
                   label=r"$2J^2/U$")
    axes[3].set_xlabel("interaction $U/J$")
    axes[3].set_ylabel(r"$J_{\mathrm{eff}}/J$")
    axes[3].set_title("(d) co-tunnelling scale")
    axes[3].legend(frameon=False, fontsize=6.5)
    axes[3].text(0.05, 0.08, fr"fit $\propto U^{{{exponent:.2f}}}$",
                 transform=axes[3].transAxes, fontsize=7, color="0.35")

    fig.tight_layout()
    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    fock_error = validate_against_fock(cfg["validation"])
    result = bm.write_results(__file__, {
        "figure": cfg["output"],
        "main_carrier_dimension": d,
        "validation_max_fock_error": fock_error,
        "strong_coupling": {
            "U_values": Us.tolist(), "J_eff": Jeff.tolist(),
            "fit_exponent": exponent, "fit_prefactor": prefactor,
            "prediction_prefactor": 2 * J ** 2,
        },
    })
    print(f"wrote {out} and {result} (Fock error {fock_error:.2e}, exponent {exponent:.3f})")


if __name__ == "__main__":
    main()
