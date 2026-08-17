"""Non-Gaussian-input correlator validation under passive optics."""
import numpy as np
import matplotlib.pyplot as plt

from gbosons import benchmarking as bm
from gbosons import core, fock_ref
from gbosons.plotting import PAPER_COLORS, paper_style, figures_dir, load_config

C = {
    "gsim": PAPER_COLORS["method"],
    "exact": PAPER_COLORS["theory"],
    "fock": PAPER_COLORS["reference"],
}


def beamsplitter(theta):
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]], dtype=complex)


def main():
    cfg = load_config(__file__)
    paper_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.6))

    hom = cfg["hom"]
    th = np.linspace(0, np.pi / 2, hom["n_theta"])
    values = np.array([core.number_number_correlator(beamsplitter(t), [1, 1])[0, 1]
                       for t in th])
    axL.plot(th, np.cos(2 * th) ** 2, color=C["exact"], lw=2.0,
             label=r"$\cos^2(2\theta)$")
    axL.plot(th, values, color=C["gsim"], ls="--", label="moment propagation")
    thf = np.array([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
    fock_values = np.array([fock_ref.nij_fock(beamsplitter(t), [1, 1],
                                              cutoff=hom["fock_cutoff"])[0, 1] for t in thf])
    axL.plot(thf, fock_values, "o", ms=4, color=C["fock"], label="Fock reference")
    axL.axvline(np.pi / 4, color="grey", lw=0.6, ls=":")
    axL.set_xlabel(r"beamsplitter angle $\theta$")
    axL.set_ylabel(r"$\langle \hat n_1\hat n_2\rangle$")
    axL.set_title(r"(a) Hong--Ou--Mandel, $|1,1\rangle$")
    axL.legend(frameon=False, fontsize=7)

    val = cfg["validation"]
    rng = np.random.default_rng(val["seed"])
    predicted, reference = [], []
    for _ in range(val["instances"]):
        U = core.haar_unitary(val["modes"], rng)
        predicted.extend(core.number_number_correlator(U, val["occupation"]).ravel())
        reference.extend(fock_ref.nij_fock(U, val["occupation"],
                                          cutoff=val["fock_cutoff"]).ravel())
    predicted = np.asarray(predicted); reference = np.asarray(reference)
    lo, hi = min(predicted.min(), reference.min()), max(predicted.max(), reference.max())
    axR.plot([lo, hi], [lo, hi], color="0.65", lw=1)
    axR.scatter(reference, predicted, s=9, alpha=0.65, color=C["gsim"])
    axR.set_xlabel("Fock reference")
    axR.set_ylabel("moment propagation")
    axR.set_title("(b) random-interferometer trust gate")
    max_error = float(np.max(np.abs(predicted - reference)))
    axR.text(0.05, 0.90, fr"max $|\Delta|={max_error:.1e}$",
             transform=axR.transAxes, fontsize=7)

    fig.tight_layout()
    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    hom_error = float(np.max(np.abs(values - np.cos(2 * th) ** 2)))
    hom_fock_error = float(np.max(np.abs(fock_values - np.cos(2 * thf) ** 2)))
    result = bm.write_results(__file__, {
        "figure": cfg["output"], "hom_max_analytic_error": hom_error,
        "hom_max_fock_error": hom_fock_error,
        "random_interferometer_max_fock_error": max_error,
        "validation_instances": val["instances"], "seed": val["seed"],
    })
    print(f"wrote {out} and {result} (random-instance error {max_error:.2e})")


if __name__ == "__main__":
    main()
