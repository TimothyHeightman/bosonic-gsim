"""Figure 1 -- Gaussian family: exactness of the symplectic transfer and the
polynomial scaling of the coincidence-matrix evaluation.

Panel (b) times the O(n^3) coincidence kernel (three dense n x n products in
``number_number_correlator``). For a clean, reproducible scaling curve we pin
BLAS to a single thread, warm it up once, then report the median and
interquartile range over accumulated timing blocks."""
import os
# pin BLAS to one thread BEFORE numpy imports -> clean single-thread n^3 scaling
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib.pyplot as plt
from gbosons import benchmarking as bm
from gbosons import core, fock_ref
from gbosons.plotting import PAPER_COLORS, PAPER_FILLS, paper_style, figures_dir, load_config

C = {
    "gsim": PAPER_COLORS["method"],
    "exact": PAPER_COLORS["theory"],
    "fock": PAPER_COLORS["reference"],
}


def main():
    cfg = load_config(__file__)
    paper_style()
    rng = np.random.default_rng(cfg["seed"])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.6))

    # (a) single-mode squeezing <n(t)> = sinh^2(r t)
    sq = cfg["squeezing"]
    xi = sq["xi"]
    ts = np.linspace(0, sq["t_max"], sq["n_time"])
    A, B = core.single_mode_squeeze_generator(xi)
    gsim = [core.mean_n_single_mode_vacuum(core.bogoliubov_transfer(A, B, t=t)) for t in ts]
    axL.plot(ts, np.sinh(xi * ts) ** 2, color=C["exact"], lw=2.0, label=r"analytic $\sinh^2(rt)$")
    axL.plot(ts, gsim, color=C["gsim"], ls="--", label=r"g-sim ($2\times2$ transfer)")
    tf = np.linspace(0.1, sq["t_max"], sq["fock_points"])
    fock_values = [fock_ref.squeeze_mean_n_fock(xi, t, cutoff=sq["fock_cutoff"]) for t in tf]
    axL.plot(tf, fock_values,
             "o", ms=4, color=C["fock"], label=f"Fock ({sq['fock_cutoff']}-dim)", zorder=5)
    axL.set_xlabel("time $t$"); axL.set_ylabel(r"$\langle \hat n(t)\rangle$")
    axL.set_title("(a) single-mode squeezing")
    axL.legend(frameon=False, loc="upper left")

    # (b) single-thread wall-clock for the full n x n coincidence matrix vs n
    sc = cfg["scaling"]
    ns = np.array(sc["modes"])
    reps, target = sc["repeats"], sc["target_seconds"]
    Wbig = core.haar_unitary(int(ns.max()), rng)             # global BLAS warmup
    core.number_number_correlator(Wbig, np.ones(int(ns.max())))
    walls, q25, q75, timing_results = [], [], [], []
    for n in ns:
        U = core.haar_unitary(int(n), rng); occ = np.ones(int(n))
        timing = bm.timed_call(lambda: core.number_number_correlator(U, occ),
                               repeats=reps, warmups=1, target_seconds=target)
        walls.append(timing["median_seconds"]); q25.append(timing["q25_seconds"])
        q75.append(timing["q75_seconds"])
        timing_results.append({"modes": int(n), **timing})
    walls = np.array(walls)

    axR.loglog(ns, walls, "o-", color=C["gsim"], ms=4, label="g-sim (1 BLAS thread)")
    guide = walls[-1] * (ns / ns[-1]) ** 3.0                 # true kernel cost: 3 GEMMs
    axR.loglog(ns, guide, ":", color="grey", label=r"$\propto n^3$ (3 GEMMs)")
    slope = np.polyfit(np.log(ns), np.log(walls), 1)[0]
    axR.fill_between(ns, q25, q75, color=PAPER_FILLS["method"], alpha=0.24)
    axR.set_xlabel("modes $n$"); axR.set_ylabel("seconds (median of %d)" % reps)
    axR.set_title("(b) full $n\\times n$ coincidence matrix")
    axR.legend(frameon=False, loc="upper left")
    axR.text(0.96, 0.06, fr"slope $\approx n^{{{slope:.1f}}}$; input sector $\binom{{2n-1}}{{n}}$",
             transform=axR.transAxes, ha="right", va="bottom", fontsize=7, color="grey")

    fig.tight_layout()
    out = figures_dir() / cfg["output"]
    fig.savefig(out)
    analytic_error = float(np.max(np.abs(np.asarray(gsim) - np.sinh(xi * ts) ** 2)))
    fock_error = float(np.max(np.abs(np.asarray(fock_values) - np.sinh(xi * tf) ** 2)))
    result = bm.write_results(__file__, {
        "figure": cfg["output"], "seed": cfg["seed"],
        "squeezing_max_analytic_error": analytic_error,
        "squeezing_max_fock_error": fock_error,
        "coincidence_fit_exponent": float(slope),
        "coincidence_timings": timing_results,
    })
    print(f"wrote {out} and {result} (scaling n^{slope:.2f})")


if __name__ == "__main__":
    main()
