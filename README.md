# bosonic-gsim

Lie-algebraic (g-sim) classical simulation of bosonic systems, beyond Gaussian dynamics.

Reference implementation and reproducible numerics for the paper *Lie-algebraic
classical simulation of bosonic systems beyond Gaussian dynamics*.

> **Paper:** arXiv:XXXX.XXXXX — *link to be added once the preprint is public.*

---

# For humans

## What this is

Classical simulability depends on both the dynamics and the observable being
measured. This code propagates observables in the Heisenberg picture inside the
smallest operator space closed under the circuit generators, the *reachable
operator module*. When that space is polynomial-dimensional, exact expectation
values, fixed-order correlators and gradients follow from finite-dimensional
linear algebra, with no Fock-space cutoff.

The implementation uses the most economical equivalent carrier in each regime:

| Regime | Mechanism | Module |
|---|---|---|
| Quadratic (Gaussian) dynamics | polynomial degree is preserved | `core` |
| Number-conserving dynamics at bounded photon number | confinement to fixed photon-number sectors, including Kerr and pair hopping | `bounded_n` |
| Bounded-photon dynamics perturbed by squeezing | parity-resolved photon-number bands | `banded` |
| Nilpotent polynomial phase dynamics | weighted degree filtration | `nilpotent_phase` |

Correctness is anchored throughout against `fock_ref`, a brute-force
truncated-Fock simulator, on systems small enough to afford it.

## Setup

Requires Python 3.9+. Dependencies are NumPy, SciPy, Matplotlib and PyYAML.

```bash
git clone https://github.com/TimothyHeightman/bosonic-gsim.git
cd bosonic-gsim
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Without installing, prefix commands with `PYTHONPATH=src` instead.

## Quick start

```bash
python experiments/fig8_beyond_gaussian/run.py
```

That runs a four-mode non-Gaussian number-correlation check against an
independent Fock calculation in about a second, writes a figure, and records the
residual in `results.json` next to it.

## Reproducing the results

### Small systems — runs on a laptop

Every experiment lives in its own directory holding `config.yaml` (all
parameters: sizes, couplings, seeds, time grids, plot ranges), `run.py`, and
`results.json` with the validation residuals, dimensions, seeds and timing
metadata from its last run.

Run one, or run all of them from seed to plot:

```bash
python experiments/fig3_doublon/run.py
python experiments/run_all.py
```

Approximate wall-clock per experiment, single run each on a laptop:

Directory names follow the figure numbering of the manuscript. Figure 1 is a
schematic with no associated computation, and Figure 6 is assembled from two
experiments.

| Manuscript | Experiment | Time |
|---|---|---|
| Fig. 2 | `fig2_bounded_n` | 5 s |
| Fig. 3 | `fig3_doublon` | 5 s |
| Fig. 4 | `fig4_squeezing` | 13 s |
| Fig. 5 | `fig5_otoc` | 86 s |
| Fig. 6 | `fig6_doublon_topology` | 57 s |
| Fig. 6 | `fig6_chiral_transport` | 75 s |
| Fig. 7 | `fig7_gaussian` | 6 s |
| Fig. 8 | `fig8_beyond_gaussian` | 1 s |
| Fig. 9 | `fig9_kerr_control_2d` | 44 s |
| Fig. 10 | `fig10_nilpotent_phase` | 3 s |
| | **all ten** | **~5 min** |

Figures are written to `notes/figures/`, which is created on first run. Results
are deterministic: seeds are explicit and live in `config.yaml`. To change a
system size, coupling or sweep range, edit `config.yaml` and re-run; nothing is
hard-coded in `run.py`.

### Extended figures from the production data

The manuscript's large-scale panels are built from a compact, checksummed
evidence bundle committed under
`experiments/extended_results/data/production-20260714/`. Rebuilding them takes about 95 seconds
on a mac with M2 chip.

```bash
python -m experiments.extended_results.build
```

The sibling `extract` step is **not** runnable from this repository. It reads the
raw multi-gigabyte production tree, which is not distributed here. The bundle it
produces is already committed, so `build` is the entry point you want.

### Large systems 

The paper's largest calculations (operator spreading to 400 modes, bounded
sector unions over the full mode ranges, the topological doublon sweeps, and the
squeezing-band scaling) were produced by a manifest-driven SLURM suite on the
LiCCA cluster at the University of Augsburg.

Those job scripts, the cluster environment locks and the raw production tree are
**not** included here: they are specific to that machine's partitions, module
system and filesystem layout, and would not run elsewhere unmodified. The Python
driver they call *is* included, under `experiments/cluster/`.

To reproduce the large-scale results, or to adapt the suite to your own cluster,
open an issue on this repository or contact the corresponding author listed on
the paper.

## Validation

Every exact numerical path is checked against an analytic result, an independent
truncated-Fock calculation, or both wherever affordable. Each experiment
directory carries a `results.json` with residuals, carrier dimensions, seeds and
timing samples.

```bash
python -m unittest discover -s tests
```

Three tests need artifacts a fresh clone does not have, and will error until you
generate them or are working in the internal repository:

- `test_cluster_suite.test_submitted_jobs_do_not_resolve_helpers_from_slurm_spool`
  reads the SLURM job scripts, which are not distributed here.
- `test_extended_results.test_extended_figures_exist_separately` needs the
  extended figures built first (`python -m experiments.extended_results.build`).
- `test_extended_results.test_protected_sources_are_unchanged` checksums the
  canonical figure PDFs, which are build outputs and are not committed here.

Everything else passes on a clean checkout.

## Layout

```
src/gbosons/          the simulation library (installable package)
  core.py               Gaussian family: symplectic / Bogoliubov transfer
  bounded_n.py          fixed sectors and finite sector unions, incl. Kerr
  banded.py             parity-resolved photon-number bands under squeezing
  nilpotent_phase.py    exact propagation for the nilpotent phase family
  variational.py        differentiable g-sim, exact reverse-mode gradients
  topology.py           gauge-invariant lattice Chern diagnostics
  lattices.py           Hofstadter and SSH single-particle hopping matrices
  dla_closure.py        finite / infinite dynamical Lie algebra certificate
  fock_ref.py           brute-force truncated-Fock reference (ground truth)
  benchmarking.py       reproducible median/IQR timings and run metadata
  plotting.py           shared figure style and experiment helpers
experiments/          one directory per figure: config.yaml + run.py + results.json
  run_all.py            regenerates every figure
  extended_results/     builds the large-scale panels from committed data
  cluster/              Python driver for the manifest-based production suite
tests/                unit tests
```

## Citing

<!-- Fill in once the preprint is posted. -->

```bibtex
@article{bosonic-gsim,
  title  = {Lie-algebraic classical simulation of bosonic systems beyond Gaussian dynamics},
  author = {TODO},
  year   = {TODO},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

# For agents

Read this section before editing anything. It is the context needed to be
productive here and, more importantly, to avoid silently regressing results that
are already published.

## The one rule

**Nothing is trusted until it is checked against `src/gbosons/fock_ref.py`.**

That module is a brute-force truncated-Fock simulator and is the correctness
ground truth for the whole repository. Every fast path exists because it agrees
with `fock_ref` on a system small enough to afford the exact calculation. Any new
simulation code must add a `fock_ref` cross-check on a small case before it is
used for anything.

Do not "optimise" a kernel and report success on the basis that it runs. Run the
reference.

## Architecture

- `gbosons.core` — Gaussian family. Symplectic / Bogoliubov transfer
  `S(t) = exp(t Omega G)`; closed-form intensity correlator
  `number_number_correlator`; `haar_unitary`.
- `gbosons.bounded_n` — fixed photon-number sectors and finite sector unions,
  including inter-sector annihilation and quadrature maps with explicit
  sector-block metadata. Sparse number-conserving generators (`hop_matrix`,
  `passive_hamiltonian`, `cross_kerr_hamiltonian`), Schrödinger `evolve` via
  sparse `expm_multiply`, and the matrix-unit adjoint primitives
  (`adjoint_action`, `heisenberg_observable`).
- `gbosons.banded` — parity-aware carriers reachable by pair creation or
  annihilation in at most `k` steps, plus a legacy full-interval basis retained
  for compatibility.
- `gbosons.nilpotent_phase` — exact polynomial propagation for the nilpotent
  phase family (position-diagonal phases, shears, displacements).
- `gbosons.variational` — differentiable g-sim. Exact reverse-mode gradients of
  a transfer-matrix observable: dense `loss_and_grad` via `expm_frechet`,
  `layered_loss_and_grad`, and the large-`n` `sparse_layered_loss_and_grad`
  (adjoint-state method through sparse `expm_multiply`, with a diagonal fast
  path). Also the finite-difference ground truth and `adam`.
- `gbosons.topology` — gauge-invariant lattice Chern diagnostics for isolated
  multiplets.
- `gbosons.lattices` — `hofstadter` (Peierls phases), `ssh`, `edge_sites`,
  `chiral_winding`.
- `gbosons.dla_closure` — finite/infinite dynamical Lie algebra certificate.
  Generates the Lie algebra of polynomial generators in `(x, p)` under the
  Poisson bracket, exactly over the rationals, and reports closure and dimension.
  Run it directly: `python -m gbosons.dla_closure`.
- `gbosons.plotting` — `paper_style()`, `figures_dir()`, `load_config(__file__)`.
- `gbosons.benchmarking` — `write_results` and the timing helpers.

## Running things

- Without installing: prefix with `PYTHONPATH=src`.
- After `pip install -e .`: `import gbosons` works directly.
- `experiments/run_all.py` adds `src` to `PYTHONPATH` itself, so it works either
  way.
- Tests are `unittest`, not pytest: `python -m unittest discover -s tests -t .`.
- Figures go to `notes/figures/`, which `figures_dir()` creates on demand. That
  directory is a build output and is not tracked.
- **Regenerating figures costs real time** (about 5 minutes for all ten, and 95
  seconds for the extended build). Run the single experiment you need, not
  `run_all.py`, unless you were asked for a full regeneration. Note that running
  an experiment rewrites its tracked `results.json`.

## Invariants — do not regress these

Each was a real bug that was found and fixed. They are load-bearing.

1. `sector_basis` and `banded_basis` row order is a **contract**. All lookups go
   through the returned `index` dict. Reorder only if you rebuild `index`.
2. In `bounded_n.cross_kerr_hamiltonian` and `banded.cross_kerr`, `chi[k, l]` is
   the **full** coefficient of `n_k n_l`, and the matrix may be supplied in
   **either triangle** (off-diagonal entries are summed, `chi_kl + chi_lk`). Do
   not assume upper-triangular.
3. `fock_ref.nij_fock` and `fock_ref.kerr_circuit_fock` require
   `cutoff >= sum(occ) + 1`, because photons bunch. Under-truncation raises a
   `ValueError`. Keep it that way; do not soften it to a warning.
4. `core.mean_n_single_mode_vacuum` is **single-mode only** (2x2 transfer). For
   `n > 1`, sum `|S[i, n+k]|^2` over the pairing block instead. The function
   guards its own shape.
5. `lattices.hofstadter`: in the Landau gauge used here the effective plaquette
   flux is `-flux`. The magnitude is `flux` and the chirality reverses with its
   sign. Pin any external Chern-number sign convention against this, not the
   other way round.
6. `fock_ref.otoc_fock` is the OTOC ground truth. It uses the **same** squared
   commutator `|| [n_i(t), n_c] psi ||^2` with `n_i(t) = e^{iHt} n_i e^{-iHt}` as
   the sector engine, and g-sim must match it to about `1e-13`. The Gaussian
   baseline is the **quadratic part only**, with Kerr dropped, because Kerr is
   not in `sp(2n)`.
7. Timing panels pin BLAS to **one thread** wherever dense linear algebra is
   used, warm up each problem size first, and report median and interquartile
   range over at least seven repeated measurements. Single-shot timings are not
   acceptable evidence.
8. Reverse-mode variational gradients must match finite differences to about
   `1e-10`. This is what panel (a) of `fig9_kerr_control_2d` checks. The
   gradient is the adjoint of the exponential: dense via
   `expm_frechet(A.conj().T, ...)`, sparse via the costate sweep
   `lam <- expm_multiply(+i theta G^T, lam)` with `grad = 2 Im<lam|G|psi>`. Do
   not "simplify" the conjugate transpose, the `+i` backward sign, or the
   Wirtinger factor of 2. All three are load-bearing.

## Adding an experiment

Create `experiments/figN_name/` mirroring `experiments/fig7_gaussian/`:

- `config.yaml` — **all** parameters: sizes, couplings, seeds, time grids, plot
  ranges. Nothing hard-coded in `run.py`.
- `run.py` — import `paper_style`, `figures_dir`, `load_config` from
  `gbosons.plotting`; `cfg = load_config(__file__)`; compute, seeded from `cfg`;
  plot; `fig.savefig(figures_dir() / cfg["output"])`. Define `main()` and guard
  it with `if __name__ == "__main__":`. Keep Matplotlib mathtext-safe, so no
  `\mathcal` in axis labels.
- `results.json` — write validation residuals, dimensions, seeds, timing samples
  and environment metadata through `gbosons.benchmarking.write_results`.

## What is not in this repository

Do not infer these paths exist, and do not write code that depends on them.

- **The manuscript source.** No tex. The paper is the specification;
  this repository is the implementation. (`notes/figures/` is created at runtime
  as a figure output directory only.)
- **The SLURM job scripts and cluster environment locks.** The Python driver in
  `experiments/cluster/` is here, but the shell suite that submits it is not; it
  is machine-specific to the LiCCA cluster at Augsburg. Consequently
  `tests/test_cluster_suite.py` has one test
  (`test_submitted_jobs_do_not_resolve_helpers_from_slurm_spool`) that reads
  `cluster/jobs/*.sh` and errors with `FileNotFoundError` on a fresh clone. That
  is expected here. **Do not "fix" it by deleting the test, stubbing the shell
  scripts, or loosening the assertion** — it guards a real property of the
  submission suite in the internal repository.
- **The raw production tree.** `experiments/extended_results/extract.py`
  references `cluster_results/<RUN_ID>`, a multi-gigabyte directory that is not
  distributed. **`extract` cannot run here.** Its output is already committed
  under `experiments/extended_results/data/production-20260714/`, so use
  `python -m experiments.extended_results.build` instead.
- **`build_fig6_alternative`.** `experiments/extended_results/README.md` mentions
  this module, but it is not shipped. That reference is stale; ignore it rather
  than trying to recreate it.

If a user asks for large-scale runs, the answer is either to request the run scripts via
a GitHub issue or from the corresponding author or attempt to reconstruct
the cluster suite from the driver.

## Conventions

- Seeds are explicit, live in `config.yaml`, and results are deterministic.
  Preserve this. Never introduce an unseeded random draw.
- Do not commit build artifacts: `__pycache__`, LaTeX auxiliaries, generated
  figures. See `.gitignore`.
- Report what actually happened. If a check fails, say so with the number. A
  residual that moved from `1e-13` to `1e-6` is a regression, not a rounding
  difference, and must be surfaced rather than absorbed into a tolerance.
