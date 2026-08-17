# Extended publication results

This directory builds manuscript figures that combine the polished local
experiments with the validated production-cluster evidence. It is deliberately
separate from both the canonical figures and the diagnostic `_cluster.pdf`
outputs.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.extended_results.extract
PYTHONPATH=src .venv/bin/python -m experiments.extended_results.build
PYTHONPATH=src .venv/bin/python -m experiments.extended_results.build_fig6_alternative
```

The extraction step reads
`cluster_results/20260714T212532Z-production-08c652e`, verifies its successful
validation report, and writes a compact, versioned evidence bundle under
`data/production-20260714/`. The build step writes only to
`notes/figures/extended_results/`.

The compact bundle records the source run, task and validation Git revisions,
checksums, and the exact cluster records used in the figures. The 26 GB raw
production tree remains outside Git.

The shared manuscript palette is defined in `src/gbosons/plotting.py`.
Publication lines use the dark colors; their light companions are used for
uncertainty bands and secondary accents. Continuous heatmaps retain
task-appropriate sequential or diverging colormaps.

The alternative Figure 6 builder preserves panels (a)--(c) of the extended
OTOC figure and replaces the threshold-front fit in panel (d) by the integrated
commutator weight \(W_\psi(t)=\sum_i C_\psi(i,t)\).

The main-text topology summary is written to
`fig10_topological_doublon_extended.pdf`. It combines the open-boundary
spectral comparison, the certified doublon-multiplet Chern number, and
flux-reversed edge transport. The detailed Figures 7, 7b, and 8 remain
available as supporting outputs.
