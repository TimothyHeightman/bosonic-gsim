"""Shared plotting style and experiment helpers for reproducing the paper figures.

Every experiment in ``experiments/`` calls :func:`paper_style` for a consistent
look, loads its parameters from a sibling ``config.yaml`` via :func:`load_config`,
and writes its figure into ``notes/figures/`` (resolved by :func:`figures_dir`),
where the LaTeX source picks it up.
"""
from pathlib import Path
import matplotlib as mpl
import yaml

# User-selected manuscript palette. Dark colors carry lines and markers; their
# light companions are reserved for uncertainty bands and secondary accents.
DARK_COLORS = {
    "azure_blue": "#4863A0",
    "basil_green": "#829F82",
    "dull_purple": "#7F525D",
    "camel_brown": "#C19A6B",
    "indian_red": "#CD5C5C",
}
LIGHT_COLORS = {
    "blue_koi": "#659EC7",
    "iguana_green": "#9CB071",
    "old_rose": "#C08081",
    "burlywood": "#DEB887",
    "pink_coral": "#E77471",
}
DARK_PALETTE = tuple(DARK_COLORS.values())
LIGHT_PALETTE = tuple(LIGHT_COLORS.values())

# Stable semantic roles used across otherwise independent experiments.
PAPER_COLORS = {
    "method": DARK_COLORS["azure_blue"],
    "baseline": DARK_COLORS["basil_green"],
    "nonlinear": DARK_COLORS["dull_purple"],
    "reference": DARK_COLORS["camel_brown"],
    "theory": DARK_COLORS["indian_red"],
}
PAPER_FILLS = {
    "method": LIGHT_COLORS["blue_koi"],
    "baseline": LIGHT_COLORS["iguana_green"],
    "nonlinear": LIGHT_COLORS["old_rose"],
    "reference": LIGHT_COLORS["burlywood"],
    "theory": LIGHT_COLORS["pink_coral"],
}

PAPER_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
}


def paper_style():
    """Apply the paper's matplotlib style (non-interactive Agg backend)."""
    mpl.use("Agg")
    mpl.rcParams.update(PAPER_RC)


def figures_dir():
    """Return ``<repo>/notes/figures`` as a ``Path``, creating it if needed."""
    d = Path(__file__).resolve().parents[2] / "notes" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_config(run_file):
    """Load the ``config.yaml`` sitting next to the calling ``run.py``."""
    cfg_path = Path(run_file).resolve().parent / "config.yaml"
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)
