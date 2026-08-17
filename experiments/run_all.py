"""Regenerate every paper figure by running each experiment, from seed to plot.

Discovers ``experiments/fig*/run.py`` and runs each in a fresh subprocess. The
repo's ``src`` is added to PYTHONPATH so this works whether or not the package
has been ``pip install``-ed. Figures are written to ``notes/figures/``.
"""
import os
import subprocess
import sys
from pathlib import Path


def main():
    exp_root = Path(__file__).resolve().parent
    runs = sorted(exp_root.glob("fig*/run.py"))
    if not runs:
        sys.exit("no experiments found under experiments/fig*/run.py")

    env = dict(os.environ)
    src = str(exp_root.parent / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    print(f"Running {len(runs)} experiments...")
    failed = []
    for run in runs:
        print(f"\n=== {run.parent.name} ===", flush=True)
        if subprocess.run([sys.executable, str(run)], env=env).returncode != 0:
            failed.append(run.parent.name)

    if failed:
        sys.exit(f"\nFAILED: {', '.join(failed)}")
    print("\nAll experiments completed; figures are in notes/figures/.")


if __name__ == "__main__":
    main()
