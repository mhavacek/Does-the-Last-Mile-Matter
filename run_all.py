"""Run the complete NHTS first/last-mile analysis workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "src/01_load_and_inspect.py",
    "src/02_prepare_analysis_dataset.py",
    "src/03_descriptive_analysis.py",
    "src/04_modeling.py",
    "src/05_generate_tables_and_figures.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===", flush=True)
        try:
            subprocess.run([sys.executable, script], cwd=root, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\nWorkflow stopped while running {script}. See the message above for details.")
            raise SystemExit(exc.returncode) from exc
    print("\nWorkflow complete. See outputs/ for tables, figures, logs, and analysis notes.")


if __name__ == "__main__":
    main()
