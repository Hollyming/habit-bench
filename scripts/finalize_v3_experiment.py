#!/usr/bin/env python
"""Deprecated alias for current four-domain non-human supplementary analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--node",
        help="Deprecated and ignored; current H multi-Replica runner merges globally.",
    )
    args = parser.parse_args()
    output_root = args.output_root or args.suite_root / "supplementary"
    print(
        "warning: finalize_v3_experiment.py is deprecated; running the current "
        "four-domain non-human sidecar",
        file=sys.stderr,
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_supplementary_analysis.py"),
            "--suite-root",
            str(args.suite_root),
            "--output-root",
            str(output_root),
            "--domains",
            "food,finance,software,travel",
            "--bootstrap-samples",
            "10000",
            "--seed",
            "42",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
