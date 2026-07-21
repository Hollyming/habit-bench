#!/usr/bin/env python
"""Validate and summarize one or more HABIT-Bench domain datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.core.dataset import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-probes", type=int)
    args = parser.parse_args()
    reports = [
        load_dataset(
            path, max_users=args.max_users, max_probes=args.max_probes
        ).manifest
        for path in args.dataset_dirs
    ]
    print(json.dumps({"status": "pass", "datasets": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
