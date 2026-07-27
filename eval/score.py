#!/usr/bin/env python
"""Score existing HABIT-Bench choice predictions with strict full coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.core.dataset import load_dataset
from eval.core.io import read_jsonl
from eval.core.scoring import score_predictions, write_score_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--domain-filter")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-probes", type=int)
    args = parser.parse_args()

    bundle = load_dataset(
        args.dataset_dir,
        domain_filter=args.domain_filter,
        max_users=args.max_users,
        max_probes=args.max_probes,
    )
    detailed, metrics, rows = score_predictions(
        read_jsonl(args.predictions), bundle, args.method_name
    )
    write_score_outputs(args.output_dir, detailed, metrics, rows)
    print(json.dumps(metrics["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
