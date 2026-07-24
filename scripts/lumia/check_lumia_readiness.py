#!/usr/bin/env python
"""Check Lumia paths required by the current HABIT-Bench evaluation pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PATHS = [
    ROOT / "eval/run.py",
    ROOT / "eval/score.py",
    ROOT / "eval/core/dataset.py",
    ROOT / "eval/core/answering.py",
    ROOT / "eval/official_adapters/mem0.py",
    ROOT / "eval/official_adapters/amem.py",
    ROOT / "eval/official_adapters/graphiti.py",
    ROOT / "eval/official_adapters/secom.py",
    ROOT / "eval/official_adapters/omem.py",
    ROOT / "domain/food/food_habit_lifelines_stress/public/lifelines.jsonl",
    ROOT / "domain/finance-software/habit_bench_multidogo_finance_software_long_hard_diverse_v0_5/public/lifelines.jsonl",
    Path("/data1/public/hf/Qwen/Qwen3-8B/config.json"),
    Path("/home/jmzhang/models/e5-base-v2/config.json"),
]
REQUIRED_MODULES = ["openai", "transformers"]


def main() -> None:
    paths = {str(path): path.exists() for path in REQUIRED_PATHS}
    modules = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}
    report = {
        "status": "pass" if all(paths.values()) and all(modules.values()) else "fail",
        "paths": paths,
        "modules": modules,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
