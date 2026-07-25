#!/usr/bin/env python3
"""Record the finalized v0.5 no-memory calibration in its release manifest."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V05 = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_5"
METRICS = V05 / "evaluation_results" / "no_memory_v05_release_20260725" / "taskmaster_v04" / "no_memory" / "merged" / "metrics.json"


def main() -> None:
    metrics = json.loads(METRICS.read_text())
    false = next(item for item in metrics["groups"] if item["group_field"] == "probe_type" and item["group"] == "false_personalization")
    manifest_path = V05 / "reports" / "v0_5_release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["no_memory_calibration"] = {
        "model": "Qwen3-8B",
        "method": "no_memory",
        "run": "no_memory_v05_release_20260725",
        "overall": metrics["overall"],
        "false_personalization": false,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["no_memory_calibration"], ensure_ascii=False))


if __name__ == "__main__":
    main()
