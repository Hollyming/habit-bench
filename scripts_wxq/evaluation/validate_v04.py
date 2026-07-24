#!/usr/bin/env python3
"""Validate and summarize the finalized Taskmaster planning-defaults v0.4 release."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.core.dataset import load_dataset  # noqa: E402


DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"


def main() -> None:
    bundle = load_dataset(DATASET)
    by_type = Counter(str((probe.get("metadata") or {}).get("probe_type", "unknown")) for probe in bundle.probes)
    by_user = Counter(probe["user_id"] for probe in bundle.probes)
    expected = {"users": 6, "sessions": 771, "probes": 157}
    observed = {key: bundle.manifest[key] for key in expected}
    if observed != expected:
        raise SystemExit(f"Unexpected finalized release counts: expected={expected}, observed={observed}")
    if by_type.get("false_personalization") != 17:
        raise SystemExit(f"Expected 17 false-personalization probes, found {by_type.get('false_personalization', 0)}")
    print(json.dumps({
        "status": "pass", "dataset": bundle.manifest,
        "probe_type_counts": dict(sorted(by_type.items())),
        "probes_by_user": dict(sorted(by_user.items())),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
