#!/usr/bin/env python3
"""Ensure v0.5 differs from v0.4 only at its six released replacement probes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V04 = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
V05 = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_5"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    manifest = json.loads((V05 / "reports" / "v0_5_release_manifest.json").read_text())
    replacement_ids = {item["replaces_probe_id"] for item in manifest["replacements"]}
    v04 = {row["probe_id"]: row for row in read_jsonl(V04 / "public" / "probes.jsonl")}
    v05 = read_jsonl(V05 / "public" / "probes.jsonl")
    restored = []
    for row in v05:
        if row["probe_id"] not in replacement_ids:
            restored.append(v04[row["probe_id"]])
            continue
        row["metadata"]["dataset_version"] = "taskmaster_planning_defaults_v0_5"
        restored.append(row)
    write_jsonl(V05 / "public" / "probes.jsonl", restored)
    print(json.dumps({"restored_v04_rows": len(restored) - len(replacement_ids), "v05_replacement_rows": len(replacement_ids)}))


if __name__ == "__main__":
    main()
