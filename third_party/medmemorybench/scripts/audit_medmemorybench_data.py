#!/usr/bin/env python3
"""Audit a materialized MedMemoryBench data tree and emit a JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EVAL_FILES = (
    "generated_dialogues.json",
    "generated_dialogues_with_noise.json",
    "generated_queries.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        prefix = path.read_bytes()[:80]
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Git LFS object is not materialized: {path}") from exc
        raise


def audit(data_root: Path) -> dict:
    personas = sorted(
        (path for path in data_root.glob("persona_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not personas:
        raise RuntimeError(f"No persona directories found under {data_root}")

    totals = Counter()
    query_types = Counter()
    per_persona = []
    files = []

    for persona_dir in personas:
        persona_id = int(persona_dir.name.split("_")[-1])
        eval_dir = persona_dir / "eval"
        missing = [name for name in EVAL_FILES if not (eval_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Persona {persona_id} missing eval files: {missing}")

        efficient = load_json(eval_dir / EVAL_FILES[0]).get("sessions", [])
        mixed = load_json(eval_dir / EVAL_FILES[1]).get("sessions", [])
        queries = load_json(eval_dir / EVAL_FILES[2]).get("queries", [])

        persona_query_types = Counter(
            str(query.get("query_type", "unknown")) for query in queries
        )
        regular_mixed = sum(
            "noise_id" not in session and "noise_family_id" not in session
            for session in mixed
        )
        noise_health = sum("noise_id" in session for session in mixed)
        noise_family = sum("noise_family_id" in session for session in mixed)

        row = {
            "persona_id": persona_id,
            "efficient_sessions": len(efficient),
            "mixed_sessions": len(mixed),
            "mixed_regular_sessions": regular_mixed,
            "mixed_health_noise_sessions": noise_health,
            "mixed_family_noise_sessions": noise_family,
            "queries": len(queries),
            "query_types": dict(sorted(persona_query_types.items())),
        }
        per_persona.append(row)
        totals.update({key: value for key, value in row.items() if key != "persona_id" and isinstance(value, int)})
        query_types.update(persona_query_types)

        for name in EVAL_FILES:
            path = eval_dir / name
            files.append(
                {
                    "path": str(path.relative_to(data_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "persona_count": len(personas),
        "persona_ids": [row["persona_id"] for row in per_persona],
        "totals": dict(sorted(totals.items())),
        "query_types": dict(sorted(query_types.items())),
        "per_persona": per_persona,
        "eval_files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = audit(args.data_root)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
