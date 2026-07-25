#!/usr/bin/env python3
"""Rescore persisted LoCoMo retrievals with one reader and evidence budget.

This postprocessor deliberately does not rebuild or query a memory system.  It
uses the ranked ``retrieved_memories`` persisted by MedMemoryBench, applies one
tokenizer-defined prefix budget, invokes one reader prompt, and recomputes the
official LoCoMo QA metric deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.prompts_qa import QA_TEMPLATES


SYSTEM_PROMPT = (
    "Answer the question using only the supplied memory evidence. "
    "Give only the shortest complete answer, with no explanation. "
    "If the answer is not supported, say exactly: No information available. "
    "Put only that answer in the required JSON field named answer."
)
USER_PROMPT_TEMPLATE = "Memory evidence:\n{evidence}\n\n{question_prompt}"
READER_ANSWER_MAX_CHARS = 512
READER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "locomo_short_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": READER_ANSWER_MAX_CHARS,
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


def build_question_prompt(row: dict[str, Any]) -> str:
    query_type = str(row.get("query_type", "default"))
    template = QA_TEMPLATES.get(
        f"locomo_{query_type}_qa", QA_TEMPLATES["locomo_default_qa"]
    )
    return template.format(
        memory_source="the supplied memory evidence",
        question=str(row.get("question", "")).strip(),
    )


def build_user_prompt(row: dict[str, Any], evidence: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        evidence=evidence, question_prompt=build_question_prompt(row)
    )


def parse_reader_response(content: str) -> str:
    """Extract and enforce the schema-constrained short answer."""
    payload = json.loads(content)
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        raise ValueError("Reader response must contain only the answer field")
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Reader returned an empty answer")
    answer = answer.strip()
    if len(answer) > READER_ANSWER_MAX_CHARS:
        raise ValueError("Reader answer exceeded the schema character limit")
    return answer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_snapshot(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready" or not manifest.get("revision"):
        raise ValueError("Reader model manifest is not a complete ready snapshot")
    small_files = manifest.get("small_files") or {}
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "repo": manifest.get("repo"),
        "revision": manifest["revision"],
        "config_sha256": (small_files.get("config.json") or {}).get("sha256"),
        "tokenizer_sha256": (small_files.get("tokenizer.json") or {}).get(
            "sha256"
        ),
    }


def _memory_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item).strip()
    for key in ("memory", "content", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def build_bounded_evidence(
    memories: list[Any], tokenizer: Any, token_budget: int
) -> tuple[str, int, int]:
    """Return ranked evidence prefix, used tokens, and non-empty item count."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    pieces: list[str] = []
    used = 0
    included = 0
    for index, item in enumerate(memories, start=1):
        text = _memory_text(item)
        if not text:
            continue
        prefix = f"[Memory {index}]\n"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if used + len(prefix_ids) >= token_budget:
            break
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        remaining = token_budget - used - len(prefix_ids)
        take = min(len(text_ids), remaining)
        if take <= 0:
            break
        bounded_text = tokenizer.decode(text_ids[:take], skip_special_tokens=True)
        pieces.append(prefix + bounded_text)
        used += len(prefix_ids) + take
        included += 1
        if take < len(text_ids):
            break
    return "\n\n".join(pieces), used, included


def official_score(row: dict[str, Any], prediction: str) -> float:
    from metrics.locomo_metrics import compute_f1, compute_multi_hop_f1

    details = row.get("evaluation_details") or {}
    category = int(details.get("category"))
    answer = str(row.get("expected_answer", ""))
    if category == 3:
        answer = answer.split(";", 1)[0].strip()
    if category == 1:
        return float(compute_multi_hop_f1(prediction, answer))
    if category in (2, 3, 4):
        return float(compute_f1(prediction, answer))
    if category == 5:
        lowered = prediction.lower()
        return float(
            "no information available" in lowered or "not mentioned" in lowered
        )
    raise ValueError(f"Unsupported LoCoMo category: {category!r}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_type", "unknown"))].append(row)

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        correct = sum(bool(item["common_reader_is_correct"]) for item in items)
        return {
            "total": total,
            "mean_official_score": (
                sum(float(item["common_reader_official_score"]) for item in items)
                / total
                if total
                else 0.0
            ),
            "threshold_0_5_correct": correct,
            "threshold_0_5_accuracy": correct / total if total else 0.0,
            "mean_evidence_tokens": (
                sum(int(item["common_reader_evidence_tokens"]) for item in items)
                / total
                if total
                else 0.0
            ),
        }

    summary = stats(rows)
    summary["by_type"] = {
        name: stats(items) for name, items in sorted(grouped.items())
    }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("dataset_name") != "locomo":
        raise ValueError("Input must be a LoCoMo query-answer artifact")
    source_rows = payload.get("queries", [])
    if not source_rows:
        raise ValueError("Input contains no queries")
    if args.expected_total is not None and len(source_rows) != args.expected_total:
        raise ValueError(
            f"Coverage mismatch: expected {args.expected_total}, found {len(source_rows)}"
        )
    query_ids = [str(row.get("query_id", "")) for row in source_rows]
    if not all(query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError("Input query IDs are empty or duplicated")

    rows = [
        row
        for index, row in enumerate(source_rows)
        if index % args.shard_count == args.shard_index
    ]
    if not rows:
        raise ValueError("Selected shard contains no queries")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    prompt_material = {
        "system": SYSTEM_PROMPT,
        "user_template": USER_PROMPT_TEMPLATE,
        "response_format": READER_RESPONSE_FORMAT,
        "qa": {
            key: value
            for key, value in sorted(QA_TEMPLATES.items())
            if key.startswith("locomo_") and key.endswith("_qa")
        },
    }
    prompt_sha256 = hashlib.sha256(
        json.dumps(prompt_material, sort_keys=True).encode("utf-8")
    ).hexdigest()
    source_sha256 = _sha256(args.input)
    model_snapshot = load_model_snapshot(args.model_manifest)
    implementation_path = Path(__file__).resolve()
    metric_path = (
        args.official_metric.resolve()
        if args.official_metric is not None
        else implementation_path.parents[1] / "metrics" / "locomo_metrics.py"
    )
    if not metric_path.is_file():
        raise FileNotFoundError(f"Official LoCoMo metric source not found: {metric_path}")
    config = {
        "model": args.model,
        "model_snapshot": model_snapshot,
        "tokenizer_path": str(Path(args.tokenizer).resolve()),
        "evidence_token_budget": args.evidence_tokens,
        "max_completion_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "enable_thinking": False,
        "correctness_threshold": 0.5,
        "response_format": READER_RESPONSE_FORMAT,
        "prompt_sha256": prompt_sha256,
        "implementation_sha256": _sha256(implementation_path),
        "official_metric_path": str(metric_path),
        "official_metric_sha256": _sha256(metric_path),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        old_config = old.get("common_reader", {}).get("config", {})
        if old_config != config:
            raise ValueError("Resume output common-reader config mismatch")
        if old.get("source", {}).get("sha256") != source_sha256:
            raise ValueError("Resume output source hash mismatch")
        completed = {str(row["query_id"]): row for row in old.get("queries", [])}

    started = time.time()
    output_rows: list[dict[str, Any]] = []

    def write_output(complete: bool) -> dict[str, Any]:
        ordered_rows = sorted(
            output_rows, key=lambda item: query_ids.index(str(item["query_id"]))
        )
        output = {
            "contract": "medmemorybench.locomo_common_reader.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": complete,
            "dataset_name": "locomo",
            "method_name": payload.get("method_name"),
            "source": {
                "path": str(args.input.resolve()),
                "sha256": source_sha256,
                "total_queries": len(source_rows),
            },
            "common_reader": {
                "config": config,
                "elapsed_seconds": time.time() - started,
                "summary": summarize(ordered_rows),
            },
            "queries": ordered_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
        return output

    for ordinal, row in enumerate(rows, start=1):
        query_id = str(row["query_id"])
        if query_id in completed:
            output_rows.append(completed[query_id])
            continue
        evidence, evidence_tokens, evidence_items = build_bounded_evidence(
            list(row.get("retrieved_memories") or []),
            tokenizer,
            args.evidence_tokens,
        )
        user_prompt = build_user_prompt(row, evidence)
        last_error: Exception | None = None
        response = None
        prediction = ""
        finish_reason = None
        for attempt in range(1, args.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    response_format=READER_RESPONSE_FORMAT,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                finish_reason = response.choices[0].finish_reason
                raw_completion = (response.choices[0].message.content or "").strip()
                if finish_reason == "length":
                    raise RuntimeError("Reader completion reached the token limit")
                if not raw_completion:
                    raise RuntimeError("Reader returned an empty completion")
                prediction = parse_reader_response(raw_completion)
                break
            except Exception as exc:  # keep failure visible in the shard artifact
                last_error = exc
                response = None
                prediction = ""
                if attempt < args.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        if response is None:
            raise RuntimeError(f"Reader failed for {query_id}: {last_error}")
        if not prediction:
            raise RuntimeError(f"Reader failed for {query_id}: {last_error}")
        score = official_score(row, prediction)
        usage = response.usage
        result_row = dict(row)
        result_row.update(
            {
                "common_reader_output": prediction,
                "common_reader_official_score": score,
                "common_reader_is_correct": score >= 0.5,
                "common_reader_evidence_tokens": evidence_tokens,
                "common_reader_evidence_items": evidence_items,
                "common_reader_prompt_tokens": usage.prompt_tokens if usage else None,
                "common_reader_completion_tokens": (
                    usage.completion_tokens if usage else None
                ),
                "common_reader_finish_reason": finish_reason,
            }
        )
        output_rows.append(result_row)
        if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(rows):
            print(
                json.dumps(
                    {
                        "progress": f"{ordinal}/{len(rows)}",
                        "query_id": query_id,
                        "evidence_tokens": evidence_tokens,
                    }
                ),
                flush=True,
            )
            write_output(complete=False)

    output = write_output(complete=True)
    return {
        "output": str(args.output.resolve()),
        "queries": len(output_rows),
        "summary": output["common_reader"]["summary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--official-metric", type=Path)
    parser.add_argument("--evidence-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index/count are invalid")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    print(json.dumps(run(parsed), ensure_ascii=False, indent=2))
