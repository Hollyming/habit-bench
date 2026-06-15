#!/usr/bin/env python
"""Evaluate lightweight memory-baseline adapters on HABIT-Bench.

These are method-inspired proxies, not official package integrations. They are
useful for testing the benchmark/evaluator loop and for diagnosing whether the
curated split separates explicit retrieval from habit boundary/counterevidence
behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "user",
    "with",
    "without",
}


def tokenize(text: str) -> List[str]:
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 2 and tok not in STOPWORDS
    ]


def token_counter(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / max(na * nb, 1e-9)


def text_of_messages(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class MemoryItem:
    session_id: str
    session_index: int
    text: str
    kind: str
    recency: float


class Baseline:
    name = "baseline"

    def __init__(self, sessions_by_user: Dict[str, List[Dict[str, Any]]]):
        self.sessions_by_user = sessions_by_user

    def answer(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def visible_sessions(self, probe: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_idx = probe["visible_history_scope"]["max_session_index"]
        return [
            s
            for s in self.sessions_by_user[probe["user_id"]]
            if s["session_index"] <= max_idx
        ]

    def pick_by_scores(
        self,
        probe: Dict[str, Any],
        scores: Dict[str, float],
        evidence: Sequence[str],
        debug: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        choices = probe["choices"]
        # Stable deterministic tie-breaker by choice order.
        best = max(choices, key=lambda c: (scores.get(c["choice_id"], 0.0), -ord(c["choice_id"][0])))
        return {
            "probe_id": probe["probe_id"],
            "choice_id": best["choice_id"],
            "scores": scores,
            "evidence_session_ids": list(evidence)[:5],
            "debug": debug or {},
            "cost": self.estimate_cost(probe, list(evidence)[:5], debug or {}),
        }

    def estimate_cost(self, probe: Dict[str, Any], evidence: Sequence[str], debug: Dict[str, Any]) -> Dict[str, Any]:
        sessions = self.visible_sessions(probe)
        visible_text = "\n".join(text_of_messages(s["messages"]) for s in sessions)
        evidence_set = set(evidence)
        retrieved_text = "\n".join(text_of_messages(s["messages"]) for s in sessions if s["session_id"] in evidence_set)
        return {
            "visible_history_sessions": len(sessions),
            "visible_history_tokens_est": len(tokenize(visible_text)),
            "retrieved_sessions": len(evidence),
            "retrieved_tokens_est": len(tokenize(retrieved_text)),
            "stored_items_est": debug.get("stored_items_est", len(sessions)),
        }


class NoMemoryLexical(Baseline):
    name = "no_memory_lexical"

    def answer(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        q = token_counter(probe["query"])
        scores = {
            c["choice_id"]: cosine_counter(q, token_counter(c["text"]))
            for c in probe["choices"]
        }
        return self.pick_by_scores(probe, scores, [])

    def estimate_cost(self, probe: Dict[str, Any], evidence: Sequence[str], debug: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "visible_history_sessions": 0,
            "visible_history_tokens_est": 0,
            "retrieved_sessions": 0,
            "retrieved_tokens_est": 0,
            "stored_items_est": 0,
        }


class FullHistoryRetrieval(Baseline):
    name = "full_history_segment_retrieval"

    def answer(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        sessions = self.visible_sessions(probe)
        query_vec = token_counter(probe["query"])
        ranked = []
        for s in sessions:
            text = text_of_messages(s["messages"])
            ranked.append((cosine_counter(query_vec, token_counter(text)), s, text))
        ranked.sort(key=lambda x: x[0], reverse=True)
        context = "\n".join(text for _, _, text in ranked[:8])
        ctx_vec = token_counter(context + "\n" + probe["query"])
        scores = {
            c["choice_id"]: cosine_counter(ctx_vec, token_counter(c["text"]))
            for c in probe["choices"]
        }
        return self.pick_by_scores(probe, scores, [s["session_id"] for _, s, _ in ranked[:5]])

    def estimate_cost(self, probe: Dict[str, Any], evidence: Sequence[str], debug: Dict[str, Any]) -> Dict[str, Any]:
        sessions = self.visible_sessions(probe)
        visible_text = "\n".join(text_of_messages(s["messages"]) for s in sessions)
        return {
            "visible_history_sessions": len(sessions),
            "visible_history_tokens_est": len(tokenize(visible_text)),
            "retrieved_sessions": len(sessions),
            "retrieved_tokens_est": len(tokenize(visible_text)),
            "stored_items_est": len(sessions),
        }


def is_positive_preference_feedback(text: str) -> bool:
    lower = text.lower()
    positive = any(
        phrase in lower
        for phrase in [
            "perfect",
            "good",
            "great",
            "exactly",
            "thanks",
            "right default",
            "works",
            "what i need",
            "the structure i need",
        ]
    )
    negative_scope = any(
        phrase in lower
        for phrase in [
            "different context",
            "exception",
            "do not turn",
            "do not remember",
            "just for today",
            "no need to remember",
        ]
    )
    return positive and not negative_scope


def extract_positive_memory_items(sessions: Sequence[Dict[str, Any]]) -> List[MemoryItem]:
    items = []
    total = max(len(sessions) - 1, 1)
    for s in sessions:
        messages = s["messages"]
        feedback = messages[-1]["content"] if len(messages) > 2 and messages[-1]["role"] == "user" else ""
        if not is_positive_preference_feedback(feedback):
            continue
        assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        text = f"{messages[0]['content']} {assistant} {feedback}"
        items.append(
            MemoryItem(
                session_id=s["session_id"],
                session_index=s["session_index"],
                text=text,
                kind="positive_preference",
                recency=s["session_index"] / total,
            )
        )
    return items


def extract_all_note_items(sessions: Sequence[Dict[str, Any]]) -> List[MemoryItem]:
    total = max(len(sessions) - 1, 1)
    items = []
    for s in sessions:
        text = text_of_messages(s["messages"])
        items.append(
            MemoryItem(
                session_id=s["session_id"],
                session_index=s["session_index"],
                text=text,
                kind="episode_note",
                recency=s["session_index"] / total,
            )
        )
    return items


def score_choices_against_items(
    probe: Dict[str, Any],
    items: Sequence[MemoryItem],
    top_k: int,
    query_weight: float = 0.25,
    recency_weight: float = 0.0,
) -> Tuple[Dict[str, float], List[str], Dict[str, Any]]:
    query_vec = token_counter(probe["query"])
    ranked = []
    for item in items:
        sim = cosine_counter(query_vec, token_counter(item.text)) + recency_weight * item.recency
        ranked.append((sim, item))
    ranked.sort(key=lambda x: x[0], reverse=True)
    retrieved = [item for _, item in ranked[:top_k]]
    retrieved_text = "\n".join(item.text for item in retrieved)
    retrieved_vec = token_counter(retrieved_text)
    scores = {}
    for choice in probe["choices"]:
        choice_vec = token_counter(choice["text"])
        scores[choice["choice_id"]] = (
            cosine_counter(retrieved_vec, choice_vec)
            + query_weight * cosine_counter(query_vec, choice_vec)
        )
    return scores, [item.session_id for item in retrieved], {"retrieved_kinds": [i.kind for i in retrieved]}


class Mem0LikeFactMemory(Baseline):
    name = "mem0_like_fact_memory"

    def answer(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        items = extract_positive_memory_items(self.visible_sessions(probe))
        scores, evidence, debug = score_choices_against_items(probe, items, top_k=12, query_weight=0.15)
        debug["stored_items_est"] = len(items)
        return self.pick_by_scores(probe, scores, evidence, debug)


class ZepLikeTemporalGraph(Baseline):
    name = "zep_like_temporal_graph"

    def answer(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        items = extract_positive_memory_items(self.visible_sessions(probe))
        scores, evidence, debug = score_choices_against_items(
            probe,
            items,
            top_k=10,
            query_weight=0.20,
            recency_weight=0.15,
        )
        debug["stored_items_est"] = len(items)
        return self.pick_by_scores(probe, scores, evidence, debug)


class AMemLikeNoteLinking(Baseline):
    name = "a_mem_like_note_linking"

    def answer(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        # Stores all notes, but retrieval still favors semantically similar notes
        # and has no explicit verifier for scope/exception semantics.
        items = extract_all_note_items(self.visible_sessions(probe))
        scores, evidence, debug = score_choices_against_items(probe, items, top_k=6, query_weight=0.20)
        debug["stored_items_est"] = len(items)
        return self.pick_by_scores(probe, scores, evidence, debug)


class SeComLikeSegmentMemory(Baseline):
    name = "secom_like_segment_memory"

    def answer(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        sessions = self.visible_sessions(probe)
        segments = []
        total = max(len(sessions) - 1, 1)
        for s in sessions:
            for m in s["messages"]:
                segments.append(
                    MemoryItem(
                        session_id=s["session_id"],
                        session_index=s["session_index"],
                        text=m["content"],
                        kind=f"segment_{m['role']}",
                        recency=s["session_index"] / total,
                    )
                )
        scores, evidence, debug = score_choices_against_items(probe, segments, top_k=16, query_weight=0.25)
        debug["stored_items_est"] = len(segments)
        return self.pick_by_scores(probe, scores, evidence, debug)


class RMMLikeReflectiveSummary(Baseline):
    name = "rmm_like_reflective_summary"

    def answer(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        # Summarizes positive preference evidence into a compact profile. This
        # intentionally loses boundary and exception episodes.
        items = extract_positive_memory_items(self.visible_sessions(probe))
        profile_text = " ".join(item.text for item in items)
        profile_vec = token_counter(profile_text)
        query_vec = token_counter(probe["query"])
        scores = {}
        for choice in probe["choices"]:
            choice_vec = token_counter(choice["text"])
            scores[choice["choice_id"]] = (
                cosine_counter(profile_vec, choice_vec)
                + 0.10 * cosine_counter(query_vec, choice_vec)
            )
        return self.pick_by_scores(
            probe,
            scores,
            [item.session_id for item in items[-5:]],
            {"profile_items": len(items), "stored_items_est": 1},
        )


class OMemLikeUserProfile(Baseline):
    name = "o_mem_like_user_profile"

    def answer(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        # Profile-centric memory: strong preference consolidation, weak scope
        # control. Use all positive preferences, with a small recency preference.
        items = extract_positive_memory_items(self.visible_sessions(probe))
        recent_items = sorted(items, key=lambda i: i.session_index, reverse=True)[:20]
        profile_text = " ".join(item.text for item in recent_items)
        profile_vec = token_counter(profile_text)
        scores = {}
        for choice in probe["choices"]:
            choice_vec = token_counter(choice["text"])
            scores[choice["choice_id"]] = cosine_counter(profile_vec, choice_vec)
            lower = choice["text"].lower()
            if "ask for clarification" in lower or "decline" in lower:
                scores[choice["choice_id"]] -= 0.03
        return self.pick_by_scores(
            probe,
            scores,
            [item.session_id for item in recent_items[:5]],
            {"profile_items": len(recent_items), "stored_items_est": 1},
        )


BASELINES = [
    NoMemoryLexical,
    FullHistoryRetrieval,
    Mem0LikeFactMemory,
    ZepLikeTemporalGraph,
    AMemLikeNoteLinking,
    SeComLikeSegmentMemory,
    RMMLikeReflectiveSummary,
    OMemLikeUserProfile,
]


def load_dataset(dataset_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    sessions = read_jsonl(dataset_dir / "public" / "lifelines.jsonl")
    sessions_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        sessions_by_user[session["user_id"]].append(session)
    for user_sessions in sessions_by_user.values():
        user_sessions.sort(key=lambda row: row["session_index"])
    probes = read_jsonl(dataset_dir / "public" / "probes.jsonl")
    keys = {row["public_probe_id"]: row for row in read_jsonl(dataset_dir / "private" / "probe_key.jsonl")}
    return sessions_by_user, probes, keys


def evaluate_predictions(predictions: List[Dict[str, Any]], keys: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detailed = []
    for pred in predictions:
        key = keys[pred["probe_id"]]
        correct = pred["choice_id"] == key["gold_choice_id"]
        detailed.append(
            {
                **pred,
                "gold_choice_id": key["gold_choice_id"],
                "correct": correct,
                "probe_type": key["probe_type"],
                "capability_group": key["capability_group"],
                "habit_family": key.get("habit_family", "unknown"),
                "gold_action": key["gold_action"],
                "stress_variant": key.get("stress_variant", "unknown"),
            }
        )

    summary_rows = []
    for group_field in ["overall", "probe_type", "capability_group", "habit_family", "stress_variant"]:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in detailed:
            key = "overall" if group_field == "overall" else row[group_field]
            buckets[key].append(row)
        for bucket, rows in sorted(buckets.items()):
            summary_rows.append(
                {
                    "group_field": group_field,
                    "group": bucket,
                    "n": len(rows),
                    "accuracy": round(sum(1 for r in rows if r["correct"]) / len(rows), 4),
                    "avg_retrieved_tokens_est": round(
                        sum(r.get("cost", {}).get("retrieved_tokens_est", 0) for r in rows) / len(rows),
                        2,
                    ),
                    "avg_stored_items_est": round(
                        sum(r.get("cost", {}).get("stored_items_est", 0) for r in rows) / len(rows),
                        2,
                    ),
                }
            )
    return detailed, summary_rows


def run(args: argparse.Namespace) -> None:
    dataset_dir: Path = args.dataset_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions_by_user, probes, keys = load_dataset(dataset_dir)

    all_summary = []
    report_sections = []
    for baseline_cls in BASELINES:
        baseline = baseline_cls(sessions_by_user)
        started = time.time()
        predictions = []
        for probe in probes:
            pred = baseline.answer(probe)
            pred["baseline"] = baseline.name
            predictions.append(pred)
        detailed, summary_rows = evaluate_predictions(predictions, keys)
        elapsed = time.time() - started
        for row in summary_rows:
            row["baseline"] = baseline.name
            row["elapsed_sec"] = round(elapsed, 3)
        all_summary.extend(summary_rows)
        write_jsonl(output_dir / f"{baseline.name}_predictions.jsonl", detailed)
        report_sections.append((baseline.name, elapsed, summary_rows))

    write_csv(output_dir / "metrics_summary.csv", all_summary)
    diagnostic_rows = make_diagnostic_summary(all_summary)
    write_csv(output_dir / "diagnostic_summary.csv", diagnostic_rows)
    write_report(output_dir / "baseline_report.md", report_sections, diagnostic_rows)
    print(f"Wrote {output_dir / 'metrics_summary.csv'}")


def make_diagnostic_summary(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_baseline: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    overall_by_baseline: Dict[str, Dict[str, Any]] = {}
    for row in summary_rows:
        if row["group_field"] == "capability_group":
            by_baseline[row["baseline"]][row["group"]] = row
        if row["group_field"] == "overall":
            overall_by_baseline[row["baseline"]] = row
    stress_groups = [
        "habit_boundary_false_personalization",
        "counterevidence_exception",
        "habit_drift",
        "false_personalization_privacy",
    ]
    rows = []
    for baseline, groups in sorted(by_baseline.items()):
        explicit = float(groups.get("explicit_fact_preference_retrieval", {}).get("accuracy", 0.0))
        direct = float(groups.get("habit_direct_use", {}).get("accuracy", 0.0))
        stress_num = 0.0
        stress_den = 0
        for group in stress_groups:
            if group not in groups:
                continue
            n = int(groups[group]["n"])
            stress_num += float(groups[group]["accuracy"]) * n
            stress_den += n
        stress = stress_num / stress_den if stress_den else 0.0
        false_personalization_groups = [
            "habit_boundary_false_personalization",
            "false_personalization_privacy",
        ]
        fp_num = 0.0
        fp_den = 0
        for group in false_personalization_groups:
            if group not in groups:
                continue
            n = int(groups[group]["n"])
            fp_num += float(groups[group]["accuracy"]) * n
            fp_den += n
        fp_acc = fp_num / fp_den if fp_den else 0.0
        rows.append(
            {
                "baseline": baseline,
                "explicit_retrieval_accuracy": round(explicit, 4),
                "habit_direct_accuracy": round(direct, 4),
                "habit_stress_accuracy_weighted": round(stress, 4),
                "explicit_minus_stress_gap": round(explicit - stress, 4),
                "false_personalization_control_accuracy": round(fp_acc, 4),
                "avg_retrieved_tokens_est": round(
                    float(overall_by_baseline.get(baseline, {}).get("avg_retrieved_tokens_est", 0.0)),
                    2,
                ),
                "avg_stored_items_est": round(
                    float(overall_by_baseline.get(baseline, {}).get("avg_stored_items_est", 0.0)),
                    2,
                ),
            }
        )
    return rows


def write_report(
    path: Path,
    report_sections: List[Tuple[str, float, List[Dict[str, Any]]]],
    diagnostic_rows: List[Dict[str, Any]],
) -> None:
    dataset_name = path.parent.parent.name
    lines = [
        f"# {dataset_name} Baseline Report",
        "",
        "These are lightweight method-inspired baselines, not official package integrations.",
        "They are intended to validate the benchmark/evaluator loop before scaling.",
        "",
        "## Accuracy By Capability",
        "",
        "| baseline | explicit | direct | boundary/false-pers | exception | drift | privacy/false-pers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    wanted = {
        "explicit_fact_preference_retrieval": "explicit",
        "habit_direct_use": "direct",
        "habit_boundary_false_personalization": "boundary",
        "counterevidence_exception": "exception",
        "habit_drift": "drift",
        "false_personalization_privacy": "privacy",
    }
    for baseline_name, _, summary_rows in report_sections:
        by_group = {
            row["group"]: row["accuracy"]
            for row in summary_rows
            if row["group_field"] == "capability_group"
        }
        values = [by_group.get(group, float("nan")) for group in wanted]
        lines.append(
            "| "
            + baseline_name
            + " | "
            + " | ".join("nan" if math.isnan(v) else f"{v:.3f}" for v in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic Gap",
            "",
            "| baseline | explicit acc | habit stress acc | gap | false-personalization control acc |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostic_rows:
        lines.append(
            "| {baseline} | {explicit_retrieval_accuracy:.3f} | {habit_stress_accuracy_weighted:.3f} | {explicit_minus_stress_gap:.3f} | {false_personalization_control_accuracy:.3f} |".format(
                **row
            )
        )
    variant_names = sorted(
        {
            row["group"]
            for _, _, summary_rows in report_sections
            for row in summary_rows
            if row["group_field"] == "stress_variant"
        }
    )
    lines.extend(["", "## Variant Sensitivity", ""])
    if variant_names:
        lines.append("| baseline | " + " | ".join(f"{name} acc" for name in variant_names) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in variant_names) + " |")
    else:
        lines.append("| baseline | variant acc |")
        lines.append("| --- | ---: |")
    for baseline_name, _, summary_rows in report_sections:
        by_variant = {
            row["group"]: row["accuracy"]
            for row in summary_rows
            if row["group_field"] == "stress_variant"
        }
        values = [by_variant.get(name, float("nan")) for name in variant_names]
        lines.append(
            "| "
            + baseline_name
            + " | "
            + " | ".join("nan" if math.isnan(value) else f"{value:.3f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Cost Proxy",
            "",
            "| baseline | avg retrieved tokens | avg stored items |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in diagnostic_rows:
        lines.append(
            f"| {row['baseline']} | {row['avg_retrieved_tokens_est']:.1f} | {row['avg_stored_items_est']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The strongest current evidence is for an explicit-vs-habit gap: fact/profile-style memories do well on explicit retrieval but degrade on boundary, exception, drift, and privacy/false-personalization stress cases.",
            "Full-history and segment-retrieval baselines can handle some boundary/exception cases, which means later versions should add paraphrased, unseen, and cost-controlled stress tests before making claims about official systems.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("./runs/habit_bench_curated_v0_1"))
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/habit_bench_curated_v0_1/baseline_results"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
