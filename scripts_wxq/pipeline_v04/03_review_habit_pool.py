#!/usr/bin/env python3
"""Second-pass semantic review and consolidation of v0.4 grounded habits."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_client import post_chat, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def review_batch(batch: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    compact = []
    for row in batch:
        compact.append({
            "habit_instance_id": row["habit_instance_id"], "family": row["family"],
            "preference_value": row["preference_value"], "name": row["name"],
            "condition": row["condition"], "default_action": row["default_action"],
            "boundary_condition": row["boundary_condition"], "exception_condition": row["exception_condition"],
            "source_examples": row["source_examples"],
        })
    prompt = {
        "task": "Independently review grounded synthetic travel-habit candidates before longitudinal generation.",
        "criteria": [
            "Keep only reusable scoped decision defaults supported across at least three distinct instruction_id values.",
            "Reject dates, destinations, routes, current party size, requested room count, and one-off numeric price caps.",
            "Reject a candidate when the source examples merely share a slot value but conflict in polarity or do not establish a reusable preference.",
            "Room or bed configuration can be kept only if it is a genuine recurring preference, not mechanically implied by the current number of travelers.",
            "Revise over-generalized wording to the narrowest reusable behavior the sources support; do not invent a stronger habit.",
            "Assign canonical_value as a short semantic key so synonyms such as king bed/king-sized bed or Delta/Delta Air Lines get the same key.",
            "Return {reviews:[{habit_instance_id,decision,reason,canonical_value,preference_value,name,condition,default_action,boundary_condition,exception_condition,quality_score}]}; decision is keep, revise, or reject.",
        ],
        "candidates": compact,
    }
    result = post_chat(
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        messages=[
            {"role": "system", "content": "You are a conservative benchmark curator. Return strict JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        max_tokens=args.max_tokens, timeout=args.timeout, retries=args.retries, transport=args.transport,
        reasoning_effort=args.reasoning_effort,
    )["json"]
    reviews = result.get("reviews", [])
    by_id = {row.get("habit_instance_id"): row for row in reviews if isinstance(row, dict)}
    output = []
    for source in batch:
        review = by_id.get(source["habit_instance_id"])
        if not review or review.get("decision") not in {"keep", "revise", "reject"}:
            output.append({"habit_instance_id": source["habit_instance_id"], "decision": "invalid_output"})
            continue
        row = dict(source)
        row["semantic_review"] = {
            "decision": review["decision"], "reason": review.get("reason"),
            "quality_score": review.get("quality_score"), "reviewed_by": args.model,
        }
        row["canonical_value"] = re.sub(r"[^a-z0-9]+", "_", str(review.get("canonical_value", "")).lower()).strip("_")
        if review["decision"] == "revise":
            for key in ["preference_value", "name", "condition", "default_action", "boundary_condition", "exception_condition"]:
                if review.get(key): row[key] = str(review[key]).strip()
        row["decision"] = review["decision"]
        output.append(row)
    return output


def review_cache_path(batch: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, Path]:
    fingerprint = hashlib.sha256(json.dumps({
        "revision": "v04_semantic_review_r2", "model": args.model,
        "reasoning_effort": args.reasoning_effort, "batch": batch,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    path = args.dataset / "work" / "habit_semantic_review_cache" / f"{fingerprint}.json"
    return fingerprint, path


def review_batch_cached(batch: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    fingerprint, path = review_cache_path(batch, args)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["rows"]
    rows = review_batch(batch, args)
    write_json(path, {"cache_fingerprint": fingerprint, "rows": rows})
    return rows


def consolidate_family_cached(family_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    family = family_rows[0]["family"]
    fingerprint = hashlib.sha256(json.dumps({
        "revision": "v04_global_family_consolidation_r1", "model": args.model,
        "reasoning_effort": args.reasoning_effort, "rows": family_rows,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    path = args.dataset / "work" / "habit_global_consolidation_cache" / f"{family}_{fingerprint}.json"
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))["groups"]
    prompt = {
        "task": f"Globally consolidate every eligible {family} candidate across all prior review batches.",
        "requirements": [
            "Group only semantically equivalent concrete behaviors and aliases; do not merge different defaults merely because they share this family.",
            "Every supplied habit_instance_id must appear exactly once.",
            "Return {groups:[{family,canonical_value,member_ids,representative_id,merge_rationale}]}",
        ],
        "candidates": [{k: row[k] for k in ["habit_instance_id", "family", "preference_value", "condition", "default_action"]} for row in family_rows],
    }
    result = post_chat(
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        messages=[
            {"role": "system", "content": "You perform family-global semantic clustering for a benchmark. Return strict JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        max_tokens=7000, timeout=args.timeout, retries=args.retries, transport=args.transport,
        reasoning_effort=args.reasoning_effort,
    )["json"]
    groups = result.get("groups", [])
    write_json(path, {"cache_fingerprint": fingerprint, "groups": groups})
    return groups


def adjudicate_release_family_cached(family_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resolve residual aliases and default/fallback fragments before release.

    The first global pass deliberately merges only obvious aliases.  This pass
    reasons over the provisional release rows as decision policies, which is
    necessary for cases such as ``prefer nonstop`` plus ``allow one stop when
    needed``.  Contrasting values that would produce different probe answers
    remain separate alternatives rather than being collapsed for balance.
    """
    family = family_rows[0]["family"]
    compact = [{
        "habit_instance_id": row["habit_instance_id"],
        "name": row["name"],
        "preference_value": row["preference_value"],
        "condition": row["condition"],
        "default_action": row["default_action"],
        "boundary_condition": row["boundary_condition"],
        "exception_condition": row["exception_condition"],
        "source_examples": [
            {k: ex.get(k) for k in ["instruction_id", "conversation_id", "user_utterance"]}
            for ex in row.get("source_examples", [])
        ],
    } for row in family_rows]
    fingerprint = hashlib.sha256(json.dumps({
        "revision": "v04_release_policy_adjudication_r1", "model": args.model,
        "reasoning_effort": args.reasoning_effort, "family": family, "rows": compact,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    path = args.dataset / "work" / "habit_release_adjudication_cache" / f"{family}_{fingerprint}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["decisions"]
    prompt = {
        "task": f"Perform the final release adjudication for every provisional {family} travel-habit policy.",
        "rules": [
            "Every input habit_instance_id must occur exactly once across decisions[].input_ids.",
            "Use action=merge only for true aliases, semantically indistinguishable probe answers, or fragments that must be one conditional default/fallback policy.",
            "Do not merge contrasting user alternatives merely because they share a family: morning versus evening, aisle versus window, and genuinely different thresholds can remain separate.",
            "When a default and fallback are merged, write one complete conditional graph: default_action, boundary_condition, and exception_condition must preserve when the fallback applies.",
            "Use action=reject for a one-off transaction constraint, obsolete/unrealistic value, weakly grounded slot coincidence, incoherent policy, or item answerable entirely from the current request without longitudinal memory.",
            "For each kept/merged policy assign alternative_group_key. Policies in the same non-empty alternative group are mutually exclusive values for one user in the same scope, but remain useful pool alternatives across different users.",
            "Do not strengthen claims beyond the supplied evidence. Prefer a soft default when evidence is mixed.",
            "Return {decisions:[{action:keep|merge|reject,input_ids:[...],representative_id,rationale,canonical_value,name,preference_value,condition,default_action,boundary_condition,exception_condition,alternative_group_key}]}",
        ],
        "candidates": compact,
    }
    result = post_chat(
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        messages=[
            {"role": "system", "content": "You are the final conservative curator of a long-term-memory benchmark. Return strict JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        max_tokens=9000, timeout=args.timeout, retries=args.retries, transport=args.transport,
        reasoning_effort=args.reasoning_effort,
    )["json"]
    decisions = result.get("decisions", [])
    write_json(path, {"cache_fingerprint": fingerprint, "decisions": decisions})
    return decisions


def adjudicate_cross_family_release_cached(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    compact = [{
        "habit_instance_id": row["habit_instance_id"], "family": row["family"],
        "name": row["name"], "preference_value": row["preference_value"],
        "condition": row["condition"],
    } for row in rows]
    stop = set("prefer prefers preferred preference hotel hotels flight flights room rooms policy default when with and or the a an to of for in on is are as include includes included access option options that".split())
    def tokens(item: dict[str, Any]) -> set[str]:
        text = " ".join(str(item.get(key, "")) for key in ["name", "preference_value", "condition"]).lower()
        return {word for word in re.findall(r"[a-z0-9]+", text) if len(word) > 2 and word not in stop}
    parent = list(range(len(compact)))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]; index = parent[index]
        return index
    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right: parent[right] = left
    token_sets = [tokens(item) for item in compact]
    for left in range(len(compact)):
        for right in range(left + 1, len(compact)):
            if compact[left]["family"] == compact[right]["family"]:
                continue
            common = token_sets[left] & token_sets[right]
            similarity = len(common) / max(1, len(token_sets[left] | token_sets[right]))
            suspicious = similarity >= 0.25 or "breakfast" in common or {"pool", "fitness"}.issubset(common) or {"afternoon", "departure"}.issubset(common)
            if suspicious: union(left, right)
    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(compact): components[find(index)].append(item)
    decisions = []
    for component in components.values():
        if len(component) == 1:
            item = component[0]
            decisions.append({
                "action": "keep", "input_ids": [item["habit_instance_id"]],
                "representative_id": item["habit_instance_id"],
                "rationale": "No semantically similar cross-family release candidate was retrieved; this item already passed xhigh family review.",
                "family": item["family"], "alternative_group_key": "",
            })
            continue
        fingerprint = hashlib.sha256(json.dumps({
            "revision": "v04_cross_family_suspicious_cluster_r3", "model": args.model,
            "reasoning_effort": args.reasoning_effort, "component": component,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        path = args.dataset / "work" / "habit_cross_family_adjudication_cache" / f"cluster_{fingerprint}.json"
        if path.exists():
            decisions.extend(json.loads(path.read_text(encoding="utf-8"))["decisions"])
            continue
        prompt = {
            "task": "Adjudicate this small cluster of potentially duplicate travel habits retrieved from different taxonomy families.",
            "rules": [
                "Every input habit_instance_id must occur exactly once across decisions[].input_ids.",
                "Merge exact or policy-equivalent duplicates. Keep policies separate when they produce different answers, such as general departure timing versus return-only timing.",
                "Reject a trip-context or atmosphere label if the policy is merely a general amenity preference; otherwise assign one precise canonical family.",
                "Do not rewrite policy text. Return {decisions:[{action:keep|merge|reject,input_ids:[...],representative_id,rationale,family,alternative_group_key}]}",
            ],
            "candidate_cluster": component,
        }
        result = post_chat(
            base_url=args.base_url, api_key=args.api_key, model=args.model,
            messages=[
                {"role": "system", "content": "You are the final cross-taxonomy curator of a rigorous longitudinal-memory benchmark. Return strict JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            max_tokens=2500, timeout=args.timeout, retries=args.retries, transport=args.transport,
            reasoning_effort=args.reasoning_effort,
        )["json"]
        cluster_decisions = result.get("decisions", [])
        write_json(path, {"cache_fingerprint": fingerprint, "decisions": cluster_decisions})
        decisions.extend(cluster_decisions)
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=os.getenv("HABITBENCH_GEN_MODEL", "gpt-5.5"))
    parser.add_argument("--base-url", default=os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=9000)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--transport", choices=["curl", "urllib"], default="curl")
    parser.add_argument("--reasoning-effort", default=os.getenv("HABITBENCH_REASONING_EFFORT", "xhigh"))
    args = parser.parse_args()
    if not args.base_url or not args.api_key: raise SystemExit("Set HABITBENCH_BASE_URL and HABITBENCH_API_KEY")
    source = read_jsonl(args.dataset / "private" / "grounded_habit_instances.jsonl")
    batches = [source[i:i + args.batch_size] for i in range(0, len(source), args.batch_size)]
    reviewed = []
    failed_batches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(review_batch_cached, batch, args): batch for batch in batches}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                reviewed.extend(future.result())
            except Exception as exc:
                failed_batches.append(futures[future])
                print(f"semantic_review_batch_deferred error={exc}", flush=True)
            print(f"semantic_review_progress {index}/{len(batches)}", flush=True)
    for batch in failed_batches:
        print(f"semantic_review_split_recovery size={len(batch)}", flush=True)
        recovered = []
        for candidate in batch:
            recovered.extend(review_batch_cached([candidate], args))
        fingerprint, path = review_cache_path(batch, args)
        write_json(path, {"cache_fingerprint": fingerprint, "rows": recovered, "recovered_by_split": True})
        reviewed.extend(recovered)
    order = {row["habit_instance_id"]: i for i, row in enumerate(source)}
    reviewed.sort(key=lambda row: order[row["habit_instance_id"]])
    write_jsonl(args.dataset / "review" / "grounded_habit_semantic_review.jsonl", reviewed)
    eligible = [row for row in reviewed if row.get("decision") in {"keep", "revise"} and row.get("canonical_value")]
    eligible_by_family = defaultdict(list)
    for row in eligible: eligible_by_family[row["family"]].append(row)
    groups = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(consolidate_family_cached, family_rows, args): family for family, family_rows in eligible_by_family.items()}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            groups.extend(future.result())
            print(f"global_family_consolidation_progress {index}/{len(futures)}", flush=True)
    eligible_by_id = {row["habit_instance_id"]: row for row in eligible}
    seen, pool = set(), []
    for group in groups:
        member_ids = group.get("member_ids", [])
        representative_id = group.get("representative_id")
        if not member_ids or representative_id not in member_ids or any(member_id not in eligible_by_id or member_id in seen for member_id in member_ids):
            raise SystemExit("invalid global semantic consolidation output")
        families = {eligible_by_id[member_id]["family"] for member_id in member_ids}
        if len(families) != 1:
            raise SystemExit("global consolidation merged different families")
        seen.update(member_ids)
        best = max(
            (eligible_by_id[member_id] for member_id in member_ids),
            key=lambda row: float(row.get("semantic_review", {}).get("quality_score") or 0),
        )
        best = dict(best)
        best["canonical_value"] = re.sub(r"[^a-z0-9]+", "_", str(group.get("canonical_value", "")).lower()).strip("_")
        best["global_semantic_group"] = {
            "member_ids": member_ids, "representative_id": representative_id,
            "merge_rationale": group.get("merge_rationale"), "reviewed_by": args.model,
        }
        pool.append(best)
    if seen != set(eligible_by_id):
        raise SystemExit("global semantic consolidation did not cover every eligible habit")
    release_exclusions = []
    filtered_pool = []
    for row in pool:
        canonical = row["canonical_value"]
        reason = None
        if canonical == "current_trip_hotel_nightly_budget":
            reason = "explicit current-trip constraint is answerable without longitudinal memory"
        elif "continental_airlines" in canonical:
            reason = "obsolete carrier is not a realistic current booking default"
        elif canonical == "hotel_tour_availability":
            reason = "source service description was over-generalized into a hotel habit"
        if reason:
            release_exclusions.append({"habit_instance_id": row["habit_instance_id"], "family": row["family"], "canonical_value": canonical, "reason": reason})
        else:
            filtered_pool.append(row)
    pool = filtered_pool

    # A second, policy-level release adjudication catches residual near aliases,
    # nested thresholds, and default/fallback fragments that a plain synonym
    # clustering pass cannot represent.  It also records mutually exclusive
    # alternatives without incorrectly deleting useful cross-user diversity.
    provisional_by_id = {row["habit_instance_id"]: row for row in pool}
    provisional_by_family = defaultdict(list)
    for row in pool:
        provisional_by_family[row["family"]].append(row)
    release_decisions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(adjudicate_release_family_cached, rows, args): family
            for family, rows in provisional_by_family.items()
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            release_decisions.extend(future.result())
            print(f"release_policy_adjudication_progress {index}/{len(futures)}", flush=True)
    covered, final_pool = set(), []
    for decision in release_decisions:
        action = decision.get("action")
        input_ids = decision.get("input_ids")
        representative_id = decision.get("representative_id")
        if action not in {"keep", "merge", "reject"} or not isinstance(input_ids, list) or not input_ids:
            raise SystemExit("invalid release policy adjudication output")
        if any(item not in provisional_by_id or item in covered for item in input_ids):
            raise SystemExit("release policy adjudication has missing or repeated ids")
        families = {provisional_by_id[item]["family"] for item in input_ids}
        if len(families) != 1:
            raise SystemExit("release policy adjudication crossed families")
        covered.update(input_ids)
        if action == "reject":
            for item in input_ids:
                release_exclusions.append({
                    "habit_instance_id": item, "family": provisional_by_id[item]["family"],
                    "canonical_value": provisional_by_id[item]["canonical_value"],
                    "reason": str(decision.get("rationale") or "final release adjudication rejected"),
                })
            continue
        if representative_id not in input_ids:
            raise SystemExit("release policy representative is not a member")
        if action == "keep" and len(input_ids) != 1:
            raise SystemExit("keep decision must cover exactly one input")
        row = dict(provisional_by_id[representative_id])
        for key in ["canonical_value", "name", "preference_value", "condition", "default_action", "boundary_condition", "exception_condition"]:
            if not str(decision.get(key, "")).strip():
                raise SystemExit(f"release policy decision lacks {key}")
            row[key] = str(decision[key]).strip()
        row["canonical_value"] = re.sub(r"[^a-z0-9]+", "_", row["canonical_value"].lower()).strip("_")
        row["release_policy_adjudication"] = {
            "action": action, "input_ids": input_ids,
            "rationale": decision.get("rationale"),
            "alternative_group_key": re.sub(
                r"[^a-z0-9]+", "_", str(decision.get("alternative_group_key", "")).lower()
            ).strip("_"),
            "reviewed_by": args.model,
        }
        final_pool.append(row)
    if covered != set(provisional_by_id):
        raise SystemExit("release policy adjudication did not cover every provisional habit")
    pool = final_pool

    # Family-level review cannot detect the same semantic policy retrieved
    # through different taxonomy directories.  A final global pass is a hard
    # release gate, not an optional audit.
    pre_cross_by_id = {row["habit_instance_id"]: row for row in pool}
    cross_decisions = adjudicate_cross_family_release_cached(pool, args)
    cross_covered, cross_pool = set(), []
    for decision in cross_decisions:
        action = decision.get("action")
        input_ids = decision.get("input_ids")
        representative_id = decision.get("representative_id")
        if action not in {"keep", "merge", "reject"} or not isinstance(input_ids, list) or not input_ids:
            raise SystemExit("invalid cross-family adjudication output")
        if any(item not in pre_cross_by_id or item in cross_covered for item in input_ids):
            raise SystemExit("cross-family adjudication has missing or repeated ids")
        cross_covered.update(input_ids)
        if action == "reject":
            for item in input_ids:
                source_row = pre_cross_by_id[item]
                release_exclusions.append({
                    "habit_instance_id": item, "family": source_row["family"],
                    "canonical_value": source_row["canonical_value"],
                    "reason": str(decision.get("rationale") or "cross-family release adjudication rejected"),
                })
            continue
        if representative_id not in input_ids or (action == "keep" and len(input_ids) != 1):
            raise SystemExit("invalid cross-family representative or keep group")
        row = dict(pre_cross_by_id[representative_id])
        if not str(decision.get("family", "")).strip():
            raise SystemExit("cross-family decision lacks family")
        row["family"] = str(decision["family"]).strip()
        combined_evidence, combined_examples, seen_evidence, seen_examples = [], [], set(), set()
        for item in input_ids:
            for evidence in pre_cross_by_id[item].get("source_evidence", []):
                key = (evidence.get("conversation_id"), evidence.get("turn_index"))
                if key not in seen_evidence:
                    combined_evidence.append(evidence); seen_evidence.add(key)
            for example in pre_cross_by_id[item].get("source_examples", []):
                key = (example.get("conversation_id"), example.get("turn_index"))
                if key not in seen_examples:
                    combined_examples.append(example); seen_examples.add(key)
        row["source_evidence"] = combined_evidence
        row["source_examples"] = combined_examples
        row["cross_family_release_adjudication"] = {
            "action": action, "input_ids": input_ids, "rationale": decision.get("rationale"),
            "alternative_group_key": re.sub(r"[^a-z0-9]+", "_", str(decision.get("alternative_group_key", "")).lower()).strip("_"),
            "reviewed_by": args.model,
        }
        cross_pool.append(row)
    if cross_covered != set(pre_cross_by_id):
        raise SystemExit("cross-family adjudication did not cover every provisional habit")
    pool = cross_pool
    pool.sort(key=lambda row: (row["family"], row["canonical_value"]))
    write_jsonl(args.dataset / "review" / "global_semantic_groups.jsonl", groups)
    write_jsonl(args.dataset / "review" / "curated_release_exclusions.jsonl", release_exclusions)
    write_jsonl(args.dataset / "review" / "release_policy_adjudication.jsonl", release_decisions)
    write_jsonl(args.dataset / "review" / "cross_family_release_adjudication.jsonl", cross_decisions)
    write_jsonl(args.dataset / "private" / "curated_habit_pool.jsonl", pool)
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "input_candidates": len(source), "decision_counts": dict(Counter(row.get("decision") for row in reviewed)),
        "eligible_before_semantic_merge": len(eligible), "curated_unique_habits": len(pool),
        "release_exclusions": len(release_exclusions),
        "pre_cross_family_habits": len(pre_cross_by_id),
        "family_counts": dict(Counter(row["family"] for row in pool)),
    }
    write_json(args.dataset / "reports" / "grounded_habit_semantic_review_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()
