#!/usr/bin/env python3
"""Generate v0.4 profiles, arcs, and full sessions directly with an LLM.

This file intentionally contains no dialogue templates.  The only authored
text is generation instructions and validators; all benchmark messages are
returned directly by the configured generation model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from api_client import post_chat, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "runs_wxq" / "taskmaster_planning_defaults_v0_4"
GENERATION_METHOD = "llm_direct_from_grounded_dossier_and_chronological_arc"
PIPELINE_REVISION = "v04_gpt_e2e_grounded_7_15_route_r17_iterative_probe_repair"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def private_probe_shuffle_seed(dataset: Path) -> tuple[str, str]:
    """Return a persistent private seed and its publishable fingerprint."""
    path = dataset / "private" / "probe_shuffle_seed.txt"
    if path.exists():
        seed = path.read_text(encoding="utf-8").strip()
    else:
        seed = os.urandom(32).hex()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(seed + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
    if len(seed) < 32:
        raise ValueError("private probe shuffle seed must contain at least 32 characters")
    return seed, hashlib.sha256(seed.encode("utf-8")).hexdigest()


def private_shuffle_rank(seed: str, namespace: str, item: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{item}".encode("utf-8")).hexdigest()


def remap_choice_references(text: Any, old_to_new: dict[str, str]) -> Any:
    """Remap explicit Choice/Option letters after private option shuffling."""
    if not isinstance(text, str):
        return text
    pattern = re.compile(r"\b(choice|option)\s+([ABCD])\b", re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        old_id = match.group(2).upper()
        return f"{match.group(1)} {old_to_new.get(old_id, old_id)}"

    return pattern.sub(replace, text)


def cache_fingerprint(args: argparse.Namespace, stage: str, payload: Any) -> str:
    material = json.dumps({
        "pipeline_revision": PIPELINE_REVISION, "stage": stage, "model": args.model,
        "reasoning_effort": args.reasoning_effort, "payload": payload,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def exact_qwen_token_count(args: argparse.Namespace, texts: list[str]) -> int:
    helper = Path(__file__).with_name("count_qwen_tokens.py")
    proc = subprocess.run(
        [str(args.tokenizer_python), str(helper), str(args.tokenizer)],
        input=json.dumps({"texts": texts}, ensure_ascii=False), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"exact Qwen tokenizer helper failed: {proc.stderr[-1200:]}")
    return int(json.loads(proc.stdout)["total"])


def api_json(args: argparse.Namespace, system: str, payload: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    if args.transport == "curl_stream":
        request = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system + " Return strict JSON only."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "response_format": {"type": "json_object"},
            "stream": True,
        }
        url = args.base_url.rstrip("/") + "/chat/completions"
        last_error = None
        for attempt in range(args.retries + 1):
            proc = subprocess.run(
                [
                    "curl", "-sS", "-N", "--http1.1", "--connect-timeout", "30",
                    "--max-time", str(args.timeout),
                    "-H", f"Authorization: Bearer {args.api_key}",
                    "-H", "Content-Type: application/json", "--data-binary", "@-", url,
                ],
                input=json.dumps(request, ensure_ascii=False), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            content_parts = []
            try:
                # Standard SSE chat-completions stream. Reasoning chunks keep
                # the connection alive but only final content is assembled.
                for line in proc.stdout.splitlines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if isinstance(delta.get("content"), str):
                            content_parts.append(delta["content"])
                content = "".join(content_parts).strip()
                if proc.returncode == 0 and content:
                    return json.loads(content)
                # Some compatible providers ignore stream=true and return one
                # ordinary JSON response; accept that form as a fallback.
                if proc.returncode == 0 and proc.stdout.lstrip().startswith("{"):
                    body = json.loads(proc.stdout)
                    return json.loads(body["choices"][0]["message"]["content"])
                last_error = f"stream_exit_{proc.returncode}:{proc.stderr[:500]}:content_chars={len(content)}"
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                last_error = f"bad_stream_json:{type(exc).__name__}:{str(exc)[:300]}:stdout_tail={proc.stdout[-500:]}"
            if attempt < args.retries:
                time.sleep(min(2 * (attempt + 1), 10))
        raise RuntimeError(last_error or "unknown_stream_error")
    return post_chat(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        messages=[
            {"role": "system", "content": system + " Return strict JSON only."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        max_tokens=max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        transport=args.transport,
        reasoning_effort=args.reasoning_effort,
    )["json"]


def validate_profile(raw: Any, user_id: str, candidates: dict[str, dict[str, Any]], model: str, effort: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("habits"), list):
        raise ValueError(f"invalid profile for {user_id}")
    if not 5 <= len(raw["habits"]) <= 8:
        raise ValueError(f"{user_id} must have a natural 5-8 habits, got {len(raw['habits'])}")
    used = set()
    habits = []
    family_values: set[tuple[str, str]] = set()
    family_scopes: dict[str, set[str]] = defaultdict(set)
    for index, habit in enumerate(raw["habits"]):
        source_id = habit.get("source_habit_instance_id")
        if source_id not in candidates or source_id in used:
            raise ValueError(f"{user_id} cites missing or repeated source habit {source_id}")
        source = candidates[source_id]
        family = source["family"]
        alternative_group_key = str(
            source.get("release_policy_adjudication", {}).get("alternative_group_key", "")
        ).strip()
        required = ["scope_key", "scope_condition", "strength"]
        if any(not str(habit.get(key, "")).strip() for key in required):
            raise ValueError(f"incomplete habit in {user_id}")
        if habit["strength"] not in {"primary", "secondary", "background"}:
            raise ValueError(f"invalid habit strength in {user_id}")
        key = (family, str(source.get("canonical_value") or habit["preference_value"]).strip().lower())
        if key in family_values:
            raise ValueError(f"duplicate family/value in {user_id}: {key}")
        scope_key = re.sub(r"[^a-z0-9]+", "_", str(habit["scope_key"]).lower()).strip("_")
        if not scope_key or scope_key in family_scopes[family] or len(family_scopes[family]) >= 2:
            raise ValueError(f"invalid or repeated scope for family {family} in {user_id}")
        family_values.add(key)
        family_scopes[family].add(scope_key)
        used.add(source_id)
        habits.append(
            {
                "habit_id": f"{user_id}_h{index:02d}",
                "user_id": user_id,
                "family": family,
                "alternative_group_key": alternative_group_key,
                "scope_key": scope_key,
                "strength": str(habit["strength"]),
                "testable": bool(habit.get("testable", habit["strength"] != "background")),
                "scope_condition": str(habit["scope_condition"]).strip(),
                "preference_value": str(source["preference_value"]).strip(),
                "name": str(source["name"]).strip(),
                "condition": str(source["condition"]).strip(),
                "default_action": str(source["default_action"]).strip(),
                "boundary_condition": str(source["boundary_condition"]).strip(),
                "exception_condition": str(source["exception_condition"]).strip(),
                "source_habit_instance_id": source_id,
                "source_bundle_id": source["bundle_id"],
            }
        )
    if not 4 <= len(family_scopes) <= 7:
        raise ValueError(f"{user_id} should naturally cover 4-7 families, got {len(family_scopes)}")
    if sum(bool(habit["testable"]) for habit in habits) < 5:
        raise ValueError(f"{user_id} needs at least 5 independently testable habits")
    longitudinal_plan = raw.get("longitudinal_plan")
    if not isinstance(longitudinal_plan, dict):
        raise ValueError(f"{user_id} lacks longitudinal_plan")
    target_sessions = int(longitudinal_plan.get("target_sessions", 0))
    if not 100 <= target_sessions <= 150:
        raise ValueError(f"{user_id} target_sessions must be 100-150, got {target_sessions}")
    if not str(longitudinal_plan.get("travel_tempo", "")).strip():
        raise ValueError(f"{user_id} lacks a travel tempo")
    target_characters = int(longitudinal_plan.get("target_history_characters", 0))
    if not 220_000 <= target_characters <= 400_000:
        raise ValueError(f"{user_id} target_history_characters must be 220k-400k")
    return {
        "user_id": user_id,
        "persona": raw.get("persona", {}),
        "travel_contexts": raw.get("travel_contexts", []),
        "communication_style": raw.get("communication_style", {}),
        "habit_interactions": raw.get("habit_interactions", []),
        "longitudinal_plan": longitudinal_plan,
        "habits": habits,
        "generation_method": GENERATION_METHOD,
        "generated_by": {"model": model, "reasoning_effort": effort},
    }


def review_profile(args: argparse.Namespace, profile: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> str | None:
    sources = [
        {
            **{k: candidates[h["source_habit_instance_id"]][k] for k in ["habit_instance_id", "family", "preference_value", "condition", "default_action"]},
            "alternative_group_key": candidates[h["source_habit_instance_id"]].get("release_policy_adjudication", {}).get("alternative_group_key", ""),
        }
        for h in profile["habits"]
    ]
    result = api_json(args, "You independently audit a synthetic longitudinal user dossier for semantic consistency.", {
        "task": "Audit the dossier. Reject cloned benchmark structure, unsupported source drift, overlapping same-family scopes, default/fallback splitting, or an incoherent persona.",
        "dossier": profile,
        "grounded_sources": sources,
        "requirements": [
            "Every selected habit must remain semantically faithful to its grounded source.",
            "Two habits in one family are valid only for genuinely disjoint recurring contexts, not synonymous scope labels.",
            "Two policies in the same non-empty alternative_group_key are mutually exclusive in one scope; accept both only when the dossier establishes genuinely disjoint recurring contexts.",
            "A default and its fallback tolerance must be one graph.",
            "The habit count and 100-150 session target must follow the persona rather than a balancing quota.",
            "Return {decision:accept|reject,reason,conflicting_habit_ids:[]}.",
        ],
    }, 3000)
    return None if result.get("decision") == "accept" else str(result.get("reason") or "profile_semantic_review_rejected")


def profile_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    habits = read_jsonl(args.dataset / "private" / "curated_habit_pool.jsonl")
    if len(habits) < 10:
        raise SystemExit("need at least 10 accepted grounded habits before profile generation")
    by_id = {row["habit_instance_id"]: row for row in habits}
    rng = random.Random(args.seed)
    shuffled = habits[:]
    rng.shuffle(shuffled)
    profiles = []
    source_usage: Counter[str] = Counter()
    for user_index in range(args.users):
        user_id = f"tm_pd_v04_user_{user_index:03d}"
        # Candidate rotation is only source allocation. The model decides the
        # coherent non-fixed habit set and authors the dossier.
        ranked = sorted(shuffled, key=lambda row: source_usage[row["habit_instance_id"]])
        # First expose one relatively underused value per family, then add
        # alternatives. This prevents cloned users without imposing a fixed
        # frequency or forcing the model to select any candidate.
        offered, represented = [], set()
        for row in ranked:
            if row["family"] not in represented:
                offered.append(row); represented.add(row["family"])
        for row in ranked:
            if row not in offered and len(offered) < min(24, len(shuffled)):
                offered.append(row)
        payload = {
            "task": "Create one realistic synthetic long-term travel-planning user dossier from grounded habit candidates.",
            "user_id": user_id,
            "requirements": [
                "Choose a natural count of 5 to 8 habits across only 4 to 7 relevant families. Derive the count from the persona and recurring travel life; do not aim for a midpoint, maximum, or table balance.",
                "A family is taxonomy only; preserve a concrete scoped preference value for every selected habit.",
                "Most selected families should contribute one habit. A family may contribute two only when their scope_key values describe genuinely disjoint recurring contexts, such as work_travel versus family_leisure, and their actions can differ without contradiction.",
                "Never split a default and its fallback/tolerance into separate habits: for example prefer_nonstop plus tolerate_one_stop is one stop-tolerance graph with a boundary, not two habits.",
                "Do not cover a family merely because it is offered. Real users have no stable default for many travel decisions.",
                "Avoid contradictory habits unless their conditions clearly separate contexts such as work, family, solo, or urgent travel.",
                "Make the user feel like one person with recurring but varied travel needs, not a collection of independent templates.",
                "Do not add habits unsupported by a supplied source_habit_instance_id.",
                "Choose target_sessions naturally between 100 and 150 and target_history_characters between 220000 and 400000 from travel tempo; users must not all have the same length. The completed history must exceed the local Qwen3-8B context rather than fit inside one window.",
                "Assign strength=primary|secondary|background from persona salience. Background habits make histories realistic and need not all be probed.",
                "Do not rewrite the supplied policy graph. Select it by source_habit_instance_id and provide only the recurring persona scope in which that policy applies.",
                "Return {persona,travel_contexts,communication_style,habit_interactions,habit_count_rationale,longitudinal_plan:{target_sessions,target_history_characters,travel_tempo,typical_episode_shape},habits:[{source_habit_instance_id,scope_key,scope_condition,strength,testable}]}",
            ],
            "grounded_candidates": [
                {
                    **{k: row[k] for k in ["habit_instance_id", "family", "name", "preference_value", "condition", "default_action", "boundary_condition", "exception_condition"]},
                    "alternative_group_key": row.get("release_policy_adjudication", {}).get("alternative_group_key", ""),
                }
                for row in offered
            ],
            "existing_users_to_avoid_cloning": [
                {
                    "user_id": row["user_id"],
                    "target_sessions": row["longitudinal_plan"]["target_sessions"],
                    "source_habit_instance_ids": [habit["source_habit_instance_id"] for habit in row["habits"]],
                    "persona": row.get("persona"),
                }
                for row in profiles
            ],
        }
        profile_cache = args.dataset / "work" / "profiles" / f"{user_id}.json"
        profile_fp = cache_fingerprint(args, "profile", payload)
        cached_profile = None
        if profile_cache.exists():
            cached_row = json.loads(profile_cache.read_text(encoding="utf-8"))
            if cached_row.get("cache_fingerprint") == profile_fp:
                cached_profile = cached_row.get("profile")
        last_error, profile = None, cached_profile
        for attempt in range(3 if profile is None else 0):
            request_payload = dict(payload)
            if last_error:
                request_payload["previous_output_rejection"] = str(last_error)
            try:
                raw = api_json(args, "You design coherent longitudinal benchmark users grounded in real travel-task evidence.", request_payload, 5500)
                profile = validate_profile(raw, user_id, by_id, args.model, args.reasoning_effort)
                semantic_error = review_profile(args, profile, by_id)
                if semantic_error:
                    raise ValueError(semantic_error)
                break
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                profile, last_error = None, exc
        if profile is None:
            raise ValueError(f"profile generation failed after 3 attempts for {user_id}: {last_error}")
        profiles.append(profile)
        source_usage.update(habit["source_habit_instance_id"] for habit in profile["habits"])
        write_json(profile_cache, {"cache_fingerprint": profile_fp, "profile": profiles[-1]})
        print(f"profile_progress {len(profiles)}/{args.users}", flush=True)
    if len({profile["longitudinal_plan"]["target_sessions"] for profile in profiles}) < min(3, len(profiles)):
        for profile in profiles[-2:]:
            (args.dataset / "work" / "profiles" / f"{profile['user_id']}.json").unlink(missing_ok=True)
        raise ValueError("user histories are still too uniform in length")
    for left_index, left in enumerate(profiles):
        left_ids = {habit["source_habit_instance_id"] for habit in left["habits"]}
        for right in profiles[left_index + 1:]:
            right_ids = {habit["source_habit_instance_id"] for habit in right["habits"]}
            if len(left_ids & right_ids) / max(1, len(left_ids | right_ids)) > 0.60:
                (args.dataset / "work" / "profiles" / f"{right['user_id']}.json").unlink(missing_ok=True)
                raise ValueError(f"cloned habit composition: {left['user_id']} {right['user_id']}")
    write_jsonl(args.dataset / "private" / "user_dossiers.jsonl", profiles)
    return profiles


def compact_arc_source_palette(
    profile: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    source_cards: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a dialogue-grounding palette without exposing habit labels.

    The event author sees real Taskmaster interaction fragments, but it does
    not see the hidden habit graph or any support/exception coverage target.
    """
    source_ids = []
    for habit in profile["habits"]:
        source = candidates[habit["source_habit_instance_id"]]
        source_ids.extend(example["conversation_id"] for example in source["source_examples"])
    palette = []
    for conversation_id in dict.fromkeys(source_ids):
        card = source_cards.get(conversation_id)
        if not card:
            continue
        turns = card.get("turns", [])
        # A compact contiguous fragment preserves dialogue/task realism while
        # keeping event-planning calls small enough for xhigh reliability.
        fragment = turns[:6]
        palette.append({
            "conversation_id": conversation_id,
            "instruction_id": card.get("instruction_id"),
            "source_domain": card.get("source_domain"),
            "turns": [{"role": row.get("role"), "text": row.get("text")} for row in fragment],
        })
    if len(palette) < 8:
        raise ValueError(f"insufficient Taskmaster grounding palette for {profile['user_id']}")
    return palette


def event_author_view(profile: dict[str, Any]) -> dict[str, Any]:
    """Return the complete dossier used by the 7.15 event-first pass."""
    return {
        "user_id": profile["user_id"],
        "persona": profile.get("persona"),
        "travel_contexts": profile.get("travel_contexts"),
        "communication_style": profile.get("communication_style"),
        "habit_interactions": profile.get("habit_interactions"),
        "habits": profile.get("habits"),
        "longitudinal_plan": profile.get("longitudinal_plan"),
    }


def validate_natural_arc_events(
    raw: Any,
    profile: dict[str, Any],
    start: int,
    count: int,
    allowed_source_ids: set[str],
) -> list[dict[str, Any]]:
    events = raw.get("events") if isinstance(raw, dict) else None
    if not isinstance(events, list) or len(events) != count:
        raise ValueError(f"arc block expected {count} events, got {len(events) if isinstance(events, list) else 'invalid'}")
    output = []
    for offset, event in enumerate(events):
        episode_id = str(event.get("episode_id", "")).strip()
        days_after_previous = event.get("days_after_previous")
        if not episode_id or not isinstance(days_after_previous, int) or not 0 <= days_after_previous <= 180:
            raise ValueError("invalid episode_id or days_after_previous")
        scenario = str(event.get("scenario_brief", "")).strip()
        interaction_goal = str(event.get("interaction_goal", "")).strip()
        if len(scenario) < 40 or len(interaction_goal) < 20:
            raise ValueError("natural event lacks a concrete scenario or interaction goal")
        grounding_ids = list(dict.fromkeys(event.get("grounding_source_ids", [])))
        if not grounding_ids or len(grounding_ids) > 3 or any(item not in allowed_source_ids for item in grounding_ids):
            raise ValueError("natural event has missing or invalid Taskmaster grounding")
        output.append(
            {
                "user_id": profile["user_id"],
                "session_index": start + offset,
                "domain": event.get("domain", "mixed_travel"),
                "scenario_brief": scenario,
                "continuity_hook": str(event.get("continuity_hook", "")).strip(),
                "episode_id": episode_id,
                "days_after_previous": days_after_previous,
                "grounding_source_ids": grounding_ids,
                "interaction_goal": interaction_goal,
                "session_length": event.get("session_length"),
                "event_first_provenance": {
                    "dossier_conditioned_without_signal_labels": True,
                    "model": profile["generated_by"]["model"],
                    "reasoning_effort": profile["generated_by"]["reasoning_effort"],
                },
            }
        )
        if output[-1]["session_length"] not in {"micro", "short", "medium", "long"}:
            raise ValueError("arc event lacks a valid variable session_length")
    return output


def validate_arc_mapping(raw: Any, profile: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations = raw.get("event_annotations") if isinstance(raw, dict) else None
    if not isinstance(annotations, list) or len(annotations) != len(events):
        raise ValueError("habit mapping must return one annotation per natural event")
    expected = {event["session_index"] for event in events}
    by_index = {int(row.get("session_index", -1)): row for row in annotations if isinstance(row, dict)}
    if set(by_index) != expected:
        raise ValueError("habit mapping session indices do not match natural events")
    habit_ids = {habit["habit_id"] for habit in profile["habits"]}
    output = []
    for event in events:
        annotation = by_index[event["session_index"]]
        raw_signals = annotation.get("habit_signals", [])
        if not isinstance(raw_signals, list):
            raise ValueError("habit_signals must be a list")
        signals, seen = [], set()
        for item in raw_signals:
            habit_id, signal = item.get("habit_id"), item.get("signal_type")
            if habit_id not in habit_ids or habit_id in seen:
                raise ValueError("habit mapping cites an unknown or repeated habit")
            if signal not in {"support", "boundary", "exception", "revision"}:
                raise ValueError(f"invalid per-habit signal {signal}")
            evidence_intent = str(item.get("evidence_intent", "")).strip()
            if len(evidence_intent) < 15:
                raise ValueError("mapped signal lacks a concrete evidence intent")
            seen.add(habit_id)
            signals.append({"habit_id": habit_id, "signal_type": signal, "evidence_intent": evidence_intent})
        updates = annotation.get("state_updates", [])
        if not isinstance(updates, list):
            raise ValueError("state_updates must be a list")
        revision_ids = {item["habit_id"] for item in signals if item["signal_type"] == "revision"}
        update_ids = {item.get("habit_id") for item in updates if isinstance(item, dict)}
        if revision_ids != update_ids or len(update_ids) != len(updates):
            raise ValueError("each revision signal requires exactly one matching state update")
        for update in updates:
            required = ["preference_value", "condition", "default_action", "boundary_condition", "exception_condition", "revision_reason"]
            if update.get("habit_id") not in habit_ids or any(not str(update.get(key, "")).strip() for key in required):
                raise ValueError("incomplete temporal habit state update")
        mapped = dict(event)
        mapped.update({
            "habit_signals": signals,
            "linked_habit_ids": [item["habit_id"] for item in signals],
            "state_updates": updates,
            "natural_disclosure_plan": str(annotation.get("natural_disclosure_plan", "")).strip(),
            "habit_mapping_provenance": {
                "independent_after_event_authoring": True,
                "model": profile["generated_by"]["model"],
                "reasoning_effort": profile["generated_by"]["reasoning_effort"],
            },
        })
        output.append(mapped)
    return output


def arc_stage(args: argparse.Namespace, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = {
        row["habit_instance_id"]: row
        for row in read_jsonl(args.dataset / "private" / "curated_habit_pool.jsonl")
    }
    source_cards = {
        row["conversation_id"]: row
        for row in read_jsonl(args.dataset / "sources" / "taskmaster_source_cards.jsonl")
    }
    selected_user_ids = {profile["user_id"] for profile in profiles}
    all_events = [
        row for row in read_jsonl(args.dataset / "private" / "chronological_arc.jsonl")
        if row.get("user_id") not in selected_user_ids
    ]
    all_versions = [
        row for row in read_jsonl(args.dataset / "private" / "habit_version_history.jsonl")
        if row.get("user_id") not in selected_user_ids
    ]
    for profile in profiles:
        natural_events = []
        target_sessions = int(profile["longitudinal_plan"]["target_sessions"])
        complete_cache = args.dataset / "work" / "arcs" / f"{profile['user_id']}_complete.json"
        complete_fp = cache_fingerprint(args, "complete_true_event_first_arc", {
            "profile": profile, "target_sessions": target_sessions, "block_size": args.arc_block_size,
        })
        if complete_cache.exists():
            cached_arc = json.loads(complete_cache.read_text(encoding="utf-8"))
            if cached_arc.get("cache_fingerprint") == complete_fp:
                all_events.extend(cached_arc["events"])
                all_versions.extend(cached_arc["versions"])
                print(f"arc_resume {profile['user_id']} {len(cached_arc['events'])}/{target_sessions}", flush=True)
                continue
        palette = compact_arc_source_palette(profile, candidates, source_cards)
        allowed_source_ids = {row["conversation_id"] for row in palette}
        # 7.15 route: author the natural chronological travel life from the
        # complete dossier, but never provide signal labels or coverage counts.
        current_versions = {habit["habit_id"]: 1 for habit in profile["habits"]}
        current_states = {habit["habit_id"]: dict(habit) for habit in profile["habits"]}
        version_history = [dict(habit, version=1, effective_from_session=0) for habit in profile["habits"]]
        for start in range(0, target_sessions, args.arc_block_size):
            count = min(args.arc_block_size, target_sessions - start)
            phase = "early" if start < target_sessions / 3 else "middle" if start < 2 * target_sessions / 3 else "late"
            palette_offset = (start // args.arc_block_size) * max(1, len(palette) // 3)
            rotating_palette = [palette[(palette_offset + index) % len(palette)] for index in range(min(3, len(palette)))]
            payload = {
                "task": "Design the next chronological block of one realistic long-term travel-planning relationship from the complete user dossier. This is an event plan, not dialogue and not a memory-label plan.",
                "phase": phase,
                "session_index_start": start,
                "exact_event_count": count,
                "user_dossier": event_author_view(profile),
                "recent_events": natural_events[-10:],
                "taskmaster_grounding_palette": rotating_palette,
                "whole_history_character_target": profile["longitudinal_plan"]["target_history_characters"],
                "requirements": [
                    "Return exactly the requested number of distinct events in chronological order, grouped into realistic trip episodes.",
                    "Each event needs episode_id and days_after_previous. Reuse an episode_id for planning, follow-up, booking changes, and pre-trip checks; use irregular gaps within and between episodes.",
                    "Let the person's recurring work, conference, family, and personal travel generate the event mix; include flights, hotels, combined trips, refinements, bookings, changes, and pre-trip follow-ups only when they fit.",
                    "Use the dossier habits only as ordinary character behavior and decision context. Do not mention memory, support, boundary, exception, revision, coverage, tests, probes, habit IDs, or labels anywhere in the output.",
                    "Do not mechanically cycle destinations, trip purposes, constraints, amenities, or interaction shapes.",
                    "Some sessions may be ordinary logistical follow-ups with no stable preference implicated.",
                    "Assign session_length micro, short, medium, or long from the interaction itself. Preserve occasional very brief follow-ups and occasional deep planning; do not target a fixed class matrix.",
                    "Ground each event in one to three supplied Taskmaster conversation IDs, using them only for task realism; do not copy wording, destinations, dates, prices, or names.",
                    "scenario_brief must contain concrete trip context, current constraints, and what changed or remains unresolved. interaction_goal describes what this session should accomplish.",
                    "Return {events:[{episode_id,days_after_previous,domain,scenario_brief,continuity_hook,grounding_source_ids:[conversation_id],interaction_goal,session_length}]}",
                ],
            }
            arc_block_cache = args.dataset / "work" / "arcs" / "natural" / f"{profile['user_id']}_{start:04d}.json"
            arc_block_fp = cache_fingerprint(args, "habit_blind_natural_arc_block", payload)
            block, last_error = None, None
            if arc_block_cache.exists():
                cached_block = json.loads(arc_block_cache.read_text(encoding="utf-8"))
                if cached_block.get("cache_fingerprint") == arc_block_fp:
                    block = cached_block.get("events")
            for attempt in range(3 if block is None else 0):
                if last_error:
                    payload["previous_output_rejection"] = str(last_error)
                raw = api_json(args, "You create varied chronological travel event arcs from a coherent dossier, without dialogue templates, signal labels, or coverage quotas.", payload, 4500)
                try:
                    candidate = validate_natural_arc_events(raw, profile, start, count, allowed_source_ids)
                    length_counts = Counter(event["session_length"] for event in candidate)
                    if count >= 6 and (len(length_counts) < 2 or max(length_counts.values()) > int(count * 0.80)):
                        raise ValueError(f"session lengths lack natural variation {dict(length_counts)}")
                    episode_counts = Counter(event["episode_id"] for event in candidate)
                    if count >= 10 and not max(3, count // 7) <= len(episode_counts) <= max(5, int(count * 0.70)):
                        raise ValueError(f"unrealistic episode count {len(episode_counts)} for {count} sessions")
                    gap_counts = Counter(event["days_after_previous"] for event in candidate)
                    if count >= 10 and (len(gap_counts) < 4 or max(gap_counts.values()) > int(count * 0.55)):
                        raise ValueError(f"timestamp gaps are too mechanical {dict(gap_counts)}")
                    normalized = [re.sub(r"[^a-z0-9]+", " ", event["scenario_brief"].lower()).strip() for event in candidate]
                    if len(normalized) != len(set(normalized)):
                        raise ValueError("natural arc block contains duplicate scenario briefs")
                    block = candidate
                    break
                except ValueError as exc:
                    last_error = exc
            if block is None:
                raise ValueError(f"dossier-conditioned natural arc block failed after 3 attempts: {last_error}")
            natural_events.extend(block)
            write_json(arc_block_cache, {"cache_fingerprint": arc_block_fp, "events": block})
            print(f"natural_arc_progress {profile['user_id']} {len(natural_events)}/{target_sessions}", flush=True)

        # No pre-dialogue habit labels are created. Evidence is classified only
        # after the final session text exists.
        all_events.extend(natural_events)
        all_versions.extend(version_history)
        write_json(complete_cache, {"cache_fingerprint": complete_fp, "events": natural_events, "versions": version_history})
        write_json(args.dataset / "reports" / f"{profile['user_id']}_arc_quality.json", {
            "user_id": profile["user_id"], "status": "pass", "event_first": True,
            "dossier_conditioned": True, "pre_dialogue_signal_labels": False,
            "natural_events": len(natural_events),
        })
    write_jsonl(args.dataset / "private" / "chronological_arc_natural.jsonl", all_events)
    write_jsonl(args.dataset / "private" / "chronological_arc.jsonl", all_events)
    write_jsonl(args.dataset / "private" / "habit_version_history.jsonl", all_versions)
    return all_events


def source_excerpts(
    profile: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    source_cards: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for habit in profile["habits"]:
        source = candidates[habit["source_habit_instance_id"]]
        grounded_dialogues = []
        for example in source["source_examples"]:
            card = source_cards.get(example["conversation_id"])
            if card:
                grounded_dialogues.append({
                    "conversation_id": card["conversation_id"],
                    "instruction_id": card["instruction_id"],
                    "turns": card["turns"][:24],
                })
        output[habit["habit_id"]] = grounded_dialogues
    return output


def validate_sessions(raw: Any, profile: dict[str, Any], events: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    sessions = raw.get("sessions") if isinstance(raw, dict) else None
    if not isinstance(sessions, list) or len(sessions) != len(events):
        raise ValueError("session block count mismatch")
    by_index = {int(row.get("session_index", -1)): row for row in sessions}
    output = []
    for event in events:
        index = event["session_index"]
        row = by_index.get(index)
        if row is None or not isinstance(row.get("messages"), list):
            raise ValueError(f"missing session {index}")
        messages = row["messages"]
        length_limits = {
            # Arc length classes are generation targets, not exact character
            # templates. Keep the authored target bands in the prompt while
            # allowing modest natural overflow at validation time.
            "micro": (4, 8, 300, 1200),
            "short": (8, 12, 1000, 2600),
            "medium": (12, 20, 2000, 4600),
            "long": (18, 28, 3500, 7500),
        }
        min_turns, max_turns, min_chars, max_chars = length_limits[event["session_length"]]
        total_chars = sum(len(str(message.get("content", ""))) for message in messages)
        if not min_turns <= len(messages) <= max_turns or not min_chars <= total_chars <= max_chars:
            raise ValueError(
                f"session {index} length mismatch class={event['session_length']} "
                f"messages={len(messages)} chars={total_chars}"
            )
        for turn, message in enumerate(messages):
            expected = "user" if turn % 2 == 0 else "assistant"
            if message.get("role") != expected or len(str(message.get("content", "")).strip()) < 8:
                raise ValueError(f"malformed turn in session {index}")
        if len(messages) % 2 != 0 or messages[-1].get("role") != "assistant":
            raise ValueError(f"session {index} must end with assistant")
        output.append(
            {
                "user_id": profile["user_id"],
                "session_id": f"{profile['user_id']}_s{index:04d}",
                "session_index": index,
                "domain": event["domain"],
                "messages": messages,
                "generation_provenance": {
                    "method": GENERATION_METHOD,
                    "model": model,
                    "reasoning_effort": "xhigh",
                    "template_or_rewrite_stage": False,
                },
                "memory_annotations": {
                    "evidence_summary": str(row.get("evidence_summary", "")).strip(),
                    "pre_dialogue_signal_labels_used": False,
                },
            }
        )
    return output


def independently_verify_sessions(
    args: argparse.Namespace,
    profile: dict[str, Any],
    events: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    previously_user_established_habit_ids: set[str],
) -> tuple[dict[int, dict[str, Any]], str | None]:
    payload_sessions = []
    for event, session in zip(events, sessions):
        payload_sessions.append({
            "session_index": event["session_index"],
            "candidate_habits": profile["habits"],
            "user_established_before_block": sorted(previously_user_established_habit_ids),
            "messages": session["messages"],
        })
    raw = api_json(args, "You independently label only what final user utterances actually establish; ignore any generation plan.", {
        "task": "Classify per-habit evidence in each final dialogue without seeing its planned labels.",
        "sessions": payload_sessions,
        "requirements": [
            "For each session, return only habits actually evidenced by user utterances.",
            "The quoted user words must themselves provide strong semantic evidence for the exact habit graph. Do not rely on an assistant summary, the dossier, a merely compatible scenario, or broad goals such as predictability to fill missing semantics.",
            "Merely mentioning the relevant attribute is insufficient. A rating habit needs the concrete threshold or an unambiguous endorsement of that threshold; an included-breakfast habit needs included/free breakfast rather than breakfast in general; a carrier, seat, timing, or layover habit needs its concrete value or an unambiguous endorsement of that exact option.",
            "support requires the user to request, choose, endorse, or naturally reaffirm the concrete default action. A question such as 'would a one-stop be okay?' is not support unless the user also endorses the answer.",
            "boundary requires user language that establishes scope or a tradeoff boundary; a price question or missing information alone is not a boundary. exception requires an explicit one-trip override. revision requires an explicit durable change, not a one-trip override.",
            "Provide an exact contiguous evidence_quote copied from a user message and confidence from 0 to 1 for every signal. Emit only signals with confidence >= 0.90.",
            "If no habit is evidenced, return habit_signals=[].",
            "Audit information origin in chronological order. For a habit not listed in user_established_before_block, set leakage=true if the assistant attributes the dossier-specific value to the user, treats it as an already-known requirement, or silently filters/recommends around it before user evidence establishes it. Once a user has established it, the assistant may summarize or apply it in this and later sessions.",
            "Normal option elicitation is not leakage: the assistant may neutrally ask about an attribute or list factual attributes of multiple concrete options (for example, one hotel includes breakfast and another does not). A later explicit user choice or endorsement can be valid user-originated evidence. Merely echoing the offered words without a meaningful choice is still insufficient evidence.",
            "Also set leakage=true when the dialogue exposes benchmark/memory/hidden-profile meta-language or mechanically enumerates dossier facts unrelated to solving the current travel task. An ordinary user-originated explicit travel preference, concrete choice, or natural callback is valid evidence and is not leakage.",
            "Return {verdicts:[{session_index,habit_signals:[{habit_id,signal_type,evidence_quote,confidence}],leakage,notes}]}",
        ],
    }, 10000)
    verdicts = raw.get("verdicts") if isinstance(raw, dict) else None
    if not isinstance(verdicts, list):
        return {}, "verifier_invalid_output"
    by_index = {int(row.get("session_index", -1)): row for row in verdicts if isinstance(row, dict)}
    verified = {}
    valid_habit_ids = {habit["habit_id"] for habit in profile["habits"]}
    for event, session in zip(events, sessions):
        index = event["session_index"]
        verdict = by_index.get(index)
        if verdict is None or not isinstance(verdict.get("habit_signals"), list):
            return {}, f"verifier_missing_session_{index}"
        if not isinstance(verdict.get("leakage", False), bool):
            return {}, f"verifier_non_boolean_leakage_{index}"
        actual = {}
        user_text = "\n".join(message["content"] for message in session["messages"] if message["role"] == "user").lower()
        for item in verdict["habit_signals"]:
            habit_id, signal = item.get("habit_id"), item.get("signal_type")
            quote = str(item.get("evidence_quote", "")).strip()
            confidence = item.get("confidence")
            if habit_id not in valid_habit_ids or signal not in {"support", "boundary", "exception", "revision"}:
                return {}, f"verifier_invalid_signal_session_{index}"
            if not isinstance(confidence, (int, float)) or not 0.90 <= float(confidence) <= 1.0:
                return {}, f"verifier_low_or_invalid_confidence_session_{index}"
            if len(quote) < 8 or quote.lower() not in user_text:
                return {}, f"verifier_untraceable_quote_session_{index}"
            actual[habit_id] = {
                "habit_id": habit_id, "signal_type": signal,
                "evidence_quote": quote, "confidence": float(confidence),
            }
        if bool(verdict.get("leakage")):
            return {}, f"unnatural_habit_leakage_session_{index}:{verdict.get('notes')}"
        verified[index] = {
            "habit_signals": list(actual.values()),
            "leakage": False,
            "verifier_notes": verdict.get("notes"),
            "verified_by": {"model": args.model, "reasoning_effort": args.reasoning_effort, "independent_call": True},
        }
    return verified, None


def session_stage(args: argparse, profiles: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_root = args.session_shard_dir or args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    source_cards = {row["conversation_id"]: row for row in read_jsonl(args.dataset / "sources" / "taskmaster_source_cards.jsonl")}
    events_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_user[event["user_id"]].append(event)
    selected_user_ids = {profile["user_id"] for profile in profiles}
    all_sessions = [] if args.session_shard_dir else [
        row for row in read_jsonl(args.dataset / "private" / "sessions_with_annotations.jsonl")
        if row.get("user_id") not in selected_user_ids
    ]
    try:
        qwen_config = json.loads((args.tokenizer / "config.json").read_text(encoding="utf-8"))
        qwen_context = int(qwen_config["max_position_embeddings"])
    except Exception as exc:
        raise SystemExit(f"cannot load required local Qwen config: {exc}")
    history_length_path = output_root / "reports" / "history_length_qwen_tokens.json"
    history_lengths = (
        json.loads(history_length_path.read_text(encoding="utf-8"))
        if history_length_path.exists() else {}
    )
    history_lengths = {
        user_id: row for user_id, row in history_lengths.items()
        if user_id not in selected_user_ids
    }
    start_time = datetime(2024, 1, 8, 9, tzinfo=timezone.utc)
    for profile in profiles:
        user_sessions = []
        continuity_summary = json.dumps({
            "persona": profile.get("persona"), "travel_contexts": profile.get("travel_contexts"),
            "habit_interactions": profile.get("habit_interactions"),
        }, ensure_ascii=False)
        user_events = sorted(events_by_user[profile["user_id"]], key=lambda row: row["session_index"])
        for block_start in range(0, len(user_events), args.session_block_size):
            block = user_events[block_start : block_start + args.session_block_size]
            grounding_ids = list(dict.fromkeys(
                conversation_id
                for event in block
                for conversation_id in event.get("grounding_source_ids", [])
            ))
            grounded = [
                {
                    "conversation_id": source_cards[conversation_id]["conversation_id"],
                    "instruction_id": source_cards[conversation_id]["instruction_id"],
                    "turns": source_cards[conversation_id]["turns"][:24],
                }
                for conversation_id in grounding_ids
                if conversation_id in source_cards
            ]
            previously_established = {
                signal["habit_id"]
                for prior_session in user_sessions
                for signal in prior_session.get("memory_annotations", {}).get("verified_habit_signals", [])
            }
            payload = {
                "task": "Directly write complete multi-turn user-agent sessions for this chronological block.",
                "user_dossier": profile,
                "event_plans": block,
                "rotating_grounded_taskmaster_dialogues": grounded,
                "recent_session_tail": [
                    {"session_index": row["session_index"], "messages": row["messages"][-2:]}
                    for row in user_sessions[-3:]
                ],
                "longitudinal_continuity_summary": continuity_summary,
                "requirements": [
                    "Generate every utterance directly; do not emit a template, outline, placeholders, or text to be rewritten later.",
                    "Return exactly one session per event plan, with its session_index unchanged.",
                    "Honor each event's session_length: micro=4-8 messages/300-1000 total characters; short=8-12/1000-2200; medium=12-20/2000-4000; long=18-28/3500-6500. Begin with user and end with assistant.",
                    "Vary openings, turn counts, user verbosity, corrections, interruptions, and task outcomes as natural conversation demands.",
                    "Ground task realism in the excerpts, but do not copy their wording, destinations, dates, or transaction details.",
                    "Let dossier habits affect choices only when naturally relevant to the event. Do not make the user announce or enumerate habits, and do not force every session to reveal one.",
                    "Do not pack multiple dossier preferences into a session merely to make them observable. Solve the event first; zero, one, or several preference signals may emerge only as the concrete task requires.",
                    "The assistant must not describe a preference as 'your usual preference', 'your default', 'you normally', 'you always', or similar profile-summary language. It may apply prior user choices quietly, offer relevant options, or ask a natural clarification, leaving the user to confirm the choice.",
                    "Treat dossier habits as latent authoring context, not facts the assistant automatically knows. Before user establishment, the assistant must not attribute a dossier-specific carrier, threshold, timing, seat, amenity, or tolerance value to the user or silently use it as a filter. It may neutrally ask about the attribute or present genuinely relevant options with differing factual attributes; let the user originate any preference through a meaningful choice. After user establishment, the assistant may summarize or apply it.",
                    "A durable preference already expressed in an earlier session usually does not need to be restated verbatim. Show continuity through a choice, correction, or tradeoff only when the present decision makes it useful; otherwise leave it implicit.",
                    "Do not target a missing habit, evidence count, history phase, or fixed disclosure schedule. Preference evidence may arise only from the current event's genuine decision needs.",
                    "Do not repeatedly organize replies as 'current hard constraints', hard-versus-soft checklists, or a fixed recommendation template. Vary the conversational organization with the task.",
                    "A one-trip override must remain local unless the user naturally and explicitly makes a durable change.",
                    "Maintain persona continuity without recapping the dossier.",
                    "Update a compact continuity summary with ongoing trips, durable user facts, resolved plans, and unresolved follow-ups; do not restate hidden labels.",
                    "Return {sessions:[{session_index,messages:[{role,content}],evidence_summary}],updated_continuity_summary}",
                ],
            }
            block_cache = output_root / "work" / "session_blocks" / f"{profile['user_id']}_{block_start:04d}.json"
            block_fp = cache_fingerprint(args, "verified_session_block", payload)
            generated, verification, updated_summary, last_error = None, None, None, None
            if block_cache.exists():
                cached_block = json.loads(block_cache.read_text(encoding="utf-8"))
                if cached_block.get("cache_fingerprint") == block_fp:
                    generated, verification = cached_block.get("sessions"), cached_block.get("verification")
                    updated_summary = cached_block.get("updated_continuity_summary")
            for attempt in range(3 if generated is None else 0):
                if last_error:
                    # Never expose a rejected habit/value or coverage deficit
                    # to the next authoring call; that creates targeted repair
                    # language and makes the benchmark answer easier.
                    payload["previous_output_rejection"] = (
                        "A prior attempt failed an independent structural or naturalness check. "
                        "Regenerate from the event needs without adding, removing, repeating, or "
                        "enumerating any preference merely to address this notice."
                    )
                raw = api_json(args, "You directly author natural long-term travel conversations; no templating or paraphrase stage exists.", payload, 22000)
                try:
                    candidate = validate_sessions(raw, profile, block, args.model)
                    candidate_summary = str(raw.get("updated_continuity_summary", "")).strip()
                    if not 100 <= len(candidate_summary) <= 5000:
                        raise ValueError("missing or malformed longitudinal continuity summary")
                    verified, verification_error = independently_verify_sessions(
                        args, profile, block, candidate, previously_established
                    )
                    if verification_error:
                        write_json(
                            output_root / "reports" / "session_block_rejections" /
                            f"{profile['user_id']}_{block_start:04d}_attempt_{attempt + 1}.json",
                            {
                                "reason": verification_error,
                                "sessions": candidate,
                                "updated_continuity_summary": candidate_summary,
                            },
                        )
                        raise ValueError(verification_error)
                    for row in candidate:
                        row["memory_annotations"]["verified_habit_signals"] = verified[row["session_index"]]["habit_signals"]
                        row["memory_annotations"]["independent_verification"] = verified[row["session_index"]]["verified_by"]
                    generated, verification, updated_summary = candidate, verified, candidate_summary
                    break
                except ValueError as exc:
                    last_error = str(exc)
            if generated is None:
                raise ValueError(f"session block failed independent verification after 3 fresh generations: {last_error}")
            user_sessions.extend(generated)
            continuity_summary = updated_summary
            write_json(block_cache, {"cache_fingerprint": block_fp, "sessions": generated, "verification": verification, "updated_continuity_summary": updated_summary})
            print(f"session_progress {profile['user_id']} {len(user_sessions)}/{len(user_events)}", flush=True)
        event_by_index = {event["session_index"]: event for event in user_events}
        for row in user_sessions:
            row["memory_annotations"]["episode_id"] = event_by_index[row["session_index"]]["episode_id"]
        verified_counts = Counter(
            (signal["habit_id"], signal["signal_type"])
            for row in user_sessions
            for signal in row["memory_annotations"].get("verified_habit_signals", [])
        )
        verified_episode_counts: dict[str, int] = {}
        support_only_habits: list[str] = []
        missing_evidence = []
        for habit in profile["habits"]:
            if not habit.get("testable", True):
                continue
            habit_id = habit["habit_id"]
            evidence_rows = [
                row for row in user_sessions
                if any(signal["habit_id"] == habit_id for signal in row["memory_annotations"].get("verified_habit_signals", []))
            ]
            phases = {
                "early" if row["session_index"] < len(user_events) / 3
                else "middle" if row["session_index"] < 2 * len(user_events) / 3
                else "late"
                for row in evidence_rows
            }
            evidence_indices = sorted(row["session_index"] for row in evidence_rows)
            temporal_span = evidence_indices[-1] - evidence_indices[0] if evidence_indices else 0
            evidence_episodes = {
                row["memory_annotations"]["episode_id"] for row in evidence_rows
            }
            support_episodes = {
                row["memory_annotations"]["episode_id"]
                for row in evidence_rows
                if any(
                    signal["habit_id"] == habit_id and signal["signal_type"] == "support"
                    for signal in row["memory_annotations"].get("verified_habit_signals", [])
                )
            }
            verified_episode_counts[habit_id] = len(evidence_episodes)
            if len(evidence_episodes) < 3:
                missing_evidence.append((habit_id, f"verified_episodes={len(evidence_episodes)}"))
            if len(phases) < 2:
                missing_evidence.append((habit_id, f"verified_phases<2:{sorted(phases)}"))
            if temporal_span < len(user_events) / 3:
                missing_evidence.append((habit_id, f"verified_temporal_span={temporal_span}"))
            if len(support_episodes) < 2:
                missing_evidence.append((habit_id, f"verified_support_episodes={len(support_episodes)}"))
            if not (verified_counts[(habit_id, "boundary")] or verified_counts[(habit_id, "exception")]):
                # Do not manufacture a trip-specific override merely to make
                # every hidden policy graph look structurally identical.  A
                # naturally support-only habit remains testable, but probe
                # generation must not ask an unsupported boundary/exception.
                support_only_habits.append(habit_id)
        write_json(output_root / "reports" / f"{profile['user_id']}_verified_history_evidence.json", {
            "user_id": profile["user_id"],
            "status": "pass" if not missing_evidence else "reject",
            "pre_dialogue_signal_labels_used": False,
            "missing": missing_evidence,
            "verified_signal_counts": {f"{habit_id}|{signal}": count for (habit_id, signal), count in verified_counts.items()},
            "verified_episode_counts": verified_episode_counts,
            "support_only_habits": support_only_habits,
        })
        if missing_evidence:
            raise ValueError(f"final dialogue history failed post-hoc evidence gate for {profile['user_id']}: {missing_evidence}")
        current_time = start_time + timedelta(days=int(profile["user_id"][-3:]) * 17)
        for row in sorted(user_sessions, key=lambda item: item["session_index"]):
            gap = event_by_index[row["session_index"]]["days_after_previous"]
            current_time += timedelta(days=gap, hours=4 if gap == 0 else 0)
            row["timestamp"] = current_time.isoformat()
        total_chars = sum(len(message["content"]) for row in user_sessions for message in row["messages"])
        target_characters = int(profile["longitudinal_plan"]["target_history_characters"])
        if total_chars < target_characters:
            raise ValueError(
                f"{profile['user_id']} history is only {total_chars} characters, below dossier target {target_characters}"
            )
        total_tokens = exact_qwen_token_count(args, ["\n".join(message["content"] for message in row["messages"]) for row in user_sessions])
        minimum_long_history_tokens = int(1.25 * qwen_context)
        if total_tokens < minimum_long_history_tokens:
            raise ValueError(f"{profile['user_id']} history is {total_tokens} Qwen tokens, below 1.25x context ({minimum_long_history_tokens})")
        history_lengths[profile["user_id"]] = {"characters": total_chars, "qwen_tokens": total_tokens, "qwen_context": qwen_context}
        all_sessions.extend(user_sessions)
    write_jsonl(output_root / "private" / "sessions_with_annotations.jsonl", all_sessions)
    public = [{k: row[k] for k in ["user_id", "session_id", "session_index", "timestamp", "domain", "messages"]} for row in all_sessions]
    write_jsonl(output_root / "public" / "lifelines.jsonl", public)
    write_json(output_root / "reports" / "history_length_qwen_tokens.json", history_lengths)
    return all_sessions


def recovery_stage(
    args: argparse.Namespace,
    profiles: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
) -> None:
    """Recover stable habits from public history without exposing hidden labels."""
    sessions_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        sessions_by_user[row["user_id"]].append(row)
    all_recovered, reports = [], []
    for profile in profiles:
        user_id = profile["user_id"]
        user_sessions = sorted(sessions_by_user[user_id], key=lambda row: row["session_index"])
        public_history = [
            {
                "session_id": row["session_id"],
                "session_index": row["session_index"],
                "timestamp": row["timestamp"],
                "messages": row["messages"],
            }
            for row in user_sessions
        ]
        recovery_payload = {
            "task": "Recover this user's durable travel-planning habits from the complete chronological user-agent history.",
            "user_id": user_id,
            "complete_public_history": public_history,
            "requirements": [
                "You are not given a dossier, habit inventory, signal labels, source graph, or target count. Infer only recurring user-specific policies supported by user utterances.",
                "A habit must describe a concrete preference value or decision policy, its recurring scope, default action, and any boundary, exception, or revision that the history actually supports.",
                "Do not promote a one-trip constraint, destination fact, family fact, or assistant suggestion into a durable habit.",
                "Do not split one default and its tolerance/fallback into separate habits. Do not merge semantically different policies merely because both concern flights or hotels.",
                "For every recovered habit cite 3 to 6 exact contiguous user quotes from distinct sessions and recurring situations. Quotes must be copied exactly and paired with session_id.",
                "Return only stable habits with confidence >=0.75. The number of habits must emerge from the history, not a presumed benchmark structure.",
                "Return {recovered_habits:[{name,family,preference_value,scope_condition,default_action,boundary_condition,exception_condition,evidence:[{session_id,quote}],confidence}]}",
            ],
        }
        cache_path = args.dataset / "work" / "habit_recovery" / f"{user_id}.json"
        recovery_fp = cache_fingerprint(args, "habit_blind_full_history_recovery", recovery_payload)
        raw = None
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_fingerprint") == recovery_fp:
                raw = cached.get("raw_recovery")
        if raw is None:
            raw = api_json(
                args,
                "You independently recover durable user habits from dialogue history only; hidden benchmark state is unavailable.",
                recovery_payload,
                18000,
            )
            write_json(cache_path, {"cache_fingerprint": recovery_fp, "raw_recovery": raw})
        rows = raw.get("recovered_habits") if isinstance(raw, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"habit-blind recovery returned no habits for {user_id}")
        sessions_by_id = {row["session_id"]: row for row in user_sessions}
        # Recovery is a diagnostic induction task.  One hallucinated citation
        # must not abort audits for the remaining habits or users, but it must
        # never enter the adjudicated set.  Pre-filter every candidate using
        # the same exact-quote and cross-episode contract applied below.
        recovery_item_rejections = []
        traceable_rows = []
        for index, item in enumerate(rows):
            reasons = []
            required = ["name", "family", "preference_value", "scope_condition", "default_action"]
            if not isinstance(item, dict) or any(not str(item.get(key, "")).strip() for key in required):
                reasons.append("malformed_required_fields")
            confidence = item.get("confidence") if isinstance(item, dict) else None
            if not isinstance(confidence, (int, float)) or not 0.75 <= float(confidence) <= 1.0:
                reasons.append("invalid_confidence")
            evidence = item.get("evidence") if isinstance(item, dict) else None
            if not isinstance(evidence, list) or not 3 <= len(evidence) <= 6:
                reasons.append("invalid_evidence_count")
                evidence = []
            cited_sessions, cited_episodes = set(), set()
            for citation in evidence:
                session_id = citation.get("session_id") if isinstance(citation, dict) else None
                quote = str(citation.get("quote", "")).strip() if isinstance(citation, dict) else ""
                source = sessions_by_id.get(session_id)
                if source is None or len(quote) < 8:
                    reasons.append(f"invalid_citation:{session_id}")
                    continue
                user_text = "\n".join(
                    message["content"] for message in source["messages"] if message["role"] == "user"
                )
                if quote not in user_text:
                    reasons.append(f"untraceable_quote:{session_id}")
                    continue
                cited_sessions.add(session_id)
                cited_episodes.add(source["memory_annotations"]["episode_id"])
            if len(cited_sessions) < 3:
                reasons.append(f"distinct_sessions_lt_3:{len(cited_sessions)}")
            if len(cited_episodes) < 3:
                reasons.append(f"distinct_episodes_lt_3:{len(cited_episodes)}")
            if reasons:
                recovery_item_rejections.append({
                    "candidate_index": index,
                    "name": item.get("name") if isinstance(item, dict) else None,
                    "reasons": list(dict.fromkeys(reasons)),
                })
            else:
                traceable_rows.append(item)
        rows = traceable_rows
        if not rows:
            raise ValueError(f"habit-blind recovery returned no traceable habits for {user_id}")
        recovered = []
        for index, item in enumerate(rows):
            required = ["name", "family", "preference_value", "scope_condition", "default_action"]
            if not isinstance(item, dict) or any(not str(item.get(key, "")).strip() for key in required):
                raise ValueError(f"malformed recovered habit {user_id}:{index}")
            confidence = item.get("confidence")
            evidence = item.get("evidence")
            if not isinstance(confidence, (int, float)) or not 0.75 <= float(confidence) <= 1.0:
                raise ValueError(f"invalid recovered confidence {user_id}:{index}")
            if not isinstance(evidence, list) or not 3 <= len(evidence) <= 6:
                raise ValueError(f"invalid recovered evidence count {user_id}:{index}")
            cited_sessions, cited_episodes, clean_evidence = set(), set(), []
            for citation in evidence:
                session_id = citation.get("session_id") if isinstance(citation, dict) else None
                quote = str(citation.get("quote", "")).strip() if isinstance(citation, dict) else ""
                source = sessions_by_id.get(session_id)
                if source is None or len(quote) < 8:
                    raise ValueError(f"invalid recovery citation {user_id}:{index}")
                user_text = "\n".join(m["content"] for m in source["messages"] if m["role"] == "user")
                if quote not in user_text:
                    raise ValueError(f"untraceable recovery quote {user_id}:{index}:{session_id}")
                cited_sessions.add(session_id)
                cited_episodes.add(source["memory_annotations"]["episode_id"])
                clean_evidence.append({"session_id": session_id, "quote": quote})
            if len(cited_sessions) < 3 or len(cited_episodes) < 3:
                raise ValueError(f"recovered habit lacks three independent episodes {user_id}:{index}")
            recovered.append({
                "recovered_habit_id": f"{user_id}_recovered_h{index:02d}",
                **{key: str(item.get(key, "")).strip() for key in required},
                "boundary_condition": str(item.get("boundary_condition", "")).strip(),
                "exception_condition": str(item.get("exception_condition", "")).strip(),
                "evidence": clean_evidence,
                "evidence_episode_count": len(cited_episodes),
                "confidence": float(confidence),
                "recovery_provenance": {
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "habit_graph_visible": False,
                    "annotations_visible": False,
                    "complete_history_visible": True,
                },
            })
        hidden = [
            {
                key: habit.get(key)
                for key in [
                    "habit_id", "family", "preference_value", "scope_condition", "condition",
                    "default_action", "boundary_condition", "exception_condition", "testable", "strength",
                ]
            }
            for habit in profile["habits"]
        ]
        match_payload = {
            "task": "Semantically adjudicate a blind habit recovery against the grounded hidden graph.",
            "hidden_grounded_habits": hidden,
            "blind_recovered_habits": recovered,
            "requirements": [
                "Match semantic policies, not wording. Family alone is never sufficient: concrete value, scope, and default action must agree.",
                "A hidden habit may match one recovered habit or a small set only when the recovery separated scope from boundary; flag improper default/fallback splitting.",
                "For every hidden habit return core_match and scope_match. boundary_match and exception_match may be not_observable when the cited history contains no such evidence.",
                "Adjudicate every recovered habit as matched_hidden, unsupported_extra, or not_stable. Background hidden habits are valid matches but are not required to be recovered.",
                "Return {hidden_matches:[{habit_id,recovered_habit_ids,core_match,scope_match,boundary_match,exception_match,confidence,reason}],recovered_adjudications:[{recovered_habit_id,status,matched_habit_ids,reason}],improper_splits:[{recovered_habit_ids,reason}]}",
            ],
        }
        match_cache_path = args.dataset / "work" / "habit_recovery" / f"{user_id}_match.json"
        match_fp = cache_fingerprint(args, "independent_recovery_graph_match", match_payload)
        match_raw = None
        if match_cache_path.exists():
            cached_match = json.loads(match_cache_path.read_text(encoding="utf-8"))
            if cached_match.get("cache_fingerprint") == match_fp:
                match_raw = cached_match.get("raw_match")
        if match_raw is None:
            match_raw = api_json(
                args,
                "You audit semantic recovery after the blind extraction is complete; do not excuse value or scope mismatches.",
                match_payload,
                12000,
            )
            write_json(match_cache_path, {"cache_fingerprint": match_fp, "raw_match": match_raw})
        hidden_matches = match_raw.get("hidden_matches") if isinstance(match_raw, dict) else None
        adjudications = match_raw.get("recovered_adjudications") if isinstance(match_raw, dict) else None
        improper_splits = match_raw.get("improper_splits") if isinstance(match_raw, dict) else None
        if not isinstance(hidden_matches, list) or not isinstance(adjudications, list) or not isinstance(improper_splits, list):
            raise ValueError(f"invalid recovery adjudication for {user_id}")
        match_by_id = {row.get("habit_id"): row for row in hidden_matches if isinstance(row, dict)}
        missing, weak, unsupported = [], [], []

        def is_full_semantic_match(value: Any) -> bool:
            if value is True:
                return True
            return str(value).strip().lower() in {"full", "match", "matched", "true"}

        for habit in profile["habits"]:
            if not habit.get("testable", True):
                continue
            verdict = match_by_id.get(habit["habit_id"])
            if verdict is None:
                missing.append(habit["habit_id"])
                continue
            confidence = verdict.get("confidence")
            if (
                not is_full_semantic_match(verdict.get("core_match"))
                or not is_full_semantic_match(verdict.get("scope_match"))
                or not isinstance(confidence, (int, float))
                or float(confidence) < 0.80
            ):
                weak.append({"habit_id": habit["habit_id"], "verdict": verdict})
        recovered_confidence = {row["recovered_habit_id"]: row["confidence"] for row in recovered}
        for verdict in adjudications:
            if not isinstance(verdict, dict):
                raise ValueError(f"malformed recovered-habit adjudication for {user_id}")
            rid = verdict.get("recovered_habit_id")
            if verdict.get("status") == "unsupported_extra" and recovered_confidence.get(rid, 0) >= 0.85:
                unsupported.append(verdict)
        status = "pass" if not (missing or weak or unsupported or improper_splits) else "reject"
        report = {
            "user_id": user_id,
            "status": status,
            "blind_recovered_count": len(recovered),
            "hidden_testable_count": sum(bool(h.get("testable", True)) for h in profile["habits"]),
            "missing_testable_habits": missing,
            "weak_testable_matches": weak,
            "unsupported_high_confidence_recoveries": unsupported,
            "improper_splits": improper_splits,
            "recovery_item_rejections": recovery_item_rejections,
            "hidden_matches": hidden_matches,
            "recovered_adjudications": adjudications,
            "recovery_model": args.model,
            "reasoning_effort": args.reasoning_effort,
        }
        write_json(args.dataset / "reports" / f"{user_id}_habit_blind_recovery.json", report)
        all_recovered.extend({"user_id": user_id, **row} for row in recovered)
        reports.append(report)
        # A rejected user must not prevent the remaining completed lifelines
        # from receiving the same blind audit.  Preserve every per-user report
        # and fail the stage only after the corpus-level summary is written.
        print(
            f"recovery_progress {user_id} recovered={len(recovered)} status={status}",
            flush=True,
        )
    write_jsonl(args.dataset / "private" / "blind_recovered_habits.jsonl", all_recovered)
    rejected_users = [row["user_id"] for row in reports if row.get("status") != "pass"]
    summary_status = "pass" if not rejected_users else "reject"
    write_json(args.dataset / "reports" / "habit_blind_recovery_summary.json", {
        "status": summary_status,
        "role": "diagnostic_not_probe_admission_gate",
        "rejected_users": rejected_users,
        "users": reports,
    })


def probe_job(args: argparse.Namespace, profile: dict[str, Any], habit: dict[str, Any], sessions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence, valid_session_ids, session_to_episode = [], set(), {}
    for row in sessions:
        annotations = row.get("memory_annotations", {})
        verified_signals = {item["habit_id"]: item for item in annotations.get("verified_habit_signals", [])}
        if habit["habit_id"] not in verified_signals:
            continue
        valid_session_ids.add(row["session_id"])
        episode_id = annotations.get("episode_id")
        session_to_episode[row["session_id"]] = episode_id
        evidence.append({
            "session_id": row["session_id"], "session_index": row["session_index"],
            "episode_id": episode_id,
            "signal_type": verified_signals[habit["habit_id"]]["signal_type"],
            "evidence_quote": verified_signals[habit["habit_id"]]["evidence_quote"],
            "evidence_summary": annotations.get("evidence_summary"),
            "user_turn_samples": [m["content"] for m in row["messages"] if m["role"] == "user"][-2:],
        })
    phases = [
        any(e["session_index"] < profile["longitudinal_plan"]["target_sessions"] / 3 for e in evidence),
        any(profile["longitudinal_plan"]["target_sessions"] / 3 <= e["session_index"] < 2 * profile["longitudinal_plan"]["target_sessions"] / 3 for e in evidence),
        any(e["session_index"] >= 2 * profile["longitudinal_plan"]["target_sessions"] / 3 for e in evidence),
    ]
    evidence_episodes = {e["episode_id"] for e in evidence if e.get("episode_id")}
    evidence_signal_types = {e["signal_type"] for e in evidence}
    evidence_indices = sorted(e["session_index"] for e in evidence)
    temporal_span = evidence_indices[-1] - evidence_indices[0] if evidence_indices else 0
    if len(evidence_episodes) < 3 or sum(phases) < 2 or temporal_span < profile["longitudinal_plan"]["target_sessions"] / 3:
        raise ValueError(
            f"insufficient long-range episode evidence for {habit['habit_id']}: "
            f"episodes={len(evidence_episodes)} phases={phases} span={temporal_span}"
        )
    probe_cache = args.dataset / "work" / "probe_sets" / f"{habit['habit_id']}.json"
    probe_fp = cache_fingerprint(
        args,
        "contrastively_judged_memory_necessary_probe_set",
        {"profile": profile, "habit": habit, "evidence": evidence},
    )
    if probe_cache.exists():
        cached_probe = json.loads(probe_cache.read_text(encoding="utf-8"))
        cached_private = cached_probe.get("private", [])
        difficulty_contract_ok = bool(cached_private) and all(
            row.get("generator_closest_distractor_choice_id")
            and row.get("query_only_judge", {}).get("choice_id") == "UNRESOLVED"
            and row.get("query_only_judge", {}).get("answerable_without_history") is False
            and row.get("query_only_judge", {}).get("generic_best_exists") is False
            and len(row.get("query_only_judge", {}).get("plausible_choice_ids", [])) >= 2
            and row.get("independent_gold_judge", {}).get("difficulty") in {"medium", "hard"}
            and row.get("independent_gold_judge", {}).get("closest_distractor_choice_id")
            for row in cached_private
        )
        if cached_probe.get("cache_fingerprint") == probe_fp and difficulty_contract_ok:
            return cached_probe["public"], cached_probe["private"]
    max_probe_count = min(8, max(4, len(evidence_episodes) + (1 if habit.get("strength") == "primary" else 0)))
    min_probe_count = min(4, max_probe_count)
    candidate_count = max_probe_count + 2
    available_probe_types = {"direct_use", "cross_context_transfer", "evidence_disambiguation"}
    if "boundary" in evidence_signal_types:
        available_probe_types.update({"boundary", "conflict_resolution"})
    if "exception" in evidence_signal_types:
        available_probe_types.update({"exception", "conflict_resolution"})
    if "revision" in evidence_signal_types:
        available_probe_types.add("post_revision_current_default")
    raw = api_json(args, "You write and label rigorous long-history travel-habit benchmark probes.", {
        "task": f"Write exactly {candidate_count} difficult multiple-choice candidate probes for one habit; an independent judge will retain between {min_probe_count} and {max_probe_count} based on quality.",
        "user_id": profile["user_id"], "hidden_habit": habit, "evidence_sessions": evidence,
        "available_probe_types": sorted(available_probe_types),
        "requirements": [
            "Choose a varied evidence-appropriate subset of types; do not mechanically emit a fixed type matrix for every habit.",
            f"Use direct_use for at most {max(2, candidate_count // 4)} candidates. Most candidates must require scope transfer, evidence disambiguation, a supported boundary/exception, or conflict resolution.",
            "Each query must describe a concrete new travel decision, not repeat an evidence transaction and not simply ask what the user generally prefers.",
            "Every probe must require at least two reasoning steps, such as identifying the applicable scope and then resolving a price/time/location tradeoff, or distinguishing a durable default from a supported exception.",
            "Memory necessity is a hard requirement: a capable travel planner who sees only the query and choices must be unable to select a unique best option, while a planner who also sees the user's history must be able to do so.",
            "Do not state the target habit, its threshold, preferred value, default action, or applicable exception directly in the query. The query may state genuine trip constraints, but those constraints alone must leave at least two options comparably defensible.",
            "Construct each item in this order: first make the gold and closest distractor a pair with comparable generic travel utility but different personal-policy implications; then add two other plausible actions; finally write a query whose explicit constraints do not break the core pair's tie.",
            "All four choices must be plausible comparable actions. Balance price, timing, routing, location, quality, and convenience so the gold does not dominate on multiple generic dimensions and no option is obviously best or worst without user history.",
            "The gold and closest distractor must form a genuine tradeoff pair: generic travel reasoning cannot break the tie, but the user's longitudinal policy can. Keep specificity, tone, and length comparable and avoid lexical, answer-length, moral, or safety giveaways.",
            "Include one strong near-miss distractor that follows part of the user's policy but fails on the current scope, boundary, exception, or higher-priority constraint. Other distractors must also be realistic travel choices.",
            "Do not copy an evidence sentence, reuse its destination/date/transaction, or make the gold option paraphrase the habit more explicitly than distractors.",
            "Cite at least three supplied evidence session_ids from three different episode_ids; never cite future or unrelated evidence.",
            "Distinguish one-trip exceptions from permanent revisions.",
            "Return {probes:[{probe_type,query,choices:[{choice_id,text}],gold_choice_id,gold_action,gold_evidence_session_ids,label_rationale,closest_distractor_choice_id,difficulty_rationale}]}",
        ],
    }, 20000)
    rows = raw.get("probes") if isinstance(raw, dict) else None
    if not isinstance(rows, list) or len(rows) != candidate_count:
        raise ValueError(f"invalid probe set for {habit['habit_id']}")

    def candidate_error(row: Any) -> str | None:
        if not isinstance(row, dict) or row.get("probe_type") not in available_probe_types:
            return "invalid_probe_type"
        choices = row.get("choices", [])
        choice_ids = [choice.get("choice_id") for choice in choices]
        cited = list(dict.fromkeys(row.get("gold_evidence_session_ids", [])))
        if len(choices) != 4 or set(choice_ids) != {"A", "B", "C", "D"} or row.get("gold_choice_id") not in choice_ids:
            return "invalid_choices"
        closest_distractor = row.get("closest_distractor_choice_id")
        if closest_distractor not in choice_ids or closest_distractor == row.get("gold_choice_id"):
            return "invalid_closest_distractor"
        choice_lengths = [len(str(choice.get("text", "")).strip()) for choice in choices]
        if min(choice_lengths) < 8 or max(choice_lengths) > 2.5 * min(choice_lengths):
            return "choice_length_giveaway"
        if len(str(row.get("query", "")).strip()) < 30:
            return "query_too_short"
        if len(str(row.get("difficulty_rationale", "")).strip()) < 30:
            return "missing_difficulty_rationale"
        if len(cited) < 3 or any(sid not in valid_session_ids for sid in cited):
            return "invalid_evidence_citations"
        if len({session_to_episode[sid] for sid in cited}) < 3:
            return "evidence_not_three_episodes"
        return None

    candidate_rows = {f"c{index:02d}": row for index, row in enumerate(rows)}
    query_only_passed: dict[str, dict[str, Any]] = {}
    diagnostics_path = args.dataset / "work" / "probe_diagnostics" / f"{habit['habit_id']}.json"
    diagnostic_rounds: list[dict[str, Any]] = []
    target_before_history_judge = min(candidate_count, min_probe_count + 2)
    for refinement_round in range(3):
        audit_rows, local_errors = [], {}
        for candidate_id, row in candidate_rows.items():
            if candidate_id in query_only_passed:
                continue
            error = candidate_error(row)
            if error:
                local_errors[candidate_id] = error
                continue
            audit_rows.append({
                "probe_id": candidate_id,
                "query": str(row["query"]).strip(),
                "choices": row["choices"],
            })
        query_only_judged = {}
        if audit_rows:
            query_only = api_json(args, "You audit whether a multiple-choice travel question genuinely requires private user history. You have no user history, habit graph, annotations, or proposed gold label.", {
                "task": "Assess every probe using only its query and choices. Do not guess a personal preference that is not stated in the query.",
                "probes": audit_rows,
                "requirements": [
                    "Return {answers:[{probe_id,choice_id,answerable_without_history,generic_best_exists,plausible_choice_ids,rationale,leakage_signals}]}",
                    "Set choice_id=UNRESOLVED when two or more choices remain reasonably defensible without user history; never guess an unstated personal preference.",
                    "Set answerable_without_history=true and return A/B/C/D whenever explicit query constraints, ordinary dominance, common-sense travel planning, or wording cues identify one best choice.",
                    "Set generic_best_exists=true if one choice is materially better on the stated objective or dominates on price, time, routing, location, quality, convenience, safety, or feasibility.",
                    "List every still-plausible choice_id. leakage_signals are diagnostic: report any suspicious cue, and set answerable_without_history=true whenever that cue is sufficient to select an answer.",
                ],
            }, 10000)
            query_only_judged = {
                row.get("probe_id"): row for row in query_only.get("answers", []) if isinstance(row, dict)
            }
        failed_feedback = []
        round_accepted = []
        for candidate_id, row in candidate_rows.items():
            if candidate_id in query_only_passed:
                continue
            if candidate_id in local_errors:
                failed_feedback.append({
                    "candidate_id": candidate_id,
                    "failure": local_errors[candidate_id],
                    "candidate": row,
                })
                continue
            verdict = query_only_judged.get(candidate_id)
            plausible_raw = verdict.get("plausible_choice_ids", []) if verdict else []
            plausible = list(dict.fromkeys(plausible_raw)) if isinstance(plausible_raw, list) else []
            passed = bool(
                verdict
                and verdict.get("choice_id") == "UNRESOLVED"
                and verdict.get("answerable_without_history") is False
                and verdict.get("generic_best_exists") is False
                and len(plausible) >= 2
                and all(choice_id in {"A", "B", "C", "D"} for choice_id in plausible)
            )
            if passed:
                leakage_raw = verdict.get("leakage_signals", [])
                leakage = leakage_raw if isinstance(leakage_raw, list) else ([str(leakage_raw)] if leakage_raw else [])
                query_only_passed[candidate_id] = {
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "choice_id": "UNRESOLVED",
                    "answerable_without_history": False,
                    "generic_best_exists": False,
                    "plausible_choice_ids": plausible,
                    "rationale": verdict.get("rationale"),
                    "leakage_signals": leakage,
                }
                round_accepted.append(candidate_id)
            else:
                failed_feedback.append({
                    "candidate_id": candidate_id,
                    "failure": "answerable_without_history_or_invalid_audit",
                    "query_only_verdict": verdict,
                    "candidate": row,
                })
        diagnostic_rounds.append({
            "round": refinement_round,
            "accepted_this_round": round_accepted,
            "accepted_total": sorted(query_only_passed),
            "failed": failed_feedback,
        })
        write_json(diagnostics_path, {
            "user_id": profile["user_id"],
            "habit_id": habit["habit_id"],
            "status": "in_progress",
            "rounds": diagnostic_rounds,
        })
        if len(query_only_passed) >= target_before_history_judge or refinement_round == 2:
            break
        rewrite = api_json(args, "You repair benchmark candidates so that personal longitudinal memory is necessary, while preserving a valid hidden-history gold label.", {
            "task": "Rewrite every failed candidate. Return one replacement for each candidate_id and do not alter candidates that already passed.",
            "hidden_habit": habit,
            "evidence_sessions": evidence,
            "available_probe_types": sorted(available_probe_types),
            "failed_candidates_with_audit": failed_feedback,
            "requirements": [
                "Return {rewritten_candidates:[{candidate_id,probe_type,query,choices:[{choice_id,text}],gold_choice_id,gold_action,gold_evidence_session_ids,label_rationale,closest_distractor_choice_id,difficulty_rationale}]}",
                "Preserve each supplied candidate_id exactly and return every failed candidate_id once.",
                "First construct a gold/near-miss pair with comparable generic utility and a real tradeoff; only the user's hidden longitudinal policy may break their tie.",
                "Remove explicit target preferences, thresholds, defaults, and exception decisions from the query. Stated trip constraints alone must leave at least two choices defensible.",
                "Do not let one option dominate on price, schedule, route, location, quality, convenience, feasibility, or number of stated benefits.",
                "Use four similarly detailed realistic actions, cite at least three supplied sessions from three episodes, and keep the intended gold consistent with the evidence.",
            ],
        }, 20000)
        rewritten = rewrite.get("rewritten_candidates", []) if isinstance(rewrite, dict) else []
        rewritten_by_id = {
            row.get("candidate_id"): {k: v for k, v in row.items() if k != "candidate_id"}
            for row in rewritten if isinstance(row, dict)
        }
        expected_ids = {row["candidate_id"] for row in failed_feedback}
        if set(rewritten_by_id) != expected_ids:
            raise ValueError(
                f"candidate repair returned wrong ids for {habit['habit_id']}: "
                f"expected={sorted(expected_ids)} got={sorted(rewritten_by_id)}"
            )
        candidate_rows.update(rewritten_by_id)

    if len(query_only_passed) < min_probe_count:
        write_json(diagnostics_path, {
            "user_id": profile["user_id"], "habit_id": habit["habit_id"],
            "status": "reject", "rounds": diagnostic_rounds,
            "reason": f"only {len(query_only_passed)}/{candidate_count} candidates require history",
        })
        raise ValueError(
            f"only {len(query_only_passed)}/{candidate_count} candidates require history after repair for "
            f"{habit['habit_id']}"
        )

    public, private = [], []
    for index, candidate_id in enumerate(sorted(query_only_passed)):
        row = candidate_rows[candidate_id]
        choices = row["choices"]
        cited = list(dict.fromkeys(row.get("gold_evidence_session_ids", [])))
        probe_id = f"{habit['habit_id']}_p{index:02d}"
        public.append({
            "probe_id": probe_id, "user_id": profile["user_id"], "split": "test",
            "query": str(row["query"]).strip(), "choices": choices,
            "visible_history_scope": {"max_session_index": profile["longitudinal_plan"]["target_sessions"] - 1},
            "metadata": {
                "dataset_version": "taskmaster_planning_defaults_v0_4",
                "probe_type": row["probe_type"],
                "generated_by": args.model,
                "blind_recovery_role": "diagnostic_only",
            },
            "evaluation_contract": {"answer_format": "return one choice_id", "validator_type": "choice_equals"},
        })
        private.append({
            "probe_id": probe_id, "user_id": profile["user_id"], "habit_id": habit["habit_id"],
            "habit_family": habit["family"], "probe_type": row["probe_type"],
            "gold_choice_id": row["gold_choice_id"], "gold_action": row.get("gold_action"),
            "gold_evidence_session_ids": cited, "label_rationale": row.get("label_rationale"),
            "generator_closest_distractor_choice_id": row["closest_distractor_choice_id"],
            "generator_difficulty_rationale": row.get("difficulty_rationale"),
            "hidden_habit_graph": habit, "label_source": "llm_generation_from_hidden_habit_and_verified_evidence",
            "query_only_judge": query_only_passed[candidate_id],
        })
    judge_evidence = [
        {
            key: row[key]
            for key in ["session_id", "session_index", "episode_id", "evidence_quote", "user_turn_samples"]
        }
        for row in evidence
    ]
    judge = api_json(args, "You independently solve benchmark questions from user history evidence; hidden habits and proposed gold labels are unavailable.", {
        "task": "Independently choose the best option and supporting evidence for every probe using only the supplied user-history excerpts.",
        "user_history_evidence": judge_evidence,
        "probes": [
            {"probe_id": row["probe_id"], "query": row["query"], "choices": row["choices"]}
            for row in public
        ],
        "requirements": [
            "Return {answers:[{probe_id,choice_id,evidence_session_ids,rationale,difficulty,ambiguous,closest_distractor_choice_id,closest_distractor_rationale}]}",
            "Infer the user's applicable policy from the quoted history only; do not infer from answer position or wording length.",
            "If the excerpts do not uniquely support one option, return choice_id=UNRESOLVED rather than guessing.",
            "Rate difficulty as easy, medium, or hard for a capable model that has the supplied excerpts. Mark ambiguous=true if two options remain comparably defensible.",
            "Identify the strongest wrong option and explain why it is tempting but loses under the applicable scope, tradeoff, boundary, or exception.",
        ],
    }, 14000)
    judged = {row.get("probe_id"): row for row in judge.get("answers", []) if isinstance(row, dict)}
    private_by_candidate_id = {row["probe_id"]: row for row in private}
    query_only_accepted_ids = [row["probe_id"] for row in private]
    accepted_ids, rejection_reasons = [], Counter()
    for probe_id in query_only_accepted_ids:
        key = private_by_candidate_id[probe_id]
        verdict = judged.get(key["probe_id"])
        if not verdict or verdict.get("choice_id") != key["gold_choice_id"]:
            rejection_reasons["gold_disagreement_or_unresolved"] += 1
            continue
        if verdict.get("difficulty") not in {"medium", "hard"} or verdict.get("ambiguous") is not False:
            rejection_reasons["easy_or_ambiguous"] += 1
            continue
        judge_closest = verdict.get("closest_distractor_choice_id")
        if judge_closest not in {"A", "B", "C", "D"} or judge_closest == verdict.get("choice_id"):
            rejection_reasons["invalid_closest_distractor"] += 1
            continue
        if len(str(verdict.get("closest_distractor_rationale", "")).strip()) < 20:
            rejection_reasons["missing_distractor_rationale"] += 1
            continue
        judge_citations = set(verdict.get("evidence_session_ids", []))
        if not judge_citations or not judge_citations.issubset(valid_session_ids):
            rejection_reasons["invalid_judge_evidence"] += 1
            continue
        key["independent_gold_judge"] = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "rationale": verdict.get("rationale"),
            "difficulty": verdict.get("difficulty"),
            "closest_distractor_choice_id": judge_closest,
            "closest_distractor_rationale": verdict.get("closest_distractor_rationale"),
        }
        accepted_ids.append(key["probe_id"])
    if len(accepted_ids) < min_probe_count:
        raise ValueError(
            f"only {len(accepted_ids)}/{candidate_count} hard unambiguous candidates passed for "
            f"{habit['habit_id']}: {dict(rejection_reasons)}"
        )
    public_by_id = {row["probe_id"]: row for row in public}
    private_by_id = {row["probe_id"]: row for row in private}
    best_combo, best_score = None, None
    for selected_count in range(min(max_probe_count, len(accepted_ids)), min_probe_count - 1, -1):
        for combo in itertools.combinations(accepted_ids, selected_count):
            combo_types = {public_by_id[probe_id]["metadata"]["probe_type"] for probe_id in combo}
            direct_count = sum(
                public_by_id[probe_id]["metadata"]["probe_type"] == "direct_use" for probe_id in combo
            )
            if len(combo_types) < min(3, len(available_probe_types)) or direct_count > max(1, selected_count // 4):
                continue
            hard_count = sum(
                private_by_id[probe_id]["independent_gold_judge"]["difficulty"] == "hard"
                for probe_id in combo
            )
            score = (selected_count, len(combo_types), hard_count)
            if best_score is None or score > best_score:
                best_combo, best_score = combo, score
        if best_combo is not None:
            break
    if best_combo is None:
        raise ValueError(
            f"{len(accepted_ids)} accepted candidates cannot form a diverse {min_probe_count}-{max_probe_count} set for {habit['habit_id']}"
        )
    selected_public, selected_private = [], []
    for index, old_probe_id in enumerate(best_combo):
        public_row, private_row = public_by_id[old_probe_id], private_by_id[old_probe_id]
        new_probe_id = f"{habit['habit_id']}_p{index:02d}"
        public_row["probe_id"] = new_probe_id
        private_row["probe_id"] = new_probe_id
        selected_public.append(public_row)
        selected_private.append(private_row)
    selection_stats = {
        "candidate_count": candidate_count,
        "query_only_unresolved_count": len(query_only_accepted_ids),
        "accepted_before_diversity_selection": len(accepted_ids),
        "selected_count": len(selected_public),
        "candidate_rejection_reasons": dict(rejection_reasons),
        "selected_type_counts": dict(Counter(row["metadata"]["probe_type"] for row in selected_public)),
        "selected_difficulty_counts": dict(Counter(row["independent_gold_judge"]["difficulty"] for row in selected_private)),
    }
    write_json(
        probe_cache,
        {
            "cache_fingerprint": probe_fp,
            "public": selected_public,
            "private": selected_private,
            "selection_stats": selection_stats,
        },
    )
    write_json(diagnostics_path, {
        "user_id": profile["user_id"],
        "habit_id": habit["habit_id"],
        "status": "pass",
        "rounds": diagnostic_rounds,
        "selection_stats": selection_stats,
    })
    return selected_public, selected_private


def probe_stage(args: argparse.Namespace, profiles: list[dict[str, Any]], sessions: list[dict[str, Any]]) -> None:
    sessions_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        sessions_by_user[row["user_id"]].append(row)
    versions = read_jsonl(args.dataset / "private" / "habit_version_history.jsonl")
    final_states = {}
    for row in versions:
        habit_id = row["habit_id"]
        if habit_id not in final_states or int(row["version"]) > int(final_states[habit_id]["version"]):
            final_states[habit_id] = row
    all_jobs = []
    for profile in profiles:
        for habit in profile["habits"]:
            if not habit.get("testable", True):
                continue
            active = dict(habit)
            active.update({k: v for k, v in final_states.get(habit["habit_id"], {}).items() if k in {"version", "preference_value", "condition", "default_action", "boundary_condition", "exception_condition"}})
            all_jobs.append((profile, active))
    excluded_habit_ids = {
        item.strip() for item in args.exclude_probe_habits.split(",") if item.strip()
    }
    known_habit_ids = {habit["habit_id"] for _, habit in all_jobs}
    unknown_exclusions = sorted(excluded_habit_ids - known_habit_ids)
    if unknown_exclusions:
        raise ValueError(f"unknown --exclude-probe-habits ids: {unknown_exclusions}")
    jobs = [
        (profile, habit)
        for profile, habit in all_jobs
        if habit["habit_id"] not in excluded_habit_ids
    ]
    public, private = [], []
    failures = []
    def run_with_retry(profile: dict[str, Any], habit: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        last_error = None
        for attempt in range(3):
            try:
                return probe_job(args, profile, habit, sessions_by_user[profile["user_id"]])
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                print(
                    f"probe_retry {habit['habit_id']} attempt={attempt + 1}/3 reason={str(exc)[:500]}",
                    flush=True,
                )
        raise ValueError(f"probe set failed after independent-judge retries for {habit['habit_id']}: {last_error}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.probe_workers) as executor:
        futures = {executor.submit(run_with_retry, p, h): (p, h) for p, h in jobs}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            profile, habit = futures[future]
            try:
                pub_rows, key_rows = future.result()
            except Exception as exc:
                failures.append({
                    "user_id": profile["user_id"],
                    "habit_id": habit["habit_id"],
                    "error": str(exc),
                })
                print(
                    f"probe_failed {habit['habit_id']} completed={completed}/{len(jobs)} "
                    f"reason={str(exc)[:500]}",
                    flush=True,
                )
                continue
            public.extend(pub_rows); private.extend(key_rows)
            print(f"probe_progress {completed}/{len(jobs)}", flush=True)
    write_json(args.dataset / "reports" / "probe_generation_failures.json", {
        "status": "reject" if failures else ("pass_with_exclusions" if excluded_habit_ids else "pass"),
        "target_habit_count": len(all_jobs),
        "job_count": len(jobs),
        "completed_habit_count": len(jobs) - len(failures),
        "failed_habit_count": len(failures),
        "excluded_habit_count": len(excluded_habit_ids),
        "excluded_habit_ids": sorted(excluded_habit_ids),
        "exclusion_policy": (
            "explicit_quality_exclusion_after_repeated strict memory-necessity and diversity failures"
            if excluded_habit_ids else "none"
        ),
        "failures": failures,
    })
    if failures:
        raise ValueError(
            f"{len(failures)} probe sets failed strict generation; "
            "successful per-habit caches were preserved for targeted retry"
        )
    shuffle_seed, shuffle_seed_fingerprint = private_probe_shuffle_seed(args.dataset)
    public.sort(key=lambda row: private_shuffle_rank(shuffle_seed, "probe_row", row["probe_id"]))
    private.sort(key=lambda row: private_shuffle_rank(shuffle_seed, "probe_row", row["probe_id"]))
    private_by_id = {row["probe_id"]: row for row in private}
    labels = ["A", "B", "C", "D"]
    for probe in public:
        key = private_by_id[probe["probe_id"]]
        probe.setdefault("metadata", {}).pop("recovery_gate", None)
        probe["metadata"]["blind_recovery_role"] = "diagnostic_only"
        old_choices = list(probe["choices"])
        shuffled_choices = sorted(
            old_choices,
            key=lambda choice: private_shuffle_rank(
                shuffle_seed, f"probe_choices:{probe['probe_id']}", choice["choice_id"]
            ),
        )
        old_to_new = {
            choice["choice_id"]: label for label, choice in zip(labels, shuffled_choices)
        }
        probe["choices"] = [
            {"choice_id": label, "text": choice["text"]}
            for label, choice in zip(labels, shuffled_choices)
        ]
        key["generator_closest_distractor_choice_id"] = old_to_new[
            key["generator_closest_distractor_choice_id"]
        ]
        key["independent_gold_judge"]["closest_distractor_choice_id"] = old_to_new[
            key["independent_gold_judge"]["closest_distractor_choice_id"]
        ]
        key["query_only_judge"]["plausible_choice_ids"] = [
            old_to_new[choice_id]
            for choice_id in key["query_only_judge"]["plausible_choice_ids"]
        ]
        key["gold_choice_id"] = old_to_new[key["gold_choice_id"]]
        key["label_rationale"] = remap_choice_references(key.get("label_rationale"), old_to_new)
        key["generator_difficulty_rationale"] = remap_choice_references(
            key.get("generator_difficulty_rationale"), old_to_new
        )
        key["query_only_judge"]["rationale"] = remap_choice_references(
            key["query_only_judge"].get("rationale"), old_to_new
        )
        key["query_only_judge"]["leakage_signals"] = [
            remap_choice_references(item, old_to_new)
            for item in key["query_only_judge"].get("leakage_signals", [])
        ]
        key["independent_gold_judge"]["rationale"] = remap_choice_references(
            key["independent_gold_judge"].get("rationale"), old_to_new
        )
        key["independent_gold_judge"]["closest_distractor_rationale"] = remap_choice_references(
            key["independent_gold_judge"].get("closest_distractor_rationale"), old_to_new
        )
        key["choice_position_balancing"] = "private_seeded_per_probe_shuffle"
        key["choice_reference_remapped"] = True
    write_jsonl(args.dataset / "public" / "probes.jsonl", public)
    write_jsonl(args.dataset / "private" / "probe_key.jsonl", private)
    write_json(args.dataset / "reports" / "probe_contrastive_validation.json", {
        "status": "pass_with_exclusions" if excluded_habit_ids else "pass",
        "validation_contract": "query-only unresolved; history-aware judge uniquely agrees with gold",
        "target_habit_count": len(all_jobs),
        "evaluated_habit_count": len(jobs),
        "excluded_habit_ids": sorted(excluded_habit_ids),
        "probe_count": len(public),
        "probe_shuffle_seed_fingerprint": shuffle_seed_fingerprint,
        "gold_position_counts": dict(Counter(row["gold_choice_id"] for row in private)),
        "query_only_unresolved_count": sum(
            row.get("query_only_judge", {}).get("choice_id") == "UNRESOLVED" for row in private
        ),
        "query_only_answerable_count": sum(
            row.get("query_only_judge", {}).get("answerable_without_history") is True for row in private
        ),
        "history_aware_gold_agreement_count": sum(bool(row.get("independent_gold_judge")) for row in private),
        "difficulty_counts": dict(Counter(
            row.get("independent_gold_judge", {}).get("difficulty") for row in private
        )),
        "habit_counts": dict(Counter(row["habit_id"] for row in private)),
        "probe_type_counts": dict(Counter(row["probe_type"] for row in private)),
    })


def load_profiles(dataset: Path) -> list[dict[str, Any]]:
    return read_jsonl(dataset / "private" / "user_dossiers.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["profiles", "arcs", "sessions", "recovery", "probes", "all"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=os.getenv("HABITBENCH_GEN_MODEL", "gpt-5.5"))
    parser.add_argument("--base-url", default=os.getenv("HABITBENCH_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("HABITBENCH_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--users", type=int, default=6)
    parser.add_argument(
        "--only-users",
        default="",
        help="Comma-separated user_ids to append/regenerate for arcs or sessions; other users are preserved.",
    )
    parser.add_argument("--arc-block-size", type=int, default=6)
    parser.add_argument("--session-block-size", type=int, default=3)
    parser.add_argument(
        "--session-shard-dir", type=Path,
        help="For stage=sessions, write one selected user's outputs to an isolated shard.",
    )
    parser.add_argument("--probe-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--transport", choices=["curl", "curl_stream", "urllib"], default="curl_stream")
    parser.add_argument("--reasoning-effort", default=os.getenv("HABITBENCH_REASONING_EFFORT", "xhigh"))
    parser.add_argument(
        "--allow-recovery-drift",
        action="store_true",
        help="Generate explicitly exploratory probes after a recorded recovery reject; never marks recovery as passed.",
    )
    parser.add_argument(
        "--exclude-probe-habits",
        default="",
        help="Comma-separated testable habit ids explicitly excluded after documented strict probe-generation failure.",
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("/data1/public/hf/Qwen/Qwen3-8B"))
    parser.add_argument("--tokenizer-python", type=Path, default=Path("/home/xqwang/miniconda3/envs/grpo/bin/python"))
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("Set HABITBENCH_BASE_URL and HABITBENCH_API_KEY")
    args.dataset.mkdir(parents=True, exist_ok=True)
    profiles = profile_stage(args) if args.stage in {"profiles", "all"} else load_profiles(args.dataset)
    if not profiles:
        raise SystemExit("no user dossiers available")
    manifest_profiles = list(profiles)
    if args.session_shard_dir and args.stage != "sessions":
        raise SystemExit("--session-shard-dir is supported only for stage=sessions")
    if args.only_users and args.stage not in {"arcs", "sessions"}:
        raise SystemExit("--only-users is supported only for arcs and sessions; run recovery/probes once on all completed users")
    if args.only_users and args.stage != "profiles":
        requested_user_ids = {item.strip() for item in args.only_users.split(",") if item.strip()}
        available_user_ids = {profile["user_id"] for profile in profiles}
        missing_user_ids = sorted(requested_user_ids - available_user_ids)
        if missing_user_ids:
            raise SystemExit(f"--only-users contains unknown ids: {missing_user_ids}")
        profiles = [profile for profile in profiles if profile["user_id"] in requested_user_ids]
        if not profiles:
            raise SystemExit("--only-users selected no profiles")
    if args.session_shard_dir and len(profiles) != 1:
        raise SystemExit("--session-shard-dir requires exactly one profile selected with --only-users")
    events = arc_stage(args, profiles) if args.stage in {"arcs", "all"} else read_jsonl(args.dataset / "private" / "chronological_arc.jsonl")
    sessions = read_jsonl(args.dataset / "private" / "sessions_with_annotations.jsonl")
    if args.stage in {"sessions", "all"}:
        if not events:
            raise SystemExit("no chronological arc available")
        sessions = session_stage(args, profiles, events)
    if args.stage in {"recovery", "all"}:
        if not sessions:
            raise SystemExit("no generated sessions available")
        recovery_stage(args, profiles, sessions)
    if args.stage in {"probes", "all"}:
        if not sessions:
            raise SystemExit("no generated sessions available")
        recovery_summary = args.dataset / "reports" / "habit_blind_recovery_summary.json"
        recovery_passed = (
            recovery_summary.exists()
            and json.loads(recovery_summary.read_text(encoding="utf-8")).get("status") == "pass"
        )
        if not recovery_passed:
            per_user_reports = [
                args.dataset / "reports" / f"{profile['user_id']}_habit_blind_recovery.json"
                for profile in profiles
            ]
            recovery_audited = all(
                path.exists()
                and json.loads(path.read_text(encoding="utf-8")).get("status") in {"pass", "reject"}
                for path in per_user_reports
            )
            if not (args.allow_recovery_drift and recovery_audited):
                raise SystemExit(
                    "habit-blind recovery did not pass; finish all per-user recovery audits and use "
                    "--allow-recovery-drift to generate explicitly drift-tagged probes"
                )
        probe_stage(args, profiles, sessions)
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "generation_method": GENERATION_METHOD,
        "dialogue_templates_used": False,
        "rewrite_or_paraphrase_stage_used": False,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "allow_recovery_drift": args.allow_recovery_drift,
        "excluded_probe_habit_ids": [
            item.strip() for item in args.exclude_probe_habits.split(",") if item.strip()
        ],
        "target_users": len(manifest_profiles),
        "target_sessions_by_user": {
            profile["user_id"]: profile["longitudinal_plan"]["target_sessions"]
            for profile in manifest_profiles
        },
    }
    manifest_root = args.session_shard_dir or args.dataset
    write_json(manifest_root / "reports" / "e2e_generation_manifest.json", manifest)


if __name__ == "__main__":
    main()
