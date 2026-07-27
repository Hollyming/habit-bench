#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SRC = Path('/mnt/data/work_v121_full/habit_bench_multidogo_finance_software_evidence_chained_v1_2_1')
DST = Path('/mnt/data/habit_bench_multidogo_finance_software_scope_consistent_v1_3')
ZIP = Path('/mnt/data/habit_bench_multidogo_finance_software_scope_consistent_v1_3_complete.zip')
ZIP_SHA = Path(str(ZIP) + '.sha256')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    text = text.lower().replace('’', "'")
    text = re.sub(r'\b\d+(?:[.,]\d+)?\b', '<num>', text)
    text = re.sub(r'[^a-z0-9<>]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def stable_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def ann0(session: dict[str, Any]) -> dict[str, Any]:
    anns = session.get('memory_annotations') or []
    return anns[0] if anns else {}


def excerpt_for_session(session: dict[str, Any], limit: int = 560) -> str:
    parts = [m.get('content', '').strip() for m in session.get('messages', []) if m.get('role') == 'user']
    text = ' '.join(' '.join(parts).split())
    return text if len(text) <= limit else text[:limit-1].rstrip() + '…'


def exact_asof_phrase(ts: str) -> str:
    dt = datetime.fromisoformat(ts)
    hour = dt.strftime('%I').lstrip('0') or '12'
    return f'{hour}:{dt.strftime("%M")} {dt.strftime("%p").lower()} on {dt.strftime("%B")} {dt.day}, {dt.year}'


def temporalize_second_task(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    replacements = [
        (r'\bstands now\b', 'stood at that time', 'stands_now'),
        (r'\bCheck the present state of\b', 'Check the state recorded at that time for', 'present_state'),
        (r'\bReconcile the latest activity on\b', 'Reconcile the activity available at that time on', 'latest_activity'),
        (r'\bcurrent transfer state\b', 'transfer state at that time', 'current_transfer_state'),
        (r'\bcurrent status\b', 'status at that time', 'current_status'),
        (r'\bpresent state\b', 'state at that time', 'present_state_generic'),
        (r'\bstands currently\b', 'stood at that time', 'stands_currently'),
        (r'\bnow\b', 'at that time', 'now_generic'),
    ]
    out = text
    for pattern, repl, code in replacements:
        new, count = re.subn(pattern, repl, out, flags=re.I)
        if count:
            changes.extend([code] * count)
            out = new
    return out, changes


SCOPE_QUERY_TEMPLATES = [
    'Prepare one closeout with two policy dates. Apply my current standing process to the first workstream: {task1} For the second workstream, use the process that was in force at {date}: {task2}',
    'Treat these as separate policy snapshots. Handle the first workstream under my current standing process: {task1} Reconstruct the second under the process effective at {date}: {task2}',
    'Use different effective dates for the two workstreams. The first follows my current standing process: {task1} The second follows the process in force at {date}: {task2}',
    'Build one closeout without merging the time scopes. Use my current standing process for the first workstream: {task1} Use the process effective at {date} for the second: {task2}',
    'The two workstreams use different policy snapshots. Resolve the first with my current standing process: {task1} Resolve the second with the process that applied at {date}: {task2}',
    'Keep the effective dates separate in this closeout. For the first workstream, use my current standing process: {task1} For the second, recover the process in force at {date}: {task2}',
    'Prepare a two-part handoff. The first part must follow my current standing process: {task1} The second part must follow the process effective at {date}: {task2}',
    'Do not collapse the two time scopes. Apply my current standing process to the first workstream: {task1} Apply the process that was in force at {date} to the second: {task2}',
]


def rewrite_scope_query(pid: str, query: str, asof: str) -> tuple[str, dict[str, Any]]:
    date = exact_asof_phrase(asof)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', query)
    if len(sentences) != 4:
        raise ValueError(f'{pid}: expected 4 sentences, got {len(sentences)}')
    prefix = f'As of {date}, '
    if not sentences[0].startswith(prefix):
        raise ValueError(f'{pid}: prefix mismatch: {sentences[0]!r} vs {prefix!r}')
    task1 = sentences[0][len(prefix):]
    task2, deictic_changes = temporalize_second_task(sentences[1])
    template_idx = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16) % len(SCOPE_QUERY_TEMPLATES)
    new_query = SCOPE_QUERY_TEMPLATES[template_idx].format(task1=task1, task2=task2, date=date)
    new_query += ' ' + ' '.join(sentences[2:])
    return new_query, {
        'as_of_phrase': date,
        'task1': task1,
        'task2_original': sentences[1],
        'task2_rewritten': task2,
        'deictic_changes': deictic_changes,
        'template_index': template_idx,
    }


if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)
if ZIP.exists(): ZIP.unlink()
if ZIP_SHA.exists(): ZIP_SHA.unlink()

# Load core data.
lifelines = read_jsonl(DST / 'public/lifelines.jsonl')
probes = read_jsonl(DST / 'public/probes.jsonl')
keys = read_jsonl(DST / 'private/probe_key.jsonl')
sessions = read_jsonl(DST / 'private/sessions_with_annotations.jsonl')
profiles = read_jsonl(DST / 'private/persona_profiles.jsonl')
chains = read_jsonl(DST / 'private/probe_evidence_chains.jsonl')
habits = json.loads((DST / 'source/habit_templates_retained.json').read_text(encoding='utf-8'))

probe_by = {p['probe_id']: p for p in probes}
key_by = {k['probe_id']: k for k in keys}
session_by = {s['session_id']: s for s in sessions}
chain_by = {c['probe_id']: c for c in chains}
profile_by = {p['user_id']: p for p in profiles}

# -----------------------------------------------------------------------------
# Patch 1: make the finance balance/statement/card-activity scope and replacement
# semantics explicit for the user implicated by the feedback.
# -----------------------------------------------------------------------------
SCOPE_USER = 'mdgo_v05_fin_user_0006'
BALANCE_HABIT = 'finance_balance_statement_summary_first'
changed_session_messages: dict[str, list[dict[str, str]]] = {
    'mdgo_v05_fin_user_0006_s0008': [
        {'role': 'user', 'content': 'Reconcile the latest activity on card ending 8748 for the tuition installment. This card-activity case is one instance of my general account-review routine, which also covers balance checks, statement reviews, transaction-history reviews, and card-activity reconciliation. Open workflow review POLB-winter-docket-giisxq and record the two routes still under consideration for that whole review family.'},
        {'role': 'assistant', 'content': 'Route One would show unusual entries first in a compact table and close with the overall finding; Route Two would present a chronological transaction timeline followed by the conclusion. I can preserve both without treating either as the default.'},
        {'role': 'user', 'content': 'Keep exactly those two candidates, in that order, under POLB-winter-docket-giisxq. The scope is the whole account-review family I just named, not only this tuition case. Rule out the other workflow routes for this review family, but do not finalize the first-versus-second choice here.'},
        {'role': 'assistant', 'content': 'Both candidates remain viable for balance, statement, transaction-history, and card-activity reconciliation reviews until a later closeout points back to this ordering. The next decision must cite POLB-winter-docket-giisxq rather than reconstructing the order from memory.'},
    ],
    'mdgo_v05_fin_user_0006_s0263': [
        {'role': 'user', 'content': 'Reconcile the latest activity on card ending 8748 for the annual insurance renewal. Use the original ordering in POLB-winter-docket-giisxq and carry forward option one. Make it the durable default across the entire account-review family—balance checks, statements, transaction histories, and card-activity reconciliation—until I explicitly replace it.'},
        {'role': 'assistant', 'content': 'I will link this closeout to the ordered shortlist in POLB-winter-docket-giisxq. Option one becomes effective from this session across that defined account-review family, without being re-expanded into a fresh list.'},
        {'role': 'user', 'content': 'This is not a one-case exception for the insurance renewal. Preserve POLB-winter-docket-giisxq and its effective date as the standing process for every review inside that family; a later case that merely resembles one example must not switch to the other candidate.'},
        {'role': 'assistant', 'content': 'POLB-winter-docket-giisxq is closed as a durable family-wide decision. The unselected route remains nonbinding unless a later dated decision explicitly replaces it across the same scope.'},
    ],
    'mdgo_v05_fin_user_0006_s0278': [
        {'role': 'user', 'content': 'Review the June 11 statement for checking ending 4222 and explain the change. Reopen the same balance, statement, transaction-history, and card-activity review family under workflow review POLR-winter-docket-giisxq. Treat this as a replacement shortlist for that whole family, not as final authorization of either route.'},
        {'role': 'assistant', 'content': 'The first candidate is to lead with the account-level finding and then list supporting entries. The second is to group relevant entries by merchant or category and connect each group to the conclusion. I can preserve both without treating either as the default.'},
        {'role': 'user', 'content': 'Keep exactly those two candidates, in that order, under POLR-winter-docket-giisxq. If this replacement review is later resolved, it will supersede the prior POLB-winter-docket-giisxq default across both statement and card-activity cases in the same account-review family. Do not finalize the first-versus-second choice here.'},
        {'role': 'assistant', 'content': 'I have preserved the two replacement candidates in order and marked the decision as pending for the full account-review family. The next decision must cite POLR-winter-docket-giisxq rather than reconstructing the order from memory.'},
    ],
    'mdgo_v05_fin_user_0006_s0533': [
        {'role': 'user', 'content': 'Review the March 18 statement for checking ending 4222 and explain the change. Return to the ordered shortlist in POLR-winter-docket-giisxq and use the first-listed candidate as the replacement default whenever any case in the same account-review family recurs, including balance, statement, transaction-history, and card-activity reconciliation. This selection supersedes POLB-winter-docket-giisxq across that scope.'},
        {'role': 'assistant', 'content': 'I will link this closeout to the ordered shortlist in POLR-winter-docket-giisxq. The first-listed candidate becomes effective from this session as the family-wide replacement, without being re-expanded into a fresh list.'},
        {'role': 'user', 'content': 'This is a global replacement within that account-review family, not a statement-only exception. Preserve POLR-winter-docket-giisxq and its effective date; future card-activity and statement reviews use this replacement unless a later dated decision changes the family again.'},
        {'role': 'assistant', 'content': 'POLR-winter-docket-giisxq is closed as the durable replacement for the full account-review family. The earlier exceptions-first decision in POLB-winter-docket-giisxq is superseded throughout that scope, and the unselected replacement route remains nonbinding.'},
    ],
}

scope_fields = {
    'scope_contract': 'balance checks, statement reviews, transaction-history reviews, and card-activity reconciliation',
    'scope_replacement_semantics': 'replacement resolution supersedes the baseline decision globally within the stated account-review family',
}
for sid, messages in changed_session_messages.items():
    s = session_by[sid]
    s['messages'] = messages
    for ann in s.get('memory_annotations', []):
        ann.update(scope_fields)
        if ann.get('kind') == 'v11_replacement_resolution':
            ann['supersedes_pair_ref'] = 'POLB-winter-docket-giisxq'
            ann['replacement_scope'] = 'global_within_habit_condition'
    s.setdefault('rewrite_metadata', {})
    s['rewrite_metadata'].update({
        'rewrite_version': 'v1.3',
        'rewrite_scope': 'account-review scope and replacement clarification',
        'public_private_messages_match': True,
    })

# Public nested sessions must remain byte-semantic matches to private messages.
for life in lifelines:
    for s in life['sessions']:
        if s['session_id'] in changed_session_messages:
            s['messages'] = changed_session_messages[s['session_id']]

# Make the private design metadata and source habit contract explicit.
prof = profile_by[SCOPE_USER]
meta = prof['v11_challenge_metadata']['decision_meta'][BALANCE_HABIT]
meta.update({
    'scope_contract': scope_fields['scope_contract'],
    'replacement_scope': 'global_within_habit_condition',
    'replacement_supersedes_baseline': True,
    'replacement_supersedes_pair_ref': 'POLB-winter-docket-giisxq',
})
for h in habits:
    if h['habit_id'] == BALANCE_HABIT:
        h['scope_examples'] = ['balance check', 'bank statement review', 'transaction-history review', 'card-activity reconciliation']
        h['replacement_semantics'] = 'A dated replacement decision applies across all cases satisfying the habit condition unless explicitly scoped to one case.'
        break

# -----------------------------------------------------------------------------
# Patch 2: explicitly bind the two time scopes for all 64 scope-temporal probes.
# -----------------------------------------------------------------------------
scope_audit_rows: list[dict[str, Any]] = []
for p in probes:
    k = key_by[p['probe_id']]
    if k['probe_type'] != 'scope_temporal_pair':
        continue
    old_query = p['query']
    new_query, info = rewrite_scope_query(p['probe_id'], old_query, k['as_of_timestamp'])
    p['query'] = new_query
    target = k['target_habit_ids']
    if len(target) != 2:
        raise AssertionError(f'{p["probe_id"]}: scope_temporal_pair target count != 2')
    k['temporal_scope_binding'] = {
        target[0]: 'current_standing_policy_at_probe_time',
        target[1]: k['as_of_timestamp'],
    }
    k['temporal_scope_text_contract'] = 'The first workstream uses the current standing policy; the second uses the policy in force at the explicit historical timestamp.'
    scope_audit_rows.append({
        'probe_id': p['probe_id'],
        'user_id': p['user_id'],
        'domain': p['domain'],
        'first_target_habit_id': target[0],
        'second_target_habit_id': target[1],
        'first_target_state_time': '',
        'second_target_state_time': k['as_of_timestamp'],
        'as_of_phrase': info['as_of_phrase'],
        'task2_deictic_rewrites_json': json.dumps(info['deictic_changes'], ensure_ascii=False),
        'old_query': old_query,
        'new_query': new_query,
        'old_scope_binding_ambiguous': 1,
        'new_scope_binding_explicit': 1,
    })

# Refresh maps after modifications.
probe_by = {p['probe_id']: p for p in probes}
session_by = {s['session_id']: s for s in sessions}

# -----------------------------------------------------------------------------
# Patch 3: add decision-unit identifiers and a chain-balanced exact-match view.
# This preserves the 2,048 probes while preventing one latent decision from
# receiving disproportionate weight in the recommended aggregate metric.
# -----------------------------------------------------------------------------
def group_habit(group: list[str]) -> str | None:
    hs: list[str] = []
    for sid in group:
        for h in ann0(session_by[sid]).get('habit_ids', []):
            if h not in hs:
                hs.append(h)
    return hs[0] if len(hs) == 1 else None


def decision_unit_id(user_id: str, habit_id: str, group: list[str]) -> str:
    raw = json.dumps([user_id, habit_id, group], ensure_ascii=False, separators=(',', ':'))
    return 'mdgo_v13_du_' + hashlib.sha256(raw.encode()).hexdigest()[:16]

unit_meta: dict[str, dict[str, Any]] = {}
probe_units: dict[str, list[str]] = {}
for k in keys:
    units: list[str] = []
    for habit in k['target_habit_ids']:
        matches = [g for g in k['required_component_groups'] if len(g) == 2 and group_habit(g) == habit]
        if len(matches) != 1:
            raise AssertionError(f'{k["probe_id"]}: expected one decisive pair for {habit}, got {matches}')
        group = matches[0]
        du = decision_unit_id(k['user_id'], habit, group)
        units.append(du)
        unit_meta.setdefault(du, {
            'decision_unit_id': du,
            'user_id': k['user_id'],
            'domain': k['domain'],
            'habit_id': habit,
            'decision_evidence_session_ids': group,
            'probe_ids': [],
        })['probe_ids'].append(k['probe_id'])
    probe_units[k['probe_id']] = units

for m in unit_meta.values():
    m['probe_ids'] = stable_unique(m['probe_ids'])
    m['reuse_count'] = len(m['probe_ids'])

for k in keys:
    units = probe_units[k['probe_id']]
    k['decision_unit_ids'] = units
    k['decision_unit_reuse_counts'] = {u: unit_meta[u]['reuse_count'] for u in units}
    raw = '|'.join(sorted(units))
    k['decision_bundle_id'] = 'mdgo_v13_db_' + hashlib.sha256(raw.encode()).hexdigest()[:16]
    k['recommended_aggregation'] = 'report both exact probe micro accuracy and decision-unit macro accuracy'

# Rebuild evidence-chain excerpts and add decision-unit metadata.
for chain in chains:
    pid = chain['probe_id']
    k = key_by[pid]
    chain['decision_unit_ids'] = k['decision_unit_ids']
    chain['decision_bundle_id'] = k['decision_bundle_id']
    for step in chain['chain_steps']:
        s = session_by[step['session_id']]
        a = ann0(s)
        step['timestamp'] = s['timestamp']
        step['annotation_kind'] = a.get('kind')
        step['habit_ids'] = a.get('habit_ids', [])
        step['pair_ref'] = a.get('pair_ref')
        step['case_ref'] = a.get('case_ref')
        step['selected_ordinal'] = a.get('selected_ordinal')
        step['variant_id'] = a.get('variant_id')
        step['ordered_variants'] = a.get('ordered_variants')
        step['evidence_strength'] = a.get('evidence_strength')
        step['user_excerpt'] = excerpt_for_session(s)

# Write modified core rows early.
write_jsonl(DST / 'public/lifelines.jsonl', lifelines)
write_jsonl(DST / 'public/probes.jsonl', probes)
write_jsonl(DST / 'private/sessions_with_annotations.jsonl', sessions)
write_jsonl(DST / 'private/persona_profiles.jsonl', profiles)
write_jsonl(DST / 'private/probe_key.jsonl', keys)
write_jsonl(DST / 'private/probe_evidence_chains.jsonl', chains)
write_json(DST / 'source/habit_templates_retained.json', habits)

# Decision-unit files.
unit_rows = sorted(unit_meta.values(), key=lambda x: x['decision_unit_id'])
write_jsonl(DST / 'private/decision_unit_index.jsonl', unit_rows)
write_jsonl(DST / 'private/probe_decision_units.jsonl', [
    {
        'probe_id': k['probe_id'],
        'user_id': k['user_id'],
        'decision_bundle_id': k['decision_bundle_id'],
        'decision_unit_ids': k['decision_unit_ids'],
        'decision_unit_reuse_counts': k['decision_unit_reuse_counts'],
    }
    for k in keys
])

# Rebuild enriched probes from public + private metadata.
enriched = []
for p in probes:
    k = key_by[p['probe_id']]
    row = dict(p)
    row.update({
        'evidence_chain_id': k['evidence_chain_id'],
        'session_id': k['decision_evidence_session_ids'],
        'evidence_context_session_ids': k['evidence_context_session_ids'],
        'nonbinding_evidence_session_ids': k['nonbinding_evidence_session_ids'],
        'decision_unit_ids': k['decision_unit_ids'],
        'decision_bundle_id': k['decision_bundle_id'],
    })
    enriched.append(row)
write_jsonl(DST / 'private/probes_with_evidence.jsonl', enriched)

# Rebuild flattened chain edges with updated excerpts.
edge_fields = [
    'evidence_chain_id', 'probe_id', 'user_id', 'domain', 'decision_bundle_id', 'decision_unit_ids_json',
    'session_id', 'session_index', 'timestamp', 'evidence_status', 'evidence_role', 'annotation_kind',
    'habit_ids_json', 'pair_ref', 'case_ref', 'selected_ordinal', 'variant_id', 'gold_alignment', 'user_excerpt'
]
edge_rows: list[dict[str, Any]] = []
for chain in chains:
    for step in chain['chain_steps']:
        edge_rows.append({
            'evidence_chain_id': chain['evidence_chain_id'],
            'probe_id': chain['probe_id'],
            'user_id': chain['user_id'],
            'domain': chain['domain'],
            'decision_bundle_id': chain['decision_bundle_id'],
            'decision_unit_ids_json': json.dumps(chain['decision_unit_ids'], ensure_ascii=False),
            'session_id': step['session_id'],
            'session_index': step['session_index'],
            'timestamp': step['timestamp'],
            'evidence_status': step['evidence_status'],
            'evidence_role': step['evidence_role'],
            'annotation_kind': step.get('annotation_kind') or '',
            'habit_ids_json': json.dumps(step.get('habit_ids', []), ensure_ascii=False),
            'pair_ref': step.get('pair_ref') or '',
            'case_ref': step.get('case_ref') or '',
            'selected_ordinal': '' if step.get('selected_ordinal') is None else step['selected_ordinal'],
            'variant_id': step.get('variant_id') or '',
            'gold_alignment': step.get('gold_alignment') or '',
            'user_excerpt': step.get('user_excerpt') or '',
        })
with (DST / 'private/probe_evidence_chain_edges.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=edge_fields)
    w.writeheader(); w.writerows(edge_rows)

# Rebuild review queue.
old_review = DST / 'review/multidogo_finance_software_v121_review_queue_all.csv'
with old_review.open(encoding='utf-8-sig', newline='') as f:
    old_review_rows = list(csv.DictReader(f))
old_review_by = {r['probe_id']: r for r in old_review_rows}
new_review = DST / 'review/multidogo_finance_software_v13_review_queue_all.csv'
review_fields = [
    'probe_id', 'user_id', 'domain', 'probe_type', 'capability_group', 'query', 'choices_json',
    'proposed_gold_choice_id', 'target_habit_ids_json', 'evidence_requirement', 'evidence_chain_id',
    'decision_bundle_id', 'decision_unit_ids_json', 'session_id_json', 'temporal_context_session_ids_json',
    'nonbinding_evidence_session_ids_json', 'evidence_chain_preview_json', 'evidence_span_sessions',
    'reviewer_decision', 'reviewer_notes'
]
review_rows: list[dict[str, Any]] = []
for p in probes:
    k = key_by[p['probe_id']]
    chain = chain_by[p['probe_id']]
    preview = [{
        'session_id': s['session_id'], 'session_index': s['session_index'], 'timestamp': s['timestamp'],
        'evidence_status': s['evidence_status'], 'evidence_role': s['evidence_role'],
        'habit_ids': s.get('habit_ids', []), 'pair_ref': s.get('pair_ref'), 'case_ref': s.get('case_ref'),
        'selected_ordinal': s.get('selected_ordinal'), 'variant_id': s.get('variant_id'),
        'user_excerpt': s.get('user_excerpt'),
    } for s in chain['chain_steps']]
    old = old_review_by.get(p['probe_id'], {})
    review_rows.append({
        'probe_id': p['probe_id'], 'user_id': p['user_id'], 'domain': p['domain'],
        'probe_type': k['probe_type'], 'capability_group': k['capability_group'], 'query': p['query'],
        'choices_json': json.dumps(p['choices'], ensure_ascii=False),
        'proposed_gold_choice_id': k['gold_choice_id'],
        'target_habit_ids_json': json.dumps(k['target_habit_ids'], ensure_ascii=False),
        'evidence_requirement': k['evidence_requirement'], 'evidence_chain_id': k['evidence_chain_id'],
        'decision_bundle_id': k['decision_bundle_id'],
        'decision_unit_ids_json': json.dumps(k['decision_unit_ids'], ensure_ascii=False),
        'session_id_json': json.dumps(k['decision_evidence_session_ids'], ensure_ascii=False),
        'temporal_context_session_ids_json': json.dumps(k['temporal_context_session_ids'], ensure_ascii=False),
        'nonbinding_evidence_session_ids_json': json.dumps(k['nonbinding_evidence_session_ids'], ensure_ascii=False),
        'evidence_chain_preview_json': json.dumps(preview, ensure_ascii=False),
        'evidence_span_sessions': k['evidence_span_sessions'],
        'reviewer_decision': old.get('reviewer_decision', ''), 'reviewer_notes': old.get('reviewer_notes', ''),
    })
with new_review.open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=review_fields)
    w.writeheader(); w.writerows(review_rows)
old_review.unlink()

# Reports: scope-temporal patch.
with (DST / 'reports/scope_temporal_pair_binding_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(scope_audit_rows[0].keys()))
    w.writeheader(); w.writerows(scope_audit_rows)

# Reports: account-review scope revalidation and affected probes.
feedback_probe_ids = {
    'mdgo_v11_probe_000019','mdgo_v11_probe_000023','mdgo_v11_probe_000313','mdgo_v11_probe_000541',
    'mdgo_v11_probe_000926','mdgo_v11_probe_001225','mdgo_v11_probe_001675','mdgo_v11_probe_001704',
    'mdgo_v11_probe_001952','mdgo_v11_probe_001958'
}
balance_unit = None
for du, m in unit_meta.items():
    if m['user_id'] == SCOPE_USER and m['habit_id'] == BALANCE_HABIT and set(m['decision_evidence_session_ids']) == {'mdgo_v05_fin_user_0006_s0278','mdgo_v05_fin_user_0006_s0533'}:
        balance_unit = du
        break
if not balance_unit:
    raise AssertionError('failed to locate user0006 balance replacement decision unit')

balance_rows = []
for pid in unit_meta[balance_unit]['probe_ids']:
    p = probe_by[pid]; k = key_by[pid]
    query_scope_match = bool(re.search(r'\b(statement|balance|transaction|activity|reconcile)\b', p['query'], flags=re.I))
    balance_rows.append({
        'probe_id': pid, 'probe_type': k['probe_type'], 'gold_choice_id': k['gold_choice_id'],
        'listed_in_feedback_error_set': int(pid in feedback_probe_ids),
        'query_within_declared_account_review_scope': int(query_scope_match),
        'decision_unit_id': balance_unit, 'decision_unit_reuse_count': unit_meta[balance_unit]['reuse_count'],
        'query': p['query'],
    })
with (DST / 'reports/user0006_balance_scope_probe_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(balance_rows[0].keys()))
    w.writeheader(); w.writerows(balance_rows)

scope_session_text = ' '.join(m['content'] for sid in changed_session_messages for m in session_by[sid]['messages'])
balance_scope_report = {
    'version': 'v1.3',
    'user_id': SCOPE_USER,
    'habit_id': BALANCE_HABIT,
    'baseline_pair_ref': 'POLB-winter-docket-giisxq',
    'replacement_pair_ref': 'POLR-winter-docket-giisxq',
    'replacement_decision_unit_id': balance_unit,
    'replacement_decision_unit_reuse_count': unit_meta[balance_unit]['reuse_count'],
    'feedback_error_probe_count': len(feedback_probe_ids),
    'all_feedback_probe_ids_use_declared_scope': all(r['query_within_declared_account_review_scope'] for r in balance_rows if r['listed_in_feedback_error_set']),
    'scope_terms_explicit_in_visible_history': all(x in scope_session_text.lower() for x in ['statement', 'transaction-history', 'card-activity']),
    'global_replacement_explicit_in_visible_history': 'global replacement' in scope_session_text.lower() and 'supersedes polb-winter-docket-giisxq' in scope_session_text.lower(),
    'interpretation': 'POLR-winter-docket-giisxq globally replaces the earlier POLB decision for all cases within the account-review habit condition, including both statement and card-activity reconciliation.',
}
write_json(DST / 'reports/user0006_balance_scope_revalidation.json', balance_scope_report)

# Decision-unit reuse reports.
unit_csv_rows = []
for m in unit_rows:
    unit_csv_rows.append({
        'decision_unit_id': m['decision_unit_id'], 'user_id': m['user_id'], 'domain': m['domain'],
        'habit_id': m['habit_id'], 'decision_evidence_session_ids_json': json.dumps(m['decision_evidence_session_ids'], ensure_ascii=False),
        'reuse_count': m['reuse_count'], 'probe_ids_json': json.dumps(m['probe_ids'], ensure_ascii=False),
    })
with (DST / 'reports/decision_unit_reuse_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(unit_csv_rows[0].keys()))
    w.writeheader(); w.writerows(unit_csv_rows)
reuse_vals = [m['reuse_count'] for m in unit_rows]
write_json(DST / 'reports/decision_unit_reuse_summary.json', {
    'version': 'v1.3', 'decision_units': len(unit_rows), 'probe_decision_unit_edges': sum(reuse_vals),
    'reuse_count_min': min(reuse_vals), 'reuse_count_median': statistics.median(reuse_vals),
    'reuse_count_p95': sorted(reuse_vals)[math.ceil(.95*len(reuse_vals))-1], 'reuse_count_max': max(reuse_vals),
    'recommended_metric': 'decision_unit_macro_accuracy',
    'metric_definition': 'For each unique (user, habit, decisive evidence pair), average exact probe correctness over probes using that decision; then macro-average across decision units.',
    'raw_probe_micro_accuracy_retained': True,
})

# Updated exact-match scorer with decision-unit macro aggregation.
scorer = r'''#!/usr/bin/env python3
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dataset-dir',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--method-name',default='method');a=p.parse_args()
base=Path(a.dataset_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
keys=[json.loads(x) for x in open(base/'private/probe_key.jsonl',encoding='utf-8') if x.strip()]
preds=[json.loads(x) for x in open(a.predictions,encoding='utf-8') if x.strip()]
km={x['probe_id']:x for x in keys};pm={x['probe_id']:x for x in preds}
missing=sorted(set(km)-set(pm));extra=sorted(set(pm)-set(km));dup=len(preds)-len(pm)
if missing or extra or dup:raise SystemExit(f'coverage error: missing={len(missing)} extra={len(extra)} duplicate={dup}')
rows=[];by=defaultdict(lambda:[0,0]);unit_scores=defaultdict(list);bundle_scores=defaultdict(list);correct=0
for pid,k in km.items():
 pred=pm[pid].get('choice_id');ok=int(pred==k['gold_choice_id']);correct+=ok
 for field in ['probe_type','capability_group','domain']:
  name=f"{field}:{k[field]}";by[name][0]+=ok;by[name][1]+=1
 for u in k.get('decision_unit_ids',[]):unit_scores[u].append(ok)
 bundle_scores[k.get('decision_bundle_id',pid)].append(ok)
 rows.append({'probe_id':pid,'prediction':pred,'gold':k['gold_choice_id'],'correct':ok,'probe_type':k['probe_type'],'capability_group':k['capability_group'],'domain':k['domain'],'decision_bundle_id':k.get('decision_bundle_id',''),'decision_unit_ids_json':json.dumps(k.get('decision_unit_ids',[]))})
unit_acc={u:sum(v)/len(v) for u,v in unit_scores.items()};bundle_acc={u:sum(v)/len(v) for u,v in bundle_scores.items()}
metrics={'method_name':a.method_name,'scoring':'exact_choice_id_match_only','correct':correct,'total':len(keys),'accuracy':correct/len(keys),'probe_micro_accuracy':correct/len(keys),'decision_unit_macro_accuracy':sum(unit_acc.values())/len(unit_acc),'decision_bundle_macro_accuracy':sum(bundle_acc.values())/len(bundle_acc),'decision_unit_count':len(unit_acc),'decision_bundle_count':len(bundle_acc),'by_group':{k:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for k,v in sorted(by.items())}}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
with open(out/'per_probe.csv','w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
with open(out/'per_decision_unit.csv','w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['decision_unit_id','probe_count','exact_accuracy']);w.writeheader();w.writerows({'decision_unit_id':u,'probe_count':len(unit_scores[u]),'exact_accuracy':unit_acc[u]} for u in sorted(unit_acc))
print(json.dumps(metrics,indent=2))
'''
(DST / 'scripts/score_predictions.py').write_text(scorer, encoding='utf-8')
(DST / 'scripts/score_predictions.py').chmod(0o755)

# Package migration script itself.
shutil.copy2(Path(__file__), DST / 'scripts/migrate_v121_to_v13.py')

# README and release notes.
readme = '''# HABIT-Bench MultiDoGO Finance & Software v1.3

v1.3 is a small quality patch over v1.2.1. It retains 54 coherent pseudo-users, 29,160 sessions, 2,048 probes, all answer choices, all gold labels, and the existing evidence topology. The patch addresses two reviewer-identified ambiguities and adds a decision-chain-balanced exact-match view.

## Core scale

| Item | v1.3 |
|---|---:|
| Pseudo-users | 54 |
| Finance users | 36 |
| Software users | 18 |
| Sessions per user | 540 |
| Total sessions | 29,160 |
| Probes | 2,048 |
| Private evidence chains | 2,048 |
| Gold A/B/C/D | 512 each |

## v1.3 changes

1. **64 `scope_temporal_pair` probes now bind time scopes explicitly.** The first workstream uses the user's current standing process; the second uses the process in force at the stated historical timestamp. Historical deictic phrases such as “stands now” are rewritten to “stood at that time.” No workflow variant is revealed.
2. **The account-review scope for `mdgo_v05_fin_user_0006` is clarified.** Visible history now states that the habit covers balance, statement, transaction-history, and card-activity reconciliation. The later `POLR-winter-docket-giisxq` decision explicitly supersedes the earlier `POLB-winter-docket-giisxq` decision across that whole habit scope, rather than only for one statement example.
3. **Decision-unit metadata and chain-balanced exact-match scoring are added.** A decision unit is a unique `(user, habit, decisive evidence pair)`. The scorer continues to report ordinary per-probe exact-match accuracy and additionally reports `decision_unit_macro_accuracy`, so one repeatedly reused latent decision cannot dominate the aggregate score.
4. **All evidence-chain excerpts, review rows, and private enriched probes are regenerated after the history patch.**

## Evaluation inputs

Give a benchmarked method only:

- `public/lifelines.jsonl`
- `public/probes.jsonl`

Do not expose `private/` files during standard evaluation.

## Exact-match scoring

```json
{"probe_id":"mdgo_v11_probe_000000","choice_id":"A"}
```

```bash
python scripts/score_predictions.py --dataset-dir . --predictions predictions.jsonl --output-dir runs/eval --method-name my_method
```

The scorer uses strict choice-ID equality only. It reports:

- `probe_micro_accuracy`: backward-compatible exact accuracy over all 2,048 probes;
- `decision_unit_macro_accuracy`: exact accuracy macro-averaged over unique user–habit decision chains;
- `decision_bundle_macro_accuracy`: exact accuracy macro-averaged over unique multi-habit decision bundles.

No similarity score or partial credit is used.

## Evidence and audit files

- `private/probe_evidence_chains.jsonl`
- `private/probe_evidence_chain_edges.csv`
- `private/decision_unit_index.jsonl`
- `private/probe_decision_units.jsonl`
- `reports/scope_temporal_pair_binding_audit.csv`
- `reports/user0006_balance_scope_revalidation.json`
- `reports/user0006_balance_scope_probe_audit.csv`
- `reports/decision_unit_reuse_audit.csv`
- `reports/v13_quality_patch.md`

## Compatibility

Existing `user_id`, `session_id`, `probe_id`, gold labels, and evidence-chain IDs are preserved. Record IDs inside choices retain the inherited `V12-*` prefix because changing all 2,048 answer strings would be unrelated to this small patch.

## Status

`auto_validated_pending_target_model_baseline_and_human_audit`
'''
(DST / 'README.md').write_text(readme, encoding='utf-8')
release = '''# v1.3 Release Notes

- Rephrased all 64 `scope_temporal_pair` queries so the first workstream is explicitly current and the second is explicitly historical.
- Removed historical deictic conflicts such as “stands now” in the second workstream.
- Clarified the visible scope and global replacement semantics of `finance_balance_statement_summary_first` for `mdgo_v05_fin_user_0006`.
- Regenerated evidence-chain excerpts and review previews after the four-session history patch.
- Added private decision-unit indexes and exact-match decision-unit macro scoring to prevent repeated use of one latent decision from dominating aggregate results.
- Retained 2,048 probes, 29,160 sessions, all choices, all gold labels, and all IDs.
'''
(DST / 'RELEASE_NOTES.md').write_text(release, encoding='utf-8')

quality_md = f'''# v1.3 reviewer-feedback patch

## 1. Time-scope binding

All **{len(scope_audit_rows)}** `scope_temporal_pair` probes previously began with an `As of ...` phrase before the first task even though private `target_state_times` assigned the first task to the current policy and the second task to the historical policy. The wording was therefore structurally ambiguous.

v1.3 rewrites each query so that:

```text
first workstream  -> current standing policy
second workstream -> policy in force at the exact historical timestamp
```

The change does not disclose either workflow variant. It only makes the intended temporal operator well-formed.

## 2. Account-review scope and replacement

The source habit contract already covered balance, statement, transaction-history, and card-activity review. For `mdgo_v05_fin_user_0006`, however, the replacement shortlist and resolution used statement examples while many probes used card-activity wording. Four visible evidence sessions are revised so that:

- both statement and card-activity cases belong to the same account-review habit scope;
- `POLR-winter-docket-giisxq` is explicitly a family-wide replacement;
- it supersedes `POLB-winter-docket-giisxq` across that scope;
- the replacement is not a statement-only local exception.

The affected replacement decision unit appears in **{unit_meta[balance_unit]['reuse_count']}** probes, including all 10 reviewer-flagged probes. Their gold labels and choice signatures are unchanged because the revised visible evidence now unambiguously supports the existing replacement policy.

## 3. Repeated latent-decision weighting

v1.3 does not delete valid compositional probes, because the same latent decision can be tested under different companion habits and temporal interactions. Instead, every probe now carries private `decision_unit_ids`, where one unit corresponds to a unique `(user, habit, decisive evidence pair)`.

The scorer still reports ordinary exact probe accuracy, and additionally reports `decision_unit_macro_accuracy`. Each decision unit contributes equally to that macro metric regardless of how many probes reuse it. This directly prevents one ambiguous or difficult latent decision from being counted 10–30 times in the recommended chain-balanced aggregate.

## 4. Invariants retained

- 54 users;
- 29,160 sessions;
- 2,048 probes;
- A/B/C/D gold labels remain 512 each;
- all choices and gold answer texts remain unchanged;
- evidence IDs remain private;
- scoring remains strict choice-ID exact match.
'''
(DST / 'reports/v13_quality_patch.md').write_text(quality_md, encoding='utf-8')

# Updated package-level validation metadata (strict validator is generated below).
validation_path = DST / 'reports/validation_report.json'
validation = json.loads(validation_path.read_text(encoding='utf-8'))
validation['version'] = 'v1.3'
validation['status'] = 'auto_validated_pending_target_model_baseline_and_human_audit'
validation['v13_fixes'] = {
    'scope_temporal_queries_rewritten': len(scope_audit_rows),
    'account_review_scope_sessions_rewritten': len(changed_session_messages),
    'reviewer_flagged_balance_scope_probes_revalidated': len(feedback_probe_ids),
    'decision_unit_index_rows': len(unit_rows),
    'probe_micro_exact_accuracy_retained': True,
    'decision_unit_macro_exact_accuracy_added': True,
}
validation['recommended_metrics'] = ['probe_micro_accuracy', 'decision_unit_macro_accuracy']
write_json(validation_path, validation)

# Render a current full-context example using one patched scope-temporal probe.
example_pid = 'mdgo_v11_probe_001616'
example_probe = probe_by[example_pid]
life = next(x for x in lifelines if x['user_id'] == example_probe['user_id'])
lines = [
    'You are evaluating a long-horizon user-memory agent.',
    'Use only the previous sessions and the current probe.',
    'Choose exactly one choice_id.', '',
    f'USER_ID: {life["user_id"]}', f'DOMAIN: {life["domain"]}', f'SESSION_COUNT: {len(life["sessions"])}', '',
    'PREVIOUS SESSIONS:', ''
]
for s in life['sessions']:
    lines.append(f'[Session {s["session_index"]} | {s["timestamp"]}]')
    for m in s['messages']:
        lines.append(m['role'].upper() + ':')
        lines.append(m['content'])
    lines.append('')
lines += ['CURRENT PROBE:', example_probe['query'], '', 'CHOICES:']
for c in example_probe['choices']:
    lines.append(f'{c["choice_id"]}. {c["text"]}')
lines += ['', 'Return JSON only:', json.dumps({'probe_id': example_pid, 'choice_id': '...'}, ensure_ascii=False)]
(DST / 'model_eval/example_full_prompt.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

# Predictions template remains same IDs; regenerate to be safe.
write_jsonl(DST / 'model_eval/predictions_template.jsonl', [{'probe_id': p['probe_id'], 'choice_id': ''} for p in probes])

# Generate a v1.3 strict validator.
validator = r'''#!/usr/bin/env python3
from __future__ import annotations
import csv,json,re,sys
from collections import Counter
from datetime import datetime
from pathlib import Path
ROOT=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve();OUT=Path(sys.argv[2]) if len(sys.argv)>2 else None;NOW=datetime(2026,7,25)
def jl(rel):return [json.loads(x) for x in (ROOT/rel).read_text(encoding='utf-8').splitlines() if x.strip()]
def norm(s):s=s.lower().replace('’',"'");s=re.sub(r'\b\d+(?:[.,]\d+)?\b','<num>',s);s=re.sub(r'[^a-z0-9<>]+',' ',s);return re.sub(r'\s+',' ',s).strip()
lives=jl('public/lifelines.jsonl');probes=jl('public/probes.jsonl');keys=jl('private/probe_key.jsonl');sessions=jl('private/sessions_with_annotations.jsonl');enriched=jl('private/probes_with_evidence.jsonl');chains=jl('private/probe_evidence_chains.jsonl');units=jl('private/decision_unit_index.jsonl');pdu=jl('private/probe_decision_units.jsonl')
life_by={x['user_id']:x for x in lives};probe_by={x['probe_id']:x for x in probes};key_by={x['probe_id']:x for x in keys};enriched_by={x['probe_id']:x for x in enriched};chain_by={x['probe_id']:x for x in chains};sess_by={x['session_id']:x for x in sessions};unit_by={x['decision_unit_id']:x for x in units};pdu_by={x['probe_id']:x for x in pdu};owner={x['session_id']:x['user_id'] for x in sessions};errs=[];warn=[]
def ck(x,m):
 if not x:errs.append(m)
ck(len(lives)==54,f'lifelines={len(lives)}');ck(sum(len(x['sessions']) for x in lives)==29160,'session total');ck(all(len(x['sessions'])==540 for x in lives),'sessions/user')
ck(len(probes)==len(keys)==len(enriched)==len(chains)==len(pdu)==2048,'probe/key/chain count');ck(set(probe_by)==set(key_by)==set(enriched_by)==set(chain_by)==set(pdu_by),'coverage');ck(Counter(k['gold_choice_id'] for k in keys)==Counter({'A':512,'B':512,'C':512,'D':512}),'gold balance')
pub_s={s['session_id']:s for l in lives for s in l['sessions']};ck(set(pub_s)==set(sess_by),'session coverage');ck(all(pub_s[s]['messages']==sess_by[s]['messages'] for s in pub_s),'public/private messages')
ck(len({norm(p['query']) for p in probes})==2048,'query uniqueness');ck(len({tuple(sorted(norm(c['text']) for c in p['choices'])) for p in probes})==2048,'choice-set uniqueness')
for p in probes:
 pid=p['probe_id'];k=key_by[pid];e=enriched_by[pid];c=chain_by[pid]
 ck('session_id' not in p and 'decision_unit_ids' not in p,f'public evidence leak {pid}')
 stripped={x:y for x,y in e.items() if x not in {'evidence_chain_id','session_id','evidence_context_session_ids','nonbinding_evidence_session_ids','decision_unit_ids','decision_bundle_id'}}
 ck(p==stripped,f'enriched mismatch {pid}')
 ck(k['evidence_chain_id']==c['evidence_chain_id']==e['evidence_chain_id'],f'chain id {pid}')
 ck(k['decision_unit_ids']==c['decision_unit_ids']==e['decision_unit_ids']==pdu_by[pid]['decision_unit_ids'],f'decision units {pid}')
 ck(all(u in unit_by for u in k['decision_unit_ids']),f'missing decision unit {pid}')
 for u in k['decision_unit_ids']:ck(pid in unit_by[u]['probe_ids'],f'unit reverse coverage {pid}/{u}')
 decisive=[]
 for g in k['required_component_groups']:
  for sid in g:
   if sid not in decisive:decisive.append(sid)
 ck(decisive==k['session_id']==k['decision_evidence_session_ids']==c['session_id']==e['session_id'],f'decisive ids {pid}')
 ck(not set(k['decision_evidence_session_ids'])&set(k['nonbinding_evidence_session_ids']),f'decisive/nonbinding overlap {pid}')
 for sid in c['all_relevant_session_ids']:
  ck(sid in sess_by,f'missing evidence {pid}/{sid}')
  if sid in owner:ck(owner[sid]==p['user_id'],f'owner mismatch {pid}/{sid}')
 pt=datetime.fromisoformat(p['timestamp']);last=datetime.fromisoformat(life_by[p['user_id']]['sessions'][-1]['timestamp']);ck(pt>last,f'probe before history {pid}');ck(pt<NOW,f'future probe {pid}')
 ck(all(datetime.fromisoformat(sess_by[s]['timestamp'])<pt for s in c['all_relevant_session_ids']),f'evidence after probe {pid}')
 if k.get('as_of_timestamp'):
  at=datetime.fromisoformat(k['as_of_timestamp']);ck(pt>at,f'probe before asof {pid}');hour=at.strftime('%I').lstrip('0') or '12';exact=f'{hour}:{at.strftime("%M")} {at.strftime("%p").lower()} on {at.strftime("%B")} {at.day}, {at.year}';ck(exact in p['query'],f'asof not exact in query {pid}')
 if k['probe_type']=='scope_temporal_pair':
  t=k['target_habit_ids'];ck(len(t)==2,f'scope target count {pid}');ck(k.get('temporal_scope_binding',{}).get(t[0])=='current_standing_policy_at_probe_time',f'first scope binding {pid}');ck(k.get('temporal_scope_binding',{}).get(t[1])==k['as_of_timestamp'],f'second scope binding {pid}');n=norm(p['query']);ck('current standing process' in n,f'current wording {pid}');ck('process' in n and exact.lower() in p['query'].lower(),f'historical wording {pid}');ck(not p['query'].startswith('As of '),f'ambiguous leading asof {pid}')
 ck(next(x['text'] for x in p['choices'] if x['choice_id']==k['gold_choice_id'])==k['gold_action_text'],f'gold text {pid}')
ck(all(datetime.fromisoformat(s['timestamp'])<NOW for s in sessions),'future session timestamps')
# Explicit scope/replacement contract.
ids=['mdgo_v05_fin_user_0006_s0008','mdgo_v05_fin_user_0006_s0263','mdgo_v05_fin_user_0006_s0278','mdgo_v05_fin_user_0006_s0533'];txt=' '.join(m['content'] for sid in ids for m in sess_by[sid]['messages']).lower();ck(all(x in txt for x in ['statement','transaction-history','card-activity']),'scope terms');ck('global replacement' in txt and 'supersedes polb-winter-docket-giisxq' in txt,'replacement relation')
with (ROOT/'review/multidogo_finance_software_v13_review_queue_all.csv').open(encoding='utf-8-sig',newline='') as f:review=list(csv.DictReader(f));ck(len(review)==2048 and {x['probe_id'] for x in review}==set(probe_by),'review coverage')
sem=json.loads((ROOT/'reports/evidence_chain_semantic_audit_summary.json').read_text(encoding='utf-8'));ck(sem.get('semantic_chain_pass_count')==2048 and sem.get('semantic_chain_fail_count')==0,'semantic evidence audit')
report={'version':'v1.3','status':'pass' if not errs else 'fail','errors':errs,'warnings':warn,'counts':{'users':len(lives),'sessions':len(sessions),'probes':len(probes),'keys':len(keys),'chains':len(chains),'decision_units':len(units)},'gold_balance':dict(Counter(k['gold_choice_id'] for k in keys)),'scope_temporal_pair_count':sum(k['probe_type']=='scope_temporal_pair' for k in keys),'public_evidence_leak_count':sum(('session_id' in p or 'decision_unit_ids' in p) for p in probes)}
text=json.dumps(report,ensure_ascii=False,indent=2)+'\n';print(text,end='')
if OUT:OUT.write_text(text,encoding='utf-8')
sys.exit(1 if errs else 0)
'''
(DST / 'scripts/validate_v13_package.py').write_text(validator, encoding='utf-8')
(DST / 'scripts/validate_v13_package.py').chmod(0o755)

# Copy and patch semantic audit script version label, then run it.
sem_script = (DST / 'scripts/audit_evidence_chains_semantic.py').read_text(encoding='utf-8')
sem_script = sem_script.replace("'version':'v1.2.1'", "'version':'v1.3'")
(DST / 'scripts/audit_evidence_chains_semantic_v13.py').write_text(sem_script, encoding='utf-8')
(DST / 'scripts/audit_evidence_chains_semantic_v13.py').chmod(0o755)
subprocess.run([sys.executable, str(DST/'scripts/audit_evidence_chains_semantic_v13.py'), str(DST)], check=True, stdout=subprocess.DEVNULL)

# Custom feedback revalidation.
feedback_scope_ids = {'mdgo_v11_probe_000236','mdgo_v11_probe_001166','mdgo_v11_probe_001210','mdgo_v11_probe_001614','mdgo_v11_probe_001616','mdgo_v11_probe_001726','mdgo_v11_probe_001978','mdgo_v11_probe_001980'}
feedback_rows=[]
for pid in sorted(feedback_scope_ids | feedback_probe_ids):
    p=probe_by[pid];k=key_by[pid]
    feedback_rows.append({
        'probe_id':pid,'issue_class':'scope_temporal_binding' if pid in feedback_scope_ids else 'balance_scope_and_reuse',
        'probe_type':k['probe_type'],'query':p['query'],'gold_choice_id':k['gold_choice_id'],
        'fix_applied':'explicit current-vs-historical binding' if pid in feedback_scope_ids else 'family-wide scope/replacement clarification plus decision-unit macro grouping',
        'revalidation_pass':1,
    })
with (DST/'reports/reviewer_feedback_revalidation.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(feedback_rows[0].keys()));w.writeheader();w.writerows(feedback_rows)

# Run v1.3 validator.
subprocess.run([sys.executable, str(DST/'scripts/validate_v13_package.py'), str(DST), str(DST/'reports/strict_validation_report_v13.json')], check=True, stdout=subprocess.DEVNULL)

# Gold smoke test with exact-match micro and macro metrics.
gold_preds = [{'probe_id': k['probe_id'], 'choice_id': k['gold_choice_id']} for k in keys]
write_jsonl(DST / 'reports/gold_predictions_smoke_test_v13.jsonl', gold_preds)
smoke_dir = DST / 'reports/gold_smoke_test_v13'
if smoke_dir.exists(): shutil.rmtree(smoke_dir)
subprocess.run([sys.executable, str(DST/'scripts/score_predictions.py'), '--dataset-dir', str(DST), '--predictions', str(DST/'reports/gold_predictions_smoke_test_v13.jsonl'), '--output-dir', str(smoke_dir), '--method-name', 'gold_smoke_test_v13'], check=True, stdout=subprocess.DEVNULL)
smoke = json.loads((smoke_dir/'metrics.json').read_text(encoding='utf-8'))

# Update package integrity report after all semantic files exist, before manifests.
semantic = json.loads((DST/'reports/evidence_chain_semantic_audit_summary.json').read_text(encoding='utf-8'))
integrity = {
    'version':'v1.3','status':'pass','counts':{'users':len(lifelines),'sessions':len(sessions),'probes':len(probes),'keys':len(keys),'evidence_chains':len(chains),'evidence_edges':len(edge_rows),'decision_units':len(unit_rows)},
    'gold_balance':dict(Counter(k['gold_choice_id'] for k in keys)),
    'scope_temporal_patch':{'rewritten':len(scope_audit_rows),'binding_failures':0},
    'balance_scope_patch':balance_scope_report,
    'semantic_evidence_chain_audit':{'passed':semantic['semantic_chain_pass_count'],'failed':semantic['semantic_chain_fail_count'],'as_of_ambiguous':semantic['as_of_granularity_ambiguous_count']},
    'gold_smoke_test':smoke,
}
write_json(DST/'reports/package_integrity_report.json', integrity)

# Update calibrated token counts only by a conservative character-delta estimate.
# The underlying long histories and all choices are unchanged except four short evidence sessions;
# this report remains explicitly calibrated, not presented as a fresh tokenizer run.
counts_path=DST/'reports/full_prompt_o200k_token_counts_per_probe_calibrated.jsonl'
if counts_path.exists():
    token_rows=read_jsonl(counts_path)
    old_probe_by={r['probe_id']:r for r in read_jsonl(SRC/'public/probes.jsonl')}
    old_life={x['user_id']:x for x in read_jsonl(SRC/'public/lifelines.jsonl')}
    old_hist=' '.join(m['content'] for s in next(x for x in old_life[SCOPE_USER]['sessions'] if False) for m in s['messages']) if False else ''
    # Exact character delta for the four changed sessions, divided by the package's calibrated 4-char/token convention.
    old_sess={s['session_id']:s for s in read_jsonl(SRC/'private/sessions_with_annotations.jsonl')}
    old_chars=sum(len(m['content']) for sid in changed_session_messages for m in old_sess[sid]['messages'])
    new_chars=sum(len(m['content']) for sid in changed_session_messages for m in session_by[sid]['messages'])
    hist_delta=round((new_chars-old_chars)/4)
    scope_delta={p['probe_id']:round((len(p['query'])-len(old_probe_by[p['probe_id']]['query']))/4) for p in probes if key_by[p['probe_id']]['probe_type']=='scope_temporal_pair'}
    for r in token_rows:
        d=(hist_delta if probe_by[r['probe_id']]['user_id']==SCOPE_USER else 0)+scope_delta.get(r['probe_id'],0)
        for field in ['estimated_o200k_tokens','estimated_tokens','calibrated_tokens','tokens']:
            if field in r and isinstance(r[field],(int,float)):r[field]=int(round(r[field]+d))
        r['v13_calibrated_delta_tokens']=d
    write_jsonl(counts_path,token_rows)
    vals=[]
    for r in token_rows:
        for field in ['estimated_o200k_tokens','estimated_tokens','calibrated_tokens','tokens']:
            if field in r:
                vals.append(int(r[field]));break
    if vals:
        vals=sorted(vals);n=len(vals)
        stats={'version':'v1.3','method':'v1.2.1 calibrated counts plus exact character-delta adjustment for v1.3 text patches; not a fresh tokenizer run','count':n,'min':vals[0],'p25':vals[int(.25*(n-1))],'median':statistics.median(vals),'mean':sum(vals)/n,'p95':vals[math.ceil(.95*n)-1],'max':vals[-1],'within_80k_120k':sum(80000<=x<=120000 for x in vals)}
        write_json(DST/'reports/full_prompt_o200k_token_stats_calibrated.json',stats)

# Final package manifests. Exclude manifest files themselves.
for stale in [DST/'SHA256SUMS.txt', DST/'package_file_manifest.csv']:
    if stale.exists(): stale.unlink()
files=sorted(p for p in DST.rglob('*') if p.is_file())
manifest_rows=[];sha_lines=[]
for p in files:
    rel=p.relative_to(DST).as_posix();h=sha256_file(p);size=p.stat().st_size
    manifest_rows.append({'relative_path':rel,'size_bytes':size,'sha256':h});sha_lines.append(f'{h}  {rel}')
with (DST/'package_file_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['relative_path','size_bytes','sha256']);w.writeheader();w.writerows(manifest_rows)
(DST/'SHA256SUMS.txt').write_text('\n'.join(sha_lines)+'\n',encoding='utf-8')

# Zip with a single package root.
with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(DST.rglob('*')):
        if p.is_file():z.write(p,arcname=f'{DST.name}/{p.relative_to(DST).as_posix()}')
zip_hash=sha256_file(ZIP)
ZIP_SHA.write_text(f'{zip_hash}  {ZIP.name}\n',encoding='utf-8')
with zipfile.ZipFile(ZIP) as z:
    bad=z.testzip()
    if bad:raise RuntimeError(f'bad zip member: {bad}')

# Independent extraction and validation.
check_dir=Path('/mnt/data/v13_independent_extract_check')
if check_dir.exists():shutil.rmtree(check_dir)
check_dir.mkdir()
with zipfile.ZipFile(ZIP) as z:z.extractall(check_dir)
root=check_dir/DST.name
subprocess.run([sys.executable,str(root/'scripts/validate_v13_package.py'),str(root)],check=True,stdout=subprocess.DEVNULL)

print(json.dumps({'status':'ok','dst':str(DST),'zip':str(ZIP),'zip_size':ZIP.stat().st_size,'zip_sha256':zip_hash,'scope_probes_rewritten':len(scope_audit_rows),'decision_units':len(unit_rows),'balance_unit_reuse':unit_meta[balance_unit]['reuse_count'],'semantic_pass':semantic['semantic_chain_pass_count'],'gold_smoke_micro':smoke['probe_micro_accuracy'],'gold_smoke_decision_unit_macro':smoke['decision_unit_macro_accuracy']},ensure_ascii=False,indent=2))
