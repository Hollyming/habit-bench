#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

parser = argparse.ArgumentParser(description='Migrate a v1.1 package directory to the v1.2 evidence-chain schema.')
parser.add_argument('--src', required=True, help='Extracted v1.1 package directory')
parser.add_argument('--dst', required=True, help='Output v1.2 package directory')
args = parser.parse_args()
SRC = Path(args.src).resolve()
DST = Path(args.dst).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def stable_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def annotation_role(kind: str | None) -> str:
    return {
        'v11_baseline_candidate': 'baseline_ordered_shortlist',
        'v11_baseline_resolution': 'baseline_ordinal_resolution',
        'v11_replacement_candidate': 'replacement_ordered_shortlist',
        'v11_replacement_resolution': 'replacement_ordinal_resolution',
        'v11_oneoff_decoy': 'one_case_exception_nonbinding',
        'v11_rejected_decoy': 'assistant_suggestion_rejected',
        'v11_reference_case': 'historical_reference_case',
        'v11_pair_rehearsal': 'nondecisive_pair_rehearsal',
        'tentative': 'weak_tentative_signal',
        'support': 'legacy_support',
        'boundary': 'legacy_boundary',
        'exception': 'legacy_exception',
        'weak_support': 'legacy_weak_support',
        'identity_anchor': 'identity_anchor',
        'v11_context_only': 'context_only',
    }.get(kind, kind or 'unannotated_context')


def excerpt_for_session(session: dict[str, Any], limit: int = 560) -> str:
    parts = []
    for message in session.get('messages', []):
        if message.get('role') == 'user':
            parts.append(message.get('content', '').strip())
    text = ' '.join(x for x in parts if x)
    text = ' '.join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + '…'
    return text


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)

# Rename package-level version-specific items while keeping inherited scripts/reports for provenance.
old_review = DST / 'review/multidogo_finance_software_v11_review_queue_all.csv'
new_review = DST / 'review/multidogo_finance_software_v12_review_queue_all.csv'

lifelines = read_jsonl(DST / 'public/lifelines.jsonl')
probes = read_jsonl(DST / 'public/probes.jsonl')
keys = read_jsonl(DST / 'private/probe_key.jsonl')
sessions = read_jsonl(DST / 'private/sessions_with_annotations.jsonl')
profiles = read_jsonl(DST / 'private/persona_profiles.jsonl')

life_by_user = {x['user_id']: x for x in lifelines}
probe_by_id = {x['probe_id']: x for x in probes}
key_by_id = {x['probe_id']: x for x in keys}
session_by_id = {x['session_id']: x for x in sessions}

# 1) Fix temporal inconsistency: every probe timestamp must be strictly later than
#    its full visible history and any requested as-of timestamp.
temporal_rows: list[dict[str, Any]] = []
fixed_temporal = 0
for ordinal, probe in enumerate(probes):
    key = key_by_id[probe['probe_id']]
    old_ts = datetime.fromisoformat(probe['timestamp'])
    last_ts = datetime.fromisoformat(life_by_user[probe['user_id']]['sessions'][-1]['timestamp'])
    asof_raw = key.get('as_of_timestamp')
    asof_ts = datetime.fromisoformat(asof_raw) if asof_raw else None
    floor = last_ts
    if asof_ts and asof_ts > floor:
        floor = asof_ts
    new_ts = old_ts
    if new_ts <= floor:
        # Preserve deterministic variation while making the probe retrospective rather than future-dated.
        new_ts = floor + timedelta(days=1, hours=ordinal % 7, minutes=ordinal % 11)
        probe['timestamp'] = new_ts.isoformat()
        fixed_temporal += 1
    temporal_rows.append({
        'probe_id': probe['probe_id'],
        'user_id': probe['user_id'],
        'probe_type': key['probe_type'],
        'old_probe_timestamp': old_ts.isoformat(),
        'new_probe_timestamp': new_ts.isoformat(),
        'last_visible_session_timestamp': last_ts.isoformat(),
        'as_of_timestamp': asof_raw or '',
        'timestamp_changed': int(new_ts != old_ts),
        'new_probe_after_history': int(new_ts > last_ts),
        'new_probe_after_as_of': int((asof_ts is None) or (new_ts > asof_ts)),
    })

write_jsonl(DST / 'public/probes.jsonl', probes)

# 2) Build role-labelled evidence chains and enrich the private probe schema.
chains: list[dict[str, Any]] = []
enriched_probes: list[dict[str, Any]] = []
edge_rows: list[dict[str, Any]] = []
new_keys: list[dict[str, Any]] = []
role_counts: Counter[str] = Counter()
status_counts: Counter[str] = Counter()
decisive_count_dist: Counter[int] = Counter()
context_count_dist: Counter[int] = Counter()
nonbinding_count_dist: Counter[int] = Counter()
nonbinding_same_as_gold = 0
nonbinding_counter_to_gold = 0

for ordinal, key in enumerate(keys):
    pid = key['probe_id']
    probe = probe_by_id[pid]
    chain_id = f'mdgo_v12_echain_{ordinal:06d}'
    decisive = stable_unique(sid for group in key.get('required_component_groups', []) for sid in group)
    nonbinding = stable_unique(key.get('adversarial_decoy_session_ids', []))
    all_relevant = stable_unique(key.get('gold_evidence_session_ids', []))
    temporal_context = [sid for sid in all_relevant if sid not in set(decisive) and sid not in set(nonbinding)]

    decisive_count_dist[len(decisive)] += 1
    context_count_dist[len(temporal_context)] += 1
    nonbinding_count_dist[len(nonbinding)] += 1

    gold_sig = key['choice_policy_signatures'][key['gold_choice_id']]['variants']
    steps: list[dict[str, Any]] = []
    for sid in sorted(all_relevant, key=lambda x: int(session_by_id[x]['session_index'])):
        session = session_by_id[sid]
        anns = session.get('memory_annotations') or []
        if sid in decisive:
            status = 'decisive'
        elif sid in nonbinding:
            status = 'nonbinding_interference'
        else:
            status = 'temporal_context'
        status_counts[status] += 1

        # Evidence sessions in this construction carry a single principal annotation.
        ann = anns[0] if anns else {}
        kind = ann.get('kind')
        role = annotation_role(kind)
        role_counts[role] += 1
        habit_ids = ann.get('habit_ids', [])
        variant_id = ann.get('variant_id')
        alignment = 'not_applicable'
        if status == 'nonbinding_interference' and variant_id:
            relevant = [h for h in habit_ids if h in gold_sig]
            if relevant:
                if any(gold_sig[h] == variant_id for h in relevant):
                    alignment = 'same_surface_variant_as_gold_but_nonbinding'
                    nonbinding_same_as_gold += 1
                else:
                    alignment = 'counter_variant_to_gold'
                    nonbinding_counter_to_gold += 1

        step = {
            'session_id': sid,
            'session_index': int(session['session_index']),
            'timestamp': session['timestamp'],
            'evidence_status': status,
            'evidence_role': role,
            'annotation_kind': kind,
            'habit_ids': habit_ids,
            'pair_ref': ann.get('pair_ref'),
            'case_ref': ann.get('case_ref'),
            'selected_ordinal': ann.get('selected_ordinal'),
            'variant_id': variant_id,
            'ordered_variants': ann.get('ordered_variants'),
            'evidence_strength': ann.get('evidence_strength'),
            'gold_alignment': alignment,
            'user_excerpt': excerpt_for_session(session),
        }
        steps.append(step)
        edge_rows.append({
            'evidence_chain_id': chain_id,
            'probe_id': pid,
            'user_id': key['user_id'],
            'domain': key['domain'],
            'session_id': sid,
            'session_index': session['session_index'],
            'timestamp': session['timestamp'],
            'evidence_status': status,
            'evidence_role': role,
            'annotation_kind': kind or '',
            'habit_ids_json': json.dumps(habit_ids, ensure_ascii=False),
            'pair_ref': ann.get('pair_ref') or '',
            'case_ref': ann.get('case_ref') or '',
            'selected_ordinal': ann.get('selected_ordinal') if ann.get('selected_ordinal') is not None else '',
            'variant_id': variant_id or '',
            'gold_alignment': alignment,
            'user_excerpt': excerpt_for_session(session),
        })

    chain = {
        'evidence_chain_id': chain_id,
        'probe_id': pid,
        'user_id': key['user_id'],
        'domain': key['domain'],
        # Requested field. It is a list because habit induction is multi-session.
        'session_id': decisive,
        'decision_evidence_session_ids': decisive,
        'temporal_context_session_ids': temporal_context,
        'nonbinding_evidence_session_ids': nonbinding,
        'all_relevant_session_ids': all_relevant,
        'required_component_groups': key.get('required_component_groups', []),
        'target_habit_ids': key.get('target_habit_ids', []),
        'probe_type': key.get('probe_type'),
        'capability_group': key.get('capability_group'),
        'as_of_timestamp': key.get('as_of_timestamp'),
        'reference_session_id': key.get('reference_session_id'),
        'visible_history_scope': key.get('visible_history_scope'),
        'chain_steps': steps,
    }
    chains.append(chain)

    enriched = dict(probe)
    enriched.update({
        'evidence_chain_id': chain_id,
        'session_id': decisive,
        'evidence_context_session_ids': all_relevant,
        'nonbinding_evidence_session_ids': nonbinding,
    })
    enriched_probes.append(enriched)

    new_key = dict(key)
    new_key.update({
        'evidence_chain_id': chain_id,
        # Requested field; decisive source sessions only.
        'session_id': decisive,
        'decision_evidence_session_ids': decisive,
        'temporal_context_session_ids': temporal_context,
        'nonbinding_evidence_session_ids': nonbinding,
        'evidence_context_session_ids': all_relevant,
        'gold_evidence_session_ids_semantics': (
            'Backward-compatible full relevant context. It includes decisive sessions, '
            'temporal context, and explicitly nonbinding interference; use session_id or '
            'decision_evidence_session_ids for the decisive habit-inference chain.'
        ),
    })
    new_keys.append(new_key)

write_jsonl(DST / 'private/probe_key.jsonl', new_keys)
write_jsonl(DST / 'private/probes_with_evidence.jsonl', enriched_probes)
write_jsonl(DST / 'private/probe_evidence_chains.jsonl', chains)

edge_path = DST / 'private/probe_evidence_chain_edges.csv'
edge_path.parent.mkdir(parents=True, exist_ok=True)
edge_fields = [
    'evidence_chain_id', 'probe_id', 'user_id', 'domain', 'session_id', 'session_index',
    'timestamp', 'evidence_status', 'evidence_role', 'annotation_kind', 'habit_ids_json',
    'pair_ref', 'case_ref', 'selected_ordinal', 'variant_id', 'gold_alignment', 'user_excerpt'
]
with edge_path.open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=edge_fields)
    w.writeheader()
    w.writerows(edge_rows)

# 3) Rebuild review queue with explicit evidence roles.
with old_review.open(encoding='utf-8-sig', newline='') as f:
    old_rows = list(csv.DictReader(f))
old_by_id = {r['probe_id']: r for r in old_rows}
chain_by_probe = {c['probe_id']: c for c in chains}
review_fields = [
    'probe_id', 'user_id', 'domain', 'probe_type', 'capability_group', 'query', 'choices_json',
    'proposed_gold_choice_id', 'target_habit_ids_json', 'evidence_requirement',
    'evidence_chain_id', 'session_id_json', 'temporal_context_session_ids_json',
    'nonbinding_evidence_session_ids_json', 'evidence_chain_preview_json',
    'evidence_span_sessions', 'reviewer_decision', 'reviewer_notes'
]
review_rows = []
for probe in probes:
    pid = probe['probe_id']
    key = next(k for k in new_keys if k['probe_id'] == pid)
    chain = chain_by_probe[pid]
    preview = [
        {
            'session_id': s['session_id'],
            'session_index': s['session_index'],
            'timestamp': s['timestamp'],
            'evidence_status': s['evidence_status'],
            'evidence_role': s['evidence_role'],
            'habit_ids': s['habit_ids'],
            'pair_ref': s['pair_ref'],
            'case_ref': s['case_ref'],
            'selected_ordinal': s['selected_ordinal'],
            'variant_id': s['variant_id'],
            'user_excerpt': s['user_excerpt'],
        }
        for s in chain['chain_steps']
    ]
    old = old_by_id[pid]
    review_rows.append({
        'probe_id': pid,
        'user_id': probe['user_id'],
        'domain': probe['domain'],
        'probe_type': key['probe_type'],
        'capability_group': key['capability_group'],
        'query': probe['query'],
        'choices_json': json.dumps(probe['choices'], ensure_ascii=False),
        'proposed_gold_choice_id': key['gold_choice_id'],
        'target_habit_ids_json': json.dumps(key['target_habit_ids'], ensure_ascii=False),
        'evidence_requirement': key['evidence_requirement'],
        'evidence_chain_id': key['evidence_chain_id'],
        'session_id_json': json.dumps(key['session_id'], ensure_ascii=False),
        'temporal_context_session_ids_json': json.dumps(key['temporal_context_session_ids'], ensure_ascii=False),
        'nonbinding_evidence_session_ids_json': json.dumps(key['nonbinding_evidence_session_ids'], ensure_ascii=False),
        'evidence_chain_preview_json': json.dumps(preview, ensure_ascii=False),
        'evidence_span_sessions': key['evidence_span_sessions'],
        'reviewer_decision': old.get('reviewer_decision', ''),
        'reviewer_notes': old.get('reviewer_notes', ''),
    })
with new_review.open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=review_fields)
    w.writeheader()
    w.writerows(review_rows)
old_review.unlink()

# 4) Audit files.
with (DST / 'reports/temporal_consistency_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(temporal_rows[0].keys()))
    w.writeheader()
    w.writerows(temporal_rows)

evidence_audit = {
    'version': 'v1.2',
    'probes': len(probes),
    'chains': len(chains),
    'edge_rows': len(edge_rows),
    'all_probes_have_session_id_field_in_private_enriched_view': all(bool(x['session_id']) for x in enriched_probes),
    'public_probes_remain_evidence_id_free': all('session_id' not in x for x in probes),
    'decision_evidence_count_distribution': dict(sorted(decisive_count_dist.items())),
    'temporal_context_count_distribution': dict(sorted(context_count_dist.items())),
    'nonbinding_evidence_count_distribution': dict(sorted(nonbinding_count_dist.items())),
    'evidence_status_counts': dict(status_counts),
    'evidence_role_counts': dict(role_counts),
    'nonbinding_steps_matching_gold_surface_variant': nonbinding_same_as_gold,
    'nonbinding_steps_using_counter_variant': nonbinding_counter_to_gold,
    'interpretation': (
        'A nonbinding event can happen to use the same workflow as the eventual gold policy; '
        'its nonbinding status comes from provenance/scope, not necessarily from variant disagreement.'
    ),
}
(DST / 'reports/evidence_chain_audit.json').write_text(
    json.dumps(evidence_audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
)

temporal_summary = {
    'version': 'v1.2',
    'probes': len(probes),
    'probe_timestamps_changed': fixed_temporal,
    'all_probe_timestamps_after_visible_history': all(
        datetime.fromisoformat(p['timestamp']) > datetime.fromisoformat(life_by_user[p['user_id']]['sessions'][-1]['timestamp'])
        for p in probes
    ),
    'all_probe_timestamps_after_as_of_timestamp': all(
        (not key_by_id[p['probe_id']].get('as_of_timestamp'))
        or datetime.fromisoformat(p['timestamp']) > datetime.fromisoformat(key_by_id[p['probe_id']]['as_of_timestamp'])
        for p in probes
    ),
}
(DST / 'reports/temporal_consistency_summary.json').write_text(
    json.dumps(temporal_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
)

quality_md = f'''# v1.2 质量复核与小版本修订说明

v1.2 是对 v1.1 的兼容性修订，不重写用户历史、query、choices、gold label 或 habit graph。

## 发现并修复的问题

1. **Probe 时间戳与 as-of 时间不一致**：v1.1 中有 {fixed_temporal} 条 probe 的公开 `timestamp` 早于或等于题目要求的 `as_of_timestamp`。这会让“回看某个时间点”的任务在元数据层面变成询问未来。v1.2 仅把这些 probe 的 timestamp 顺延到 as-of 时间之后；题目文本和答案不变。
2. **证据字段语义含混**：v1.1 的 `gold_evidence_session_ids` 同时包含决定性 shortlist/resolution、时间上下文和明确非绑定的 local exception / assistant suggestion。v1.2 保留该字段用于向后兼容，同时新增角色明确的字段和独立证据链文件。
3. **文档统计不一致**：v1.1 `RELEASE_NOTES.md` 写成 4,816 条重写 session，实际 validation 结果是 **4,797**。v1.2 统一为实际值。

## 证据链设计

- `private/probes_with_evidence.jsonl`：private enriched probe；每条 probe 新增 `session_id` 列表。
- `private/probe_evidence_chains.jsonl`：一条 probe 对应一条完整证据链。
- `private/probe_evidence_chain_edges.csv`：一行表示一条 probe–session 证据边，方便表格审核。
- `session_id` / `decision_evidence_session_ids`：真正决定 habit policy 的跨 session 证据。
- `temporal_context_session_ids`：用于理解 old/new 状态，但不属于当前答案的最小决定链。
- `nonbinding_evidence_session_ids`：一次性例外或未被用户采纳的 assistant suggestion。

## 防止评测泄漏

模型可见的 `public/probes.jsonl` **不暴露 session IDs**。需要证据链的审计、训练或可解释性实验应读取 private enriched probe / evidence-chain 文件。这样既满足证据溯源要求，也不会把检索答案直接交给被测模型。
'''
(DST / 'reports/v12_quality_audit.md').write_text(quality_md, encoding='utf-8')

# 5) Update README and release notes.
readme = f'''# HABIT-Bench MultiDoGO Finance & Software v1.2

v1.2 是 v1.1 的小版本质量修订。核心 benchmark 内容保持稳定：54 个 coherent pseudo-users、每人 540 个 sessions、29,160 个 sessions、2,048 个 probes、相同 query / choices / gold / habit graph。主要新增 role-labelled evidence chain，并修复 probe 元数据中的时间顺序问题。

## 1. 数据规模

| 项目 | v1.2 |
|---|---:|
| Pseudo-users | 54 |
| Finance users | 36 |
| Software users | 18 |
| Sessions / user | 540 |
| 总 sessions | 29,160 |
| 新写或重写的关键 evidence sessions | 4,797 |
| Probes | 2,048 |
| Finance probes | 1,368 |
| Software probes | 680 |
| Gold A / B / C / D | 各 512 |
| Evidence chains | 2,048 |

完整输入长度沿用 v1.1 的校准估计，约 85.7k–92.3k tokens；v1.2 没有修改模型可见的历史、query 或 choices 文本。

## 2. v1.2 相对 v1.1 的变化

1. 修复 {fixed_temporal} 条 probe 的时间元数据，使所有 probe timestamp 均严格晚于完整可见历史和 `as_of_timestamp`。
2. 为每条 probe 建立独立证据链，并在 private enriched probe 中新增 `session_id` 字段。该字段是列表，因为一个 habit policy 需要多个 session 联合推断。
3. 将证据明确拆分为：
   - `decision_evidence_session_ids`：决定性 shortlist / resolution / reference evidence；
   - `temporal_context_session_ids`：相关的 old/new 时间上下文；
   - `nonbinding_evidence_session_ids`：一次性例外和未被采纳的 assistant suggestion。
4. 修正 v1.1 release note 中 4,816 / 4,797 的统计不一致。
5. 新增 v1.2 validator、时间一致性报告和 evidence-chain audit。

## 3. 为什么 public probe 不直接公开 session_id

若在 `public/probes.jsonl` 中直接给出 evidence session IDs，会把需要检索的历史位置泄露给被测 memory agent。因此：

```text
public/probes.jsonl
  评测模型可见；不含 evidence IDs

private/probes_with_evidence.jsonl
  与 public probe 同内容，但新增 session_id 与 evidence_chain_id

private/probe_evidence_chains.jsonl
  完整的角色化证据链
```

这既满足证据溯源要求，也保持 benchmark 的检索与归纳难度。

## 4. 证据链字段

每条 private enriched probe 包含：

```json
{{
  "probe_id": "mdgo_v11_probe_000000",
  "session_id": ["...shortlist...", "...resolution..."],
  "evidence_chain_id": "mdgo_v12_echain_000000",
  "evidence_context_session_ids": ["..."],
  "nonbinding_evidence_session_ids": ["..."]
}}
```

`private/probe_evidence_chains.jsonl` 还会为每个 session 标注：

```text
evidence_status: decisive / temporal_context / nonbinding_interference
evidence_role: baseline shortlist / ordinal resolution / replacement / local exception / rejected suggestion / reference case
habit_ids
pair_ref / case_ref
selected_ordinal / variant_id
user_excerpt
```

## 5. Probe 类型

| Probe type | 数量 |
|---|---:|
| `triple_asof_interleaved` | 512 |
| `dual_asof_reversal` | 384 |
| `surface_decoy_pair` | 384 |
| `reference_case_reconstruction` | 320 |
| `suggestion_rejection_pair` | 256 |
| `provenance_weighted_triple` | 128 |
| `scope_temporal_pair` | 64 |

## 6. 模型评测输入

模型只能读取：

```text
public/lifelines.jsonl
public/probes.jsonl
```

预测格式：

```json
{{"probe_id":"mdgo_v11_probe_000000","choice_id":"A"}}
```

评分：

```bash
python scripts/score_predictions.py \
  --dataset-dir . \
  --predictions predictions.jsonl \
  --output-dir runs/my_v12_eval \
  --method-name my_method
```

只有 `predicted_choice_id == gold_choice_id` 才得 1 分；不采用相似度和部分分。

## 7. 文件结构

```text
public/
  lifelines.jsonl
  probes.jsonl

private/
  probe_key.jsonl
  probes_with_evidence.jsonl
  probe_evidence_chains.jsonl
  probe_evidence_chain_edges.csv
  sessions_with_annotations.jsonl
  persona_profiles.jsonl

review/
  multidogo_finance_software_v12_review_queue_all.csv

reports/
  validation_report.json
  strict_validation_report_v12.json
  temporal_consistency_audit.csv
  temporal_consistency_summary.json
  evidence_chain_audit.json
  v12_quality_audit.md
  以及继承自 v1.1 的难度、窗口、重复率与长度审计

scripts/
  migrate_v11_to_v12.py
  validate_v12_package.py
  score_predictions.py
  以及 v1.1 构建/审计脚本
```

## 8. 当前状态

```text
auto_validated_pending_target_model_baseline_and_human_audit
```

v1.2 不声称改变 v1.1 的目标模型难度。它修复的是时间元数据和 evidence provenance schema，并保持 public benchmark task 基本不变。
'''
(DST / 'README.md').write_text(readme, encoding='utf-8')

release = f'''# v1.2 Release Notes

- Retained the complete v1.1 benchmark task: 54 users, 29,160 sessions, 2,048 probes, identical queries, choices, gold labels and habit graph.
- Corrected {fixed_temporal} probe timestamps that were not later than their requested as-of dates.
- Added `session_id` to a private enriched probe view; the value is a list of decisive evidence sessions.
- Added one role-labelled evidence-chain JSONL row per probe and one flattened CSV row per probe–session evidence edge.
- Separated decisive evidence, temporal context and explicitly nonbinding interference.
- Kept `public/probes.jsonl` free of evidence IDs to prevent retrieval leakage.
- Corrected the rewritten-session count from 4,816 in the v1.1 release note to the validated value of 4,797.
- Scoring remains strict `choice_id` exact match.
'''
(DST / 'RELEASE_NOTES.md').write_text(release, encoding='utf-8')

# Copy the reproducible migration script into the package.
shutil.copy2(Path(__file__), DST / 'scripts/migrate_v11_to_v12.py')

# 6) Build a strict v1.2 validator.
validator = r'''#!/usr/bin/env python3
from __future__ import annotations
import csv, json, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT=Path(sys.argv[1] if len(sys.argv)>1 else '.')
def jl(rel): return [json.loads(x) for x in (ROOT/rel).read_text(encoding='utf-8').splitlines() if x.strip()]

lives=jl('public/lifelines.jsonl');probes=jl('public/probes.jsonl');keys=jl('private/probe_key.jsonl')
sessions=jl('private/sessions_with_annotations.jsonl');enriched=jl('private/probes_with_evidence.jsonl');chains=jl('private/probe_evidence_chains.jsonl')
life_by_user={x['user_id']:x for x in lives};probe_by_id={x['probe_id']:x for x in probes};key_by_id={x['probe_id']:x for x in keys};enriched_by_id={x['probe_id']:x for x in enriched};chain_by_id={x['probe_id']:x for x in chains};session_by_id={x['session_id']:x for x in sessions};owner={x['session_id']:x['user_id'] for x in sessions}
errs=[];warnings=[]
def check(cond,msg):
    if not cond: errs.append(msg)

check(len(lives)==54,f'lifelines={len(lives)}')
check(sum(len(x['sessions']) for x in lives)==29160,'session total')
check(all(len(x['sessions'])==540 for x in lives),'sessions/user')
check(len(probes)==len(keys)==len(enriched)==len(chains)==2048,'probe/key/chain counts')
check(set(probe_by_id)==set(key_by_id)==set(enriched_by_id)==set(chain_by_id),'coverage')
check(Counter(k['gold_choice_id'] for k in keys)==Counter({'A':512,'B':512,'C':512,'D':512}),'gold balance')

pub_sessions={s['session_id']:s for l in lives for s in l['sessions']}
check(set(pub_sessions)==set(session_by_id),'public/private session coverage')
check(all(pub_sessions[s]['messages']==session_by_id[s]['messages'] for s in pub_sessions),'public/private message equality')

for p in probes:
    pid=p['probe_id'];k=key_by_id[pid];e=enriched_by_id[pid];c=chain_by_id[pid]
    check('session_id' not in p,f'public evidence leak {pid}')
    check(p=={x:y for x,y in e.items() if x not in {'evidence_chain_id','session_id','evidence_context_session_ids','nonbinding_evidence_session_ids'}},f'enriched/public mismatch {pid}')
    check(e['evidence_chain_id']==k['evidence_chain_id']==c['evidence_chain_id'],f'chain id mismatch {pid}')
    decisive=[]
    for grp in k['required_component_groups']:
        for sid in grp:
            if sid not in decisive:decisive.append(sid)
    check(e['session_id']==k['session_id']==k['decision_evidence_session_ids']==c['session_id']==decisive,f'decisive ids {pid}')
    check(set(k['nonbinding_evidence_session_ids'])==set(k['adversarial_decoy_session_ids'])==set(c['nonbinding_evidence_session_ids']),f'nonbinding ids {pid}')
    check(set(k['evidence_context_session_ids'])==set(k['gold_evidence_session_ids'])==set(c['all_relevant_session_ids']),f'context ids {pid}')
    check(not set(k['decision_evidence_session_ids']) & set(k['nonbinding_evidence_session_ids']),f'decisive/nonbinding overlap {pid}')
    for sid in c['all_relevant_session_ids']:
        check(sid in session_by_id,f'missing chain session {pid}/{sid}')
        if sid in owner:check(owner[sid]==p['user_id'],f'owner mismatch {pid}/{sid}')
    check(len(c['chain_steps'])==len(c['all_relevant_session_ids']),f'chain step count {pid}')
    check({x['session_id'] for x in c['chain_steps']}==set(c['all_relevant_session_ids']),f'chain step coverage {pid}')
    pt=datetime.fromisoformat(p['timestamp']);last=datetime.fromisoformat(life_by_user[p['user_id']]['sessions'][-1]['timestamp'])
    check(pt>last,f'probe before history end {pid}')
    if k.get('as_of_timestamp'):check(pt>datetime.fromisoformat(k['as_of_timestamp']),f'probe before asof {pid}')
    check(all(datetime.fromisoformat(session_by_id[s]['timestamp'])<pt for s in c['all_relevant_session_ids']),f'evidence after probe {pid}')
    gold_text=next(x['text'] for x in p['choices'] if x['choice_id']==k['gold_choice_id'])
    check(gold_text==k['gold_action_text'],f'gold text {pid}')

review=ROOT/'review/multidogo_finance_software_v12_review_queue_all.csv'
with review.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
check(len(rows)==2048,'review rows')
check({x['probe_id'] for x in rows}==set(probe_by_id),'review coverage')

report={'version':'v1.2','status':'pass' if not errs else 'fail','errors':errs,'warnings':warnings,'counts':{'users':len(lives),'sessions':sum(len(x['sessions']) for x in lives),'probes':len(probes),'keys':len(keys),'evidence_chains':len(chains)},'gold_balance':dict(Counter(k['gold_choice_id'] for k in keys)),'temporal_consistency':{'all_probe_after_history':all(datetime.fromisoformat(p['timestamp'])>datetime.fromisoformat(life_by_user[p['user_id']]['sessions'][-1]['timestamp']) for p in probes),'all_probe_after_asof':all((not key_by_id[p['probe_id']].get('as_of_timestamp')) or datetime.fromisoformat(p['timestamp'])>datetime.fromisoformat(key_by_id[p['probe_id']]['as_of_timestamp']) for p in probes)},'public_evidence_leak_count':sum('session_id' in p for p in probes)}
print(json.dumps(report,ensure_ascii=False,indent=2))
if len(sys.argv)>2:Path(sys.argv[2]).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sys.exit(1 if errs else 0)
'''
(DST / 'scripts/validate_v12_package.py').write_text(validator, encoding='utf-8')
(DST / 'scripts/validate_v12_package.py').chmod(0o755)

# Update the main validation report; keep inherited difficulty audits unchanged.
old_validation = json.loads((DST / 'reports/validation_report.json').read_text(encoding='utf-8'))
old_validation['version'] = 'v1.2'
old_validation['status'] = 'auto_validated_pending_target_model_baseline_and_human_audit'
old_validation['counts']['evidence_chains'] = len(chains)
old_validation['counts']['private_enriched_probes'] = len(enriched_probes)
old_validation['v12_fixes'] = {
    'probe_timestamps_corrected': fixed_temporal,
    'all_probe_timestamps_after_visible_history': temporal_summary['all_probe_timestamps_after_visible_history'],
    'all_probe_timestamps_after_as_of_timestamp': temporal_summary['all_probe_timestamps_after_as_of_timestamp'],
    'role_labelled_evidence_chains': len(chains),
    'public_probe_evidence_ids_exposed': False,
    'validated_rewritten_evidence_session_count': 4797,
}
old_validation['evidence_chain_schema'] = {
    'enriched_probe_file': 'private/probes_with_evidence.jsonl',
    'chain_file': 'private/probe_evidence_chains.jsonl',
    'edge_file': 'private/probe_evidence_chain_edges.csv',
    'requested_session_id_field_location': 'private/probes_with_evidence.jsonl and private/probe_key.jsonl',
    'session_id_semantics': 'decisive multi-session evidence list',
}
(DST / 'reports/validation_report.json').write_text(json.dumps(old_validation, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

# The package manifest and SHA sums are written after the external validator is run.
print(json.dumps({'dst':str(DST),'fixed_temporal':fixed_temporal,'chains':len(chains),'edge_rows':len(edge_rows)},indent=2))
