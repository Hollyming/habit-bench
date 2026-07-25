#!/usr/bin/env python3
from __future__ import annotations
import csv, json, statistics, sys
from pathlib import Path

ROOT=Path(sys.argv[1] if len(sys.argv)>1 else '.')
BUDGET=int(sys.argv[2]) if len(sys.argv)>2 else 140000

def load_jsonl(rel):
    return [json.loads(x) for x in (ROOT/rel).read_text(encoding='utf-8').splitlines() if x.strip()]

lifelines=load_jsonl('public/lifelines.jsonl')
keys=load_jsonl('private/probe_key.jsonl')
life_by_user={x['user_id']:x for x in lifelines}

def render_session(s):
    lines=[f"[Session {s['session_index']} | {s['timestamp']}]" ]
    for m in s['messages']:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return '\n'.join(lines)+'\n\n'

# Precompute all maximum contiguous windows under BUDGET for each user.
windows={}
position={}
history_chars={}
for uid,life in life_by_user.items():
    ss=life['sessions']
    lens=[len(render_session(s)) for s in ss]
    history_chars[uid]=sum(lens)
    pos={s['session_id']:i for i,s in enumerate(ss)}
    position[uid]=pos
    ends=[]
    e=0; total=0
    for st in range(len(ss)):
        if e<st:
            e=st;total=0
        while e<len(ss) and total+lens[e]<=BUDGET:
            total+=lens[e];e+=1
        ends.append(e-1)
        if e>st:
            total-=lens[st]
    windows[uid]=ends

rows=[]
complete=0
best_missing=[]
best_resolved=[]
for k in keys:
    uid=k['user_id']; pos=position[uid]; groups=k.get('required_component_groups',[])
    group_pos=[[pos[s] for s in g] for g in groups]
    best=-1; best_st=0; best_en=-1
    for st,en in enumerate(windows[uid]):
        if en<st: continue
        resolved=sum(all(st<=i<=en for i in gp) for gp in group_pos)
        if resolved>best:
            best=resolved;best_st=st;best_en=en
    missing=len(groups)-best
    complete += (missing==0)
    best_missing.append(missing);best_resolved.append(best)
    rows.append({
        'probe_id':k['probe_id'],'user_id':uid,'domain':k['domain'],'probe_type':k['probe_type'],
        'required_groups':len(groups),'best_resolved_groups':best,'unresolved_groups_in_best_window':missing,
        'best_window_start_session':best_st,'best_window_end_session':best_en,
        'history_chars':history_chars[uid],'window_char_budget':BUDGET,
    })

out={
    'audit':'contiguous_history_character_window_component_completeness',
    'window_char_budget':BUDGET,
    'reason_for_budget':'Conservative construction proxy for an approximately 34k-token English history slice; the current query and choices are not charged against this budget.',
    'probes':len(keys),'complete_probes':complete,'complete_rate':complete/len(keys),
    'min_unresolved_groups_in_best_window':min(best_missing),
    'median_unresolved_groups_in_best_window':statistics.median(best_missing),
    'max_unresolved_groups_in_best_window':max(best_missing),
    'best_resolved_groups_distribution':{str(x):best_resolved.count(x) for x in sorted(set(best_resolved))},
    'unresolved_groups_distribution':{str(x):best_missing.count(x) for x in sorted(set(best_missing))},
    'history_chars_per_user':{
        'min':min(history_chars.values()),'median':statistics.median(history_chars.values()),
        'mean':statistics.mean(history_chars.values()),'max':max(history_chars.values())
    },
    'interpretation':'Evidence availability audit, not a model score. A policy component is counted as resolved only if both its ordered shortlist and its distant ordinal-resolution session are visible in the same contiguous slice.'
}
rep=ROOT/'reports'
(rep/'truncated_34k_char_window_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
with (rep/'truncated_34k_char_window_per_probe.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(out,ensure_ascii=False,indent=2))
