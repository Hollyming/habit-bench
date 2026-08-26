#!/usr/bin/env python3
from __future__ import annotations
import csv, json, re, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT=Path(sys.argv[1] if len(sys.argv)>1 else '.')

def jl(rel):
    p=ROOT/rel
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def norm(t):
    t=t.lower().replace('’',"'")
    t=re.sub(r'\b\d+(?:[.,]\d+)?\b','<num>',t)
    t=re.sub(r'[^a-z0-9<>]+',' ',t)
    return re.sub(r'\s+',' ',t).strip()

lives=jl('public/lifelines.jsonl');probes=jl('public/probes.jsonl');keys=jl('private/probe_key.jsonl');sessions=jl('private/sessions_with_annotations.jsonl');profiles=jl('private/persona_profiles.jsonl')
habits=json.loads((ROOT/'source/habit_templates_retained.json').read_text(encoding='utf-8'))
life_by_user={x['user_id']:x for x in lives};profile_by_user={x['user_id']:x for x in profiles};probe_by_id={x['probe_id']:x for x in probes};key_by_id={x['probe_id']:x for x in keys};session_by_id={x['session_id']:x for x in sessions};habit_by_id={x['habit_id']:x for x in habits}
sidx={sid:int(s['session_index']) for sid,s in session_by_id.items()};sts={sid:datetime.fromisoformat(s['timestamp']) for sid,s in session_by_id.items()};owner={sid:s['user_id'] for sid,s in session_by_id.items()}
errs=[];warnings=[]
def check(cond,msg):
    if not cond:errs.append(msg)

check(len(lives)==54,f'lifelines={len(lives)}')
check(len(profiles)==54,f'profiles={len(profiles)}')
check(sum(len(x['sessions']) for x in lives)==29160,'session total')
check(all(len(x['sessions'])==540 for x in lives),'sessions/user')
check(len(probes)==2048 and len(keys)==2048,'probe/key counts')
check(Counter(p['domain'] for p in profiles)==Counter({'finance':36,'software':18}),'domain user split')
check(Counter(k['gold_choice_id'] for k in keys)==Counter({'A':512,'B':512,'C':512,'D':512}),'gold balance')
check(set(probe_by_id)==set(key_by_id),'public/private probe coverage')

# Public/private session identity and message equality.
pub_sessions={s['session_id']:s for l in lives for s in l['sessions']}
check(set(pub_sessions)==set(session_by_id),'public/private session coverage')
for sid in set(pub_sessions)&set(session_by_id):
    if pub_sessions[sid]['messages']!=session_by_id[sid]['messages']:
        errs.append(f'public/private message mismatch {sid}');break
for l in lives:
    idx=[int(s['session_index']) for s in l['sessions']]
    check(idx==list(range(540)),f'noncontiguous indices {l["user_id"]}')
    ts=[datetime.fromisoformat(s['timestamp']) for s in l['sessions']]
    check(ts==sorted(ts),f'nonmonotonic timestamps {l["user_id"]}')

# Split-decision graph correctness.
for p in profiles:
    uid=p['user_id'];meta=p.get('v11_challenge_metadata',{}).get('decision_meta',{})
    active=p.get('active_habit_ids',[])
    check(set(meta)==set(active),f'decision meta coverage {uid}')
    for h in active:
        if h not in meta:continue
        m=meta[h];valid={v['variant_id'] for v in habit_by_id[h]['policy_variants']}
        for field in ['baseline_candidate_session_id','baseline_resolution_session_id','oneoff_decoy_session_id','rejected_decoy_session_id']:
            sid=m.get(field);check(sid in session_by_id,f'{uid}/{h} missing {field}');
            if sid in owner:check(owner[sid]==uid,f'{uid}/{h} owner {field}')
        if m.get('baseline_candidate_session_id') in sidx and m.get('baseline_resolution_session_id') in sidx:
            check(sidx[m['baseline_resolution_session_id']]-sidx[m['baseline_candidate_session_id']]>220,f'{uid}/{h} baseline distance')
        pair=m.get('baseline_ordered_variants',[]);check(len(pair)==2 and len(set(pair))==2 and set(pair)<=valid,f'{uid}/{h} baseline pair')
        check(m.get('baseline_variant_id') in pair,f'{uid}/{h} baseline selected not in pair')
        # Annotation and ordinal consistency.
        rs=session_by_id.get(m.get('baseline_resolution_session_id'),{}).get('memory_annotations',[])
        if rs:
            a=rs[0];check(a.get('pair_ref')==m.get('baseline_pair_ref'),f'{uid}/{h} baseline ref mismatch');check(a.get('selected_ordinal')==1+pair.index(m['baseline_variant_id']),f'{uid}/{h} baseline ordinal')
        if m.get('replacement_candidate_session_id'):
            for field in ['replacement_candidate_session_id','replacement_resolution_session_id']:
                sid=m.get(field);check(sid in session_by_id,f'{uid}/{h} missing {field}');
                if sid in owner:check(owner[sid]==uid,f'{uid}/{h} owner {field}')
            if m.get('replacement_candidate_session_id') in sidx and m.get('replacement_resolution_session_id') in sidx:
                check(sidx[m['replacement_resolution_session_id']]-sidx[m['replacement_candidate_session_id']]>220,f'{uid}/{h} replacement distance')
            pair=m.get('replacement_ordered_variants',[]);check(len(pair)==2 and len(set(pair))==2 and set(pair)<=valid,f'{uid}/{h} replacement pair')
            check(m.get('replacement_variant_id')==p['active_policy_variants'][h],f'{uid}/{h} replacement != active')
            check(m.get('replacement_variant_id') in pair,f'{uid}/{h} replacement selected not pair')
            rs=session_by_id.get(m.get('replacement_resolution_session_id'),{}).get('memory_annotations',[])
            if rs:
                a=rs[0];check(a.get('pair_ref')==m.get('replacement_pair_ref'),f'{uid}/{h} replacement ref mismatch');check(a.get('selected_ordinal')==1+pair.index(m['replacement_variant_id']),f'{uid}/{h} replacement ordinal')
        else:
            check(m.get('baseline_variant_id')==p['active_policy_variants'][h],f'{uid}/{h} stable baseline != active')

# Probe semantics and truncated-window structure.
def expected_state(p,h,when):
    m=p['v11_challenge_metadata']['decision_meta'][h]
    if m.get('replacement_resolution_session_id'):
        rt=sts[m['replacement_resolution_session_id']]
        if when is None or when>=rt:return p['active_policy_variants'][h]
    return m['baseline_variant_id']
complete205=0;min_missing=99
for k in keys:
    pid=k['probe_id'];p=probe_by_id.get(pid);uid=k['user_id'];prof=profile_by_user.get(uid)
    if not p or not prof:continue
    check(p['user_id']==uid,f'{pid} user mismatch')
    check(set(c['choice_id'] for c in p['choices'])==set('ABCD') and len(p['choices'])==4,f'{pid} choices')
    check(len({norm(c['text']) for c in p['choices']})==4,f'{pid} duplicate choice')
    check(k['gold_choice_id'] in set(c['choice_id'] for c in p['choices']),f'{pid} missing gold')
    gold_text=next((c['text'] for c in p['choices'] if c['choice_id']==k['gold_choice_id']),None)
    check(gold_text==k['gold_action_text'],f'{pid} gold text')
    for sid in k['gold_evidence_session_ids']:
        check(sid in session_by_id,f'{pid} missing evidence {sid}')
        if sid in owner:check(owner[sid]==uid,f'{pid} evidence owner {sid}')
    groups=k.get('required_component_groups',[]);check(len(groups)>=2,f'{pid} too few groups')
    for grp in groups:
        check(bool(grp),f'{pid} empty group')
        for sid in grp:
            check(sid in session_by_id,f'{pid} component missing {sid}')
            if sid in owner:check(owner[sid]==uid,f'{pid} component owner {sid}')
        if len(grp)==2 and all(s in sidx for s in grp):check(abs(sidx[grp[1]]-sidx[grp[0]])>220,f'{pid} split distance {grp}')
    best=0
    for st in range(336):
        en=st+204;resolved=sum(all(st<=sidx[s]<=en for s in grp) for grp in groups)
        best=max(best,resolved)
    missing=len(groups)-best;min_missing=min(min_missing,missing);complete205+=missing==0
    check(missing>=2,f'{pid} <2 missing groups in best 205-window')
    # Gold policy signatures must equal the state implied by the dated split decisions.
    sig=k['choice_policy_signatures'][k['gold_choice_id']]['variants']
    for h in k['target_habit_ids']:
        raw=k.get('target_state_times',{}).get(h);when=datetime.fromisoformat(raw) if raw else None
        check(sig.get(h)==expected_state(prof,h,when),f'{pid} state mismatch {h}: {sig.get(h)}')

check(complete205==0,f'complete 205-window probes={complete205}')
check(min_missing>=2,f'min missing groups={min_missing}')
check(len({norm(p['query']) for p in probes})==2048,'query uniqueness')
check(len({tuple(sorted(norm(c['text']) for c in p['choices'])) for p in probes})==2048,'choice-set uniqueness')
for p in probes:
    forbidden={'gold_choice_id','probe_type','habit_template_id','capability_group','target_habit_ids','evidence_session_ids'}
    check(not forbidden.intersection(p),f'public metadata leak {p["probe_id"]}')

# Review queue coverage.
review_path=ROOT/'review/multidogo_finance_software_v11_review_queue_all.csv'
with review_path.open(encoding='utf-8-sig',newline='') as f:review=list(csv.DictReader(f))
check(len(review)==2048,'review rows')
check({r['probe_id'] for r in review}==set(probe_by_id),'review probe coverage')

report={'version':'v1.1','status':'pass' if not errs else 'fail','errors':errs,'warnings':warnings,'counts':{'users':len(lives),'sessions':sum(len(x['sessions']) for x in lives),'probes':len(probes),'keys':len(keys)},'gold_balance':dict(Counter(k['gold_choice_id'] for k in keys)),'truncated_window':{'window_sessions':205,'complete_probes':complete205,'min_unresolved_groups':min_missing}}
print(json.dumps(report,ensure_ascii=False,indent=2))
sys.exit(1 if errs else 0)
