#!/usr/bin/env python3
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
