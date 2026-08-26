#!/usr/bin/env python3
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
with (ROOT/'review/multidogo_finance_software_v14_review_queue_all.csv').open(encoding='utf-8-sig',newline='') as f:review=list(csv.DictReader(f));ck(len(review)==2048 and {x['probe_id'] for x in review}==set(probe_by),'review coverage')
sem=json.loads((ROOT/'reports/evidence_chain_semantic_audit_summary.json').read_text(encoding='utf-8'));ck(sem.get('semantic_chain_pass_count')==2048 and sem.get('semantic_chain_fail_count')==0,'semantic evidence audit')
report={'version':'v1.3','status':'pass' if not errs else 'fail','errors':errs,'warnings':warn,'counts':{'users':len(lives),'sessions':len(sessions),'probes':len(probes),'keys':len(keys),'chains':len(chains),'decision_units':len(units)},'gold_balance':dict(Counter(k['gold_choice_id'] for k in keys)),'scope_temporal_pair_count':sum(k['probe_type']=='scope_temporal_pair' for k in keys),'public_evidence_leak_count':sum(('session_id' in p or 'decision_unit_ids' in p) for p in probes)}
text=json.dumps(report,ensure_ascii=False,indent=2)+'\n';print(text,end='')
if OUT:OUT.write_text(text,encoding='utf-8')
sys.exit(1 if errs else 0)
