#!/usr/bin/env python3
from __future__ import annotations
import csv,json,re,sys
from collections import Counter
from datetime import datetime
from pathlib import Path
ROOT=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
OUT=Path(sys.argv[2]) if len(sys.argv)>2 else None
NOW=datetime(2026,7,25)
def jl(rel):return [json.loads(x) for x in (ROOT/rel).read_text(encoding='utf-8').splitlines() if x.strip()]
def norm(s):
 s=s.lower().replace('’',"'");s=re.sub(r'\b\d+(?:[.,]\d+)?\b','<num>',s);s=re.sub(r'[^a-z0-9<>]+',' ',s);return re.sub(r'\s+',' ',s).strip()
lives=jl('public/lifelines.jsonl');probes=jl('public/probes.jsonl');keys=jl('private/probe_key.jsonl');sessions=jl('private/sessions_with_annotations.jsonl');enriched=jl('private/probes_with_evidence.jsonl');chains=jl('private/probe_evidence_chains.jsonl')
life_by={x['user_id']:x for x in lives};probe_by={x['probe_id']:x for x in probes};key_by={x['probe_id']:x for x in keys};enriched_by={x['probe_id']:x for x in enriched};chain_by={x['probe_id']:x for x in chains};sess_by={x['session_id']:x for x in sessions};owner={x['session_id']:x['user_id'] for x in sessions}
errs=[];warn=[]
def ck(x,m):
 if not x:errs.append(m)
ck(len(lives)==54,f'lifelines={len(lives)}');ck(sum(len(x['sessions']) for x in lives)==29160,'session total');ck(all(len(x['sessions'])==540 for x in lives),'sessions/user')
ck(len(probes)==len(keys)==len(enriched)==len(chains)==2048,'probe/key/chain count');ck(set(probe_by)==set(key_by)==set(enriched_by)==set(chain_by),'coverage');ck(Counter(k['gold_choice_id'] for k in keys)==Counter({'A':512,'B':512,'C':512,'D':512}),'gold balance')
pub_s={s['session_id']:s for l in lives for s in l['sessions']};ck(set(pub_s)==set(sess_by),'session coverage');ck(all(pub_s[s]['messages']==sess_by[s]['messages'] for s in pub_s),'public/private messages')
ck(len({norm(p['query']) for p in probes})==2048,'query uniqueness');ck(len({tuple(sorted(norm(c['text']) for c in p['choices'])) for p in probes})==2048,'choice-set uniqueness')
for p in probes:
 pid=p['probe_id'];k=key_by[pid];e=enriched_by[pid];c=chain_by[pid]
 ck('session_id' not in p,f'public evidence leak {pid}')
 ck(p=={x:y for x,y in e.items() if x not in {'evidence_chain_id','session_id','evidence_context_session_ids','nonbinding_evidence_session_ids'}},f'enriched mismatch {pid}')
 ck(k['evidence_chain_id']==c['evidence_chain_id']==e['evidence_chain_id'],f'chain id {pid}')
 decisive=[]
 for g in k['required_component_groups']:
  for sid in g:
   if sid not in decisive:decisive.append(sid)
 ck(decisive==k['session_id']==k['decision_evidence_session_ids']==c['session_id']==e['session_id'],f'decisive ids {pid}')
 ck(not set(k['decision_evidence_session_ids'])&set(k['nonbinding_evidence_session_ids']),f'decisive/nonbinding overlap {pid}')
 for sid in c['all_relevant_session_ids']:
  ck(sid in sess_by,f'missing evidence {pid}/{sid}')
  if sid in owner:ck(owner[sid]==p['user_id'],f'owner mismatch {pid}/{sid}')
 pt=datetime.fromisoformat(p['timestamp']);last=datetime.fromisoformat(life_by[p['user_id']]['sessions'][-1]['timestamp'])
 ck(pt>last,f'probe before history {pid}');ck(pt<NOW,f'future probe {pid}')
 ck(all(datetime.fromisoformat(sess_by[s]['timestamp'])<pt for s in c['all_relevant_session_ids']),f'evidence after probe {pid}')
 if k.get('as_of_timestamp'):
  at=datetime.fromisoformat(k['as_of_timestamp']);ck(pt>at,f'probe before asof {pid}');hour=at.strftime('%I').lstrip('0') or '12';exact=f'{hour}:{at.strftime("%M")} {at.strftime("%p").lower()} on {at.strftime("%B")} {at.day}, {at.year}';ck(exact in p['query'],f'asof not exact in query {pid}')
 ck(next(x['text'] for x in p['choices'] if x['choice_id']==k['gold_choice_id'])==k['gold_action_text'],f'gold text {pid}')
 ck('V11-' not in p['query'] and all('V11-' not in x['text'] for x in p['choices']),f'old record id {pid}')
ck(all(datetime.fromisoformat(s['timestamp'])<NOW for s in sessions),'future session timestamps')
# Review and semantic audit
with (ROOT/'review/multidogo_finance_software_v121_review_queue_all.csv').open(encoding='utf-8-sig',newline='') as f:review=list(csv.DictReader(f))
ck(len(review)==2048 and {x['probe_id'] for x in review}==set(probe_by),'review coverage')
sem=json.loads((ROOT/'reports/evidence_chain_semantic_audit_summary.json').read_text(encoding='utf-8'))
ck(sem.get('semantic_chain_pass_count')==2048 and sem.get('semantic_chain_fail_count')==0,'semantic evidence audit')
report={'version':'v1.2.1','status':'pass' if not errs else 'fail','errors':errs,'warnings':warn,'counts':{'users':len(lives),'sessions':len(sessions),'probes':len(probes),'keys':len(keys),'chains':len(chains)},'gold_balance':dict(Counter(k['gold_choice_id'] for k in keys)),'timeline':{'session_min':min(s['timestamp'] for s in sessions),'session_max':max(s['timestamp'] for s in sessions),'probe_min':min(p['timestamp'] for p in probes),'probe_max':max(p['timestamp'] for p in probes),'all_before_2026_07_25':all(datetime.fromisoformat(s['timestamp'])<NOW for s in sessions) and all(datetime.fromisoformat(p['timestamp'])<NOW for p in probes)},'evidence_chain_semantic_pass':sem.get('semantic_chain_pass_count'),'public_evidence_leak_count':sum('session_id' in p for p in probes)}
text=json.dumps(report,ensure_ascii=False,indent=2)+'\n';print(text,end='')
if OUT:OUT.write_text(text,encoding='utf-8')
sys.exit(1 if errs else 0)
