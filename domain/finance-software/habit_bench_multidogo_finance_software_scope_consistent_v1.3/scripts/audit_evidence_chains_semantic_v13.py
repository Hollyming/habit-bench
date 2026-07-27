from __future__ import annotations
import calendar, csv, json, re, sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

BASE=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
OUT=BASE/'reports'
OUT.mkdir(exist_ok=True)

def jl(rel):
    return [json.loads(x) for x in (BASE/rel).read_text(encoding='utf-8').splitlines() if x.strip()]

def norm(s:str)->str:
    s=s.lower().replace('’',"'")
    s=re.sub(r'[^a-z0-9$.-]+',' ',s)
    return re.sub(r'\s+',' ',s).strip()

def ann0(s):
    a=s.get('memory_annotations') or []
    return a[0] if a else {}

def fmt_date(dt:datetime,style:int)->str:
    # portable %-d replacement
    day=str(dt.day)
    return [f'{dt.strftime("%B")} {day}, {dt.year}', f'{dt.strftime("%b")} {day}, {dt.year}', dt.strftime('%Y-%m-%d'), f'{dt.strftime("%B")} {dt.year}'][style%4]

lives=jl('public/lifelines.jsonl'); probes=jl('public/probes.jsonl'); keys=jl('private/probe_key.jsonl')
sessions=jl('private/sessions_with_annotations.jsonl'); chains=jl('private/probe_evidence_chains.jsonl'); profiles=jl('private/persona_profiles.jsonl')
habits=json.loads((BASE/'source/habit_templates_retained.json').read_text(encoding='utf-8'))
habit_by={h['habit_id']:h for h in habits}; variants={h['habit_id']:{v['variant_id']:v for v in h['policy_variants']} for h in habits}
probe_by={x['probe_id']:x for x in probes}; key_by={x['probe_id']:x for x in keys}; chain_by={x['probe_id']:x for x in chains}; sess_by={x['session_id']:x for x in sessions}; prof_by={x['user_id']:x for x in profiles}; life_by={x['user_id']:x for x in lives}

# Choice realization mapping is packaged for reproducible text/signature validation.
CHOICE=json.loads((BASE/'source/choice_realizations.json').read_text(encoding='utf-8'))

summary=Counter(); per_type=defaultdict(Counter); rows=[]; issue_examples=defaultdict(list)

def add_issue(issues:list[str], code:str, pid:str, detail:str=''):
    msg=code + (': '+detail if detail else '')
    issues.append(msg); summary[code]+=1; per_type[key_by[pid]['probe_type']][code]+=1
    if len(issue_examples[code])<12: issue_examples[code].append({'probe_id':pid,'detail':detail})

# Map all annotation decision records by user/habit for independent expected-state reconstruction.
decisions=defaultdict(lambda:defaultdict(dict))
for s in sessions:
    a=ann0(s); kind=a.get('kind'); hs=a.get('habit_ids',[])
    if len(hs)!=1: continue
    h=hs[0]
    if kind in {'v11_baseline_candidate','v11_baseline_resolution','v11_replacement_candidate','v11_replacement_resolution'}:
        decisions[s['user_id']][h][kind]=s['session_id']

for p in probes:
    pid=p['probe_id']; k=key_by[pid]; c=chain_by[pid]; issues=[]
    uid=p['user_id']; target=k['target_habit_ids']; gold=k['gold_choice_id']; goldsig=k['choice_policy_signatures'][gold]

    # 0. Basic chain/public relationship.
    if c['probe_id']!=pid or c['user_id']!=uid: add_issue(issues,'CHAIN_IDENTITY_MISMATCH',pid)
    if c['session_id']!=k['decision_evidence_session_ids']: add_issue(issues,'DECISIVE_LIST_MISMATCH',pid)
    if set(c['target_habit_ids'])!=set(target): add_issue(issues,'TARGET_HABIT_MISMATCH',pid)
    if len(set(c['session_id']))!=len(c['session_id']): add_issue(issues,'DUPLICATE_DECISIVE_SESSION',pid)

    derived={}; group_habits=[]; reference_anns=[]
    # 1. Validate each required component group as actual evidence.
    for gi,g in enumerate(k['required_component_groups']):
        if len(g)==2:
            if any(sid not in sess_by for sid in g):
                add_issue(issues,'MISSING_GROUP_SESSION',pid,f'group {gi}'); continue
            s1,s2=[sess_by[x] for x in g]; a1,a2=ann0(s1),ann0(s2)
            # tolerate order, but canonical should candidate then resolution by session index.
            pair=sorted([(s1,a1),(s2,a2)],key=lambda z:z[0]['session_index'])
            sc,ac=pair[0]; sr,ar=pair[1]
            ck=ac.get('kind'); rk=ar.get('kind')
            if ck not in {'v11_baseline_candidate','v11_replacement_candidate'} or rk not in {'v11_baseline_resolution','v11_replacement_resolution'}:
                add_issue(issues,'INVALID_PAIR_KINDS',pid,f'{ck}+{rk}')
                continue
            if ck.startswith('v11_baseline') != rk.startswith('v11_baseline'):
                add_issue(issues,'PAIR_PHASE_MISMATCH',pid,f'{ck}+{rk}')
            h1=ac.get('habit_ids',[]); h2=ar.get('habit_ids',[])
            if len(h1)!=1 or h1!=h2:
                add_issue(issues,'PAIR_HABIT_MISMATCH',pid,f'{h1}/{h2}'); continue
            h=h1[0]; group_habits.append(h)
            if h not in target: add_issue(issues,'DECISIVE_GROUP_UNRELATED_HABIT',pid,h)
            if ac.get('pair_ref')!=ar.get('pair_ref'):
                add_issue(issues,'PAIR_REF_MISMATCH',pid,f'{ac.get("pair_ref")}/{ar.get("pair_ref")}')
            ordered=ac.get('ordered_variants') or []
            ordn=ar.get('selected_ordinal'); rv=ar.get('variant_id')
            if len(ordered)!=2 or len(set(ordered))!=2 or any(v not in variants.get(h,{}) for v in ordered):
                add_issue(issues,'INVALID_ORDERED_VARIANTS',pid,f'{h}:{ordered}')
            if ordn not in {1,2}:
                add_issue(issues,'INVALID_SELECTED_ORDINAL',pid,f'{h}:{ordn}')
            else:
                dv=ordered[ordn-1] if len(ordered)>=ordn else None
                if dv!=rv: add_issue(issues,'ORDINAL_VARIANT_MISMATCH',pid,f'{h}: {dv} != {rv}')
                if h in derived and derived[h]!=dv: add_issue(issues,'MULTIPLE_DERIVED_VARIANTS',pid,f'{h}:{derived[h]}/{dv}')
                derived[h]=dv
            # Actual visible texts must materialize ordered routes and ordinal selection.
            cand_text=' '.join(m.get('content','') for m in sc.get('messages',[]))
            res_text=' '.join(m.get('content','') for m in sr.get('messages',[]))
            if ac.get('pair_ref') and ac['pair_ref'] not in cand_text: add_issue(issues,'PAIR_REF_MISSING_FROM_CANDIDATE_TEXT',pid,h)
            if ar.get('pair_ref') and ar['pair_ref'] not in res_text: add_issue(issues,'PAIR_REF_MISSING_FROM_RESOLUTION_TEXT',pid,h)
            for v in ordered:
                action=variants[h][v]['action']
                if norm(action) not in norm(cand_text):
                    add_issue(issues,'VARIANT_ACTION_MISSING_FROM_CANDIDATE_TEXT',pid,f'{h}:{v}')
            # Resolution text must make the intended ordinal legible.
            rt=norm(res_text)
            ordinal_phrases={1:['first route','former of the two','option one','first-listed candidate'],2:['second route','latter of the two','option two','second-listed candidate']}
            if ordn in {1,2} and not any(x in rt for x in ordinal_phrases[ordn]):
                add_issue(issues,'ORDINAL_NOT_EXPLICIT_IN_RESOLUTION_TEXT',pid,f'{h}: ordinal={ordn}')
        elif len(g)==1:
            sid=g[0]
            if sid not in sess_by: add_issue(issues,'MISSING_REFERENCE_SESSION',pid,sid); continue
            a=ann0(sess_by[sid]); reference_anns.append(a)
            if a.get('kind')!='v11_reference_case': add_issue(issues,'INVALID_SINGLETON_GROUP',pid,a.get('kind',''))
            if k.get('reference_session_id')!=sid: add_issue(issues,'REFERENCE_ID_MISMATCH',pid,sid)
            cref=a.get('case_ref')
            if cref and cref not in p['query']: add_issue(issues,'REFERENCE_CASE_NOT_IN_QUERY',pid,cref)
            oh=a.get('open_habit_id')
            if oh not in target: add_issue(issues,'OPEN_HABIT_NOT_TARGETED',pid,str(oh))
        else:
            add_issue(issues,'INVALID_COMPONENT_GROUP_SIZE',pid,str(len(g)))

    # Exactly one decisive pair per target habit.
    if Counter(group_habits)!=Counter(target):
        add_issue(issues,'TARGET_TO_DECISIVE_GROUP_COVERAGE',pid,f'{Counter(group_habits)} vs {Counter(target)}')

    # 2. Independently determine expected policy at target time from full annotation timeline.
    expected={}
    for h in target:
        d=decisions[uid][h]
        bcs=d.get('v11_baseline_candidate'); brs=d.get('v11_baseline_resolution')
        rcs=d.get('v11_replacement_candidate'); rrs=d.get('v11_replacement_resolution')
        if not bcs or not brs:
            add_issue(issues,'MISSING_BASELINE_DECISION_RECORD',pid,h); continue
        ba=ann0(sess_by[bcs]); bra=ann0(sess_by[brs])
        baseline=ba['ordered_variants'][bra['selected_ordinal']-1]
        t_raw=(k.get('target_state_times') or {}).get(h)
        when=datetime.fromisoformat(t_raw) if t_raw else datetime.fromisoformat(p['timestamp'])
        if rrs and when>=datetime.fromisoformat(sess_by[rrs]['timestamp']):
            ra=ann0(sess_by[rcs]); rra=ann0(sess_by[rrs]); expected[h]=ra['ordered_variants'][rra['selected_ordinal']-1]
            expected_phase='replacement'
        else:
            expected[h]=baseline; expected_phase='baseline'
        # Pair selected as decisive must correspond to expected phase.
        gh=None
        for g in k['required_component_groups']:
            if len(g)==2 and h in ann0(sess_by[g[0]]).get('habit_ids',[]): gh=ann0(sess_by[min(g,key=lambda sid:sess_by[sid]['session_index'])]).get('kind')
        if gh:
            got_phase='replacement' if 'replacement' in gh else 'baseline'
            if got_phase!=expected_phase: add_issue(issues,'WRONG_TEMPORAL_DECISION_PHASE',pid,f'{h}: {got_phase}/{expected_phase}')
    if derived!=expected:
        add_issue(issues,'DERIVED_VARIANT_NOT_EXPECTED_STATE',pid,f'derived={derived}; expected={expected}')

    # 3. Gold signature must be uniquely implied by evidence.
    goldvars=goldsig.get('variants',{})
    if {h:goldvars.get(h) for h in target}!=expected:
        add_issue(issues,'GOLD_SIGNATURE_NOT_SUPPORTED',pid,f'gold={goldvars}; expected={expected}')
    matching=[]
    for lab,sig in k['choice_policy_signatures'].items():
        if {h:sig.get('variants',{}).get(h) for h in target}==expected: matching.append(lab)
    if matching!=[gold]: add_issue(issues,'GOLD_NOT_UNIQUE_FROM_CHAIN',pid,f'matching={matching}, gold={gold}')
    # all choice signatures should be unique on target variants.
    sigtuples=[]
    for lab in 'ABCD':
        sig=k['choice_policy_signatures'][lab]
        sigtuples.append(tuple((h,sig.get('variants',{}).get(h)) for h in target))
    if len(set(sigtuples))!=4: add_issue(issues,'DUPLICATE_CHOICE_POLICY_SIGNATURE',pid)

    # 4. Actual choice text must realize the metadata signature.
    choice_map={x['choice_id']:x['text'] for x in p['choices']}
    for lab,sig in k['choice_policy_signatures'].items():
        txt=norm(choice_map[lab])
        for h,v in sig.get('variants',{}).items():
            phrase=norm(CHOICE[h][v])
            if phrase not in txt:
                add_issue(issues,'CHOICE_TEXT_SIGNATURE_MISMATCH',pid,f'{lab}:{h}:{v}')
    if choice_map[gold]!=k['gold_action_text']: add_issue(issues,'GOLD_TEXT_MISMATCH',pid)

    # 5. Nonbinding evidence must explicitly announce its nonbinding provenance/scope.
    for sid in k.get('nonbinding_evidence_session_ids',[]):
        a=ann0(sess_by[sid]); text=norm(' '.join(m.get('content','') for m in sess_by[sid]['messages']))
        if a.get('kind')=='v11_oneoff_decoy':
            if not ('for this record only' in text and ('must not change' in text or 'new default' in text)):
                add_issue(issues,'ONEOFF_NOT_EXPLICITLY_NONBINDING',pid,sid)
        elif a.get('kind')=='v11_rejected_decoy':
            if not (('proposal' in text or 'suggestion' in text) and ('unchosen' in text or 'nonbinding' in text or 'not' in text)):
                add_issue(issues,'SUGGESTION_NOT_EXPLICITLY_REJECTED',pid,sid)
        else:
            add_issue(issues,'NONBINDING_WRONG_ANNOTATION_KIND',pid,f'{sid}:{a.get("kind")}')

    # 6. Temporal query wording must encode the exact benchmark-local date and minute.
    asof=k.get('as_of_timestamp')
    temporal_ambiguous=False; temporal_precision='none'
    if asof:
        dt=datetime.fromisoformat(asof)
        hour=dt.strftime('%I').lstrip('0') or '12'
        exact=f'{hour}:{dt.strftime("%M")} {dt.strftime("%p").lower()} on {dt.strftime("%B")} {dt.day}, {dt.year}'
        temporal_precision='minute'
        if exact not in p['query']:
            temporal_ambiguous=True
            add_issue(issues,'ASOF_EXACT_TIMESTAMP_NOT_FOUND_IN_QUERY',pid,f'expected={exact}')

    # 7. Reference-case semantic support.
    if k.get('reference_session_id'):
        a=ann0(sess_by[k['reference_session_id']]); oh=a.get('open_habit_id'); rtime=datetime.fromisoformat(sess_by[k['reference_session_id']]['timestamp'])
        if oh:
            declared=a.get('variant_ids',{}).get(oh)
            if expected.get(oh)!=declared:
                add_issue(issues,'REFERENCE_CASE_VARIANT_MISMATCH',pid,f'{oh}: expected {expected.get(oh)} declared {declared}')
            t=(k.get('target_state_times') or {}).get(oh)
            if not t or datetime.fromisoformat(t)!=rtime: add_issue(issues,'REFERENCE_TARGET_TIME_MISMATCH',pid,f'{t}/{rtime.isoformat()}')

    # 8. Date/ordering sanity.
    pt=datetime.fromisoformat(p['timestamp'])
    if pt<=datetime.fromisoformat(life_by[uid]['sessions'][-1]['timestamp']): add_issue(issues,'PROBE_NOT_AFTER_HISTORY',pid)
    if any(datetime.fromisoformat(sess_by[s]['timestamp'])>=pt for s in c['all_relevant_session_ids']): add_issue(issues,'EVIDENCE_NOT_BEFORE_PROBE',pid)

    rows.append({
        'probe_id':pid,'user_id':uid,'domain':p['domain'],'probe_type':k['probe_type'],'gold_choice_id':gold,
        'target_habit_count':len(target),'decision_session_count':len(k['decision_evidence_session_ids']),
        'temporal_context_count':len(k['temporal_context_session_ids']),'nonbinding_count':len(k['nonbinding_evidence_session_ids']),
        'as_of_timestamp':asof or '','temporal_precision':temporal_precision,'temporal_ambiguous':int(temporal_ambiguous),
        'semantic_chain_pass':int(not issues),'issue_count':len(issues),'issues':' | '.join(issues),
        'derived_variants_json':json.dumps(derived,sort_keys=True),'expected_variants_json':json.dumps(expected,sort_keys=True),
    })

# write outputs
fields=list(rows[0])
with (OUT/'evidence_chain_semantic_audit_per_probe.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
report={
    'version':'v1.3','probes':len(rows),'semantic_chain_pass_count':sum(r['semantic_chain_pass'] for r in rows),
    'semantic_chain_fail_count':sum(not r['semantic_chain_pass'] for r in rows),
    'issue_counts':dict(summary),'issue_examples':dict(issue_examples),
    'by_probe_type':{t:{'count':sum(1 for r in rows if r['probe_type']==t),'pass':sum(r['semantic_chain_pass'] for r in rows if r['probe_type']==t),'fail':sum(1-r['semantic_chain_pass'] for r in rows if r['probe_type']==t),'issues':dict(per_type.get(t,{}))} for t in sorted({r['probe_type'] for r in rows})},
    'as_of_probes':sum(bool(r['as_of_timestamp']) for r in rows),'as_of_granularity_ambiguous_count':sum(r['temporal_ambiguous'] for r in rows),
    'audit_scope':[
        'required component group topology','candidate/resolution pair_ref and habit consistency','ordinal-to-variant resolution',
        'visible text realization of shortlist and ordinal','independent state reconstruction at current/as-of/reference time',
        'unique gold choice implied by evidence','choice text to policy-signature consistency','nonbinding provenance markers',
        'as-of query temporal precision','reference-case checkpoint consistency','evidence ownership and temporal ordering'
    ]
}
(OUT/'evidence_chain_semantic_audit_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
