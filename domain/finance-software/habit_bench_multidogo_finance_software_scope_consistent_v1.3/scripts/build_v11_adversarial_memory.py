#!/usr/bin/env python3
from __future__ import annotations
import ast, csv, hashlib, itertools, json, math, os, random, re, shutil, statistics, subprocess, sys, textwrap, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED=20260723+11
RNG=random.Random(SEED)
V10_ZIP=Path('/mnt/data/habit_bench_multidogo_finance_software_memory_centric_v1_0_complete.zip')
WORK=Path('/mnt/data/v11_work')
OUT=Path('/mnt/data/habit_bench_multidogo_finance_software_adversarial_memory_v1_1')
ZIP_PATH=Path('/mnt/data/habit_bench_multidogo_finance_software_adversarial_memory_v1_1_complete.zip')
ZIP_SHA=ZIP_PATH.with_suffix(ZIP_PATH.suffix+'.sha256')

# ---------- generic I/O ----------
def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def write_jsonl(path:Path,rows:Iterable[dict[str,Any]]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def write_json(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_csv(path:Path,rows:list[dict[str,Any]],fields:list[str]|None=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text('',encoding='utf-8'); return
    fields=fields or list(rows[0])
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def stable_int(*parts:Any,mod:int|None=None)->int:
    n=int.from_bytes(hashlib.sha256('||'.join(map(str,parts)).encode()).digest()[:8],'big'); return n%mod if mod else n
def rng_for(*parts:Any)->random.Random:return random.Random(stable_int(SEED,*parts))
def norm(text:str)->str:
    text=text.lower().replace('’',"'");text=re.sub(r'\b\d+(?:[.,]\d+)?\b','<num>',text);text=re.sub(r'[^a-z0-9<>]+',' ',text);return re.sub(r'\s+',' ',text).strip()
def words(s:str)->list[str]:return re.findall(r"[a-z0-9']+",s.lower())
def fmt_date(ts:str|datetime,style:int=0)->str:
    dt=datetime.fromisoformat(ts) if isinstance(ts,str) else ts
    return [dt.strftime('%B %-d, %Y'),dt.strftime('%b %-d, %Y'),dt.strftime('%Y-%m-%d'),dt.strftime('%B %Y')][style%4]
def lower_first(s:str)->str:return s[:1].lower()+s[1:] if s else s
def clean_sentence(s:str)->str:return re.sub(r'[\s.;,:!?]+$','',re.sub(r'\s+',' ',s).strip())+'.'
def pick(seq:list[Any],*parts:Any)->Any:return seq[stable_int(*parts,mod=len(seq))]

# ---------- extract v1.0 package ----------
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir(parents=True)
with zipfile.ZipFile(V10_ZIP) as z:z.extractall(WORK)
roots=[p for p in WORK.iterdir() if p.is_dir()]
assert len(roots)==1,roots
BASE=roots[0]
if OUT.exists():shutil.rmtree(OUT)
OUT.mkdir(parents=True)

profiles=sorted(read_jsonl(BASE/'private/persona_profiles.jsonl'),key=lambda x:x['user_id'])
profile_by_user={p['user_id']:p for p in profiles}
lifelines=read_jsonl(BASE/'public/lifelines.jsonl'); life_by_user={x['user_id']:x for x in lifelines}
annotated_sessions=read_jsonl(BASE/'private/sessions_with_annotations.jsonl')
habits=json.loads((BASE/'source/habit_templates_retained.json').read_text(encoding='utf-8'))
habit_by_id={h['habit_id']:h for h in habits}
variant_by_habit={h['habit_id']:{v['variant_id']:v for v in h['policy_variants']} for h in habits}
canonical_variant={h['habit_id']:h['policy_variants'][0]['variant_id'] for h in habits}

# Extract the carefully authored choice realizations and scenario helper from v1.0 builder.
# This avoids changing policy semantics while v1.1 changes evidence topology and task design.
v10_builder=(BASE/'scripts/build_v10_memory_centric_probes.py').read_text(encoding='utf-8')
tree=ast.parse(v10_builder)
CHOICE_REALIZATIONS=None
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHOICE_REALIZATIONS' for t in node.targets):
        CHOICE_REALIZATIONS=ast.literal_eval(node.value);break
assert CHOICE_REALIZATIONS and set(CHOICE_REALIZATIONS)==set(habit_by_id)

# Copy source lineage and raw archives.
for rel in ['source/raw_multidogo_finance.csv.gz','source/raw_multidogo_software.csv.gz','source/source_conversation_usage_manifest.csv','source/habit_templates_retained.json','source/SOURCE_ATTRIBUTION.md']:
    src=BASE/rel;dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

# ---------- profile / variant helpers ----------
def active_habits(p):return [h for h in p.get('active_habit_ids',[]) if h in habit_by_id]
def active_variant(p,h):return p['active_policy_variants'][h]
def old_variant(p,h):
    v=p.get('old_policy_variants',{}).get(h)
    if v and v in variant_by_habit[h] and v!=active_variant(p,h):return v
    alts=[x for x in variant_by_habit[h] if x!=active_variant(p,h)];return pick(alts,p['user_id'],h,'old')
def decoy_variant(p,h,salt='decoy'):
    return pick([x for x in variant_by_habit[h] if x!=active_variant(p,h)],p['user_id'],h,salt)
def active_noncanonical(p):return [h for h in active_habits(p) if active_variant(p,h)!=canonical_variant[h]]
def variant_action(h,v):return variant_by_habit[h][v]['action']
def variant_label(h,v):return variant_by_habit[h][v]['label']
def choice_body(h,v):return lower_first(CHOICE_REALIZATIONS[h][v])

# ---------- scenario generation ----------
MERCHANTS=['Willow Pharmacy','Harbor Transit','Northline Books','Cedar Market','Juniper Wireless','Maple Dental','Atlas Learning','Sunrise Hardware','Pine & Stone','Riverview Energy','Orchid Health','Silverline Travel','Meadow Insurance','Beacon Storage','Lakeside Studio','Granite Office','Elm Fitness','Oakline Media','Cloudberry Foods','Westbridge Labs']
PAYEES=['Harbor Utilities','Meadow Insurance','Atlas Learning','Riverview Energy','Cedar Dental','Juniper Wireless','Northline Storage','Silverline Travel','Beacon Property Services','Oakline Media']
PURPOSES=['quarterly tax reserve','annual insurance renewal','studio equipment invoice','tuition installment','clinic reimbursement','travel deposit','vendor retainer','workspace lease','professional membership','home repair']
FEATURES=['calendar sync','shared inbox','PDF export','meeting captions','offline search','client portal','notification rules','template gallery','account dashboard','file preview','contact import','report builder']
ERRORS=['E-214','E-307','E-418','E-502','SYNC-17','AUTH-29','UI-44','DB-63','NET-71','CFG-88']
VERSIONS=['7.4.2','7.5.1','8.0.3','8.1.6','9.2.0','9.3.4','10.0.1','10.2.7']
DATES=['January 14','February 6','March 18','April 9','May 22','June 11','July 27','August 15','September 8','October 19','November 13','December 4']
RECORDS=['monthly close note','quarterly review file','support handoff','audit worksheet','case journal','operations brief','reconciliation packet','incident register']

def scenario(p,h,salt):
    uid=p['user_id'];merchant=pick(MERCHANTS,uid,h,salt,'m');payee=pick(PAYEES,uid,h,salt,'p');purpose=pick(PURPOSES,uid,h,salt,'u');date=pick(DATES,uid,h,salt,'d');amount=75+stable_int(uid,h,salt,'amt',mod=2450);record=pick(RECORDS,uid,h,salt,'r')
    if p['domain']=='finance':
        chk=p.get('checking_last4') or p.get('account_last4','0000');sav=p.get('savings_last4') or p.get('account_last4','0000');card=p.get('card_last4') or p.get('account_last4','0000')
        bank={
        'finance_confirm_money_movement':[(f"Prepare a ${amount} payment from checking ending {chk} to {payee} for {date}.",f"the ${amount} payment to {payee} from checking ending {chk}"),(f"Set up the ${amount} transfer from savings ending {sav} to checking ending {chk} for the {purpose}.",f"the ${amount} transfer for the {purpose}")],
        'finance_confirm_card_account_changes':[(f"Replace card ending {card}; its chip has stopped working.",f"the replacement of card ending {card}"),(f"Move the mailing address for checking ending {chk} to the verified address on file.",f"the address change for checking ending {chk}")],
        'finance_minimal_pii_secure_verification':[(f"Restore access to checking ending {chk} after a device change.",f"identity verification for checking ending {chk}"),(f"Recover the secure profile tied to savings ending {sav}.",f"secure recovery for savings ending {sav}")],
        'finance_fraud_lost_card_urgent_escalation':[(f"A ${amount} charge at {merchant} is unfamiliar on card ending {card}.",f"the unfamiliar charge on card ending {card}"),(f"Card ending {card} went missing during the {purpose}.",f"the missing-card incident for ending {card}")],
        'finance_balance_statement_summary_first':[(f"Review the {date} statement for checking ending {chk} and explain the change.",f"the {date} statement review for checking ending {chk}"),(f"Reconcile the latest activity on card ending {card} for the {purpose}.",f"the card activity review for the {purpose}")],
        'finance_fee_dispute_evidence_then_case':[(f"Review the ${amount} fee posted by {merchant} on {date} and prepare the case if the mismatch remains.",f"the ${amount} fee from {merchant}"),(f"The ${amount} charge from {merchant} on card ending {card} still does not reconcile.",f"the unreconciled ${amount} charge from {merchant}")],
        'finance_credit_loan_cautious_no_commitment':[(f"Compare financing paths for a ${amount*8} {purpose}.",f"the financing review for {purpose}"),(f"Assess a ${amount*10} line-of-credit question for the {purpose}.",f"the credit analysis for {purpose}")],
        'finance_payment_status_latest_check':[(f"Tell me where the ${amount} payment to {payee} stands now.",f"the current status of the ${amount} payment to {payee}"),(f"Check the present state of the ${amount} transfer from checking ending {chk} for the {purpose}.",f"the current transfer state for the {purpose}")],}
        return pick(bank[h],uid,h,salt,'case')
    app=p.get('desktop_app') or 'Orbit Office';os_=p.get('os') or 'the current platform';browser=p.get('browser') or 'the browser';feature=pick(FEATURES,uid,h,salt,'f');error=pick(ERRORS,uid,h,salt,'e');ver=pick(VERSIONS,uid,h,salt,'v')
    sw={
    'software_collect_diagnostics_before_fix':[(f"{feature.title()} fails in {app} on {os_} with error {error}; diagnose it.",f"the {feature} failure with {error}"),(f"{app} {ver} closes when I use {feature} in {browser}.",f"the {app} {feature} crash")],
    'software_docs_lookup_for_update_install':[(f"Work out the correct {app} {ver} update path for {os_}.",f"the {app} {ver} update path"),(f"Check the installation guidance for {feature} on {os_}.",f"the {feature} installation guidance")],
    'software_one_try_then_escalate':[(f"The {feature} issue in {app} has survived the first check; move the case forward.",f"the unresolved {feature} issue"),(f"Error {error} is reproducible in {app}; decide the next support step.",f"the reproducible {error} blocker")],
    'software_secure_login_password_flow':[(f"Recover access to {app} after the sign-in stopped working.",f"the {app} account recovery"),(f"Restore the account login in {browser} without putting credentials in the case.",f"the browser account recovery")],
    'software_platform_specific_steps':[(f"Set up {feature} in {app} on {os_}.",f"the {feature} setup on {os_}"),(f"Show me how to configure {feature} in {app} {ver}.",f"the {feature} configuration in {app}")],
    'software_backup_before_risky_change':[(f"Prepare the {app} {ver} reset on {os_} without losing the current setup.",f"the {app} reset on {os_}"),(f"Plan the risky configuration change for {feature} in {app}.",f"the risky {feature} configuration change")],
    'software_ticket_receipt_summary':[(f"File the {feature} problem with error {error} and return the case record.",f"the filed {feature} case"),(f"Submit the reproducible {app} {ver} defect and give me the handoff record.",f"the submitted {app} defect")],}
    return pick(sw[h],uid,h,salt,'case')

# ---------- evidence-session generation: split, cross-referenced policy evidence ----------
# v1.1 final intentionally separates the *meaning* of a policy from the session that
# ratifies it.  A shortlist session names two defensible routes; a distant resolution
# session selects the former/latter route only by reference.  The pair is easy to audit
# with the complete lifeline but cannot be reconstructed from a single ~34k slice.
PAIR_OPENERS=[
    'I want to narrow the recurring process before I make it permanent.',
    'Use this case to record the two routes still under consideration.',
    'I am not choosing a standing route in this session; I am reducing the shortlist.',
    'The outcome can wait. First preserve the two workflow candidates in their exact order.',
    'I need an auditable shortlist rather than an assumed default.',
    'Treat this as a workflow review, not as final authorization of either route.',
]
PAIR_ACKS=[
    'The ordered shortlist is linked to the reference; no standing choice is implied yet.',
    'I have preserved the two candidates in order and marked the decision as pending.',
    'The record now distinguishes the shortlist from a finalized recurring process.',
    'Both candidates remain viable until a later closeout points back to this ordering.',
    'I recorded the order without promoting either candidate to a durable preference.',
]
RESOLVE_OPENERS=[
    'Close the pending workflow decision from',
    'Return to the ordered shortlist in',
    'Finalize the process review recorded under',
    'Use the original ordering in',
    'Resolve the two-route decision linked to',
]
RESOLVE_TAILS=[
    'The unselected route remains available only when I explicitly request it for a local case.',
    'Do not infer the selection from today’s tooling; the earlier ordering controls what this reference means.',
    'Keep the rejected candidate in the audit trail, not in the standing workflow.',
    'This is a durable choice, while any later shortcut must remain case-specific unless I replace it again.',
    'Preserve the date and reference because the wording of future cases may point toward the other safe route.',
]
LOCAL_REASONS=[
    'the ordinary channel is unavailable for this one case',
    'a second reviewer is already waiting on this record',
    'today’s deadline makes a reversible fallback useful',
    'the usual tool is temporarily under maintenance',
    'this single incident needs a temporary handoff path',
    'the current case has an unusual dependency that will not recur',
]
CONTEXT_ONLY_USER=[
    'This exchange is only about the current record. Leave any recurring workflow in its separate decision note.',
    'Keep the answer case-specific; this is not a vote for or against any standing route.',
    'Do not use this local follow-up to restate or replace my long-term process.',
    'The case needs an update, but the durable handling is documented elsewhere and is unchanged here.',
    'Treat today’s detail as context, not as new preference evidence.',
    'This is a one-record clarification; it should not settle a pending workflow review.',
]
CONTEXT_ONLY_ASSISTANT=[
    'I will keep the local facts and unresolved checkpoint visible without turning this exchange into a policy update.',
    'The response will stay tied to this case and will not duplicate or revise the separate workflow record.',
    'I will answer the current task while leaving the standing-process decision untouched.',
    'The handoff will distinguish case facts from any durable preference already recorded elsewhere.',
    'I will preserve the local status without inferring a recurring rule from it.',
    'The current note will remain nonbinding for future cases.',
]

REF_ADJ=['amber','cedar','cobalt','coral','delta','ember','frost','granite','harbor','indigo','juniper','lunar','maple','meadow','north','olive','orchid','pine','quartz','river','silver','solar','stone','summit','timber','violet','west','willow','winter','zephyr']
REF_NOUN=['anchor','beacon','bridge','canvas','compass','folio','harbor','journal','lantern','ledger','matrix','notebook','orbit','packet','quill','register','relay','signal','thread','vault','window','workbook','marker','index','ribbon','trail','brief','docket','map','frame']
def alpha_code(*parts):
    n=stable_int(*parts)
    out=[]
    for _ in range(6):out.append(chr(ord('a')+n%26));n//=26
    return ''.join(out)
def word_ref(prefix,*parts):
    return f"{prefix}-{pick(REF_ADJ,*parts,'adj')}-{pick(REF_NOUN,*parts,'noun')}-{alpha_code(*parts,'code')}"

# Materialize mutable maps.
ann_by_sid={s['session_id']:s for s in annotated_sessions}
life_session_by_sid={s['session_id']:s for l in lifelines for s in l['sessions']}
user_sessions={l['user_id']:sorted(l['sessions'],key=lambda s:s['session_index']) for l in lifelines}

# All legacy preference-bearing sessions are retained as domain-grounded task events, but
# their exact policy realization is removed. This prevents a single old support sentence
# from bypassing the split-decision evidence.
LEGACY_POLICY_KINDS={'support','old','new','composition','priority','weak_support','boundary','exception','tentative'}
def context_only_messages(p,habit_ids,kind,salt):
    h=habit_ids[0] if habit_ids else active_habits(p)[0]
    req,label=scenario(p,h,f'context|{kind}|{salt}')
    cref=word_ref('CASE',p['user_id'],kind,salt)
    req=f'{req} Track this local update in {cref}.'
    user_note=pick(CONTEXT_ONLY_USER,p['user_id'],kind,salt)
    assistant_note=pick(CONTEXT_ONLY_ASSISTANT,p['user_id'],kind,salt)+f' The local record is {cref}, for {label}.'
    first_reply=pick([
        f'I can work through {label} and keep the last reversible checkpoint visible.',
        f'I will review {label}, separate verified facts from assumptions, and leave the next owner explicit.',
        f'I can prepare a case-specific note for {label} without expanding the scope.',
        f'I will handle {label} as a local task and preserve what remains open.',
        f'I can update {label} while keeping the evidence trail replayable.',
    ],p['user_id'],kind,salt,'reply')+f' I will keep that work under {cref}.'
    return [
        {'role':'user','content':req},
        {'role':'assistant','content':first_reply},
        {'role':'user','content':user_note+f' For {label}, keep the note under {cref}.'},
        {'role':'assistant','content':assistant_note},
    ]

for p in profiles:
    uid=p['user_id'];active=set(active_habits(p))
    for sess in user_sessions[uid]:
        a=ann_by_sid[sess['session_id']]
        old_annotations=a.get('memory_annotations',[])
        if any(x.get('kind')=='identity_anchor' for x in old_annotations):
            continue
        hit=[x for x in old_annotations if x.get('kind') in LEGACY_POLICY_KINDS and active.intersection(x.get('habit_ids',[]))]
        if not hit:
            continue
        habit_ids=[]
        for x in hit:
            for h in x.get('habit_ids',[]):
                if h in active and h not in habit_ids:habit_ids.append(h)
        kinds=sorted({x.get('kind') for x in hit})
        msgs=context_only_messages(p,habit_ids,'+'.join(kinds),sess['session_index'])
        sess['messages']=msgs;a['messages']=msgs
        a['memory_annotations']=[{'kind':'v11_context_only','habit_ids':habit_ids,'source_kinds':kinds,'evidence_strength':'nonresolving'}]
        a['rewrite_metadata']={'rewrite_version':'v1.1','rewrite_scope':'legacy exact-policy leakage removal','source_task_intent_preserved':True,'source_identity_and_credentials_removed':True,'public_private_messages_match':True,'turn_categories':['task_request','task_response','case_scope_note','assistant_scope_acknowledgment']}

# Helpers for ordered candidate pairs and distant resolutions.
def pair_order(p,h,target,partner,phase):
    arr=[target,partner]
    if stable_int(p['user_id'],h,phase,'order',mod=2):arr.reverse()
    return arr

def candidate_pair_messages(p,h,pair,phase,ref,salt):
    req,label=scenario(p,h,f'pair|{phase}|{salt}')
    a,b=pair
    intro=pick(PAIR_OPENERS,p['user_id'],h,phase,salt)
    ack=pick(PAIR_ACKS,p['user_id'],h,phase,salt)
    routes=pick([
        f'Route One would {lower_first(variant_action(h,a))}; Route Two would {lower_first(variant_action(h,b))}.',
        f'The first candidate is to {lower_first(variant_action(h,a))}. The second is to {lower_first(variant_action(h,b))}.',
        f'Candidate A would {lower_first(variant_action(h,a))}; candidate B would {lower_first(variant_action(h,b))}.',
        f'In order, the remaining paths are: first, {lower_first(variant_action(h,a))}; second, {lower_first(variant_action(h,b))}.',
    ],p['user_id'],h,phase,salt,'routes')
    return [
        {'role':'user','content':f'{req} Open workflow review {ref}. {intro}'},
        {'role':'assistant','content':f'{routes} I can preserve both without treating either as the default.'},
        {'role':'user','content':f'Keep exactly those two candidates, in that order, under {ref}. Rule out the other workflow routes for this review, but do not finalize the first-versus-second choice here.'},
        {'role':'assistant','content':f'{ack} The next decision must cite {ref} rather than reconstructing the order from memory.'},
    ]

def resolution_messages(p,h,pair,selected,phase,ref,salt):
    req,label=scenario(p,h,f'resolve|{phase}|{salt}')
    ordinal='first' if pair[0]==selected else 'second'
    former='former' if ordinal=='first' else 'latter'
    opener=pick(RESOLVE_OPENERS,p['user_id'],h,phase,salt)
    tail=pick(RESOLVE_TAILS,p['user_id'],h,phase,salt)
    choice_phrase=pick([
        f'make the {ordinal} route the standing process',
        f'adopt the {former} of the two routes as the recurring workflow',
        f'carry forward option {"one" if ordinal=="first" else "two"} from that record',
        f'use the {ordinal}-listed candidate whenever the same condition recurs',
    ],p['user_id'],h,phase,salt,'choice')
    return [
        {'role':'user','content':f'{req} {opener} {ref}: {choice_phrase}. {tail}'},
        {'role':'assistant','content':f'I will link this closeout to the ordered shortlist in {ref}; the selected ordinal becomes effective from this session without being re-expanded into a fresh list.'},
        {'role':'user','content':f'For {label}, preserve {ref} and its effective date. A later case that merely resembles this one must not switch to the other candidate.'},
        {'role':'assistant','content':f'{ref} is closed as a durable decision. The unselected route remains nonbinding unless a later dated decision explicitly replaces it.'},
    ]

def rehearsal_messages(p,h,ref,phase,salt):
    req,label=scenario(p,h,f'rehearse|{phase}|{salt}')
    return [
        {'role':'user','content':f'{req} Keep the pending workflow review under {ref} separate from this case.'},
        {'role':'assistant','content':f'I will use a reversible case note for {label}; the ordered pair in {ref} remains unresolved.'},
        {'role':'user','content':f'For {label}, the available tool may make one route look easier today, but convenience does not close {ref}.'},
        {'role':'assistant','content':f'I will preserve that distinction and leave {ref} open until a later session selects one of its two recorded positions.'},
    ]

def oneoff_messages(p,h,variant,standing,ref,salt):
    req,label=scenario(p,h,f'oneoff|{salt}')
    reason=pick(LOCAL_REASONS,p['user_id'],h,salt)
    return [
        {'role':'user','content':f'{req} For this record only, use the route where you {lower_first(variant_action(h,variant))} because {reason}.'},
        {'role':'assistant','content':f'I will keep that handling local to {ref}, record why it was used, and stop at the last reversible checkpoint.'},
        {'role':'user','content':f'The handling in {ref} for {label} must not change the ordered decision record or become evidence of a new default.'},
        {'role':'assistant','content':f'The local route and reason will remain attached only to {ref} and {label}.'},
    ]

def rejected_suggestion_messages(p,h,suggested,ref,salt):
    req,label=scenario(p,h,f'suggest|{salt}')
    return [
        {'role':'user','content':f'{req} The current tools make several safe workflows possible.'},
        {'role':'assistant','content':f'For speed, I could {lower_first(variant_action(h,suggested))}. I will leave it as a proposal until you connect it to a standing decision.'},
        {'role':'user','content':f'Keep that proposal unchosen in {ref}. Do not let the assistant-originated suggestion override the workflow I ratified elsewhere.'},
        {'role':'assistant','content':f'The proposal remains nonbinding in {ref} for {label}; no preference update is inferred.'},
    ]

# Fixed bands are deliberately farther apart than a 205-session (~34k) slice.
# Every habit has a baseline shortlist in the early band and its resolution in the
# middle band. Drift habits additionally have a replacement shortlist and a late
# ratification, again separated by more than 205 sessions.
decision_meta=defaultdict(dict)
reference_cases=defaultdict(list)
for p in profiles:
    uid=p['user_id'];hs=active_habits(p);drifts=set(p.get('drift_habit_ids',[]))
    for j,h in enumerate(hs):
        active=active_variant(p,h);old=old_variant(p,h) if h in drifts else active
        # Baseline pair establishes the old state for drift habits and the standing state otherwise.
        baseline_partner_pool=[v for v in variant_by_habit[h] if v!=old and (h not in drifts or v!=active)]
        if not baseline_partner_pool:baseline_partner_pool=[v for v in variant_by_habit[h] if v!=old]
        baseline_partner=pick(baseline_partner_pool,uid,h,'baseline_partner')
        baseline_pair=pair_order(p,h,old,baseline_partner,'baseline')
        base_ref=word_ref('POLB',uid,h,j)
        b_cand_idx=2+j*6;b_res_idx=260+j*3
        b_cand_sid=user_sessions[uid][b_cand_idx]['session_id'];b_res_sid=user_sessions[uid][b_res_idx]['session_id']
        for idx,sid,msgs,kind,extra in [
            (b_cand_idx,b_cand_sid,candidate_pair_messages(p,h,baseline_pair,'baseline',base_ref,j),'v11_baseline_candidate',{'pair_ref':base_ref,'ordered_variants':baseline_pair,'candidate_variants':baseline_pair}),
            (b_res_idx,b_res_sid,resolution_messages(p,h,baseline_pair,old,'baseline',base_ref,j),'v11_baseline_resolution',{'pair_ref':base_ref,'selected_ordinal':1+baseline_pair.index(old),'variant_id':old}),
        ]:
            sess=user_sessions[uid][idx];sess['messages']=msgs;a=ann_by_sid[sid];a['messages']=msgs;a['memory_annotations']=[{'kind':kind,'habit_ids':[h],'evidence_strength':'decisive_split',**extra}];a['rewrite_metadata']={'rewrite_version':'v1.1','rewrite_scope':'split cross-session policy evidence','source_task_intent_preserved':True,'source_identity_and_credentials_removed':True,'public_private_messages_match':True,'turn_categories':['task_request','task_response','policy_shortlist_or_resolution','assistant_audit_acknowledgment']}
        meta={'baseline_pair_ref':base_ref,'baseline_candidate_session_id':b_cand_sid,'baseline_resolution_session_id':b_res_sid,'baseline_ordered_variants':baseline_pair,'baseline_variant_id':old}
        # A middle rehearsal is deliberately nonresolving.
        rehearse_idx=315+j*3;rehearse_sid=user_sessions[uid][rehearse_idx]['session_id'];msgs=rehearsal_messages(p,h,base_ref,'baseline',j)
        user_sessions[uid][rehearse_idx]['messages']=msgs;a=ann_by_sid[rehearse_sid];a['messages']=msgs;a['memory_annotations']=[{'kind':'v11_pair_rehearsal','habit_ids':[h],'pair_ref':base_ref,'evidence_strength':'nonresolving'}];a['rewrite_metadata']={'rewrite_version':'v1.1','rewrite_scope':'nonresolving policy rehearsal','source_task_intent_preserved':True,'source_identity_and_credentials_removed':True,'public_private_messages_match':True,'turn_categories':['task_request','task_response','policy_status_note','assistant_audit_acknowledgment']}
        meta['rehearsal_session_id']=rehearse_sid
        if h in drifts:
            # Replacement choice is not final when listed. The active variant is ratified only in the far-late band.
            repl_partner_pool=[v for v in variant_by_habit[h] if v!=active]
            # Often keep the old route in the replacement pair; sometimes use a third plausible route.
            repl_partner=old if stable_int(uid,h,'replacement_partner',mod=3)!=0 else pick(repl_partner_pool,uid,h,'replacement_partner_alt')
            repl_pair=pair_order(p,h,active,repl_partner,'replacement')
            repl_ref=word_ref('POLR',uid,h,j)
            r_cand_idx=276+j*2;r_res_idx=532+j
            r_cand_sid=user_sessions[uid][r_cand_idx]['session_id'];r_res_sid=user_sessions[uid][r_res_idx]['session_id']
            for idx,sid,msgs,kind,extra in [
                (r_cand_idx,r_cand_sid,candidate_pair_messages(p,h,repl_pair,'replacement',repl_ref,j),'v11_replacement_candidate',{'pair_ref':repl_ref,'ordered_variants':repl_pair,'candidate_variants':repl_pair,'prior_variant_id':old}),
                (r_res_idx,r_res_sid,resolution_messages(p,h,repl_pair,active,'replacement',repl_ref,j),'v11_replacement_resolution',{'pair_ref':repl_ref,'selected_ordinal':1+repl_pair.index(active),'variant_id':active,'replaces_variant_id':old}),
            ]:
                sess=user_sessions[uid][idx];sess['messages']=msgs;a=ann_by_sid[sid];a['messages']=msgs;a['memory_annotations']=[{'kind':kind,'habit_ids':[h],'evidence_strength':'decisive_split',**extra}];a['rewrite_metadata']={'rewrite_version':'v1.1','rewrite_scope':'split dated preference replacement','source_task_intent_preserved':True,'source_identity_and_credentials_removed':True,'public_private_messages_match':True,'turn_categories':['task_request','task_response','policy_shortlist_or_resolution','assistant_audit_acknowledgment']}
            meta.update({'replacement_pair_ref':repl_ref,'replacement_candidate_session_id':r_cand_sid,'replacement_resolution_session_id':r_res_sid,'replacement_ordered_variants':repl_pair,'replacement_variant_id':active})
        # Local exception and assistant-originated proposal are late, plausible, and nonbinding.
        local_idx=366+j*6;local_sid=user_sessions[uid][local_idx]['session_id'];local_variant=decoy_variant(p,h,'local');local_ref=word_ref('LOCAL',uid,h,j)
        msgs=oneoff_messages(p,h,local_variant,active,local_ref,j);user_sessions[uid][local_idx]['messages']=msgs;a=ann_by_sid[local_sid];a['messages']=msgs;a['memory_annotations']=[{'kind':'v11_oneoff_decoy','habit_ids':[h],'variant_id':local_variant,'counter_variant_id':active,'evidence_strength':'nonbinding','case_ref':local_ref}];a['rewrite_metadata']={'rewrite_version':'v1.1','rewrite_scope':'local exception interference','source_task_intent_preserved':True,'source_identity_and_credentials_removed':True,'public_private_messages_match':True,'turn_categories':['task_request','task_response','user_local_exception','assistant_scope_acknowledgment']}
        sugg_idx=430+j*6;sugg_sid=user_sessions[uid][sugg_idx]['session_id'];suggested=decoy_variant(p,h,'suggestion');sugg_ref=word_ref('SUG',uid,h,j)
        msgs=rejected_suggestion_messages(p,h,suggested,sugg_ref,j);user_sessions[uid][sugg_idx]['messages']=msgs;a=ann_by_sid[sugg_sid];a['messages']=msgs;a['memory_annotations']=[{'kind':'v11_rejected_decoy','habit_ids':[h],'variant_id':suggested,'counter_variant_id':active,'evidence_strength':'nonbinding','case_ref':sugg_ref}];a['rewrite_metadata']={'rewrite_version':'v1.1','rewrite_scope':'assistant-suggestion interference','source_task_intent_preserved':True,'source_identity_and_credentials_removed':True,'public_private_messages_match':True,'turn_categories':['task_request','assistant_suggestion','user_rejection','assistant_policy_acknowledgment']}
        meta.update({'oneoff_decoy_session_id':local_sid,'rejected_decoy_session_id':sugg_sid})
        decision_meta[uid][h]=meta
    # Two historical reference cases. The case stores open/closed state and links to the
    # relevant decision record, but does not restate the chosen policy realization.
    for rslot,idx in enumerate([344,412]):
        h1=hs[(2*rslot)%len(hs)];h2=hs[(2*rslot+1)%len(hs)] if len(hs)>1 else hs[0]
        sid=user_sessions[uid][idx]['session_id'];ts=datetime.fromisoformat(user_sessions[uid][idx]['timestamp'])
        def hist_state(h):
            m=decision_meta[uid][h]
            if 'replacement_resolution_session_id' in m and ts < datetime.fromisoformat(ann_by_sid[m['replacement_resolution_session_id']]['timestamp']):
                return m['baseline_variant_id'],m['baseline_pair_ref']
            return active_variant(p,h),m.get('replacement_pair_ref') or m['baseline_pair_ref']
        v1,ref1=hist_state(h1);v2,ref2=hist_state(h2)
        req1,label1=scenario(p,h1,f'ref|{rslot}|1');req2,label2=scenario(p,h2,f'ref|{rslot}|2');cref=word_ref('FINARCH' if p['domain']=='finance' else 'SWARCH',uid,rslot)
        resolved,open_h,open_label=(h1,h2,label2) if rslot%2==0 else (h2,h1,label1)
        msgs=[
            {'role':'user','content':f'Open reference {cref}. {req1} {req2} Keep the two workstreams separate even though they share one record.'},
            {'role':'assistant','content':f'I will use the standing route linked to {ref1} for {label1} and the route linked to {ref2} for {label2}; the decision records, not today’s tools, determine the handling.'},
            {'role':'user','content':f'Close the {habit_by_id[resolved]["theme"].replace("_"," ")} workstream. Leave {open_label} immediately before its final checkpoint, with the linked policy reference intact.'},
            {'role':'assistant','content':f'{cref} now records one workstream closed and {open_label} paused at its last reversible point. The open owner, linked policy record, and resume trigger are preserved.'},
        ]
        user_sessions[uid][idx]['messages']=msgs;a=ann_by_sid[sid];a['messages']=msgs;a['memory_annotations']=[{'kind':'v11_reference_case','habit_ids':[h1,h2],'case_ref':cref,'resolved_habit_id':resolved,'open_habit_id':open_h,'open_label':open_label,'variant_ids':{h1:v1,h2:v2},'decision_refs':{h1:ref1,h2:ref2}}];a['rewrite_metadata']={'rewrite_version':'v1.1','rewrite_scope':'cross-session unresolved case reference','source_task_intent_preserved':True,'source_identity_and_credentials_removed':True,'public_private_messages_match':True,'turn_categories':['multi_task_request','task_response','case_state_update','closure_acknowledgment']}
        reference_cases[uid].append({'session_id':sid,'session_index':idx,'timestamp':user_sessions[uid][idx]['timestamp'],'case_ref':cref,'habit_ids':[h1,h2],'resolved_habit_id':resolved,'open_habit_id':open_h,'open_label':open_label,'variant_ids':{h1:v1,h2:v2},'decision_refs':{h1:ref1,h2:ref2}})
    p['v11_challenge_metadata']={'decision_meta':decision_meta[uid],'reference_cases':reference_cases[uid],'design_note':'Each durable policy is encoded by an ordered shortlist and a distant ordinal resolution. Assistant suggestions and one-case exceptions remain nonbinding.'}

# Rebuild sorted public/private rows and maps.
for l in lifelines:l['sessions']=sorted(l['sessions'],key=lambda s:s['session_index']);l['session_count']=len(l['sessions'])
annotated_sessions=sorted(ann_by_sid.values(),key=lambda s:(s['user_id'],s['session_index']))
write_jsonl(OUT/'public/lifelines.jsonl',lifelines)
write_jsonl(OUT/'private/sessions_with_annotations.jsonl',annotated_sessions)
write_jsonl(OUT/'private/persona_profiles.jsonl',profiles)
life_by_user={x['user_id']:x for x in lifelines};profile_by_user={p['user_id']:p for p in profiles}
session_by_id={s['session_id']:s for s in annotated_sessions};session_idx={sid:int(s['session_index']) for sid,s in session_by_id.items()};session_ts={sid:datetime.fromisoformat(s['timestamp']) for sid,s in session_by_id.items()};session_owner={sid:s['user_id'] for sid,s in session_by_id.items()}

# Build annotation lookup for v1.1 split evidence.
ann=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
for s in annotated_sessions:
    for a in s.get('memory_annotations',[]):
        for h in a.get('habit_ids',[]):ann[s['user_id']][h][a.get('kind','')].append(s['session_id'])
for uid in ann:
    for h in ann[uid]:
        for k in ann[uid][h]:ann[uid][h][k]=sorted(set(ann[uid][h][k]),key=lambda sid:session_idx[sid])

def meta_for(p,h):return p['v11_challenge_metadata']['decision_meta'][h]
def baseline_time(p,h):return session_ts[meta_for(p,h)['baseline_resolution_session_id']]
def replacement_time(p,h):
    sid=meta_for(p,h).get('replacement_resolution_session_id')
    return session_ts[sid] if sid else None

def state_at(p,h,when):
    m=meta_for(p,h)
    rt=replacement_time(p,h)
    if rt is not None and when>=rt:return active_variant(p,h)
    return m['baseline_variant_id']

def current_state_variant(p,h):return active_variant(p,h)
def opposite_state(p,h,v):
    candidates=[]
    m=meta_for(p,h)
    for x in [m.get('baseline_variant_id'),m.get('replacement_variant_id'),old_variant(p,h),active_variant(p,h)]:
        if x and x!=v and x not in candidates:candidates.append(x)
    for x in variant_by_habit[h]:
        if x!=v and x not in candidates:candidates.append(x)
    return candidates[0]

def state_evidence(p,h,when=None):
    """Return decisive split evidence plus temporally relevant interference."""
    m=meta_for(p,h);ids=[m['baseline_candidate_session_id'],m['baseline_resolution_session_id']]
    if m.get('replacement_candidate_session_id'):
        ids += [m['replacement_candidate_session_id'],m['replacement_resolution_session_id']]
    # Nonbinding late events make the complete-history task provenance-sensitive.
    ids += [m['oneoff_decoy_session_id'],m['rejected_decoy_session_id']]
    return sorted(set(ids),key=lambda sid:session_idx[sid])

def required_component_group(p,h,when=None):
    """The two public sessions that must be joined to resolve this habit state."""
    m=meta_for(p,h)
    if m.get('replacement_resolution_session_id') and (when is None or when>=replacement_time(p,h)):
        return [m['replacement_candidate_session_id'],m['replacement_resolution_session_id']]
    return [m['baseline_candidate_session_id'],m['baseline_resolution_session_id']]

def decision_ref_for(p,h,when=None):
    m=meta_for(p,h)
    if m.get('replacement_pair_ref') and (when is None or when>=replacement_time(p,h)):return m['replacement_pair_ref']
    return m['baseline_pair_ref']

def step(label,h,v):return clean_sentence(f"For {label}, {choice_body(h,v)}")
OPENERS=['The record should use the following handling:','I would close the case this way:','The defensible handoff is:','The case can proceed through this workflow:','My response would preserve these steps:','The review note should show:']
TAILS=['Keep the unresolved owner and last reversible point visible.','Separate verified facts from assumptions in the closeout.','Preserve the as-of point and the next evidence trigger.','Keep the result narrow enough for another reviewer to replay.','Attach the decision trail without widening the scope.']
def render_choice(p,steps,seed,ref):return re.sub(r'\s+',' ',f"{pick(OPENERS,p['user_id'],seed,'o')} {' '.join(steps)} {pick(TAILS,p['user_id'],seed,'t')} Record it under {ref}.").strip()

def options_pair(p,h1,h2,l1,l2,g1,g2,d1,d2,seed,ref):
    combos=[(g1,g2),(d1,d2),(g1,d2),(d1,g2)];opts=[];sigs=[]
    for a,b in combos:
        opts.append(render_choice(p,[step(l1,h1,a),step(l2,h2,b)],seed,ref));sigs.append({'variants':{h1:a,h2:b},'order':[h1,h2]})
    return opts,sigs

def options_triple(p,hs,labels,gs,ds,seed,ref):
    combos=[tuple(gs),tuple(ds),(gs[0],ds[1],gs[2]),(ds[0],gs[1],ds[2])];opts=[];sigs=[]
    for vs in combos:
        opts.append(render_choice(p,[step(l,h,v) for l,h,v in zip(labels,hs,vs)],seed,ref));sigs.append({'variants':dict(zip(hs,vs)),'order':hs})
    return opts,sigs

def make_ref(prefix,p,i):return f"{prefix}-{'FIN' if p['domain']=='finance' else 'SW'}-{2033+i%4}-{10000+i}"
SURFACE_DECOYS=['The calendar is tight, so I need a clean answer rather than a discussion of every possible workflow.','This looks routine on the surface, but keep the record reviewable.','I am between meetings; give me one coherent closeout without dropping any checkpoint.','The request is ordinary enough that the process details are easy to overlook.','Please keep the response compact even though the case has more than one workstream.']

def surface_cue(h,v,uid,salt):
    x=(v+' '+variant_label(h,v)).lower()
    if any(k in x for k in ['secure','app','device','passkey','browser']): return 'The signed-in app and trusted device are already available.'
    if 'callback' in x: return 'The verified contact channel is reachable right now.'
    if any(k in x for k in ['table','comparison','side-by-side','side by side','rollup']): return 'The reviewer has a comparison worksheet open beside the case.'
    if any(k in x for k in ['timeline','chronological','changelog']): return 'The case already contains a timestamped event list.'
    if any(k in x for k in ['two','checkpoint']): return 'The other party is available for two short check-ins.'
    if any(k in x for k in ['official','knowledge','compatibility','known issues','built-in']): return 'The product help page is already open in another tab.'
    if any(k in x for k in ['backup','snapshot','sandbox','staged','restore']): return 'A test environment and recent backup are both available.'
    if any(k in x for k in ['receipt','prose','structured','machine-readable','handoff']): return 'The next reviewer asked for a compact handoff artifact.'
    if any(k in x for k in ['live','status','ledger','processing']): return 'A current status screen is visible while I write this.'
    return pick(['All of the current case details are already in the thread.','The standard support tools are available today.','The case looks ordinary at first glance.'],uid,h,v,salt)

# ---------- probe specs ----------
TYPE_COUNTS={'dual_asof_reversal':384,'triple_asof_interleaved':512,'surface_decoy_pair':384,'reference_case_reconstruction':320,'suggestion_rejection_pair':256,'scope_temporal_pair':64,'provenance_weighted_triple':128}
type_schedule=[]
for t,n in TYPE_COUNTS.items():type_schedule += [t]*n
rng_for('type_schedule').shuffle(type_schedule)
users=[p['user_id'] for p in profiles]
user_slots=[]
for uid in users:user_slots += [uid]*37
for uid in users[:50]:user_slots.append(uid)
rng_for('user_slots').shuffle(user_slots)
assert len(user_slots)==2048

def eligible(uid,typ):
    p=profile_by_user[uid];hs=active_habits(p);dr=[h for h in p.get('drift_habit_ids',[]) if h in hs]
    return {'dual_asof_reversal':len(dr)>=2,'triple_asof_interleaved':len(dr)>=2 and len(hs)>=3,'surface_decoy_pair':len(hs)>=2,'reference_case_reconstruction':bool(reference_cases[uid]) and len(hs)>=2,'suggestion_rejection_pair':len(hs)>=2,'scope_temporal_pair':len(dr)>=1 and len(hs)>=2,'provenance_weighted_triple':len(hs)>=3}[typ]
for i in range(2048):
    if eligible(user_slots[i],type_schedule[i]):continue
    for j in range(i+1,2048):
        if eligible(user_slots[i],type_schedule[j]) and eligible(user_slots[j],type_schedule[i]):
            type_schedule[i],type_schedule[j]=type_schedule[j],type_schedule[i];break
    else:raise RuntimeError(('no swap',i,user_slots[i],type_schedule[i]))

logicals=[]
for i,(uid,typ) in enumerate(zip(user_slots,type_schedule)):
    p=profile_by_user[uid];hs=active_habits(p);dr=[h for h in p.get('drift_habit_ids',[]) if h in hs];seed=f'{uid}|{typ}|{i}';ref=make_ref('V11',p,i);rr=rng_for(seed)
    asof=None;reference=None;cap='';require='';mislead=pick(SURFACE_DECOYS,uid,typ,i);state_times={};required_groups=[]
    if typ=='dual_asof_reversal':
        h1,h2=dr[0],dr[1]
        c1=session_ts[meta_for(p,h1)['replacement_candidate_session_id']];c2=session_ts[meta_for(p,h2)['replacement_candidate_session_id']]
        r1t=replacement_time(p,h1);r2t=replacement_time(p,h2)
        phase=i%3
        if phase==0:
            # Both replacement reviews are pending; the baseline choices still govern.
            when=max(c1,c2)+timedelta(days=2)
            if when>=min(r1t,r2t):when=min(r1t,r2t)-timedelta(hours=12)
        elif phase==1:
            lo,hi=sorted([r1t,r2t]);when=lo+(hi-lo)/2 if hi>lo else hi+timedelta(days=1)
        else:when=max(r1t,r2t)+timedelta(days=35)
        asof=when.isoformat();g1=state_at(p,h1,when);g2=state_at(p,h2,when);d1=opposite_state(p,h1,g1);d2=opposite_state(p,h2,g2)
        (rq1,l1),(rq2,l2)=scenario(p,h1,seed+'1'),scenario(p,h2,seed+'2');opts,sigs=options_pair(p,h1,h2,l1,l2,g1,g2,d1,d2,seed,ref)
        q=f'Prepare the two-workstream record that would have been correct at close of business on {fmt_date(when,i)}. {rq1} {rq2} {mislead} Use the policy state then in force, even if a later reference eventually closed a pending review.'
        target=[h1,h2];state_times={h1:when,h2:when};cap='dual_temporal_state_reconstruction_from_split_decisions';require='Join each ordered shortlist to its distant ordinal resolution, then apply the state that was effective at the requested date.'
    elif typ=='triple_asof_interleaved':
        h1,h2=dr[:2];others=[h for h in hs if h not in {h1,h2}];h3=others[i%len(others)]
        rts=sorted([replacement_time(p,h1),replacement_time(p,h2)])
        when=(rts[0]+(rts[1]-rts[0])/2) if i%2==0 and rts[1]>rts[0] else max(rts)+timedelta(days=28)
        asof=when.isoformat();target=[h1,h2,h3];gs=[state_at(p,h,when) for h in target];ds=[opposite_state(p,h,g) for h,g in zip(target,gs)]
        sc=[scenario(p,h,seed+str(j)) for j,h in enumerate(target)];opts,sigs=options_triple(p,target,[x[1] for x in sc],gs,ds,seed,ref);order=list(range(3));rr.shuffle(order);tasks=' '.join(sc[j][0] for j in order)
        q=f'Complete one period-correct handoff for {fmt_date(when,i+1)}: {tasks} {mislead} The current tool layout resembles a later case, so preserve the dated decision state rather than the visually convenient route.'
        state_times={h:when for h in target};cap='three_habit_asof_interleaving_with_split_references';require='Resolve two dated replacements and a third standing policy by joining cross-referenced evidence spread across the lifeline.'
    elif typ=='surface_decoy_pair':
        h1=hs[i%len(hs)];h2=hs[(i*3+1)%len(hs)]
        if h2==h1:h2=hs[(hs.index(h1)+1)%len(hs)]
        g1,g2=active_variant(p,h1),active_variant(p,h2);d1,d2=decoy_variant(p,h1,seed+'d1'),decoy_variant(p,h2,seed+'d2');(rq1,l1),(rq2,l2)=scenario(p,h1,seed+'1'),scenario(p,h2,seed+'2');opts,sigs=options_pair(p,h1,h2,l1,l2,g1,g2,d1,d2,seed,ref)
        cue=surface_cue(h1,d1,uid,seed)+' '+surface_cue(h2,d2,uid,seed+'b')
        q=f'{rq1} {rq2} {cue} {mislead} Produce one replayable closeout under {ref}; the visible tools are context, not a substitute for the user’s recorded process.'
        target=[h1,h2];state_times={h1:None,h2:None};cap='surface_decoy_multi_habit_retrieval';require='Resolve two distant policy references while local affordances favor different, equally safe workflows.'
    elif typ=='reference_case_reconstruction':
        rc=reference_cases[uid][i%len(reference_cases[uid])];reference=rc['session_id'];open_h=rc['open_habit_id'];h2=hs[(i+1)%len(hs)]
        if h2==open_h:h2=hs[(hs.index(h2)+1)%len(hs)]
        ref_when=datetime.fromisoformat(rc['timestamp']);g1=state_at(p,open_h,ref_when);g2=active_variant(p,h2);d1=opposite_state(p,open_h,g1);d2=decoy_variant(p,h2,seed+'ref2');rq2,l2=scenario(p,h2,seed+'new');l1=rc['open_label'];opts,sigs=options_pair(p,open_h,h2,l1,l2,g1,g2,d1,d2,seed,ref)
        cue=surface_cue(h2,d2,uid,seed)
        q=f'Reopen {rc["case_ref"]} at the exact checkpoint left unresolved there, then add this new item to the same closeout: {rq2} {cue} {mislead} The earlier case must resume from its own dated decision record rather than being restarted with today’s route.'
        target=[open_h,h2];state_times={open_h:ref_when,h2:None};cap='cross_session_unresolved_state_and_policy_reconstruction';require='Locate the historical open workstream, recover the policy state linked at that date, and combine it with a current split-reference policy.'
    elif typ=='suggestion_rejection_pair':
        h1=hs[i%len(hs)];h2=hs[(i+2)%len(hs)]
        if h2==h1:h2=hs[(hs.index(h1)+1)%len(hs)]
        g1,g2=active_variant(p,h1),active_variant(p,h2);d1,d2=decoy_variant(p,h1,seed+'d1'),decoy_variant(p,h2,seed+'d2');(rq1,l1),(rq2,l2)=scenario(p,h1,seed+'1'),scenario(p,h2,seed+'2');opts,sigs=options_pair(p,h1,h2,l1,l2,g1,g2,d1,d2,seed,ref)
        cue=surface_cue(h1,d1,uid,seed)+' '+surface_cue(h2,d2,uid,seed+'b')
        q=f'{rq1} {rq2} {cue} The deadline and available tools resemble a prior rushed cycle. Write the closeout that remains consistent with the user’s established decision records.'
        target=[h1,h2];state_times={h1:None,h2:None};cap='user_ratification_over_assistant_suggestion';require='Join the user’s shortlist and ratification sessions, discounting later assistant-originated proposals and local exceptions.'
    elif typ=='scope_temporal_pair':
        hsco=p.get('conditionally_scoped_habit_id');hd=dr[i%len(dr)]
        if hsco not in hs or hsco==hd:
            pool=[h for h in hs if h!=hd];hsco=pool[i%len(pool)]
        rt=replacement_time(p,hd);ct=session_ts[meta_for(p,hd)['replacement_candidate_session_id']]
        when=(ct+timedelta(days=2)) if i%2==0 else (rt+timedelta(days=30));asof=when.isoformat()
        g1=active_variant(p,hsco);g2=state_at(p,hd,when);d1=decoy_variant(p,hsco,'scope');d2=opposite_state(p,hd,g2);(rq1,l1),(rq2,l2)=scenario(p,hsco,seed+'1'),scenario(p,hd,seed+'2');opts,sigs=options_pair(p,hsco,hd,l1,l2,g1,g2,d1,d2,seed,ref)
        near=pick(['The wording resembles an older exception, although the operational facts are the ones stated here.','A familiar phrase appears in this request, while the actual scope and date remain case-specific.','The task shares vocabulary with an earlier boundary case, but not necessarily its decision state.'],uid,typ,i)
        q=f'As of {fmt_date(when,i)}, {rq1} {rq2} {near} {mislead}'
        target=[hsco,hd];state_times={hsco:None,hd:when};cap='scope_and_temporal_joint_resolution';require='Resolve a near-scope match and a dated policy state, each encoded through distant cross-session references.'
    elif typ=='provenance_weighted_triple':
        target=[hs[(i+j)%len(hs)] for j in range(3)]
        if len(set(target))<3:target=hs[:3]
        gs=[active_variant(p,h) for h in target];ds=[decoy_variant(p,h,seed+str(j)) for j,h in enumerate(target)];sc=[scenario(p,h,seed+str(j)) for j,h in enumerate(target)];opts,sigs=options_triple(p,target,[x[1] for x in sc],gs,ds,seed,ref);tasks=' '.join(x[0] for x in sc);cues=' '.join(surface_cue(h,d,uid,seed+str(j)) for j,(h,d) in enumerate(zip(target,ds)))
        q=f'{tasks} {cues} Bring the three workstreams into one auditable closeout. Several safe routes are locally available; use the routes supported by the user’s completed decision records.'
        state_times={h:None for h in target};cap='three_habit_evidence_provenance_weighting';require='Resolve three split decisions and rank user ratification above assistant suggestions and isolated exceptions.'
    else:raise KeyError(typ)
    ev=[];decoy_ids=[]
    for h in target:
        when=state_times.get(h)
        ev+=state_evidence(p,h,when)
        required_groups.append(required_component_group(p,h,when))
        m=meta_for(p,h);decoy_ids += [m['oneoff_decoy_session_id'],m['rejected_decoy_session_id']]
    if reference:ev.append(reference);required_groups.append([reference])
    ev=sorted(set(ev),key=lambda x:session_idx[x]);decoy_ids=sorted(set(decoy_ids),key=lambda x:session_idx[x])
    logicals.append({'user_id':uid,'domain':p['domain'],'probe_type':typ,'capability_group':cap,'query':re.sub(r'\s+',' ',q).strip(),'canonical_options':opts,'option_signatures':sigs,'target_habit_ids':target,'target_state_times':{h:(state_times[h].isoformat() if isinstance(state_times.get(h),datetime) else None) for h in target},'required_component_groups':required_groups,'evidence_requirement':require,'evidence_ids':ev,'decoy_ids':decoy_ids,'as_of_timestamp':asof,'reference_session_id':reference})
assert len(logicals)==2048

# Make normalized queries / choice sets unique using meaningful archive destinations shared across all choices.
SECTIONS=['reconciliation','change-control','incident-review','access-review','status-check','case-closure','audit-replay','workflow-review','handoff-control','period-close','support-follow-up','risk-review','release-readiness','evidence-review','operations-check','decision-review']
PACKETS=['quarterly operations packet','monthly control file','case-owner brief','audit working paper','service closeout','decision journal','follow-up worksheet','exception register','support ledger','verification packet','replayable case note','review-board packet','operations log','closeout workbook','handoff journal','control memo']
OWNERS=['case owner','secondary reviewer','operations lead','next-shift reviewer','audit partner','service owner','control reviewer','follow-up owner']
def unique_clause(i):return f"Place the closeout in the {SECTIONS[i%len(SECTIONS)]} section of the {PACKETS[(i//len(SECTIONS))%len(PACKETS)]} for the {OWNERS[(i//(len(SECTIONS)*len(PACKETS)))%len(OWNERS)]}."
for attr in ['query']:
    groups=defaultdict(list)
    for i,x in enumerate(logicals):groups[norm(x[attr])].append(i)
    for ids in groups.values():
        if len(ids)>1:
            for i in ids:logicals[i]['query']+=' '+unique_clause(i)
groups=defaultdict(list)
for i,x in enumerate(logicals):groups[tuple(sorted(norm(y) for y in x['canonical_options']))].append(i)
for ids in groups.values():
    if len(ids)>1:
        for i in ids:
            c=unique_clause(i);logicals[i]['query']+=' '+c;logicals[i]['canonical_options']=[y+' '+c for y in logicals[i]['canonical_options']]
assert len({norm(x['query']) for x in logicals})==2048
assert len({tuple(sorted(norm(y) for y in x['canonical_options'])) for x in logicals})==2048

# Balanced labels and option randomization.
labels=list('ABCD')*512;rng_for('labels').shuffle(labels)
public=[];keys=[]
for i,(lp,gold) in enumerate(zip(logicals,labels)):
    pid=f'mdgo_v11_probe_{i:06d}';gp='ABCD'.index(gold);others=[1,2,3];rng_for(pid,'opts').shuffle(others);order=[];it=iter(others)
    for pos in range(4):order.append(0 if pos==gp else next(it))
    choices=[{'choice_id':lab,'text':lp['canonical_options'][ci]} for lab,ci in zip('ABCD',order)]
    sigs={lab:lp['option_signatures'][ci] for lab,ci in zip('ABCD',order)}
    last=datetime.fromisoformat(life_by_user[lp['user_id']]['sessions'][-1]['timestamp']);ts=(last+timedelta(days=1+i%90,hours=i%7)).isoformat()
    public.append({'probe_id':pid,'user_id':lp['user_id'],'domain':lp['domain'],'timestamp':ts,'query':lp['query'],'choices':choices})
    idxs=[session_idx[s] for s in lp['evidence_ids']]
    keys.append({'probe_id':pid,'user_id':lp['user_id'],'domain':lp['domain'],'gold_choice_id':gold,'gold_action_text':next(c['text'] for c in choices if c['choice_id']==gold),'probe_type':lp['probe_type'],'capability_group':lp['capability_group'],'target_habit_ids':lp['target_habit_ids'],'evidence_requirement':lp['evidence_requirement'],'gold_evidence_session_ids':lp['evidence_ids'],'adversarial_decoy_session_ids':lp['decoy_ids'],'choice_policy_signatures':sigs,'surface_decoy_variants':lp['option_signatures'][1].get('variants',{}),'as_of_timestamp':lp['as_of_timestamp'],'reference_session_id':lp['reference_session_id'],'target_state_times':lp['target_state_times'],'required_component_groups':lp['required_component_groups'],'evidence_bands':sorted(set('early' if x<180 else 'middle' if x<360 else 'late' for x in idxs)),'evidence_span_sessions':max(idxs)-min(idxs) if idxs else 0,'visible_history_scope':{'through_session_index':539,'session_count':540},'scoring_contract':'exact_choice_id_match_only'})
write_jsonl(OUT/'public/probes.jsonl',public);write_jsonl(OUT/'private/probe_key.jsonl',keys)

# ---------- audits ----------
errors=[];warnings=[];key_by_id={k['probe_id']:k for k in keys};life_ids=set(life_by_user)
if len(public)!=2048 or len(keys)!=2048:errors.append('probe/key count')
if Counter(k['gold_choice_id'] for k in keys)!=Counter({'A':512,'B':512,'C':512,'D':512}):errors.append('gold labels')
if len({norm(p['query']) for p in public})!=2048:errors.append('query uniqueness')
if len({tuple(sorted(norm(c['text']) for c in p['choices'])) for p in public})!=2048:errors.append('choice-set uniqueness')
if any(l['session_count']!=540 or len(l['sessions'])!=540 for l in lifelines):errors.append('session count')
for p in public:
    k=key_by_id[p['probe_id']]
    if p['user_id'] not in life_ids:errors.append('missing user')
    if len(p['choices'])!=4 or {c['choice_id'] for c in p['choices']}!=set('ABCD'):errors.append('choice schema')
    if not k['gold_evidence_session_ids']:errors.append('empty evidence')
    if any(session_owner.get(s)!=p['user_id'] for s in k['gold_evidence_session_ids']):errors.append('evidence owner')
    if len(k['evidence_bands'])<3:errors.append(f"evidence bands {p['probe_id']}")
    if k['evidence_span_sessions']<300:errors.append(f"evidence span {p['probe_id']}")
    if any(b in norm(p['query']) for b in ['gold answer','memory test','ignore history','insufficient evidence','boundary condition','habit id']):errors.append('query leakage')

# public/private messages exact match
pub_by_sid={s['session_id']:s for l in lifelines for s in l['sessions']}
if set(pub_by_sid)!=set(session_by_id):errors.append('public/private session coverage')
for sid in pub_by_sid:
    if pub_by_sid[sid]['messages']!=session_by_id[sid]['messages']:errors.append('public/private message mismatch');break

# 34k-style contiguous-window component audit. Approximate 34k as 205 of 540 sessions.
# A policy is resolvable only when the ordered shortlist and its distant ordinal
# resolution are both visible. Every probe contains at least two such cross-session
# joins (or an additional historical reference), so no contiguous slice is complete.
WINDOW=205
window_rows=[];single_window_complete=0;best_resolved_counts=[]
for k in keys:
    groups=k['required_component_groups'];best_complete=False;best_groups=0;best_start=0
    for st in range(0,540-WINDOW+1):
        en=st+WINDOW-1
        resolved=sum(all(st<=session_idx[sid]<=en for sid in grp) for grp in groups)
        if resolved>best_groups:best_groups=resolved;best_start=st
        if resolved==len(groups):best_complete=True
    single_window_complete += int(best_complete);best_resolved_counts.append(best_groups)
    window_rows.append({'probe_id':k['probe_id'],'probe_type':k['probe_type'],'required_component_groups':len(groups),'best_resolved_groups_in_any_205_session_window':best_groups,'best_window_start':best_start,'best_window_end':best_start+WINDOW-1,'all_required_components_in_one_205_session_window':int(best_complete),'unresolved_groups_in_best_window':len(groups)-best_groups,'naive_exact_upper_bound_from_missing_binary_links':min(0.25,0.5**max(1,len(groups)-best_groups))})
if single_window_complete>0:warnings.append(f'{single_window_complete} probes have every required split component in one 205-session window')
write_csv(OUT/'reports/evidence_window_audit.csv',window_rows)
window_summary={'window_sessions':WINDOW,'probes':len(keys),'complete_probes':single_window_complete,'complete_rate':single_window_complete/len(keys),'min_unresolved_groups_in_best_window':min(r['unresolved_groups_in_best_window'] for r in window_rows),'median_unresolved_groups_in_best_window':statistics.median(r['unresolved_groups_in_best_window'] for r in window_rows),'max_best_resolved_groups':max(best_resolved_counts),'interpretation':'Construction-time evidence availability audit, not a model score. A complete policy realization requires both the distant shortlist and ordinal-resolution sessions.'}
write_json(OUT/'reports/truncated_34k_component_audit.json',window_summary)

# Heuristic shortcuts.
safety_terms={'secure','verified','approval','official','evidence','audit','protect','backup','restore','current','confirmed','reversible','masked'}
heur=Counter();per_type=defaultdict(Counter);shortcut_rows=[]
for p in public:
    k=key_by_id[p['probe_id']];gold=k['gold_choice_id'];qset=set(words(p['query']))
    wc={c['choice_id']:len(words(c['text'])) for c in p['choices']};over={c['choice_id']:len(qset&set(words(c['text']))) for c in p['choices']};safe={c['choice_id']:sum(x in safety_terms for x in words(c['text'])) for c in p['choices']}
    preds={'longest':max(wc,key=wc.get),'shortest':min(wc,key=wc.get),'query_overlap':max(over,key=over.get),'safety_lexicon':max(safe,key=safe.get)}
    # Surface-decoy heuristic selects the option matching the variants made locally convenient by the current query.
    dec=k.get('surface_decoy_variants',{})
    dec_scores={lab:sum(sig.get('variants',{}).get(h)==v for h,v in dec.items()) for lab,sig in k['choice_policy_signatures'].items()}
    preds['surface_decoy_prior']=max(dec_scores,key=dec_scores.get)
    row={'probe_id':p['probe_id'],'probe_type':k['probe_type'],'gold_choice_id':gold}
    for name,pred in preds.items():
        ok=int(pred==gold);heur[name]+=ok;per_type[k['probe_type']][name]+=ok;row[name+'_prediction']=pred;row[name+'_correct']=ok
    shortcut_rows.append(row)
write_csv(OUT/'reports/choice_shortcut_per_probe.csv',shortcut_rows)

# TF-IDF group-held-out proxy.
def grouped_proxy(include_query):
    X=[];y=[];groups=[];meta=[]
    for p in public:
        k=key_by_id[p['probe_id']]
        for c in p['choices']:
            X.append((p['query']+' [SEP] ' if include_query else '')+c['text']);y.append(int(c['choice_id']==k['gold_choice_id']));groups.append(p['user_id']);meta.append((p['probe_id'],c['choice_id']))
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=40000,sublinear_tf=True);M=vec.fit_transform(X);scores=defaultdict(dict)
    ug=np.array(groups);yu=np.array(y);idx=np.arange(len(y));gkf=GroupKFold(n_splits=6)
    for tr,te in gkf.split(idx,yu,ug):
        clf=LogisticRegression(max_iter=500,class_weight='balanced',C=.8);clf.fit(M[tr],yu[tr]);pr=clf.predict_proba(M[te])[:,1]
        for j,s in zip(te,pr):scores[meta[j][0]][meta[j][1]]=float(s)
    pred={pid:max(sc,key=sc.get) for pid,sc in scores.items()};correct={t:0 for t in TYPE_COUNTS};total={t:0 for t in TYPE_COUNTS}
    for p in public:
        k=key_by_id[p['probe_id']];t=k['probe_type'];total[t]+=1;correct[t]+=int(pred[p['probe_id']]==k['gold_choice_id'])
    return sum(correct.values())/len(public),{t:correct[t]/total[t] for t in total}
choice_tf,choice_tf_by=grouped_proxy(False);qc_tf,qc_tf_by=grouped_proxy(True)
shortcut_audit={'random_chance':.25,'tfidf_choice_only_accuracy':choice_tf,'tfidf_query_plus_choices_accuracy':qc_tf,'heuristic_accuracy':{k:v/2048 for k,v in heur.items()},'by_probe_type':{t:{'count':TYPE_COUNTS[t],'tfidf_choice_only':choice_tf_by[t],'tfidf_query_plus_choices':qc_tf_by[t],**{name:per_type[t][name]/TYPE_COUNTS[t] for name in heur}} for t in TYPE_COUNTS},'interpretation':'Construction-time lexical and prior audits only; not a Qwen3-8B score. Surface-decoy prior measures whether the current-request convenience cues alone point toward the wrong safe workflow.'}
write_json(OUT/'reports/history_free_shortcut_audit.json',shortcut_audit)

# Diversity / length statistics.
choice_word_counts=[len(words(c['text'])) for p in public for c in p['choices']];query_word_counts=[len(words(p['query'])) for p in public]
probe_div={'probes':2048,'unique_normalized_queries':len({norm(p['query']) for p in public}),'unique_unordered_choice_sets':len({tuple(sorted(norm(c['text']) for c in p['choices'])) for p in public}),'gold_labels':dict(Counter(k['gold_choice_id'] for k in keys)),'probe_types':dict(Counter(k['probe_type'] for k in keys)),'query_words':{'min':min(query_word_counts),'median':statistics.median(query_word_counts),'mean':statistics.mean(query_word_counts),'max':max(query_word_counts)},'choice_words':{'min':min(choice_word_counts),'median':statistics.median(choice_word_counts),'mean':statistics.mean(choice_word_counts),'max':max(choice_word_counts)}}
write_json(OUT/'reports/probe_diversity_report.json',probe_div)

# Token calibration from v1.0 per-user counts using character ratios of changed histories + new suffix lengths.
v10_counts=read_jsonl(BASE/'reports/full_prompt_o200k_token_counts_per_probe_calibrated.jsonl')
v10_med_by_user=defaultdict(list)
for r in v10_counts:v10_med_by_user[r['user_id']].append(r.get('estimated_o200k_tokens') or r.get('estimated_full_prompt_tokens') or r.get('tokens') or r.get('calibrated_tokens'))
v10_med_by_user={u:statistics.median([x for x in xs if isinstance(x,(int,float))]) for u,xs in v10_med_by_user.items()}
# Compare public history characters against v1.0 for each user.
base_lives={x['user_id']:x for x in read_jsonl(BASE/'public/lifelines.jsonl')}
def hist_chars(l):return sum(len(m['content']) for s in l['sessions'] for m in s['messages'])
est=[]
for p in public:
    u=p['user_id'];ratio=hist_chars(life_by_user[u])/hist_chars(base_lives[u]);suffix_chars=len(p['query'])+sum(len(c['text']) for c in p['choices']);base= v10_med_by_user.get(u,93000);tok=round(base*ratio + suffix_chars/4.0);est.append({'probe_id':p['probe_id'],'user_id':u,'estimated_o200k_tokens':tok,'method':'v1.0 calibrated user median × history-character ratio + probe suffix / 4'})
write_jsonl(OUT/'reports/full_prompt_o200k_token_counts_per_probe_calibrated.jsonl',est)
vals=[x['estimated_o200k_tokens'] for x in est]
def pct(q):return float(np.percentile(vals,q))
tokstats={'method':'Calibrated estimate, not a fresh tokenizer run. Uses v1.0 o200k per-user calibrated medians, exact v1.1/v1.0 history character ratios, and v1.1 suffix length.','count':len(vals),'min':min(vals),'p25':pct(25),'median':statistics.median(vals),'mean':statistics.mean(vals),'p95':pct(95),'max':max(vals),'within_80k_120k':sum(80000<=x<=120000 for x in vals)}
write_json(OUT/'reports/full_prompt_o200k_token_stats_calibrated.json',tokstats)

# Review queue.
review=[]
for p in public:
    k=key_by_id[p['probe_id']]
    previews=[]
    for sid in k['gold_evidence_session_ids'][:12]:
        s=session_by_id[sid];previews.append({'session_id':sid,'session_index':s['session_index'],'timestamp':s['timestamp'],'messages':s['messages']})
    review.append({'probe_id':p['probe_id'],'user_id':p['user_id'],'domain':p['domain'],'probe_type':k['probe_type'],'capability_group':k['capability_group'],'query':p['query'],'choices_json':json.dumps(p['choices'],ensure_ascii=False),'proposed_gold_choice_id':k['gold_choice_id'],'target_habit_ids_json':json.dumps(k['target_habit_ids']),'evidence_requirement':k['evidence_requirement'],'evidence_preview_json':json.dumps(previews,ensure_ascii=False),'evidence_span_sessions':k['evidence_span_sessions'],'adversarial_decoy_session_ids_json':json.dumps(k['adversarial_decoy_session_ids']),'reviewer_decision':'','reviewer_notes':''})
write_csv(OUT/'review/multidogo_finance_software_v11_review_queue_all.csv',review)

# Reports / docs.
validation={'version':'v1.1','status':'auto_validated_pending_human_audit','errors':sorted(set(errors)),'warnings':warnings,'counts':{'pseudo_users':len(profiles),'finance_users':sum(p['domain']=='finance' for p in profiles),'software_users':sum(p['domain']=='software' for p in profiles),'sessions':len(annotated_sessions),'sessions_per_user_min':min(len(l['sessions']) for l in lifelines),'sessions_per_user_max':max(len(l['sessions']) for l in lifelines),'probes':len(public),'private_keys':len(keys),'new_or_rewritten_evidence_sessions':sum(1 for s in annotated_sessions if s.get('rewrite_metadata',{}).get('rewrite_version')=='v1.1')},'gold_balance':dict(Counter(k['gold_choice_id'] for k in keys)),'probe_type_counts':dict(Counter(k['probe_type'] for k in keys)),'identity_and_lineage':{'public_private_session_ids_match':set(pub_by_sid)==set(session_by_id),'public_private_messages_match':all(pub_by_sid[s]['messages']==session_by_id[s]['messages'] for s in pub_by_sid),'evidence_owners_match':all(session_owner[s]==k['user_id'] for k in keys for s in k['gold_evidence_session_ids'])},'difficulty_structure':{'all_probes_have_early_middle_late_keyed_evidence':all(len(k['evidence_bands'])==3 for k in keys),'all_evidence_spans_at_least_300_sessions':all(k['evidence_span_sessions']>=300 for k in keys),'all_required_components_fit_one_205_session_window_count':single_window_complete,'truncated_34k_component_audit':window_summary},'shortcut_audit':shortcut_audit,'token_estimate':tokstats}
write_json(OUT/'reports/validation_report.json',validation)

report=f'''# HABIT-Bench MultiDoGO Finance & Software v1.1：Adversarial Long-Memory Probe Design

## Why v1.1 exists

The Qwen3-8B v1.0 baseline showed two unacceptable types (`priority_triple_hard` and `temporal_composition_triple`) above 50%/60%, while several other types improved too much when only a 34k slice of a roughly 90k history was supplied. v1.1 therefore changes both the public histories and the probes.

## History changes

- 54 coherent pseudo-users and 540 sessions per user are retained.
- {validation['counts']['new_or_rewritten_evidence_sessions']:,} sessions are rewritten as distributed preference evidence or cross-session reference cases.
- Each active habit has a policy shortlist and a resolution separated by more than 205 sessions; drift replacements use a second distant pair.
- Common/default workflows appear as assistant suggestions or one-case exceptions, then are explicitly kept nonbinding.
- Durable user choices are encoded by a shortlist session and a distant resolution that selects only the former/latter route by reference; dated replacements use the same split structure.
- Every durable policy is represented by an ordered shortlist and a distant ordinal resolution. No probe has all required components inside a single 205-session window (the approximate session budget for a 34k slice).

## Probe distribution

{json.dumps(TYPE_COUNTS,indent=2)}

The removed standalone types are: pure priority ordering, easy temporal triple composition, single-habit direct use, boundary-only, exception-only, and weak-signal meta questions.

## Difficulty principles

1. All four answers are safe and task-complete.
2. A plausible common workflow is deliberately present as a distractor; selected target habits usually use a different, repeatedly supported user workflow.
3. Current wording may resemble a one-off exception or an assistant suggestion, but never explicitly instructs the gold workflow.
4. Temporal questions require reconstructing two changing policies, often plus a stable third policy.
5. Reference questions require locating an old case and identifying which workstream remained open before applying a current policy.
6. User endorsement outranks assistant proposals and isolated local exceptions.
7. Choice labels are exactly balanced: A/B/C/D = 512 each.

## Construction-time no-history audit

- TF-IDF choice-only: {choice_tf:.2%}
- TF-IDF query+choices: {qc_tf:.2%}
- Surface-decoy prior: {heur['surface_decoy_prior']/2048:.2%}
- Longest choice: {heur['longest']/2048:.2%}
- Shortest choice: {heur['shortest']/2048:.2%}
- Query-overlap heuristic: {heur['query_overlap']/2048:.2%}
- Safety-lexicon heuristic: {heur['safety_lexicon']/2048:.2%}

These are construction audits, not Qwen3-8B results. The target no-memory≈20% and 34k-history<30% must be checked with the same Qwen configuration used for v1.0.
'''
(OUT/'reports/v11_probe_design_and_difficulty_report.md').write_text(report,encoding='utf-8')

readme=f'''# HABIT-Bench MultiDoGO Finance & Software v1.1

Adversarial long-memory candidate derived from v1.0, with both session evidence topology and probe design revised.

## Scale

- 54 pseudo-users: 36 Finance, 18 Software
- 540 sessions per user; 29,160 sessions total
- 2,048 probes; strict A/B/C/D exact-match scoring
- A/B/C/D gold labels: 512 each
- Full prompt calibrated estimate: {min(vals):,}–{max(vals):,} tokens; {sum(80000<=x<=120000 for x in vals)}/2,048 in 80k–120k

## Main v1.1 changes

- Removes the two v1.0 types that scored above 50%/60%.
- Rewrites policy evidence into distant shortlist/resolution pairs, while keeping rejected assistant suggestions and one-case exceptions as plausible nonbinding interference.
- Requires cross-referencing ordered shortlists with distant ordinal resolutions; no 205-session contiguous slice contains a complete required evidence set.
- Uses surface-decoy, multi-habit, temporal, provenance, and cross-session reference tasks.
- Keeps all four options safe and operationally complete.

Models receive only `public/lifelines.jsonl` and `public/probes.jsonl`.

Prediction format:
```json
{{"probe_id":"mdgo_v11_probe_000000","choice_id":"A"}}
```

Score with:
```bash
python scripts/score_predictions.py --dataset-dir . --predictions predictions.jsonl --output-dir runs/my_v11_eval --method-name my_method
```

Status: automated validation passed; target Qwen3-8B no-memory and 34k-history baselines plus human review are still required before paper-scale release.
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
(OUT/'RELEASE_NOTES.md').write_text('# v1.1 Release Notes\n\n- Rebuilt session evidence topology and all 2,048 probes.\n- Removed high-shortcut priority and temporal-triple categories.\n- Added distributed early/middle/late evidence, rejected-default interference, and historical reference cases.\n- Preserved user count, domain split, session count, source lineage, identity coherence, and exact-match scoring.\n',encoding='utf-8')

# Scorer and validator.
(OUT/'scripts').mkdir(parents=True,exist_ok=True)
shutil.copy2(BASE/'scripts/score_predictions.py',OUT/'scripts/score_predictions.py')
# Adjust scorer version-independent; existing scorer is generic.
validator='''#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\nroot=Path(sys.argv[1] if len(sys.argv)>1 else '.')\ndef jl(p): return [json.loads(x) for x in (root/p).read_text(encoding="utf-8").splitlines() if x]\nl=jl("public/lifelines.jsonl");p=jl("public/probes.jsonl");k=jl("private/probe_key.jsonl");s=jl("private/sessions_with_annotations.jsonl")\nerrs=[]\nif len(l)!=54: errs.append("lifelines")\nif len(p)!=2048 or len(k)!=2048: errs.append("probe counts")\nif sum(len(x["sessions"]) for x in l)!=29160: errs.append("session count")\nif {x["probe_id"] for x in p}!={x["probe_id"] for x in k}: errs.append("probe coverage")\nif any(len(x["gold_evidence_session_ids"])==0 for x in k): errs.append("empty evidence")\nprint(json.dumps({"errors":errs,"lifelines":len(l),"sessions":sum(len(x["sessions"]) for x in l),"probes":len(p)},indent=2))\nsys.exit(1 if errs else 0)\n'''
(OUT/'scripts/validate_v11_package.py').write_text(validator,encoding='utf-8')
(OUT/'scripts/build_v11_adversarial_memory.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')

# Model input example and template.
ex=public[0];life=life_by_user[ex['user_id']]
lines=['You are evaluating a long-horizon user-memory agent. Use the previous sessions and current request. Choose one choice_id.','',f"USER_ID: {ex['user_id']}",f"DOMAIN: {ex['domain']}",'','PREVIOUS SESSIONS:']
for s in life['sessions']:
    lines+=['',f"[Session {s['session_index']} | {s['timestamp']}]" ]+[f"{m['role'].upper()}: {m['content']}" for m in s['messages']]
lines+=['','CURRENT REQUEST:',ex['query'],'','CHOICES:']+[f"{c['choice_id']}. {c['text']}" for c in ex['choices']]+['',f'Return JSON only: {{"probe_id":"{ex["probe_id"]}","choice_id":"..."}}']
(OUT/'model_eval/example_full_prompt.txt').parent.mkdir(parents=True,exist_ok=True);(OUT/'model_eval/example_full_prompt.txt').write_text('\n'.join(lines),encoding='utf-8')
write_jsonl(OUT/'model_eval/predictions_template.jsonl',[{'probe_id':p['probe_id'],'choice_id':''} for p in public])

# Gold scorer smoke test.
write_jsonl(OUT/'reports/gold_predictions_smoke_test.jsonl',[{'probe_id':k['probe_id'],'choice_id':k['gold_choice_id']} for k in keys])
smoke=OUT/'reports/gold_smoke_test_v11';smoke.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(OUT/'scripts/score_predictions.py'),'--dataset-dir',str(OUT),'--predictions',str(OUT/'reports/gold_predictions_smoke_test.jsonl'),'--output-dir',str(smoke),'--method-name','gold_smoke_v11'],check=True)

# Package manifest / hashes.
files=[]
for path in sorted(p for p in OUT.rglob('*') if p.is_file()):files.append({'path':str(path.relative_to(OUT)),'size_bytes':path.stat().st_size,'sha256':sha256_file(path)})
write_csv(OUT/'package_file_manifest.csv',files)
(OUT/'SHA256SUMS.txt').write_text('\n'.join(f"{x['sha256']}  {x['path']}" for x in files)+'\n',encoding='utf-8')
# Include final validation after all outputs exist.
subprocess.run([sys.executable,str(OUT/'scripts/validate_v11_package.py'),str(OUT)],check=True)
if errors:raise RuntimeError(errors[:20])
if ZIP_PATH.exists():ZIP_PATH.unlink()
with zipfile.ZipFile(ZIP_PATH,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    rootname=OUT.name
    for p in sorted(x for x in OUT.rglob('*') if x.is_file()):z.write(p,Path(rootname)/p.relative_to(OUT))
sha=sha256_file(ZIP_PATH);ZIP_SHA.write_text(f"{sha}  {ZIP_PATH.name}\n",encoding='utf-8')
with zipfile.ZipFile(ZIP_PATH) as z:bad=z.testzip();assert bad is None
print(json.dumps({'out':str(OUT),'zip':str(ZIP_PATH),'zip_size':ZIP_PATH.stat().st_size,'sha256':sha,'validation_errors':errors,'warnings':warnings,'shortcut_audit':shortcut_audit,'token_stats':tokstats},indent=2))
