#!/usr/bin/env python3
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
