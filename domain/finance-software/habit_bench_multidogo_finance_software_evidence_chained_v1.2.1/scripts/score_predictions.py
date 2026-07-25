#!/usr/bin/env python3
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',required=True); p.add_argument('--predictions',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--method-name',default='method'); a=p.parse_args()
base=Path(a.dataset_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
keys=[json.loads(x) for x in open(base/'private/probe_key.jsonl',encoding='utf-8') if x.strip()]
preds=[json.loads(x) for x in open(a.predictions,encoding='utf-8') if x.strip()]
km={x['probe_id']:x for x in keys}; pm={x['probe_id']:x for x in preds}
missing=sorted(set(km)-set(pm)); extra=sorted(set(pm)-set(km)); dup=len(preds)-len(pm)
if missing or extra or dup: raise SystemExit(f'coverage error: missing={len(missing)} extra={len(extra)} duplicate={dup}')
rows=[]; by=defaultdict(lambda:[0,0]); correct=0
for pid,k in km.items():
    pred=pm[pid].get('choice_id'); ok=int(pred==k['gold_choice_id']); correct+=ok
    for field in ['probe_type','capability_group','domain']:
        name=f"{field}:{k[field]}"; by[name][0]+=ok; by[name][1]+=1
    rows.append({'probe_id':pid,'prediction':pred,'gold':k['gold_choice_id'],'correct':ok,'probe_type':k['probe_type'],'capability_group':k['capability_group'],'domain':k['domain']})
metrics={'method_name':a.method_name,'scoring':'exact_choice_id_match_only','correct':correct,'total':len(keys),'accuracy':correct/len(keys),'by_group':{k:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for k,v in sorted(by.items())}}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
with open(out/'per_probe.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(json.dumps(metrics,indent=2))
