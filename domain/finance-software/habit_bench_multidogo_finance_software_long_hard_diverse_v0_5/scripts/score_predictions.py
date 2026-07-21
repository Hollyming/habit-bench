#!/usr/bin/env python3
import argparse,json,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--dataset-dir",required=True);p.add_argument("--predictions",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--method-name",default="method");a=p.parse_args()
base=Path(a.dataset_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
keys={r["probe_id"]:r for r in (json.loads(x) for x in (base/"private/probe_key.jsonl").read_text().splitlines() if x.strip())}
preds={r["probe_id"]:r for r in (json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip())}
if set(preds)!=set(keys):raise SystemExit(f"coverage mismatch missing={len(set(keys)-set(preds))} extra={len(set(preds)-set(keys))}")
rows=[]
for pid,k in keys.items():rows.append({"probe_id":pid,"correct":int(preds[pid]["choice_id"]==k["gold_choice_id"]),"probe_type":k["probe_type"],"target_habit_count":len(k["target_habit_ids"])})
summary={"method_name":a.method_name,"total":len(rows),"accuracy":sum(r["correct"] for r in rows)/len(rows)}
for t in sorted({r["probe_type"] for r in rows}):
 s=[r for r in rows if r["probe_type"]==t];summary[f"accuracy__{t}"]=sum(r["correct"] for r in s)/len(s)
(out/"metrics.json").write_text(json.dumps(summary,indent=2))
with (out/"per_probe.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
