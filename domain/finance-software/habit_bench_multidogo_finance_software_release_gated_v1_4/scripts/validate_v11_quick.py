#!/usr/bin/env python3
import json,sys
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else '.')
def jl(p): return [json.loads(x) for x in (root/p).read_text(encoding="utf-8").splitlines() if x]
l=jl("public/lifelines.jsonl");p=jl("public/probes.jsonl");k=jl("private/probe_key.jsonl");s=jl("private/sessions_with_annotations.jsonl")
errs=[]
if len(l)!=54: errs.append("lifelines")
if len(p)!=2048 or len(k)!=2048: errs.append("probe counts")
if sum(len(x["sessions"]) for x in l)!=29160: errs.append("session count")
if {x["probe_id"] for x in p}!={x["probe_id"] for x in k}: errs.append("probe coverage")
if any(len(x["gold_evidence_session_ids"])==0 for x in k): errs.append("empty evidence")
print(json.dumps({"errors":errs,"lifelines":len(l),"sessions":sum(len(x["sessions"]) for x in l),"probes":len(p)},indent=2))
sys.exit(1 if errs else 0)
