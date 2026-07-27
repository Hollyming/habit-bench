#!/usr/bin/env python3
from __future__ import annotations
import json,re,collections,sys
from pathlib import Path

V10=Path(sys.argv[1]);V11=Path(sys.argv[2])
def jl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def norm(s):
 s=s.lower().replace('’',"'")
 s=re.sub(r'\b\d+(?:[.,]\d+)?\b','<num>',s)
 s=re.sub(r'[^a-z0-9<>\']+',' ',s)
 return re.sub(r'\s+',' ',s).strip()
def collect(root):
 ss=jl(root/'private/sessions_with_annotations.jsonl')
 first=[];users=[];assist=[]
 for s in ss:
  seen=False
  for m in s['messages']:
   if m['role']=='user':
    users.append(m['content'])
    if not seen:first.append(m['content']);seen=True
   else:assist.append(m['content'])
 return {'first_user':first,'all_user':users,'all_assistant':assist}
def metric(turns,n=8,threshold=5):
 nts=[norm(x) for x in turns]
 exact=collections.Counter(nts)
 grams=collections.Counter()
 for t in nts:
  w=t.split()
  grams.update(tuple(w[i:i+n]) for i in range(max(0,len(w)-n+1)))
 total=sum(grams.values());high=sum(c for c in grams.values() if c>=threshold)
 return {
  'turns':len(turns),'unique_exact':len(exact),
  'exact_duplicate_occurrence_rate':sum(c for c in exact.values() if c>1)/len(turns) if turns else 0,
  'exact_excess_duplicate_rate':sum(c-1 for c in exact.values() if c>1)/len(turns) if turns else 0,
  'eightgram_occurrences':total,'eightgram_high_frequency_occurrence_mass':high/total if total else 0,
  'eightgram_frequency_threshold':threshold,'max_exact_frequency':max(exact.values(),default=0),
 }
out={'definition':'8-gram mass is the share of all normalized 8-gram occurrences belonging to an 8-gram observed at least five times. Exact duplicate rates normalize numbers before comparison.','v1.0':{},'v1.1':{}}
for version,root in [('v1.0',V10),('v1.1',V11)]:
 c=collect(root)
 out[version]={k:metric(v) for k,v in c.items()}
out['comparison']={k:{'v1.0':out['v1.0'][k],'v1.1':out['v1.1'][k]} for k in ['first_user','all_user','all_assistant']}
(V11/'reports/language_repetition_comparison_v10_v11.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
