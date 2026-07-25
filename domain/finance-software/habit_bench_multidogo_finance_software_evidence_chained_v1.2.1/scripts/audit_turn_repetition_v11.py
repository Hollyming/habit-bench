import json,re,collections,statistics,pathlib,csv
base=pathlib.Path('/mnt/data/habit_bench_multidogo_finance_software_adversarial_memory_v1_1')
sessions=[json.loads(x) for x in (base/'private/sessions_with_annotations.jsonl').read_text().splitlines() if x]

def norm(s):
 s=s.lower().replace('’',"'")
 s=re.sub(r'\b\d+(?:[.,]\d+)?\b','<num>',s)
 s=re.sub(r'[^a-z0-9<>\']+',' ',s)
 return re.sub(r'\s+',' ',s).strip()

def audit(turns,n=8,minfreq=5):
 nts=[norm(x) for x in turns]
 exact=collections.Counter(nts)
 grams=collections.Counter()
 turngrams=[]
 for t in nts:
  w=t.split(); gs=set(tuple(w[i:i+n]) for i in range(max(0,len(w)-n+1)));turngrams.append(gs);grams.update(gs)
 high={g for g,c in grams.items() if c>=minfreq}
 flagged=[bool(gs&high) for gs in turngrams]
 return {
  'turns':len(turns),
  'unique_exact':len(exact),
  'exact_duplicate_turn_rate':sum(c for c in exact.values() if c>1)/len(turns) if turns else 0,
  'exact_excess_rate':sum(c-1 for c in exact.values() if c>1)/len(turns) if turns else 0,
  'high_frequency_ngram_mass':sum(flagged)/len(turns) if turns else 0,
  'high_frequency_ngrams':len(high),
  'max_exact_frequency':max(exact.values(),default=0),
  'top_exact':exact.most_common(10),
  'top_ngrams':[(' '.join(g),c) for g,c in grams.most_common(20)]
 }

all_user=[];all_ass=[];first_user=[];cats=collections.defaultdict(list)
for s in sessions:
 cats_list=s.get('rewrite_metadata',{}).get('turn_categories',[])
 ai=0
 for i,m in enumerate(s['messages']):
  if m['role']=='user':
   all_user.append(m['content'])
   if i==0:first_user.append(m['content'])
  else:all_ass.append(m['content'])
 # Map by position to categories if same length, otherwise broad rewrite scope.
 if len(cats_list)==len(s['messages']):
  for c,m in zip(cats_list,s['messages']):cats[c].append(m['content'])
 else:
  scope=s.get('rewrite_metadata',{}).get('rewrite_scope','unclassified')
  for m in s['messages']:cats[scope+'|'+m['role']].append(m['content'])

out={'all_user':audit(all_user),'first_user':audit(first_user),'all_assistant':audit(all_ass),'categories':{k:audit(v) for k,v in sorted(cats.items()) if len(v)>=20}}
(base/'reports/turn_repetition_audit_v11.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
rows=[]
for k,v in [('all_user',out['all_user']),('first_user',out['first_user']),('all_assistant',out['all_assistant'])]+list(out['categories'].items()):
 rows.append({'category':k,**{x:v[x] for x in ['turns','unique_exact','exact_duplicate_turn_rate','exact_excess_rate','high_frequency_ngram_mass','high_frequency_ngrams','max_exact_frequency']}})
with (base/'reports/turn_repetition_by_category_v11.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps({k:{x:v[x] for x in ['turns','exact_duplicate_turn_rate','high_frequency_ngram_mass','max_exact_frequency']} for k,v in out.items() if k!='categories'},indent=2))
print('categories')
for k,v in out['categories'].items():
 if v['high_frequency_ngram_mass']>.25 or v['exact_duplicate_turn_rate']>.15:
  print(k,v['turns'],round(v['high_frequency_ngram_mass'],3),round(v['exact_duplicate_turn_rate'],3),v['max_exact_frequency'])
