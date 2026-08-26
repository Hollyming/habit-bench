#!/usr/bin/env python3
import json,sys,subprocess,tempfile
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
out=Path(sys.argv[2]) if len(sys.argv)>2 else root/'reports/strict_validation_report_v14.json'
gate=json.loads((root/'reports/release_gate_validation.json').read_text())
if gate.get('status')!='pass' or gate.get('gate_pass_count')!=2048:
    raise SystemExit('v1.4 release gates failed')
with tempfile.NamedTemporaryFile(suffix='.json',delete=False) as f: tmp=Path(f.name)
subprocess.run([sys.executable,str(root/'scripts/validate_v13_package.py'),str(root),str(tmp)],check=True,stdout=subprocess.DEVNULL)
base=json.loads(tmp.read_text()); tmp.unlink(missing_ok=True)
base['version']='v1.4'; base['validator_base']='v1.3 structural invariants plus v1.4 release gates'; base['release_gates']=gate['gates']; base['v14_patch_counts']=gate['patch_counts']
out.write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(base,ensure_ascii=False,indent=2))
