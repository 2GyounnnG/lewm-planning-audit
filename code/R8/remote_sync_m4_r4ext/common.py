from __future__ import annotations
import hashlib, json, os
from pathlib import Path

STREAMS=('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2')
ARMS=('H0','REFIT_103201','REFIT_103202','REFIT_103203')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def atomic_npz(path,**arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def replan_seed(task,case_id,replan_index,stream='R3_ORIGINAL'):
    if task not in ('pusht','reacher') or stream not in STREAMS or not case_id or type(replan_index) is not int or replan_index<0:
        raise ValueError('Invalid fixed planning random-stream identity')
    namespace='R3_CEM_CASE_20261002' if stream=='R3_ORIGINAL' else stream
    return int.from_bytes(hashlib.sha256(f'{namespace}/{task}/{case_id}/{replan_index}'.encode()).digest()[:8],'big')

def fixed_subset(task,cases,n=20):
    """Metadata-only SHA ordering; PushT family round-robin; never outcome input."""
    ranked=sorted(cases,key=lambda c:(digest(['R4_CASE_SUBSET_V1',task,c]),c['case_id']))
    if task=='reacher':return ranked[:n]
    if task!='pusht' or any(c['family_id'] is None for c in cases):raise ValueError('Known PushT families required')
    groups={}
    for case in ranked:groups.setdefault(case['family_id'],[]).append(case)
    families=sorted(groups,key=lambda f:digest(['R4_CASE_SUBSET_V1',task,'family',f]))
    return [groups[f][i] for i in range(max(map(len,groups.values()))) for f in families if i<len(groups[f])][:n]

def file_record(path):
    p=Path(path);return {'bytes':p.stat().st_size,'sha256':sha256(p)}

def verify_complete(folder,identity_sha):
    folder=Path(folder);p=folder/'COMPLETE.json'
    if not p.exists():return None
    record=json.loads(p.read_text())
    if record['identity_sha256']!=identity_sha:raise RuntimeError('Completed identity changed')
    for name,expected in record['files'].items():
        if file_record(folder/name)!=expected:raise RuntimeError('Completed output changed: '+name)
    return json.loads((folder/'result.json').read_text())
