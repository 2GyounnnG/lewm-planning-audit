from __future__ import annotations
import csv, hashlib, json, os
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def record(path, root):
    st=path.stat(); return {'path':str(path.relative_to(root)),'bytes':st.st_size,'sha256':sha(path)}
def write_manifest(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
def main():
    root=Path(os.environ.get('R7_ROOT','/workspace/r7'));ops=root/'ops';reports=root/'reports'
    A=[]
    for task in ('reacher','pusht'):
        A.extend(sorted((root/'raw'/task/'FORMAL').glob('**/result.json')))
        A.extend(sorted((root/'raw'/task/'FORMAL').glob('**/trajectory.npz')))
    A_man=[record(p,root) for p in A];write_manifest(ops/'R7_A_RAW_MANIFEST.csv',A_man)
    B=[]
    for task in ('reacher','pusht','tworoom','cube'): B.append(root/'random'/task/f'{task}_RANDOM_RAW_VALUES.csv')
    B_man=[record(p,root) for p in B];write_manifest(ops/'R7_B_RAW_MANIFEST.csv',B_man)
    shared=['ops/R7_FORMAL_LAUNCH.json','ops/R7_REACHER_CASE_SELECTION.json','ops/R7_PUSHT_CASE_SELECTION.json','ops/R7_REACHER_TECH_GATE.json','ops/R7_PUSHT_TECH_GATE.json','reports/R7_STATS.json']
    common=[record(root/x,root) for x in shared]
    ar=[record(p,root) for p in sorted(reports.glob('A_*'))]+[record(ops/'R7_A_RAW_MANIFEST.csv',root)]
    br=[record(p,root) for p in sorted(reports.glob('B_*'))]+[record(ops/'R7_B_RAW_MANIFEST.csv',root)]
    sealA={'status':'SEALED','evidence_label':'PREREGISTERED_STRESS_TEST_FRESH_CASES','module':'R7_A_LONG_HORIZON','formal_result_count':len([p for p in A if p.name=='result.json']),'trajectory_count':len([p for p in A if p.name=='trajectory.npz']),'reports':ar,'common_inputs':common,'remote_originals_retained':True,'no_training':True,'no_result_selection':True}
    sealB={'status':'SEALED','evidence_label':'DESCRIPTIVE_REFERENCE','module':'R7_B_RANDOM_ACTION_LOWER_BOUND','raw_files':B_man,'reports':br,'common_inputs':common,'remote_originals_retained':True,'no_training':True,'no_result_selection':True}
    (ops/'R7_SEAL_A.json').write_text(json.dumps(sealA,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    (ops/'R7_SEAL_B.json').write_text(json.dumps(sealB,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    combined={'status':'SEALED','A':'ops/R7_SEAL_A.json','B':'ops/R7_SEAL_B.json','A_sha256':sha(ops/'R7_SEAL_A.json'),'B_sha256':sha(ops/'R7_SEAL_B.json'),'remote_root':'/workspace/r7','destroy':'not performed'}
    (ops/'R7_SEAL.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'SEALED','A_results':sealA['formal_result_count'],'B_files':len(B_man),'A_seal_sha256':combined['A_sha256'],'B_seal_sha256':combined['B_sha256']}))
if __name__=='__main__':main()
