import json,sys
from pathlib import Path
import torch,numpy as np
sys.path[:0]=['/workspace/r10/code/r10','/workspace/r10/code/r8']
import old_history_runner as h
from x1 import core
import r8_m2_x1_runner as r8
core.R3_ROOT=Path('/workspace/r10/shared_data/r3');core.ROOT=Path('/workspace/r10/model_assets/tworoom');core.policy()
model,scaler=h.load('tworoom','H0','cuda')
e=next(x for x in json.loads(Path('/workspace/r10/inputs/M2_CASE_WINDOWS_TWOROOM.json').read_text())['cases'] if x['case']['case_id']=='R8_tworoom_00475b9743bd99184577e0e2')
raw=h.CaseWindows('/workspace/r10/assets/M2/tworoom',e);raw.file=raw.arrays
out=Path('/workspace/r10/diagnostics/B_exact_r8_single')
r=r8.run_case(core,'tworoom',e['case'],'H0','R3_ORIGINAL',model,raw,out,'cuda')
refs=json.loads(Path('/workspace/r10/inputs/TWOROOM_BASELINE.json').read_text());ref=next(x for x in refs if x['case_id']==e['case']['case_id'] and x['arm']=='H0' and x['stream']=='R3_ORIGINAL')
a=np.load(out/'trajectory.npz')['returned_plans'][0,0];b=np.load('/workspace/r10/inputs/TWOROOM_FIRST_PLANS.npz')['plans'][ref['plan_index']]
print(json.dumps({'first_plan_equal':a.tobytes()==b.tobytes(),'max_diff':float(abs(a-b).max()),'model_digest':r['model_digest_before']}))
