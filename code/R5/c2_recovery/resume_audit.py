"""One explicitly authorized native resume; no changes to frozen scientific code."""
import datetime,fcntl,os,shutil,subprocess
from pathlib import Path
from c2.common import *

def main():
    out=ROOT/'audit/RESUME_ATTEMPT_1'
    out.mkdir(parents=True,exist_ok=False)
    with (ROOT/'SUPERVISOR.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=read(ROOT/'C2_CONTRACT.json');contract_before=record(ROOT/'C2_CONTRACT.json')
        locks={'C2_contract':contract_before,'C1_contract':c['c1_contract'],'C0_contract':c['c0_contract'],'C0_reset':c['c0_reset'],'C0_gate':c['c0_gate'],**{'code:'+k:v for k,v in c['code'].items()}}
        for item in locks.values():verify(item)
        completed=[]
        for p in sorted((ROOT/'closed_loop/EVAL').rglob('COMPLETE.json')):
            r=read(p)
            for item in r['files'].values():verify(item)
            result=read(p.parent/'result.json')
            if result['status']!='COMPLETE':raise RuntimeError('Unexpected completed status')
            import csv
            rows=list(csv.DictReader((p.parent/'C1_INITIAL_EQUIVALENCE_RAW.csv').open()))
            if len(rows)!=76 or any(x['bitwise_equal']!='True' for x in rows):raise RuntimeError('Prior gate differs')
            completed.append({'case_id':result['case_id'],'arm':result['arm'],'stream':result['stream'],'complete':record(p),'result':record(p.parent/'result.json'),'initial_gate':record(p.parent/'C1_INITIAL_EQUIVALENCE_RAW.csv'),'identity_sha256':r['identity_sha256'],'files':r['files']})
        if len(completed)!=280:raise RuntimeError('Expected exact frozen 280 complete before resume')
        atomic(out/'COMPLETED_280_INVENTORY.json',completed)
        saved=[]
        candidates=list((ROOT/'logs').rglob('*'))+list((ROOT/'jobs').rglob('*'))+list(ROOT.glob('*PID'))
        fail=ROOT/'closed_loop/EVAL/R4_ALT_CEM_1/H0/R3_cube_ed7d40ae89e31ab29e099b34'
        candidates+=list(fail.glob('*'))+list((ROOT/'diagnostic').rglob('*'))
        for p in sorted(set(candidates)):
            if not p.is_file() or p.name.endswith('.lock'):continue
            q=out/'preserved'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
            source=record(p);shutil.copy2(p,q)
            if sha(q)!=source['sha256']:raise RuntimeError('Backup differs')
            saved.append({'source':source,'copy':record(q)})
        failure=read(fail/'result.json')
        if failure['executed_raw_steps']!=0 or failure['replan_calls']!=0:raise RuntimeError('Failed attempt unexpectedly evaluated')
        diagnostic=read(ROOT/'diagnostic/DIAGNOSTIC_RENDER/DIAGNOSTIC_COMPLETE.json')
        atomic(out/'PRESERVED_METADATA_MANIFEST.json',saved)
        with (out/'PREPARE_VERIFY.log').open('wb') as log:
            subprocess.run(['/workspace/env/bin/python','-B','-m','c2.prepare'],check=True,stdout=log,stderr=subprocess.STDOUT)
        for item in locks.values():verify(item)
        ledger={'failed_attempt':{'initialization_mj_step_calls':len(failure['mj_step_ledger']),'physics_substeps':failure['initialization_physics_substeps'],'evaluated_raw_actions':0,'CEM_calls':0},'fixed_diagnostic':{'independent_resets':4,'initialization_mj_step_calls':diagnostic['initialization_mj_step_calls'],'physics_substeps':diagnostic['initialization_physics_substeps'],'evaluated_raw_actions':0,'CEM_calls':0}}
        receipt={'status':'READY_FOR_ONE_NATIVE_RESUME','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authority':'Parent explicit authorization: one native resume, original contract/code/GPU0/1/CPU80–111/case/seed/stream/bitwise tolerance; no second restart after any new technical failure.','reason':'Cause unproven. Four fixed no-action diagnostics exact; this is infrastructure recovery under already frozen STARTED/no-COMPLETE semantics, not a scientific protocol repair.','completed_observed':280,'remaining_evaluated_trajectories':520,'final_formal_budget':800,'skip_complete_identity_and_sha_verified':True,'all_prior_280_initial_gates_76_fields_pass':True,'frozen_sources_before_and_after_prepare_unchanged':True,'locks':locks,'preserved_metadata':record(out/'PRESERVED_METADATA_MANIFEST.json'),'completed_inventory':record(out/'COMPLETED_280_INVENTORY.json'),'preparation_check':record(out/'PREPARE_VERIFY.log'),'initialization_ledger':ledger,'diagnostic_contract':record(ROOT/'diagnostic/DIAGNOSTIC_RENDER/CONTRACT.json'),'diagnostic_receipt':record(ROOT/'diagnostic/DIAGNOSTIC_RENDER/DIAGNOSTIC_COMPLETE.json'),'source_code':record(__file__),'resume_code_lines':{'skip_verified_complete':'c2/runner.py:47-53','preserve_partial_same_identity':'c2/runner.py:54-61','exact_initial_gate':'c2/runner.py:77-84','new_world_and_close':'c2/runner.py:102,121-126'},'single_resumption_only':True,'tolerance_changed':False,'new_seed_or_case':False}
        atomic(out/'RESUME_AUDIT_RECEIPT.json',receipt)
        print(receipt,flush=True)

if __name__=='__main__':main()
