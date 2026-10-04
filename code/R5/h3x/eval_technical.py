"""One metadata-first TECH episode; base H0; no parameter update/selection."""
import os,sys,json,time
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from . import common
from .evaluate import file,state_sha,verify_code

def main():
    sys.path.insert(0,'/workspace/r4_v23_execution');verify_code('/workspace/r4_v23_execution')
    import torch
    from r3.model import load_official
    from r3.data import action_processor,image_transform
    import r4.runner as runner
    from r4.export_cases import CaseWindows
    torch.set_num_threads(4);torch.set_num_interop_threads(1);common.fp32_policy()
    assets=Path('/workspace/shared_data/r4_assets_reacher');manifest=common.read_json(assets/'CASE_WINDOWS.json');entries=[e for e in manifest['tasks']['reacher']['cases'] if e['case']['role']=='TECH'];entry=entries[0]
    m=load_official('reacher','cuda');before=state_sha(m);arm='H3X_TECH_H0';runner.ARMS=tuple(runner.ARMS)+(arm,)
    provenance={'label':common.LABEL,'classification':'TECHNICAL','model':'OFFICIAL_H0_UNMODIFIED','preset_H3_status':'NOT_TRIGGERED','wrapper':file(__file__),'case_manifest':file(assets/'CASE_WINDOWS.json'),'selection':'FIRST_METADATA_TECH_CASE','optimizer_updates':0}
    out=common.ROOT/'technical/default_history_case';r=runner.run_case(m,'reacher',entry['case'],CaseWindows(assets,entry),action_processor('reacher'),image_transform(),arm=arm,stream='R3_ORIGINAL',output=out/'trajectory',provenance=provenance,phase='TECH')
    after=state_sha(m);assert before==after;assert all(x['observed_history_frames']==1 for x in r['replans'])
    common.atomic_json(out/'EVALUATION_TECH_CHECK.json',{'status':'PASS','label':common.LABEL,'classification':'TECHNICAL','case_id':entry['case']['case_id'],'success':r['success'],'executed_raw_steps':r['executed_raw_steps'],'replan_calls':r['replan_calls'],'history_frames':[x['observed_history_frames'] for x in r['replans']],'model_all_state_before':before,'model_all_state_after':after,'complete_trajectories':1,'optimizer_updates':0,'closed_formal_target':900,'case_complete':file(out/'trajectory/COMPLETE.json'),'source_wrapper':file(__file__)})
    print('CLOSED_TECH_PASS',r['executed_raw_steps'],r['replan_calls'],r['wall_seconds'],flush=True)
if __name__=='__main__':main()
