"""Lossless descriptive trajectory ledger; no inference or frozen-table mutation."""
import json
from pathlib import Path
from c2.common import *
def main():
    c=read(ROOT/'C2_CONTRACT.json');rows=[]
    for case in sorted(c['cases']['EVAL'],key=lambda x:x['case_id']):
        for stream in ('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2'):
            base=Path('/workspace/r5/C1') if stream=='R3_ORIGINAL' else ROOT
            for arm in ARMS:
                folder=base/'closed_loop/EVAL'/stream/arm/case['case_id'];receipt=read(folder/'COMPLETE.json');src=receipt['files']['result.json'];r=read(verify(src))
                if r['status']!='COMPLETE' or (r['case_id'],r['arm'],r['stream'])!=(case['case_id'],arm,stream):raise RuntimeError('Incomplete source matrix')
                rows.append(dict(task='cube',case_id=r['case_id'],family_id=r['family_id'],arm=arm,stream=stream,**{k:r[k] for k in ('success','any_step_success','terminal_success','final_goal_error','executed_raw_steps','replan_calls','trajectory_wall_seconds','planning_synchronized_wall_seconds','environment_step_seconds','initialization_physics_substeps','executed_physics_substeps','optimizer_updates')},method_failure=json.dumps(r['failure'],sort_keys=True),reset_label=LABEL,evidence_label=EVIDENCE,source_result_sha256=src['sha256'],source_result_path=src['path'],reused_C1=stream=='R3_ORIGINAL',derived_seed_mean=False))
    if len(rows)!=1200:raise RuntimeError('Wrong repeated-measure coverage')
    out=ROOT/'reports';csv_write(out/'TRAJECTORY_AUXILIARY_RAW.csv',rows)
    atomic(out/'AUXILIARY_RAW_LEDGER_RECEIPT.json',{'status':'COMPLETE','rows':1200,'C1_reused':400,'new_C2':800,'independent_cases':100,'no_new_statistics':True,'scientific_primary_tables_unchanged':True,'source_code':record(__file__),'raw':record(out/'TRAJECTORY_AUXILIARY_RAW.csv')})
if __name__=='__main__':main()
