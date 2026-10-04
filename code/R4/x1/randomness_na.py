"""Cube's explicit missing planning-stream interface; no bootstrap or trials."""
from . import core
from .evaluate import write_csv

if __name__=='__main__':
    root=core.ROOT;gate=root/'reset_audit/RESET_FALLBACK.json';audit=core.read(gate)
    if root!=core.Path('/workspace/x1_cube') or audit['status']!='BLOCKED_RESET_FALLBACK' or audit['raw_calls']!=16:raise RuntimeError('Original Cube exact gate required')
    cases=sorted(c['case_id'] for c in core.read(root/'manifests/cube_data_roles.json')['cases']['EVAL']);assert len(cases)==100
    if list((root/'closed_loop/EVAL').rglob('COMPLETE.json')):raise RuntimeError('Unexpected Cube closed-loop observation')
    arms=['H0']+[f'REFIT_{s}' for s in core.SEEDS];streams=['R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2'];reason='RESET_FALLBACK_EXACT_CHECK_FAILED'
    def row(arm,reference,**context):
        return dict(context,task='cube',metric='success_fraction',arm=arm,estimate=None,cases=0,universe_cases=100,
            conditional95_low=None,conditional95_high=None,paired_cases=0,reference_arm=reference,difference_vs_reference=None,
            difference95_low=None,difference95_high=None,family_difference95_low=None,family_difference95_high=None,
            multiplicity='UNADJUSTED_POINTWISE',independent_unit='CASE',status='TECHNICALLY_UNEVALUABLE',reason=reason,observed_TECH_P90_seconds=None)
    random=[row(s,streams[0],policy=a,scope='SAME_MODEL_DIFFERENT_PLANNER_STREAM') for a in arms for s in streams]
    models=[row(a,'H0',scope='LEARNING_POLICIES_ALL100',stream=s) for s in streams for a in arms+['FIXED3_REFIT_MEAN_NOT_ENSEMBLE']]
    raw=[dict(task='cube',case_id=c,family_id=None,arm=a,stream=s,success=None,final_goal_error=None,executed_raw_steps=None,replan_calls=None,trajectory_wall_seconds=None,observation_status='MISSING',missing_reason=reason) for c in cases for a in arms for s in streams]
    out=root/'reports/cube_RANDOMNESS'
    for name,values in [('PLANNING_RANDOMNESS_TABLE.csv',random),('S3_LEARNING_MAIN_TABLE.csv',models),('ALL_RAW_VALUES.csv',raw)]:write_csv(out/name,values)
    core.atomic(out/'MODULE_STATUS.json',{'task':'cube','status':'COMPLETE_WITH_TECHNICAL_LIMITATIONS','reason':reason,'cases':0,'universe_cases':100,
        'observed_closed_loop_trajectories':0,'explicit_missing_rows':1200,'observed_TECH_P90_seconds':None,'second_batch_status':'TECHNICALLY_INELIGIBLE_RESET_GATE',
        'not_interpreted_as_P90_above_30_seconds':True,'bootstrap_performed':False,'new_optimizer_updates':0,'new_trials':0,
        'reset_gate':core.file_record(gate),'source':core.file_record(__file__),'tables':{n:core.file_record(out/n) for n in ('PLANNING_RANDOMNESS_TABLE.csv','S3_LEARNING_MAIN_TABLE.csv','ALL_RAW_VALUES.csv')}})
    (out/'CONCLUSION_ZH.txt').write_text('Cube 四个模型×三个规划流均保留为技术缺失，真实闭环轨迹为0。\n每行cases=0、universe_cases=100；成功率和区间为空，不填0。\n单条闭环TECH P90未观测，也未判定为超过30秒。\n原因是原source exact复位门槛失败，原16次证据SHA随表保存。\n未追加规划流、复位试验或bootstrap，所有1200个预定单元均显式缺失。\n')
    print('CUBE_PLANNING_MISSING_TABLE_COMPLETE')
