"""Descriptive closeout after the single allowed native resume failed; no inference."""
import csv,datetime
from pathlib import Path
from c2.common import *
from c2.report import MEAN,ALL_STREAMS
REASON='REPEATED_PRE_CEM_BITWISE_RENDER_MISMATCH_CAUSE_UNPROVEN'
PARTIAL='PARTIAL_TECHNICAL_DIAGNOSTIC_NOT_INFERENTIAL'

def main():
    c=read(ROOT/'C2_CONTRACT.json');out=ROOT/'reports';raw=[];coverage=[];sources=[];failures=[]
    audit=read(ROOT/'audit/RESUME_ATTEMPT_1/RESUME_AUDIT_RECEIPT.json')
    for r in audit['locks'].values():verify(r)
    prior=read(verify(audit['completed_inventory']))
    for item in prior:
        verify(item['complete'])
        for r in item['files'].values():verify(r)
    for case in sorted(c['cases']['EVAL'],key=lambda x:x['case_id']):
      for stream in STREAMS:
       for arm in ARMS:
        folder=ROOT/'closed_loop/EVAL'/stream/arm/case['case_id'];result=folder/'result.json';complete=folder/'COMPLETE.json'
        row=dict(task='cube',case_id=case['case_id'],arm=arm,stream=stream,expected=True,status='NOT_RUN',success='NA',executed_raw_steps='NA',replan_calls='NA',reason='MODULE_STOPPED_AT_TECHNICAL_GATE',source_result_sha256='',source_complete_sha256='',reset_label=LABEL)
        if complete.exists():
            receipt=read(complete)
            for r in receipt['files'].values():verify(r)
            r=read(result);src=record(result);sources.append(src)
            if r['status']!='COMPLETE':raise RuntimeError('Inconsistent completed row')
            row.update(status='OBSERVED_COMPLETE',success=r['success'],executed_raw_steps=r['executed_raw_steps'],replan_calls=r['replan_calls'],reason='',source_result_sha256=src['sha256'],source_complete_sha256=sha(complete))
            raw.append(dict(task='cube',case_id=case['case_id'],family_id=case['family_id'],arm=arm,stream=stream,evaluation_status=PARTIAL,**{k:r[k] for k in ('success','any_step_success','terminal_success','final_goal_error','executed_raw_steps','replan_calls','trajectory_wall_seconds','planning_synchronized_wall_seconds','environment_step_seconds','initialization_physics_substeps','executed_physics_substeps','optimizer_updates')},reset_label=LABEL,evidence_label=EVIDENCE,source_result_path=str(result),source_sha256=src['sha256'],source_complete_sha256=sha(complete),derived_within_case_refit_mean=False))
        elif result.exists():
            r=read(result)
            if r['status']!='INFRASTRUCTURE_FAILURE' or r['executed_raw_steps']!=0 or r['replan_calls']!=0:raise RuntimeError('Unexpected failed attempt')
            row.update(status='FAILED_BEFORE_ACTION_AND_CEM',executed_raw_steps=0,replan_calls=0,reason=REASON,source_result_sha256=sha(result))
            for label,path in [('ORIGINAL',folder/'attempts/0'),('SINGLE_NATIVE_RESUME',folder)]:
                fr=read(path/'result.json');diff=list(csv.DictReader((path/'C1_INITIAL_EQUIVALENCE_RAW.csv').open()));bad=[x for x in diff if x['bitwise_equal']!='True']
                if len(diff)!=76 or len(bad)!=1 or bad[0]['field']!='render' or int(bad[0]['different_elements'])!=13 or float(bad[0]['max_abs_difference'])!=1 or fr['executed_raw_steps']!=0 or fr['replan_calls']!=0:raise RuntimeError('Failure evidence differs')
                failures.append(dict(case_id=case['case_id'],arm=arm,stream=stream,attempt=label,utc=fr['utc'],fields_compared=76,nonrender_fields_exact=75,render_different_channels=13,render_max_abs=1,evaluated_raw_actions=0,CEM_calls=0,initialization_mj_step_calls=len(fr['mj_step_ledger']),initialization_physics_substeps=fr['initialization_physics_substeps'],result=record(path/'result.json'),field_comparison=record(path/'C1_INITIAL_EQUIVALENCE_RAW.csv')))
        coverage.append(row)
    counts={s:sum(r['status']==s for r in coverage) for s in ('OBSERVED_COMPLETE','FAILED_BEFORE_ACTION_AND_CEM','NOT_RUN')}
    if counts!={'OBSERVED_COMPLETE':280,'FAILED_BEFORE_ACTION_AND_CEM':1,'NOT_RUN':519} or len(raw)!=280 or len(failures)!=2:raise RuntimeError('Final coverage differs')
    csv_write(out/'C2_PARTIAL_RAW_VALUES.csv',raw);csv_write(out/'C2_EXPECTED_CELL_STATUS.csv',coverage);atomic(out/'TWO_FAILED_ATTEMPTS_RAW.json',failures)
    mainpath=Path('/workspace/r5/C1/reports/MAIN_TABLE.csv');old={r['arm']:r for r in csv.DictReader(mainpath.open())};flip={r['arm']:r for r in csv.DictReader(Path('/workspace/r5/C1/reports/PAIRED_FLIPS.csv').open())};table=[]
    def add(kind,arm,stream,reference):
        only=kind=='SAME_STREAM_MODEL' and stream=='R3_ORIGINAL';observed=100 if only else 70 if stream=='R4_ALT_CEM_1' and (kind=='SAME_STREAM_MODEL' or reference=='R3_ORIGINAL') else 0
        row=dict(task='cube',comparison=kind,arm=arm,stream=stream,reference=reference,cases=100,success_percent='NA',success_ci95_low='NA',success_ci95_high='NA',delta_success_pp='NA',delta_ci95_low='NA',delta_ci95_high='NA',failure_to_success='NA',success_to_failure='NA',reset_label=LABEL,evidence_label=EVIDENCE,bootstrap_replicates=5000 if only else 0,independent_unit='CASE',source_sha256=sha(mainpath) if only else digest(sources),statistics_source_sha256=c['statistics_source']['sha256'],report_source_sha256=sha(__file__),frozen_report_source_sha256=c['code']['report.py']['sha256'],estimation_status='C1_SOURCE_ONLY_UNCHANGED' if only else 'TECHNICALLY_UNEVALUABLE',observed_complete_case_count=observed,reason='' if only else REASON)
        if only:
            r=old[arm]
            for dest,source in [('success_percent','success_percent'),('success_ci95_low','success_case_ci95_low'),('success_ci95_high','success_case_ci95_high'),('delta_success_pp','delta_success_pp'),('delta_ci95_low','delta_case_ci95_low'),('delta_ci95_high','delta_case_ci95_high')]:row[dest]=r[source] or 'NA'
            if arm in flip:row.update(failure_to_success=flip[arm]['s01_H0_failure_REFIT_success'],success_to_failure=flip[arm]['s10_H0_success_REFIT_failure'])
        table.append(row)
    for stream in ALL_STREAMS:
        for arm in ARMS+(MEAN,):add('SAME_STREAM_MODEL',arm,stream,'H0')
    for arm in ARMS+(MEAN,):
        for a,b in ((0,1),(0,2),(1,2)):add('SAME_MODEL_STREAM',arm,ALL_STREAMS[b],ALL_STREAMS[a])
        add('THREE_STREAM_CASE_MEAN',arm,'MEAN_OF_FIXED_THREE_STREAMS','H0')
    csv_write(out/'CUBE_PLANNER_RANDOMNESS_TABLE.csv',table)
    (out/'CONCLUSION_ZH.txt').write_text('C2技术不可评价：800个设计单元中280条完成、1个单元在动作/CEM前失败、519未运行。\n原失败与唯一一次原生恢复均为同case的13个RGB通道相差1灰阶，75非渲染字段逐位一致。\n4次固定无动作诊断均通过，故原因仍未证实；零容差及冻结配置未变，不再重启。\n280条仅作非推断性原值证据；所有涉及C2的总体估计和区间均NA，未把缺失计为失败0。\nC1原流400条及其已交付主表保持有效；新增神经训练为0。\n')
    files={p.name:record(p) for p in out.iterdir() if p.is_file() and p.name!='MODULE_STATUS.json'}
    atomic(out/'MODULE_STATUS.json',{'status':'TECHNICALLY_UNEVALUABLE','module':'C2','expected_new_trajectories':800,'completed_observed_trajectories':280,'failed_design_cells_before_action_and_CEM':1,'failed_attempts_before_action_and_CEM':2,'not_run_design_cells':519,'formal_design_cases':100,'complete_case_prefix_performed':70,'partial_raw_status':PARTIAL,'table_rows':35,'C2_estimable_rows':0,'C1_source_only_rows':5,'raw_rows':280,'coverage_rows':800,'reason':REASON,'no_second_resume':True,'new_training_updates':0,'reused_C1_trajectories':400,'C1_sources_unchanged':True,'prior_280_complete_and_all_files_unchanged':True,'contract':record(ROOT/'C2_CONTRACT.json'),'resume_audit':record(ROOT/'audit/RESUME_ATTEMPT_1/RESUME_AUDIT_RECEIPT.json'),'diagnostic':record(ROOT/'diagnostic/DIAGNOSTIC_RENDER/DIAGNOSTIC_COMPLETE.json'),'source_code':record(__file__),'sources':sources,'files':files,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print({'status':'TECHNICALLY_UNEVALUABLE','counts':counts,'table_rows':len(table)},flush=True)

if __name__=='__main__':main()
