"""Seal completed evidence recovery while preserving scientific un-evaluability."""
import datetime,os
from pathlib import Path
from c2.common import *

def main():
    watched={'c2.runner','c2.prepare','c2.report','c2_recovery.compact','c2_recovery.render_diagnostic','c2_recovery.resume_audit','c2_recovery.technical_closeout','c2_recovery.partial_reference','c2_recovery.partial_archive','/workspace/r5/code/c2/launch.sh'}
    active=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:args=proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError,ProcessLookupError):continue
        for role in watched.intersection(args):active.append({'pid':int(proc.name),'role':role})
    if active:raise RuntimeError('C2 component still active: '+str(active))
    recovery=ROOT/'recovery';archive=read(recovery/'RECOVERY_ARCHIVE.json');verify(archive['archive']);verify(archive['manifest'])
    manifest=read(archive['manifest']['path'])
    for r in manifest['files'].values():verify(r)
    module=read(ROOT/'reports/MODULE_STATUS.json');local=read(recovery/'LOCAL_RECOVERY_CHECK.json');cpu=read(recovery/'cpu_reference/CPU_RECOVERY_CHECK.json');independent=read(ROOT/'reports/INDEPENDENT_DEGRADATION_CHECK.json')
    if module['status']!='TECHNICALLY_UNEVALUABLE' or module['completed_observed_trajectories']!=280 or local['status']!='PASS_NUMERIC_DERIVED_RECOVERY' or local['trajectories']!=280 or cpu['status']!='PASS':raise RuntimeError('Recovery/scope gate failed')
    if independent['status']!='PASS_TECHNICAL_DEGRADATION_ACCOUNTING' or independent['module_status_sha256']!=sha(ROOT/'reports/MODULE_STATUS.json'):raise RuntimeError('Independent audit differs')
    diagnostic=ROOT/'diagnostic/DIAGNOSTIC_RENDER';logreceipt=read(diagnostic/'LOCAL_RECOVERY_CHECK.json')
    if logreceipt['stable_members_hash_verified']!=24:raise RuntimeError('Diagnostic stable evidence not verified')
    final_log=next(x for x in logreceipt['records'] if x['relative_path']=='orchestrator.log')['actual']
    if sha(diagnostic/'orchestrator.log')!=final_log['sha256']:raise RuntimeError('Final diagnostic runtime log differs')
    c=read(ROOT/'C2_CONTRACT.json')
    completed=read(ROOT/'audit/RESUME_ATTEMPT_1/COMPLETED_280_INVENTORY.json')
    completed_ids={x['case_id'] for x in completed}
    subset_indices=[i for i,case in enumerate(c['cases']['EVAL']) if case['case_id'] in completed_ids]
    if subset_indices!=[i for i in range(100) if i%2==1 or i<40]:raise RuntimeError('Observed shard subset differs')
    for r in c['code'].values():verify(r)
    for name in ('c0_contract','c0_reset','c0_gate','c1_contract','c1_module_status'):verify(c[name])
    seal={'version':'R5_C2_TECHNICAL_DEGRADATION_RECOVERY_SEAL_V1','status':'SEALED_TECHNICALLY_UNEVALUABLE_WITH_PARTIAL_NUMERIC_AND_CPU_RECOVERY','scientific_status':'TECHNICALLY_UNEVALUABLE','module':'C2','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'planned_trajectories':800,'completed_observed_trajectories':280,'failed_before_action_design_cells':1,'not_run_design_cells':519,'failed_before_action_attempts':2,'additional_diagnostic_resets':4,'all_failed_and_diagnostic_evaluated_actions':0,'all_failed_and_diagnostic_CEM_calls':0,'unscored_failed_and_diagnostic_initialization_mj_step_calls':24,'unscored_failed_and_diagnostic_initialization_physics_substeps':600,'C2_inferential_rows':0,'C1_source_only_rows':5,'new_training_updates':0,'archive':archive['archive'],'recovery_manifest':archive['manifest'],'manifest_member_files':len(manifest['files']),'scientific_contract':record(ROOT/'C2_CONTRACT.json'),'module_status':record(ROOT/'reports/MODULE_STATUS.json'),'C1_reused_seal':record('/workspace/r5/C1/RECOVERY_SEAL.json'),'local_numeric_recovery':record(recovery/'LOCAL_RECOVERY_CHECK.json'),'CPU_recovery':record(recovery/'cpu_reference/CPU_RECOVERY_CHECK.json'),'CPU_scope':'fixed metadata-order first EVAL case, four fixed arms, actually completed ALT1 only, five executed replans; saved latent/action inference only; ALT2 absent','independent_degradation_check':record(ROOT/'reports/INDEPENDENT_DEGRADATION_CHECK.json'),'resume_audit':record(ROOT/'audit/RESUME_ATTEMPT_1/RESUME_AUDIT_RECEIPT.json'),'failed_attempts':record(ROOT/'reports/TWO_FAILED_ATTEMPTS_RAW.json'),'diagnostic_original_receipt':record(diagnostic/'DIAGNOSTIC_COMPLETE.json'),'diagnostic_log_hash_caveat':'Original receipt hashed orchestrator.log before the final summary print. That single runtime log subsequently grew; original recorded SHA is not declared valid. All 24 stable evidence files match, final log SHA separately bound.','diagnostic_local_hash_check':record(diagnostic/'LOCAL_RECOVERY_CHECK.json'),'diagnostic_final_runtime_log':record(diagnostic/'orchestrator.log'),'source_relocation_map':record(recovery/'bundle/SOURCE_RELOCATION_MAP.json'),'source_relocation_note':'Original failed paths now refer to resume attempt; original bytes retained under attempts/0 and audit/RESUME_ATTEMPT_1/preserved, mapped by old SHA to preserved path.','original_successful_trajectory_RGB_and_planner_images_local':False,'pixel_scope':'Original successful-case RGB, planner image arrays and source H5 remain remote. Complete source hashes and reconstruction recipe local; exact regeneration not verified or guaranteed after the render mismatch. Failed and diagnostic renders included locally.','local_numeric_scope':'All 280 completed actions/full nonpixel state/latents/goals/returned plans/RNG plus both zero-action failed attempts, four diagnostics and complete 800-key status inventory.','no_second_resume':True,'C0_C1_C2_frozen_code_and_tolerances_unchanged':True,'recovery_source':record(__file__)}
    seal['completed_case_subset']={'cases':70,'metadata_zero_based_indices':subset_indices,'definition':'First20 cases in shard0 plus all50 cases in shard1 under original shard scheduling; not the first70 global metadata cases. All4 arms completed in ALT1 only.','original_status_field_note':'complete_case_prefix_performed=70 in the preserved MODULE_STATUS denotes this shard-order subset, not a global metadata prefix; original table/status bytes retained.'}
    seal['quiescence_at_seal']={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'C2_simulation_diagnostic_reference_compactor_archive_processes':active,'seal_process_pid':os.getpid(),'note':'Only this short seal writer remains and exits after exclusive seal creation. Local download/numeric pipeline finishes after copying and checking this seal.'}
    freeze(ROOT/'RECOVERY_SEAL.json',seal);print(record(ROOT/'RECOVERY_SEAL.json'),flush=True)

if __name__=='__main__':main()
