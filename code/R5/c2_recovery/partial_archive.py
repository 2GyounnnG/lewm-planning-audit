"""280 completed trajectories plus technical-hold evidence; no claim of full C2."""
import fcntl,shutil,tarfile
from pathlib import Path
from c2.common import *
from c2_recovery.compact import compact

def main():
    recovery=ROOT/'recovery';out=recovery/'bundle';lock=(recovery/'COMPACT.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=read(ROOT/'reports/MODULE_STATUS.json')
    if status['status']!='TECHNICALLY_UNEVALUABLE' or status['completed_observed_trajectories']!=280:raise RuntimeError('Scope differs')
    complete=sorted((ROOT/'closed_loop/EVAL').rglob('COMPLETE.json'))
    if len(complete)!=280:raise RuntimeError('Coverage changed')
    for receipt in complete:compact(receipt.parent,out)
    for name in ('C2_CONTRACT.json','MAIN_DELIVERY_GATE.json'):
        shutil.copy2(ROOT/name,out/name)
    for name in ('reports','logs','jobs','diagnostic','audit'):
        shutil.copytree(ROOT/name,out/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.lock'))
    for p in ROOT.glob('*PID'):shutil.copy2(p,out/p.name)
    for p in ROOT.glob('*SUPERVISOR.log'):shutil.copy2(p,out/p.name)
    failure=ROOT/'closed_loop/EVAL/R4_ALT_CEM_1/H0/R3_cube_ed7d40ae89e31ab29e099b34'
    shutil.copytree(failure,out/'failed_attempts/R3_cube_ed7d40ae89e31ab29e099b34',dirs_exist_ok=True,ignore=shutil.ignore_patterns('RUN.lock'))
    shutil.copytree(recovery/'cpu_reference',out/'cpu_reference',dirs_exist_ok=True)
    for group in ('c2','c2_recovery'):
        shutil.copytree(Path('/workspace/r5/code')/group,out/'code'/group,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    preserved=read(ROOT/'audit/RESUME_ATTEMPT_1/PRESERVED_METADATA_MANIFEST.json')
    relocations=[{'original':item['source'],'preserved':item['copy'],'bundle_copy':str(Path('audit/RESUME_ATTEMPT_1/preserved')/Path(item['source']['path']).relative_to(ROOT))} for item in preserved]
    atomic(out/'SOURCE_RELOCATION_MAP.json',{'scope':'Original pre-resume path may now contain the second attempt; archived copies retain original bytes/SHA. No source is silently overwritten without archival.','entries':relocations})
    files={str(p.relative_to(out)):record(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='RECOVERY_MANIFEST.json'}
    c=read(ROOT/'C2_CONTRACT.json')
    manifest={'status':'READY_FOR_LOCAL_PARTIAL_RECOVERY','module_scientific_status':'TECHNICALLY_UNEVALUABLE','module':'C2','planned_trajectories':800,'completed_observed_trajectories':280,'derived_trajectories':280,'failed_before_action_design_cells':1,'failed_before_action_attempts':2,'not_run_design_cells':519,'fixed_no_action_diagnostic_resets':4,'C1_reused_seal':record('/workspace/r5/C1/RECOVERY_SEAL.json'),'local_restore_root':'bundle','original_RGB_and_source_H5_retained_remote':True,'original_successful_trajectory_RGB_local':False,'pixel_reconstruction_status':'RECIPE_AND_SOURCE_SHA_ONLY; exact pixel regeneration not verified and not guaranteed after observed render mismatch. Original successful-case pixels remain remote. Failed snapshots and 4 diagnostic renders included locally.','pixel_reconstruction_c0':c['c0_contract'],'reused_models':c['models'],'files':files,'source_code':record(__file__)}
    atomic(out/'RECOVERY_MANIFEST.json',manifest)
    archive=recovery/'C2_partial_minimal_v1.tar.gz'
    if archive.exists():raise RuntimeError('Do not replace sealed candidate archive')
    with tarfile.open(archive.with_suffix('.tmp'),'w:gz') as tf:tf.add(out,arcname='bundle')
    archive.with_suffix('.tmp').replace(archive)
    atomic(recovery/'RECOVERY_ARCHIVE.json',{'status':'READY_PARTIAL_TECHNICAL_RECOVERY','archive':record(archive),'manifest':record(out/'RECOVERY_MANIFEST.json'),'originals_not_removed':True,'planned':800,'observed':280,'members':len(files)+1})
    print(read(recovery/'RECOVERY_ARCHIVE.json'),flush=True)

if __name__=='__main__':main()
