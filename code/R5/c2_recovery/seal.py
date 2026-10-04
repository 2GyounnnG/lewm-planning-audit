"""Additive recovery seal; does not rewrite the C1 result contract/archive."""
from pathlib import Path
import datetime
from c2.common import ROOT,read,record,verify,freeze,sha

def main():
    base=ROOT/'recovery';archive=read(base/'RECOVERY_ARCHIVE.json');verify(archive['archive']);verify(archive['manifest'])
    manifest=read(archive['manifest']['path']);bad=[]
    for key,r in manifest['files'].items():
        try:verify(r)
        except Exception as e:bad.append({'file':key,'error':str(e)})
    if bad:raise RuntimeError(str(bad))
    check=read(base/'LOCAL_RECOVERY_CHECK.json');cpu=read(base/'cpu_reference/CPU_RECOVERY_CHECK.json');stat=read(ROOT/'reports/INDEPENDENT_C2_TABLE_CHECK.json')
    if check['status']!='PASS_NUMERIC_DERIVED_RECOVERY' or cpu['status']!='PASS':raise RuntimeError('Local recovery not passed')
    c=read(ROOT/'C2_CONTRACT.json');module=read(ROOT/'reports/MODULE_STATUS.json')
    if module['status']!='COMPLETE' or module['new_trajectories']!=800:raise RuntimeError('Scientific coverage incomplete')
    seal={'version':'R5_C2_RECOVERY_SEAL_V1','status':'SEALED_C2_COMPLETE_WITH_LOCAL_NUMERIC_AND_CPU_RECOVERY','module':'C2','sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'evidence_label':c['evidence_label'],'reset_label':c['reset_label'],'formal_trajectories':800,'technical_trajectories':0,'reused_C1_trajectories':400,'C1_reused_seal':record('/workspace/r5/C1/RECOVERY_SEAL.json'),'new_training_updates':0,
      'archive':archive['archive'],'recovery_manifest':archive['manifest'],'members_verified':len(manifest['files']),
      'local_numeric_recovery':record(base/'LOCAL_RECOVERY_CHECK.json'),'cpu_recovery':record(base/'cpu_reference/CPU_RECOVERY_CHECK.json'),
      'cpu_reference':record(base/'cpu_reference/REFERENCE_MANIFEST.json'),'root_independent_table_check':record(ROOT/'reports/INDEPENDENT_C2_TABLE_CHECK.json'),
      'module_status':record(ROOT/'reports/MODULE_STATUS.json'),'scientific_contract':record(ROOT/'C2_CONTRACT.json'),
      'recovery_code':{p.name:record(p) for p in Path(__file__).parent.glob('*.py')},
      'all_original_RGB_planner_image_arrays_and_source_H5_retained_remote':True,'original_RGB_and_planner_images_local':False,
      'local_contents':'800 formal derived trajectories: raw actions, physical/controller full dynamic state, common latents, goals, CEM returned plans and RNG; exact source-to-derived SHA/shape/dtype maps; source endpoint assets. Original source RGB/planner tensors remote-only; no local pixel resimulation claimed.',
      'local_CPU_scope':'metadata-order first EVAL, all four fixed models and both ALT streams, all actually executed replans; fixed R3 CPU tolerances1e-5 absolute/relative, no parameter/buffer changes',
      'formal_symmetry_checked':check['all_100_cases_both_ALT_streams_four_arm_initial_state_and_render_hash_bitwise_equal'],
      'immutable_R4_C0_sources':{'c0_contract':c['c0_contract'],'c0_reset':c['c0_reset'],'c0_gate':c['c0_gate']},'C1_C0_R4_scientific_files_modified':False}
    freeze(ROOT/'RECOVERY_SEAL.json',seal);print(record(ROOT/'RECOVERY_SEAL.json'),flush=True)
if __name__=='__main__':main()
