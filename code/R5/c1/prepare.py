import sys, socket, shutil
from pathlib import Path
import numpy as np
from .common import *

def main():
    from x1 import core,data
    from c0 import reset
    core.ROOT=Path('/workspace/x1_cube')
    gate=read(C0/'C0_GATE.json');c0=read(C0/'C0_CONTRACT.json')
    if gate['status']!='PASS_C_A_A':raise RuntimeError('C0 gate not passed')
    verify(c0['code']['reset.py'])
    if sha(reset.__file__)!=c0['code']['reset.py']['sha256']:raise RuntimeError('C0 reset changed')
    roles_path=core.ROOT/'manifests/cube_data_roles.json';roles=read(roles_path)
    if sha(roles_path)!=c0['roles']['sha256']:raise RuntimeError('Roles changed')
    if len(roles['cases']['EVAL'])!=100 or len(roles['cases']['TECH'])!=4:raise RuntimeError('Case universe changed')
    source=roles['source']
    if Path(source['path']).stat().st_size!=source['bytes']:raise RuntimeError('Source changed size')
    for r in (c0['h5_provenance']['full_sha_reused_from_prior_verified_unpack'],c0['h5_provenance']['unpack_receipt']):verify(r)
    models={'H0':read(core.ROOT/'manifests/cube_model_assets.json')}
    for seed in core.SEEDS:
        p=core.ROOT/'train'/str(seed);r=read(p/'result.json');delta=read(p/'last.json');verify(delta)
        if r['status']!='REFIT_TRAINING_COMPLETE_UNSCORED' or r['technical'] or r['actual_updates']!=30000:raise RuntimeError('Invalid frozen refit')
        models['REFIT_'+str(seed)]={'delta':delta,'training_result':record(p/'result.json')}
    cases={phase:roles['cases'][phase] for phase in ('TECH','EVAL')};assets=[]
    with data.RawH5(source['path'],keys=['pixels']+core.CONFIG['cube']['reset_keys']) as raw:
        for phase,part in cases.items():
            for case in part:
                out={};ep=case['source_episode_idx'];s=case['start_raw_index'];g=case['goal_raw_index']
                if g-s!=25:raise RuntimeError('Wrong endpoint offset')
                for prefix,index,name in [('initial',s,'start'),('goal',g,'goal')]:
                    for key in raw.column_names:out[prefix+':'+key]=raw.array(ep,key,index,index+1)[0]
                    import hashlib
                    if hashlib.sha256(out[prefix+':pixels'].tobytes()).hexdigest()!=case['reset_metadata']['source_'+name+'_pixel_sha256']:raise RuntimeError('Endpoint pixels changed')
                    for key in core.CONFIG['cube']['reset_keys']:
                        if not np.array_equal(out[prefix+':'+key],case['reset_metadata'][name][key]):raise RuntimeError('Source state changed')
                dest=ROOT/'assets'/(case['case_id']+'.npz')
                if not dest.exists():save_npz(dest,**out)
                else:
                    with np.load(dest) as f:
                        if set(f.files)!=set(out) or not all(np.array_equal(f[k],v) for k,v in out.items()):raise RuntimeError('Asset changed')
                assets.append({'phase':phase,'case_id':case['case_id'],'file':record(dest)})
    config={'version':'R5_C1_V1','task':'cube','host':socket.gethostname(),'reset_label':LABEL,'evidence_label':EVIDENCE,
      'c0_contract':record(C0/'C0_CONTRACT.json'),'c0_gate':record(C0/'C0_GATE.json'),'c0_recovery_seal':record(C0/'RECOVERY_SEAL.json'),
      'c0_reset':record(reset.__file__),'official':core.task_contract('cube'),'models':models,'roles':record(roles_path),
      'cases':cases,'assets':assets,'h5_provenance':c0['h5_provenance'],'source':source,
      'stream':'R3_ORIGINAL','case_seed':'first8 SHA256(R3_CEM_CASE_20261002/cube/{case_id}/{replan_index}), uint64 big endian',
      'technical_design':{'trajectories':4,'pairing':'original four TECH in metadata order zipped to H0,REFIT103201,REFIT103202,REFIT103203',
        'gate':'Every first planner call complete state/render BITWISE equals sealed C0 trial0 initial snapshot; official success events/steps/plan finite; no outcome success gate',
        'no_case_substitution':True,'c0_tolerance_unchanged':True},
      'formal_trajectories':400,'optimizer_updates':0,'policy':'Official World.evaluate + seedless actual-source CaseView; World.reset wrapper clears C0 internals before official callables',
      'log':'every raw action + full physical/controller snapshot/render; actual policy RGB and common encoder latent; returned plan and CEM RNG pre/post per replan',
      'resume':'completed identity/hash verified and skipped; incomplete attempt preserved in attempts/ with failure ledger before identical rerun, no scored duplication',
      'statistics':'frozen R4 case bootstrap5000, sorted100 case IDs; unadjusted conditional95; three refits averaged within case, not ensemble; no fabricated family',
      'statistics_source':record('/workspace/r4_v23_execution/analysis/statistics.py'),
      'resources':{'GPU6':list(range(80,96)),'GPU7':list(range(96,112)),'threads':1},
      'code':{p.name:record(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file() and p.suffix in ('.py','.sh')}}
    freeze(ROOT/'C1_CONTRACT.json',config)
    print({'status':'C1_CONTRACT_FROZEN','sha256':sha(ROOT/'C1_CONTRACT.json'),'assets':len(assets)},flush=True)
if __name__=='__main__':main()
