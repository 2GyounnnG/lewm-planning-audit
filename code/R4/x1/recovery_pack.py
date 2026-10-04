"""Small completed-result recovery export; source archives and large caches stay host."""
import argparse,shutil,tarfile
import numpy as np
from . import core,data

def pack(task):
    root=core.ROOT;out=root/'recovery';stage=out/'bundle';stage.mkdir(parents=True,exist_ok=True)
    status=core.read(root/'reports'/task/'MODULE_STATUS.json')
    if status['status'] not in ('COMPLETE','COMPLETE_WITH_TECHNICAL_LIMITATIONS'):raise RuntimeError('Completed main report required')
    records={}
    def add(source,relative):
        source=core.Path(source);target=stage/relative;rec=core.file_record(source)
        if target.exists():
            if core.sha(target)!=rec['sha256']:raise RuntimeError('Recovery snapshot changed: '+relative)
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        records[relative]={'original':rec,'bytes':target.stat().st_size,'sha256':core.sha(target)}
    for folder in ('manifests','reports','technical_model','reset_audit','open_loop','state'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.txt','.npz','.yaml'):add(p,str(p.relative_to(root)))
    for phase in ('train','technical_train'):
        for p in sorted((root/phase).rglob('*')):
            if p.is_file() and (p.suffix in ('.json','.jsonl','.npz') or p.name=='checkpoint_30000.pt'):
                add(p,str(p.relative_to(root)))
    for p in sorted((root/'closed_loop').rglob('*')):
        if p.is_file() and 'R3_ORIGINAL' in p.parts and p.suffix in ('.json','.npz'):add(p,str(p.relative_to(root)))
    for p in sorted((root/'logs').glob('*.log')):add(p,str(p.relative_to(root)))
    assets=core.read(root/'manifests'/f'{task}_model_assets.json')
    for name,rec in assets['files'].items():add(core.verify(rec),'official/'+name)
    for p in sorted(core.Path(__file__).parent.rglob('*.py')):add(p,'source/x1/'+str(p.relative_to(core.Path(__file__).parent)))
    for p in ('manifests/NUMERICAL_TOLERANCES.json','state/lewm_source_manifest.json','state/swm_compat_source_manifest.json','state/spt_source_manifest.json'):
        add(core.R3_ROOT/p,'r3_reference/'+p)
    roles=core.read(root/'manifests'/f'{task}_data_roles.json');cases=roles['cases']['EVAL'];cache=data.load_cache(task,'EVAL')
    config=core.read(stage/'official/config.json');A=int(config['action_encoder']['input_dim']);history=[];actions=[];target=[];hi=[];ai=[];ti=[]
    for c in cases:
        z=cache[c['episode_id']];s=c['open_loop_window_start_raw'];h=s+np.arange(3)*5;a=np.arange(s,s+35);t=c['open_loop_anchor_raw']+np.arange(1,6)*5
        history.append(z['z'][h]);actions.append(z['actions'][a].reshape(7,A));target.append(z['z'][t]);hi.append(h);ai.append(a);ti.append(t)
    generated=out/'fixed_open_loop_inputs.npz'
    if not generated.exists():data.save_npz(generated,initial_z=np.stack(history),macro_actions=np.stack(actions),target_z=np.stack(target),history_raw_indices=np.stack(hi),action_raw_indices=np.stack(ai),target_raw_indices=np.stack(ti),case_ids=np.asarray([c['case_id'] for c in cases]))
    add(generated,'replay/fixed_open_loop_inputs.npz')
    manifest={'task':task,'status':'RECOVERY_EXPORTED','scope':'Completed final predictor replacements, frozen metadata, exact100 open-loop inputs/targets/forecasts, original closed-loop logs, reset evidence; no large cache or HDF5; no optimizer/intermediate checkpoints',
        'files':records,'cases':100,'new_optimizer_updates':0,'new_cases':0,'source_archives_and_large_cache_retained_on_host':True,'training_resume_environment_recovery_claimed':False}
    core.freeze(stage/'RECOVERY_MANIFEST.json',manifest)
    archive=out/f'{task}_main.tar.gz'
    if not archive.exists():
        temp=archive.with_suffix('.tmp')
        with tarfile.open(temp,'w:gz',compresslevel=1) as tar:tar.add(stage,arcname=task)
        temp.replace(archive)
    receipt={'status':'COMPLETE','archive':core.file_record(archive),'manifest':core.file_record(stage/'RECOVERY_MANIFEST.json'),'files':len(records),'uncompressed_bytes':sum(r['bytes'] for r in records.values())}
    core.freeze(out/'RECOVERY_ARCHIVE.json',receipt);print(core.canonical(receipt).decode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=core.TASKS);a=p.parse_args();pack(a.task)
