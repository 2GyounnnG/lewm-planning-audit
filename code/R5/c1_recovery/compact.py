import argparse, hashlib, json, os, shutil, tarfile, time
from pathlib import Path
import numpy as np
from c1.common import ROOT,ARMS,read,record,sha,atomic,save_npz,verify,digest

def ahash(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def compact(folder,out):
    receipt=read(folder/'COMPLETE.json');dest=out/folder.relative_to(ROOT)
    if (dest/'DERIVATION.json').exists():
        d=read(dest/'DERIVATION.json')
        if d['source_complete']['sha256']!=sha(folder/'COMPLETE.json'):raise RuntimeError('Source receipt changed')
        for r in d['derived'].values():verify(r)
        return
    dest.mkdir(parents=True,exist_ok=True)
    for r in receipt['files'].values():verify(r)
    mapping={};derived={}
    for name in ('trajectory.npz','states.npz','plans.npz'):
        src=folder/name;arrays={};fields={}
        with np.load(src) as f:
            for key in f.files:
                v=f[key];omit=(name=='trajectory.npz' and key in ('raw_pixels','goal_pixels')) or (name=='states.npz' and key.endswith(':render')) or (name=='plans.npz' and key.endswith((':input:pixels',':input:goal')))
                duplicate=(name=='states.npz' and ':model:' in key and not key.startswith('000:'))
                if duplicate:
                    base='000:'+key.split(':',1)[1]
                    if base not in f or v.dtype!=f[base].dtype or v.shape!=f[base].shape or v.tobytes()!=f[base].tobytes():duplicate=False
                fields[key]={'dtype':str(v.dtype),'shape':list(v.shape),'array_sha256':ahash(v),'representation':'RECONSTRUCT_RGB_WITH_FROZEN_RESET_ACTIONS_AND_SOURCE_ENDPOINTS' if omit else 'IDENTICAL_INITIAL_MODEL_ARRAY' if duplicate else 'LOSSLESS_COPY','reference_key':base if duplicate else None}
                if not omit and not duplicate:arrays[key]=v.copy()
                if omit and key=='raw_pixels':arrays['raw_pixel_sha256']=np.asarray([ahash(frame) for frame in v])
                if omit and key=='goal_pixels':arrays['goal_pixel_sha256']=np.asarray(ahash(v))
        destname=name.replace('.npz','_compact.npz');save_npz(dest/destname,**arrays);derived[destname]=record(dest/destname)
        mapping[name]={'source':record(src),'fields':fields}
    for name in ('result.json','STARTED.json','COMPLETE.json','C0_INITIAL_EQUIVALENCE_RAW.csv'):
        if (folder/name).exists():shutil.copy2(folder/name,dest/name);derived[name]=record(dest/name)
    atomic(dest/'DERIVATION.json',{'status':'DERIVED_LOSSLESS_NUMERIC_WITH_RGB_REPLAY_MAP','not_original_npz':True,'original_files_retained_remote':True,'source_complete':record(folder/'COMPLETE.json'),'source_arrays':mapping,'derived':derived,'code':record(__file__),'pixel_reconstruction':'Official source endpoint pixels/goal from C1 assets. C0 sealed official reset + clear_internal + callables; replay exact raw_actions without auto-reset. Every post-action renderer frame hash is logged. Actual initial source image differs from initial rendered world: both hashes remain distinct. Planner tensors are frozen image_transform applied to corresponding raw_pixels at recorded replan raw_index.','state_reconstruction':'Every omitted model array is byte-identical to the same field at000; all nonpixel dynamical/controller state arrays retained without conversion.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    import fcntl
    root=ROOT/'recovery';root.mkdir(exist_ok=True);lock=(root/'COMPACT.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=read(ROOT/'C1_CONTRACT.json');out=root/'bundle';expected=[ROOT/'closed_loop/EVAL/R3_ORIGINAL'/arm/case['case_id'] for case in c['cases']['EVAL'] for arm in ARMS]
    expected += [ROOT/'closed_loop/TECH/R3_ORIGINAL'/arm/case['case_id'] for case,arm in zip(c['cases']['TECH'],ARMS)]
    while True:
        count=0
        for folder in expected:
            if (folder/'COMPLETE.json').exists():compact(folder,out);count+=1
        atomic(root/'COMPACT_PROGRESS.json',{'completed':count,'expected':404,'derived_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())})
        if count==404 and (ROOT/'reports/MODULE_STATUS.json').exists():break
        if not a.watch:return
        time.sleep(20)
    for name in ('C1_CONTRACT.json','TECH_GATE.json','TECH_RAW_TABLE.csv'):
        shutil.copy2(ROOT/name,out/name)
    for name in ('reports','assets','logs','jobs'):
        shutil.copytree(ROOT/name,out/name,dirs_exist_ok=True)
    for group in ('c1','c1_recovery'):
        shutil.copytree(Path('/workspace/r5/code')/group,out/'code'/group,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    files={str(p.relative_to(out)):record(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='RECOVERY_MANIFEST.json'}
    manifest={'status':'READY_FOR_LOCAL_RECOVERY','module':'C1','formal_trajectories':400,'tech_trajectories':4,'derived_trajectories':404,'local_restore_root':'bundle','original_RGB_and_source_H5_retained_remote':True,'pixel_reconstruction_c0':c['c0_contract'],'reused_models':c['models'],'files':files}
    atomic(out/'RECOVERY_MANIFEST.json',manifest)
    archive=root/'C1_minimal_v1.tar.gz'
    with tarfile.open(archive.with_suffix('.tmp'),'w:gz') as tf:tf.add(out,arcname='bundle')
    archive.with_suffix('.tmp').replace(archive)
    atomic(root/'RECOVERY_ARCHIVE.json',{'status':'READY','archive':record(archive),'manifest':record(out/'RECOVERY_MANIFEST.json'),'originals_not_removed':True,'formal':400,'tech':4,'members':len(files)+1})
    print(read(root/'RECOVERY_ARCHIVE.json'),flush=True)
if __name__=='__main__':main()
