"""Bound derived compact recovery: all numeric arrays, actual planner/goal RGB.

The complete original trajectories/full recovery archives remain on the host.
Every original array has a shape/dtype/bytes hash; omitted RGB is not claimed local.
"""
import json,time,tarfile,datetime,hashlib,shutil
from pathlib import Path
import numpy as np
from r4.common import ARMS,file_record,atomic_json,atomic_npz

def array_record(value):
    a=np.ascontiguousarray(value);return {'shape':list(a.shape),'dtype':str(a.dtype),'array_bytes_sha256':hashlib.sha256(a.tobytes()).hexdigest()}

def main():
    root=Path('/workspace/r5/H2');out=root/'compact_archives';out.mkdir(exist_ok=True);pending={(t,a) for t in ['reacher','pusht'] for a in ARMS}
    while pending:
      for task,arm in sorted(pending):
        stem=f'{task}_{arm}_COMPACT_V1';receipt_path=out/(stem+'_ARCHIVE.json')
        if receipt_path.exists():pending.remove((task,arm));continue
        audit=json.loads((root/task/'CONTROL_REUSE_AUDIT.json').read_text());controls=[r for r in audit['controls'] if r['arm']==arm];assert len(controls)==300
        folders=[root/task/'FORMAL'/r['stream']/r['case_id']/arm for r in controls]
        if not all((p/'COMPLETE.json').exists() for p in folders):continue
        stage=out/stem;stage.mkdir(exist_ok=True);members=[];cases=[]
        for folder,c in zip(folders,controls):
            complete=json.loads((folder/'COMPLETE.json').read_text())
            for n,r in complete['files'].items():assert file_record(folder/n)==r
            relative=folder.relative_to(root);dest=stage/relative;dest.mkdir(parents=True,exist_ok=True)
            with np.load(folder/'trajectory.npz',allow_pickle=False) as f:original={k:f[k] for k in f.files}
            indexes=sorted({0,*([15,20,25] if len(original['replan_raw'])>1 else [])})
            arrays={k:v for k,v in original.items() if k!='raw_pixels'}
            if 'raw_pixels' in original:
                arrays['planner_raw_indices']=np.array(indexes,dtype=np.int64);arrays['planner_pixels']=original['raw_pixels'][indexes]
                pixel_hashes=[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in original['raw_pixels']]
            else:pixel_hashes=[]
            atomic_npz(dest/'compact_trajectory.npz',**arrays)
            for p in sorted(folder.rglob('*')):
                if not p.is_file() or p.name in ['trajectory.npz','case.lock']:continue
                target=dest/p.relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            control=Path(c['control_folder']);ct=stage/'controls'/relative
            ct.mkdir(parents=True,exist_ok=True)
            for name in ['result.json','COMPLETE.json']:shutil.copyfile(control/name,ct/name)
            for src in control.glob('attempt_*/STARTED.json'):
                target=ct/src.relative_to(control);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
            cases.append({'case_id':c['case_id'],'stream':c['stream'],'arm':arm,'relative_directory':str(relative),
                'original_trajectory':{'path':str(folder/'trajectory.npz'),**file_record(folder/'trajectory.npz')},
                'compact_trajectory':{'path':str(dest/'compact_trajectory.npz'),'relative_path':str(relative/'compact_trajectory.npz'),**file_record(dest/'compact_trajectory.npz')},
                'original_arrays':{k:array_record(v) for k,v in original.items()},'original_raw_pixel_per_frame_sha256':pixel_hashes,'retained_planning_raw_indices':indexes,
                'omitted_local_arrays':['raw_pixels'],'retained_arrays':list(arrays),'control_original_result':{'path':str(control/'result.json'),**file_record(control/'result.json')}})
        for p in sorted(stage.rglob('*')):
            if p.is_file():members.append({'path':str(p),'relative_path':str(p.relative_to(stage)),**file_record(p)})
        manifest=out/(stem+'_MANIFEST.json');archive=out/(stem+'.tar.gz')
        if manifest.exists() or archive.exists():raise RuntimeError('Immutable compact manifest/archive exists')
        atomic_json(manifest,{'version':'R5_H2_COMPACT_RECOVERY_V1','task':task,'arm':arm,'case_streams':300,'cases':cases,'members':members,
            'local_scope':'All actions/states/latents/plans/candidates/costs; actual planning input images raw0 and raw[15,20,25] if replan1, plus goal RGB. Full raw RGB stays remote with array/frame SHA identity.',
            'full_original_archives_remote':'/workspace/r5/H2/recovery_archives','world_model_updates':0,'new_simulator_steps':0})
        temporary=out/(stem+'.tar.gz.partial')
        with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
            for m in members:tar.add(m['path'],arcname=m['relative_path'],recursive=False)
        temporary.replace(archive)
        receipt={'status':'COMPLETE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':task,'arm':arm,'case_streams':300,'member_count':len(members),
            'manifest':{'path':str(manifest),**file_record(manifest)},'archive':{'path':str(archive),**file_record(archive)}}
        atomic_json(receipt_path,receipt);print(json.dumps(receipt),flush=True);pending.remove((task,arm))
      if pending:time.sleep(10)
    atomic_json(out/'COMPACT_PACKING_COMPLETE.json',{'status':'COMPLETE','archives':8,'new_trajectories':2400,'full_RGB_local_recovery_claimed':False})

if __name__=='__main__':main()
