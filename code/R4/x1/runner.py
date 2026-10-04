"""python -m x1.runner; X1_ROOT and X1_R3_ROOT always select isolated roots."""
from __future__ import annotations
import argparse,json,os
from . import core

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['contract','register-model','freeze-data','cache','cache-merge','train','closed-loop','open-loop','reset-audit','report','check-model'])
    p.add_argument('task',choices=core.TASKS);p.add_argument('--device',default='cpu');p.add_argument('--h5');p.add_argument('--source-receipt')
    p.add_argument('--family-column');p.add_argument('--model-directory');p.add_argument('--model-api');p.add_argument('--model-tree')
    p.add_argument('--seed',type=int,choices=core.SEEDS,default=103201);p.add_argument('--microbatch',type=int,default=128)
    p.add_argument('--technical-steps',type=int);p.add_argument('--stop-after',type=int);p.add_argument('--slot',type=int,default=0)
    p.add_argument('--arm',default='H0');p.add_argument('--phase',choices=['TECH','EVAL'],default='EVAL')
    p.add_argument('--stream',default='R3_ORIGINAL');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    if a.task=='cube':
        # CUDA visibility does not select MuJoCo's EGL device. Keep graphics
        # on the same assigned physical card before importing rendering code.
        binding=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if binding in ('6','7'):os.environ['MUJOCO_EGL_DEVICE_ID']=binding
    if a.command=='contract':
        result=core.task_contract(a.task);core.freeze(core.ROOT/'manifests'/f'{a.task}_contract.json',result)
    elif a.command=='register-model':
        from pathlib import Path
        api=core.read(a.model_api);tree=core.read(a.model_tree);directory=Path(a.model_directory).resolve();files={}
        if not isinstance(api.get('sha'),str) or len(api['sha'])!=40:raise ValueError('Immutable official model revision required')
        if api['sha']!=core.CONFIG[a.task]['model_revision'] or api.get('id',api.get('modelId'))!=core.CONFIG[a.task]['repo']:
            raise ValueError('Official model repository/revision differs from fixed X1 source lock')
        for name in ('config.json','weights.pt'):
            matches=[x for x in tree if x['path']==name]
            if len(matches)!=1:raise ValueError('Missing official model tree entry: '+name)
            expected=matches[0];record=core.file_record(directory/name)
            if record['bytes']!=expected['size']:raise RuntimeError('Official file size differs')
            oid=(expected.get('lfs') or {}).get('oid')
            if oid and oid.removeprefix('sha256:')!=record['sha256']:raise RuntimeError('Official LFS SHA differs')
            if name=='weights.pt' and not oid:raise RuntimeError('Weights require repository LFS SHA')
            if not oid:
                import hashlib
                b=(directory/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
                if blob!=expected['oid']:raise RuntimeError('Git blob identity differs')
            files[name]={**record,'lfs_sha_verified':bool(oid)}
        if files['weights.pt']['sha256']!=core.CONFIG[a.task]['weights_sha256']:raise ValueError('Official X1 weight SHA differs')
        result={'task':a.task,'repo':core.CONFIG[a.task]['repo'],'revision':api['sha'],'files':files,
            'api':core.file_record(a.model_api),'tree':core.file_record(a.model_tree),'status':'PINNED_ASSETS_DOWNLOADED_VERIFIED'}
        core.freeze(core.ROOT/'manifests'/f'{a.task}_model_assets.json',result)
    elif a.command=='freeze-data':
        from .data import freeze_data
        result=freeze_data(a.task,a.h5,family_column=a.family_column,source_receipt=a.source_receipt)
    elif a.command=='cache':
        from .data import build_cache
        result=build_cache(a.task,a.device,a.microbatch,a.shard,a.shards)
    elif a.command=='cache-merge':
        from .data import merge_cache
        result=merge_cache(a.task,a.shards)
    elif a.command=='train':
        from .train import train
        result=train(a.task,a.seed,a.device,a.microbatch,a.technical_steps,a.stop_after,a.slot)
    elif a.command=='closed-loop':
        from .evaluate import closed_loop
        result=closed_loop(a.task,a.arm,a.device,a.shard,a.shards,a.phase,a.stream)
    elif a.command=='open-loop':
        from .evaluate import open_loop
        result=open_loop(a.task,a.device)
    elif a.command=='reset-audit':
        import fcntl
        from . import reset
        folder=core.ROOT/'reset_audit';folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            receipt=folder/'RESET_FALLBACK.json'
            if receipt.exists():
                result=core.read(receipt)
                if result['task']!=a.task or result['code_sha256']!=core.sha(reset.__file__) or result['contract']!=core.task_contract(a.task):
                    raise RuntimeError('Prior reset audit code/source differs; explicit technical review required')
                core.verify(result['roles'])
                for trial in result['trials']:core.verify(trial['arrays'])
            else:result=reset.audit(a.task)
    elif a.command=='report':
        from .report import summarize
        result=summarize(a.task)
    elif a.command=='check-model':
        from .check_model import check
        result=check(a.task,a.device)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
