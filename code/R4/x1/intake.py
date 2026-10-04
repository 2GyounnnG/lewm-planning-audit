"""Bridge the coordinator's SHA download/unpack receipts into the X1 root."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from . import core

def run(task,unpacked_root='/workspace/shared_data/unpacked'):
    import h5py,numpy as np
    remote_task='tworooms' if task=='tworoom' else 'cube';assets=Path('/workspace/shared_data/x1_assets')
    model_dir=assets/'data/source'/remote_task/'models'
    args=[sys.executable,'-m','x1.runner','register-model',task,'--model-directory',str(model_dir),
        '--model-api',str(assets/remote_task/'models_api.json'),'--model-tree',str(assets/remote_task/'models_tree.json')]
    subprocess.run(args,check=True)
    unpack=core.read(Path(unpacked_root)/remote_task/'UNPACKED.json')
    rows=unpack.get('records',unpack.get('files',[]))
    if len(rows)!=1:raise RuntimeError('Actual archive has multiple/no HDF5 members; source allocation requires audit')
    source=rows[0];download=core.read(assets/remote_task/'datasets_download.json');meta=core.read(assets/remote_task/'datasets_api.json')
    archives=list(download['files'].values())
    if len(archives)!=1 or source.get('archive_sha256')!=archives[0]['sha256']:raise RuntimeError('Unpacked member/archive binding differs')
    archive={**archives[0],'path':str(assets/archives[0]['path'])}
    receipt={'repo':meta['id'],'revision':meta['sha'],'archive':archive,'files':[source],
        'unpack_receipt':core.file_record(Path(unpacked_root)/remote_task/'UNPACKED.json'),
        'storage':'volatile tmpfs reproducible from pinned archive' if str(source['path']).startswith('/dev/shm/') else 'instance workspace'}
    receipt_path=core.ROOT/'manifests'/f'{task}_source_receipt.json';core.freeze(receipt_path,receipt)
    with h5py.File(source['path'],'r') as f:
        schema={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in f.items()}
        lengths=np.asarray(f['ep_len'][:]);summary={'task':task,'source':source,'schema':schema,'episodes':len(lengths),
            'episode_length_min':int(lengths.min()),'episode_length_max':int(lengths.max()),'source_seed_present':'seed' in f,
            'family_columns':[k for k in f if 'family' in k.lower() or 'demo' in k.lower()],
            'dataset_repo':meta['id'],'dataset_revision':meta['sha']}
        if 'seed' in f:summary['seed_first_values']=np.asarray(f['seed'][:min(10,len(f['seed']))]).tolist()
    core.freeze(core.ROOT/'manifests'/f'{task}_SCHEMA.json',summary);print(json.dumps(summary),flush=True)
    return receipt_path,source['path']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=core.TASKS);p.add_argument('--unpacked-root',default='/workspace/shared_data/unpacked');a=p.parse_args()
    receipt,h5=run(a.task,a.unpacked_root)
    print(json.dumps({'source_receipt':str(receipt),'h5':h5}),flush=True)
