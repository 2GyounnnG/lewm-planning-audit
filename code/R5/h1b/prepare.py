"""Read-only source audit, TRAIN role isolation, explicit R5 input acceptance."""
import argparse, os, socket, subprocess
from pathlib import Path
import h5py,hdf5plugin,numpy as np
from common import *

def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);protocol=read(Path(__file__).with_name('protocol.json'))
    result={'status':'PASS','utc':now(),'hostname':socket.gethostname(),'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'gpu_compute_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),'protocol':file(Path(__file__).with_name('protocol.json')),'tasks':{},'mlp_status':protocol['mlp_status']}
    for task in ['pusht','reacher','tworoom','cube']:
        c=task_config(task);roles=read(c['roles']);episodes=roles['episodes'];train=[e for e in episodes if e['role']=='REFIT_TRAIN'];ev=roles['cases']['EVAL']
        assert len(ev)==100 and {e['episode_id'] for e in train}.isdisjoint({e['episode_id'] for e in ev})
        source=read(c['root']/'manifests'/f'{task}_source_map.json')['assets'] if task in ('pusht','reacher') else {roles['source']['sha256']:roles['source']}
        expected=next(iter(source));assert c['h5'].stat().st_size==source[expected]['bytes']
        # Full source scan verifies original SHA without touching any original ledger.
        actual=sha(c['h5']);assert actual==expected,(task,'SOURCE_H5_SHA_MISMATCH')
        with h5py.File(c['h5'],'r') as h:
            labels={}
            for group,d in protocol['state_targets'][task].items():
                if d is None:labels[group]={'status':'TECHNICALLY_UNAVAILABLE','reason':'Dataset and pinned TwoRoom source contain no recorded velocity state; not reconstructed.'};continue
                ds=h[d['key']];assert ds.shape[1]>max(d['indices']);labels[group]={'status':'AVAILABLE','column':d['key'],'shape':list(ds.shape),'dtype':str(ds.dtype),'indices':d['indices']}
            assert all(int(h['ep_len'][e['source_episode_idx']])==e['length'] for e in episodes)
        cached=read(c['cache']) if c['cache'].exists() else None
        missing=[] if cached is None else [e['episode_id'] for e in train if e['episode_id'] not in cached['episodes']]
        if cached:assert not missing
        counts={str(f):sum(e['length'] for e in train if fold(e,task)==f) for f in range(5)};assert all(n>0 for n in counts.values())
        result['tasks'][task]={'roles':file(c['roles']),'normalization':file(c['normalization']),'source_h5':{'path':str(c['h5']),'sha256':actual,'bytes':c['h5'].stat().st_size},'train_episodes':len(train),'train_frames':sum(e['length'] for e in train),'eval_cases':len(ev),'fold_frames_before_anchor_filter':counts,'labels':labels,'cache':file(c['cache']) if cached else None,'cache_status':'EXISTING_COMPLETE_TRAIN_COVERAGE' if cached else 'FROZEN_ENCODER_CACHE_REQUIRED'}
        atomic(out/'INPUT_AUDIT.json',result);print(task,result['tasks'][task]['cache_status'],flush=True)
    atomic(out/'INPUT_AUDIT.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args())
