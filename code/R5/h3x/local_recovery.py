"""Local-only archive verification and saved-forecast CPU recovery orchestration.

No training, simulator calls, or modifications to any original scientific files.
Archive transfers are separately scheduled through the shared single-stream slot.
"""
import argparse, hashlib, json, os, subprocess, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (103201, 103202, 103203)
R3 = Path('/Volumes/MyProj/r3_official_lewm_predictor_refit/recovery')
PYTHON = HERE.parents[1] / 'jepa_low_label_bridge_gpu_v2/g1/.venv/bin/python'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def file(p): return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}

def verify(p,r):
    assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)

def unpack(seed,kind):
    desc=read(HERE/f'recovery_metadata/H3X_{kind}_{seed}.tar.json')
    archive=HERE/f'archives/H3X_{kind}_{seed}.tar.gz';verify(archive,desc['archive'])
    dest=HERE/('weight_recovery' if kind=='WEIGHTS' else 'recovery');dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            p=Path(m.name)
            assert not p.is_absolute() and '..' not in p.parts and p.parts[0]==str(seed)
            assert m.isfile() or m.isdir()
        t.extractall(dest,filter='data')
    out=dest/str(seed)
    manifest=out/'RECOVERY_MANIFEST.json';verify(manifest,desc['manifest' if kind=='WEIGHTS' else 'full_bundle_manifest'])
    if kind=='NUMERIC':
        source=HERE/f'weight_recovery/{seed}/train/{seed}/checkpoint_30000.pt'
        target=out/f'train/{seed}/checkpoint_30000.pt'
        verify(source,desc['excluded_already_transferred_exact_final_checkpoint'])
        if not target.exists():os.link(source,target)
    d=read(manifest)
    for r in d['files']:verify(out/r['relative_path'],r)
    return out

def weights():
    log=HERE/'recovery_validation';log.mkdir(exist_ok=True)
    for seed in SEEDS:
        archive=HERE/f'archives/H3X_WEIGHTS_{seed}.tar.gz'
        while not archive.exists():time.sleep(10)
        out=unpack(seed,'WEIGHTS')
        receipt=out/'CPU_RECOVERY_CHECK.json'
        if not receipt.exists():
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',R3_ROOT=str(R3),
                PYTHONPATH=os.pathsep.join([str(HERE.parent),str(HERE.parents[1]/'r4_v23_execution/local_deps'),str(R3)]),
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            with (log/f'CPU_{seed}.log').open('ab',buffering=0) as f:
                subprocess.run([str(PYTHON),'-B','-m','h3x.recover_v2','cpu','--seed',str(seed),'--bundle',str(out)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        d=read(receipt);gpu=read(out/f'evaluation/open_loop/{seed}/OPEN_LOOP_COMPLETE.json')
        assert d['model_all_state_before']==d['model_all_state_after']==gpu['model_all_state_before']==gpu['model_all_state_after']
        write(log/f'WEIGHTS_{seed}.json',{'status':d['status'],'seed':seed,'archive':file(archive),'manifest':file(out/'RECOVERY_MANIFEST.json'),'cpu_receipt':file(receipt),'CPU_GPU_all_model_state_sha_identical':True,'new_updates':0,'new_simulator_steps':0})
        print('CPU_WEIGHTS_VERIFIED',seed,d['status'],flush=True)
    write(log/'WEIGHTS_COMPLETE.json',{'status':'ALL_THREE_SAVED_FORECAST_CPU_CHECKS_RECORDED','seeds':list(SEEDS),'receipts':[file(log/f'WEIGHTS_{s}.json') for s in SEEDS]})

def numeric():
    import numpy as np
    from .recover_v2 import tsha
    log=HERE/'recovery_validation';log.mkdir(exist_ok=True)
    for seed in SEEDS:
        archive=HERE/f'archives/H3X_NUMERIC_{seed}.tar.gz'
        while not archive.exists():time.sleep(10)
        out=unpack(seed,'NUMERIC');d=read(out/'RECOVERY_MANIFEST.json')
        assert d['closed_trajectories']==300 and len(d['derived_trajectories'])==300
        for r in d['derived_trajectories']:
            with np.load(out/r['bundle_path'],allow_pickle=False) as f:
                assert {k:tsha(f[k]) for k in f.files}==r['retained_tensor_sha256']
        cpu=HERE/f'weight_recovery/{seed}/CPU_RECOVERY_CHECK.json'
        verify(out/f'train/{seed}/checkpoint_30000.pt',read(cpu)['checkpoint'])
        write(log/f'NUMERIC_{seed}.json',{'status':'ALL_300_DERIVED_NUMERIC_TRAJECTORIES_AND_ORIGINAL_FILES_SHA_VERIFIED','seed':seed,'archive':file(archive),'manifest':file(out/'RECOVERY_MANIFEST.json'),'cpu_receipt':file(cpu),'verified_original_files':len(d['files']),'derived_trajectories':300,'new_simulator_steps':0,'scope':'Every retained tensor exact; original pixels remote-only. No local closed-loop rerun.'})
        print('NUMERIC_VERIFIED',seed,len(d['files']),flush=True)
    write(log/'NUMERIC_COMPLETE.json',{'status':'ALL_900_DERIVED_NUMERIC_TRAJECTORIES_VERIFIED','seeds':list(SEEDS),'receipts':[file(log/f'NUMERIC_{s}.json') for s in SEEDS]})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['weights','numeric']);a=p.parse_args()
    (weights if a.mode=='weights' else numeric)()
