"""Assemble exact original full recovery bundle from V2 transport and weights."""
import os,tarfile,time
import numpy as np
from .local_recovery import HERE,SEEDS,read,write,file,verify
from .recover_v2 import tsha

def main():
    dest=HERE/'recovery';dest.mkdir(exist_ok=True);log=HERE/'recovery_validation';log.mkdir(exist_ok=True)
    for seed in SEEDS:
        archive=HERE/f'archives/H3X_NUMERIC_DEDUP_V2_{seed}.tar.gz'
        while not archive.exists():time.sleep(10)
        desc=read(HERE/f'recovery_metadata/H3X_NUMERIC_DEDUP_V2_{seed}.tar.json');verify(archive,desc['archive'])
        wb=HERE/f'weight_recovery/{seed}';verify(wb/'RECOVERY_MANIFEST.json',desc['reuse_from_weights_manifest'])
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                from pathlib import Path
                p=Path(m.name);assert not p.is_absolute() and '..' not in p.parts and p.parts[0]==str(seed)
                assert m.isfile() or m.isdir()
            t.extractall(dest,filter='data')
        out=dest/str(seed);verify(out/'RECOVERY_MANIFEST.json',desc['full_bundle_manifest'])
        for r in desc['reused_files']:
            src=wb/r['relative_path'];dst=out/r['relative_path'];verify(src,r);dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():os.link(src,dst)
            verify(dst,r)
        d=read(out/'RECOVERY_MANIFEST.json')
        for r in d['files']:verify(out/r['relative_path'],r)
        assert d['closed_trajectories']==len(d['derived_trajectories'])==300
        for r in d['derived_trajectories']:
            with np.load(out/r['bundle_path'],allow_pickle=False) as f:assert {k:tsha(f[k]) for k in f.files}==r['retained_tensor_sha256']
        cpu=wb/'CPU_RECOVERY_CHECK.json';verify(out/f'train/{seed}/checkpoint_30000.pt',read(cpu)['checkpoint'])
        write(log/f'NUMERIC_{seed}.json',{'status':'ALL_300_DERIVED_NUMERIC_TRAJECTORIES_AND_ORIGINAL_FILES_SHA_VERIFIED','seed':seed,'archive':file(archive),'transport_descriptor':file(HERE/f'recovery_metadata/H3X_NUMERIC_DEDUP_V2_{seed}.tar.json'),'manifest':file(out/'RECOVERY_MANIFEST.json'),'cpu_receipt':file(cpu),'verified_original_files':len(d['files']),'reused_byte_identical_files':len(desc['reused_files']),'derived_trajectories':300,'new_simulator_steps':0,'scope':'Exact original full recovery manifest reconstructed from nonduplicated transport plus prior weights bundle; all retained tensor hashes exact. Original pixels remote-only, no local closed-loop rerun.','code':file(__file__)})
        print('NUMERIC_V2_VERIFIED',seed,len(d['files']),flush=True)
    write(log/'NUMERIC_COMPLETE.json',{'status':'ALL_900_DERIVED_NUMERIC_TRAJECTORIES_VERIFIED','seeds':list(SEEDS),'receipts':[file(log/f'NUMERIC_{s}.json') for s in SEEDS]})

if __name__=='__main__':main()
