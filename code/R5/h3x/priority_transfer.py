"""Prepare weights first; later numeric archive excludes already sent final weights.

Both archives contain exact original bytes. Complete bundles can be assembled
locally by hardlinking/copying the verified final checkpoint from the weights
bundle into the later bundle before verifying its original full manifest.
"""
import json,os,shutil,tarfile,time
from pathlib import Path
from . import common
from .evaluate import file

def archive(folder,path,exclude_final=False):
    tmp=path.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or (exclude_final and p.name=='checkpoint_30000.pt'):continue
            tar.add(p,arcname=folder.name+'/'+str(p.relative_to(folder)),recursive=False)
    tmp.replace(path)

def weights(seed,root,out):
    bundle=out/f'weight_bundles/{seed}';bundle.mkdir(parents=True,exist_ok=True);sources=[]
    def copy(p,rel):
        target=bundle/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert common.sha256(target)==common.sha256(p);sources.append({'source':file(p),'relative_path':rel,**file(target)})
    train=root/f'artifacts/train/H3X_reacher_s{seed}';result=common.read_json(train/'result.json');assert result['actual_updates']==30000
    for n in ['checkpoint_30000.pt','result.json','updates.jsonl','monitor.jsonl','freeze_checks.jsonl','RUN_IDENTITY.json','FROZEN_INITIAL.json','PARAMETER_WHITELIST.json','sampling_counts.npz','last.json','resume.json']:copy(train/n,f'train/{seed}/{n}')
    for p in (root/f'evaluation/open_loop/{seed}').iterdir():
        if p.is_file():copy(p,f'evaluation/open_loop/{seed}/{p.name}')
    common.atomic_json(bundle/'RECOVERY_MANIFEST.json',{'status':'WEIGHTS_AND_OFFLINE_ONLY_COMPLETE','label':common.LABEL,'seed':seed,'files':sources,'derived_trajectories':[],'closed_trajectories':0,'scope':'PRIORITY_EXACT_FINAL_WEIGHTS_AND_ALL100_OFFLINE_INPUT_REFERENCES_AND_SAVED_PREDICTIONS; closed numerical recovery follows separately'})
    dest=out/f'H3X_WEIGHTS_{seed}.tar.gz';archive(bundle,dest);common.atomic_json(dest.with_suffix('.json'),{'archive':file(dest),'manifest':file(bundle/'RECOVERY_MANIFEST.json'),'seed':seed,'scope':'WEIGHTS_AND_OFFLINE_ONLY'})
    print('WEIGHTS_READY',seed,dest.stat().st_size,flush=True)

def main():
    root=common.ROOT;out=root/'recovery/priority';out.mkdir(parents=True,exist_ok=True);w=set();n=set()
    while len(n)<3:
        for seed in common.SEEDS:
            if seed not in w and (root/f'evaluation/open_loop/{seed}/OPEN_LOOP_COMPLETE.json').exists():weights(seed,root,out);w.add(seed)
            bundle=root/f'recovery/bundles/{seed}';full_archive=root/f'recovery/H3X_{seed}.tar.json'
            if seed not in n and full_archive.exists():
                dest=out/f'H3X_NUMERIC_{seed}.tar.gz';archive(bundle,dest,exclude_final=True);cp=bundle/f'train/{seed}/checkpoint_30000.pt'
                common.atomic_json(dest.with_suffix('.json'),{'archive':file(dest),'full_bundle_manifest':file(bundle/'RECOVERY_MANIFEST.json'),'seed':seed,'excluded_already_transferred_exact_final_checkpoint':file(cp),'assembly':'Hardlink or copy verified weight_bundles/SEED/train/SEED/checkpoint_30000.pt into numeric bundle at same relative path; then verify original RECOVERY_MANIFEST all files/tensors. No weight retransmission.'});n.add(seed);print('NUMERIC_READY',seed,dest.stat().st_size,flush=True)
        common.atomic_json(out/'STATUS.json',{'pid':os.getpid(),'unix':time.time(),'weights_ready':sorted(w),'numeric_ready':sorted(n)})
        if len(n)<3:time.sleep(10)
    common.atomic_json(out/'COMPLETE.json',{'status':'PRIORITY_ARCHIVES_COMPLETE','seeds':list(common.SEEDS)})
if __name__=='__main__':main()
