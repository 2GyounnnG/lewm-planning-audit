"""Exact menu evidence and explicitly derived minimal trajectories for recovery."""
import argparse,json,tarfile
from pathlib import Path
from .common import atomic_json,file_record,sha256

def package(line,assets,output):
    line,assets,output=Path(line),Path(assets),Path(output);output.mkdir(parents=True,exist_ok=True)
    groups={'menu_evidence':[],'minimum_trajectories':[]}
    for receipt in sorted((line/'trajectories').glob('**/COMPLETE.json')):
        folder=receipt.parent
        if not (folder/'trajectory.npz').exists():continue
        for path in sorted((folder/'menus').rglob('*')):
            if path.is_file():groups['menu_evidence'].append((path,Path('trajectory_menus')/path.relative_to(line/'trajectories')))
    for module in ('s2','s0','boot','s1'):
        for path in sorted((line/module).rglob('*')):
            if path.is_file() and path.suffix in ('.npz','.json','.csv'):groups['menu_evidence'].append((path,Path(module)/path.relative_to(line/module)))
    for path in sorted(assets.glob('*.json')):groups['menu_evidence'].append((path,Path('source_window_maps')/path.name))
    for name in ('MAIN_MODULE_STATUS.json','FINAL_PRIMARY_MANIFEST.json','SIM_TECHNICAL_LIMITATION.json','SIM_CEM_ESTIMATE.json'):
        path=line/name
        if path.exists():groups['menu_evidence'].append((path,Path(name)))
    for path in sorted((line/'minimum_recovery').rglob('*')):
        if path.is_file():groups['minimum_trajectories'].append((path,Path('minimum_recovery')/path.relative_to(line/'minimum_recovery')))
    receipts={}
    for name,files in groups.items():
        manifest={'version':'R4_V23_RECOVERY_ARCHIVE_V1','kind':name,'task_line':str(line),'files':{str(rel):{'source':str(path),**file_record(path)} for path,rel in files},
            'original_files_retained_on_host':True,'minimum_npz_is_explicit_derivative_not_original':True,
            'source_pixels':'Raw pixels are deterministically reconstructed using retained source-window maps and original pinned HDF5; minimal trajectories retain raw0 and every RGB frame hash. Full windows, original pixels and full300 CEM logs remain unchanged on host.',
            'actual_menu_evidence':'Complete S2 initial_proposals, exact64 menus, branch endpoints and all actual S3 menus/cost/selection are included, without dropping candidates.',
            'code_sha256':sha256(__file__)}
        mpath=output/(name+'_MANIFEST.json');atomic_json(mpath,manifest)
        apath=output/(name+'.tar.gz')
        with tarfile.open(apath,'w:gz') as archive:
            archive.add(mpath,arcname=mpath.name)
            for path,relative in files:archive.add(path,arcname=str(relative))
        receipts[name]={'archive':file_record(apath),'manifest':file_record(mpath),'files':len(files),'uncompressed_file_bytes':sum(r['bytes'] for r in manifest['files'].values())}
    atomic_json(output/'RECOVERY_ARCHIVES.json',receipts);return receipts

def main():
    p=argparse.ArgumentParser();p.add_argument('--line',required=True);p.add_argument('--assets',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(package(a.line,a.assets,a.output)),flush=True)

if __name__=='__main__':main()
