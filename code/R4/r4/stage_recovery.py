"""Stage exact sealed sidecars and source windows beside nonpixel derivatives."""
import argparse,json,shutil
from pathlib import Path
from .common import atomic_json,file_record,sha256

def stage(line,assets,out):
    line,assets,out=Path(line),Path(assets),Path(out);rows=[]
    def copy(path,relative,kind):
        path=Path(path);target=out/relative;record=file_record(path)
        if target.exists():
            if file_record(target)!=record:raise RuntimeError('Existing recovery artifact changed: '+str(target))
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            if file_record(target)!=record:raise RuntimeError('Recovery copy checksum differs')
        rows.append({'source':str(path),'recovery_relative':str(relative),'kind':kind,**record})
    # Only completed policy cases have stable menus and experiment identities.
    for receipt in sorted((line/'trajectories').glob('**/COMPLETE.json')):
        folder=receipt.parent
        if not (folder/'trajectory.npz').exists():continue
        for name,record in json.loads(receipt.read_text())['files'].items():
            if file_record(folder/name)!=record:raise RuntimeError('Source completion seal differs')
        for p in sorted((folder/'menus').rglob('*')):
            if p.is_file() and p.suffix in ('.json','.npz'):copy(p,Path('menus')/p.relative_to(line/'trajectories'),'EXACT_ORIGINAL_MENU_SIDECAR')
    # S2 menus, predictions, branch states/latents, coverage values contain no raw RGB.
    for receipt in sorted((line/'s2').glob('*/COMPLETE.json')):
        folder=receipt.parent
        for name,record in json.loads(receipt.read_text())['files'].items():
            if file_record(folder/name)!=record:raise RuntimeError('S2 source completion seal differs')
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix in ('.json','.csv','.npz'):copy(p,Path('s2')/p.relative_to(line/'s2'),'EXACT_ORIGINAL_S2')
    for name in ('s0','s1','boot','reproduction'):
        folder=line/name
        for p in sorted(folder.glob('*')):
            if p.is_file() and p.suffix in ('.json','.csv','.npz'):copy(p,Path(name)/p.name,'EXACT_ORIGINAL_MODULE_VALUE')
    for p in sorted(assets.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.npz'):copy(p,Path('source_windows')/p.relative_to(assets),'EXACT_FROZEN_SOURCE_WINDOW')
    result={'version':'R4_V23_SIDECAR_RECOVERY_V1','source_logs_retained':True,'contains_raw_trajectory_pixels':False,'source_window_pixels_retained':True,'scope':'Only completed case/fork outputs are included; repeat after remaining modules complete. Scientific outcomes unchanged.','files':rows,'bytes':sum(r['bytes'] for r in rows),'derivation_code_sha256':sha256(__file__)}
    atomic_json(out/'SIDECAR_RECOVERY_INDEX.json',result);return {k:v for k,v in result.items() if k!='files'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--line',required=True);p.add_argument('--assets',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(stage(a.line,a.assets,a.out)),flush=True)

if __name__=='__main__':main()
