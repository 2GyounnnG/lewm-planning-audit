"""Use R3 verified/resumable range I/O with X1 pinned author identities."""
import argparse, hashlib, importlib.util, json, pathlib, time, urllib.parse, os

ROOT=pathlib.Path(os.environ.get('ASSET_ROOT','/workspace/shared_data/x1_assets'))

def run(task,kind):
    spec=importlib.util.spec_from_file_location('range_io','/workspace/r4_v23_execution/ops/r3_range_io.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=ROOT
    meta=json.loads((ROOT/task/f'{kind}_api.json').read_text())
    tree=json.loads((ROOT/task/f'{kind}_tree.json').read_text())
    names=('weights.pt','config.json','README.md') if kind=='models' else tuple(r['path'] for r in tree if r['type']=='file' and r['path'].endswith('.zst'))
    if kind=='models':
        m.url_for=lambda ident:f'https://huggingface.co/{ident["repo"]}/resolve/{ident["revision"]}/{urllib.parse.quote(ident["name"],safe="")}'
    records={}
    with m.file_lock(ROOT/'state'/f'{task}_{kind}.lock',nonblocking=True):
        for name in names:
            row=next(r for r in tree if r['path']==name)
            lfs=row.get('lfs')
            ident={'task':task+'/'+kind,'repo':meta['id'],'revision':meta['sha'],'name':name,'bytes':row['size'],'lfs_sha256':lfs['oid'] if lfs else None,'git_blob_sha1':None if lfs else row['oid']}
            print(json.dumps({'task':task,'kind':kind,'name':name,'bytes':row['size'],'status':'START'}),flush=True)
            records[name]=m.download_asset(ident,workers=8,timeout=120,retries=6)
            print(json.dumps({'task':task,'kind':kind,'name':name,'status':'VERIFIED','sha256':records[name]['sha256']}),flush=True)
        m.atomic(ROOT/task/f'{kind}_download.json',{'status':'VERIFIED','revision':meta['sha'],'files':records})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['tworooms','cube','pusht','reacher']);p.add_argument('kind',choices=['models','datasets']);a=p.parse_args();run(a.task,a.kind)
