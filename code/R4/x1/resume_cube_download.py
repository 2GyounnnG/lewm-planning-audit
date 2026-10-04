"""Resume only Cube's existing verified range map with lower HTTP concurrency."""
import importlib.util,json,time
from pathlib import Path

def main():
    root=Path('/workspace/shared_data/x1_assets');spec=importlib.util.spec_from_file_location('x1_range_io','/workspace/r4_v23_execution/ops/r3_range_io.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=root
    meta=json.loads((root/'cube/datasets_api.json').read_text());tree=json.loads((root/'cube/datasets_tree.json').read_text())
    if meta['sha']!='02a19a67a0dc8c9d6215f89c19e0a597691e152a' or meta['id']!='quentinll/lewm-cube':raise RuntimeError('Cube dataset lock differs')
    record=next(r for r in tree if r['path']=='cube_single_expert.tar.zst');lfs=record['lfs']
    if lfs['oid']!='3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600':raise RuntimeError('Cube LFS SHA differs')
    identity={'task':'cube/datasets','repo':meta['id'],'revision':meta['sha'],'name':record['path'],'bytes':record['size'],
        'lfs_sha256':lfs['oid'],'git_blob_sha1':None}
    with m.file_lock(root/'state/cube_datasets.lock',nonblocking=True):
        print(json.dumps({'status':'RESUME_EXISTING_VERIFIED_RANGES','workers':2,'prior_failure':'HTTP_429','time':time.time()}),flush=True)
        result=m.download_asset(identity,workers=2,timeout=120,retries=8)
        m.atomic(root/'cube/datasets_download.json',{'status':'VERIFIED','revision':meta['sha'],'files':{record['path']:result}})
        print(json.dumps({'status':'VERIFIED','sha256':result['sha256'],'bytes':result['bytes']}),flush=True)

if __name__=='__main__':main()
