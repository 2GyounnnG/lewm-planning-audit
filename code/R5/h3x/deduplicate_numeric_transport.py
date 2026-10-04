"""Second transport only: omit every byte-identical already-recovered weights file.

No scientific file or original recovery manifest is altered; V1 archives remain.
"""
import tarfile
from pathlib import Path
from . import common
from .evaluate import file

def main():
    r=common.ROOT/'recovery';out=r/'priority'
    for seed in common.SEEDS:
        original=r/f'bundles/{seed}';wp=out/f'weight_bundles/{seed}/RECOVERY_MANIFEST.json'
        w=common.read_json(wp);f=common.read_json(original/'RECOVERY_MANIFEST.json');lookup={x['relative_path']:x for x in w['files']};reused=[]
        for rec in f['files']:
            rel=rec['relative_path']
            if rel not in lookup:continue
            old=lookup[rel];assert rec['bytes']==old['bytes'] and rec['sha256']==old['sha256']
            p=original/rel;q=out/f'weight_bundles/{seed}'/rel
            assert common.sha256(p)==common.sha256(q)==rec['sha256']
            reused.append({'relative_path':rel,'bytes':rec['bytes'],'sha256':rec['sha256']})
        names={x['relative_path'] for x in reused};archive=out/f'H3X_NUMERIC_DEDUP_V2_{seed}.tar.gz';tmp=archive.with_suffix('.tmp')
        assert not archive.exists(),'Refuse replacement of completed transport archive'
        with tarfile.open(tmp,'w:gz',compresslevel=1) as t:
            for p in sorted(original.rglob('*')):
                if p.is_file() and str(p.relative_to(original)) not in names:t.add(p,arcname=str(seed)+'/'+str(p.relative_to(original)),recursive=False)
        tmp.replace(archive)
        common.atomic_json(archive.with_suffix('.json'),{'status':'EXACT_DUPLICATE_OMISSION_TRANSPORT_V2','archive':file(archive),'seed':seed,'full_bundle_manifest':file(original/'RECOVERY_MANIFEST.json'),'reuse_from_weights_manifest':file(wp),'reused_files':reused,'original_V1_numeric_archive_retained':file(out/f'H3X_NUMERIC_{seed}.tar.gz'),'assembly':'After archive extraction copy/hardlink each reused file from verified weights bundle, then verify every original full manifest file and every retained tensor. No scientific array changed.','code':file(__file__)})
        print('DEDUP_V2',seed,archive.stat().st_size,'reused',len(reused),flush=True)

if __name__=='__main__':main()
