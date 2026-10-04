"""Pin author asset API revisions before downloading; no model deserialization."""
import concurrent.futures, hashlib, json, pathlib, urllib.parse, urllib.request

ROOT=pathlib.Path('/workspace/shared_data/x1_assets')

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'R4-v2.3-asset-audit'})
    with urllib.request.urlopen(req,timeout=90) as r:
        return json.load(r),r.headers.get('Link','')

def pin(task,kind):
    dest=ROOT/task;dest.mkdir(parents=True,exist_ok=True)
    ap=dest/f'{kind}_api.json';tp=dest/f'{kind}_tree.json'
    if ap.exists() and tp.exists():
        meta=json.loads(ap.read_text());tree=json.loads(tp.read_text())
    else:
        repo='quentinll/lewm-'+task
        meta,_=get(f'https://huggingface.co/api/{kind}/{repo}')
        assert meta['id']==repo and len(meta['sha'])==40
        assert not meta.get('private') and not meta.get('disabled')
        tree=[];url=f'https://huggingface.co/api/{kind}/{repo}/tree/{meta["sha"]}?recursive=true&expand=true'
        while url:
            part,link=get(url);tree.extend(part);url=None
            for bit in link.split(','):
                if 'rel="next"' in bit:url=bit.split('<',1)[1].split('>',1)[0]
        ap.write_text(json.dumps(meta,indent=2)+'\n');tp.write_text(json.dumps(tree,indent=2)+'\n')
    rows=[r for r in tree if r['type']=='file']
    out={'task':task,'kind':kind,'revision':meta['sha'],'files':[{k:r.get(k) for k in ('path','size','lfs','oid')} for r in rows],'bytes':sum(r['size'] for r in rows)}
    print(json.dumps(out),flush=True);return out

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(lambda a:pin(*a),[(t,k) for t in ('tworooms','cube') for k in ('models','datasets')]))
    (ROOT/'ASSET_LOCK.json').write_text(json.dumps(result,indent=2)+'\n')
