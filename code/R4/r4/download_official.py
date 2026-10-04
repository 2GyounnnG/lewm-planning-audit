"""Download immutable R3 H0 bytes directly on the authorized host, SHA checked."""
import concurrent.futures,hashlib,json,os,sys,urllib.request
from pathlib import Path
ASSETS={'pusht':('22b330c28c27ead4bfd1888615af1340e3fe9052',72290721,'48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'), 'reacher':('62adae4b71dc474ddf8f794c476ebfe737a743ca',72290849,'eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469')}
def download(task):
    root=Path(sys.argv[1]);revision,size,sha=ASSETS[task];folder=root/'official'/task;folder.mkdir(parents=True,exist_ok=True)
    for name,expected in [('weights.pt',sha),('config.json','2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09')]:
        path=folder/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:continue
        temp=folder/(name+f'.download_{os.getpid()}')
        url=f'https://huggingface.co/quentinll/lewm-{task}/resolve/{revision}/{name}'
        h=hashlib.sha256();total=0
        with urllib.request.urlopen(url,timeout=120) as response,temp.open('wb') as f:
            while True:
                b=response.read(1<<20)
                if not b:break
                h.update(b);f.write(b);total+=len(b)
            f.flush();os.fsync(f.fileno())
        if h.hexdigest()!=expected:raise RuntimeError('Official source hash mismatch')
        os.replace(temp,path);print(json.dumps({'task':task,'file':name,'bytes':total,'sha256':expected,'status':'VERIFIED'}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(download,ASSETS))
