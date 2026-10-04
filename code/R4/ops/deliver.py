"""Delivery registry and lossless table concatenation; no statistical fitting."""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json
from pathlib import Path

def record(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def register(root,name,scope,paths,conclusions):
    if len(conclusions)>5:raise ValueError('At most five conclusion lines per module')
    files=[record(p) for p in paths]
    if not files:raise ValueError('A delivery requires actual recovered evidence')
    index=Path(root)/'manifests/DELIVERY_INDEX.json'
    value=json.loads(index.read_text()) if index.exists() else {'protocol':'R4-v2.3','modules':{}}
    value['modules'][name]={'scope':scope,'recovered_verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'conclusions':conclusions}
    atomic(index,value);return value['modules'][name]

def merge(spec_path,out):
    spec=json.loads(Path(spec_path).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pending=[];technical=[];gates=[]
    for module in spec['requirements']:
        path=Path(module['status_file'])
        if not path.exists():pending.append({'module':module['name'],'reason':'STATUS_NOT_RECOVERED'});continue
        gate=json.loads(path.read_text());state=gate.get('status');gates.append(record(path))
        if state=='COMPLETE':continue
        if state in module.get('allowed_technical_statuses',[]) and gate.get('reason'):
            technical.append({'module':module['name'],'status':state,'reason':gate['reason']})
        else:pending.append({'module':module['name'],'reason':state or 'MISSING_STATUS'})
    if pending:
        result={'status':'WAITING_MAIN_RESULTS','pending':pending,'technical_limitations':technical}
        atomic(out/'MERGE_STATUS.json',result);return result
    output_records=[];inputs=[]
    for table in spec['tables']:
        rows=[]
        for source in table['sources']:
            path=Path(source['path']);rec=record(path);inputs.append(rec)
            with path.open(newline='') as f:
                for r in csv.DictReader(f):
                    rows.append(dict(r,delivery_source=source['label'],delivery_source_sha256=rec['sha256']))
        fields=list(dict.fromkeys(k for r in rows for k in r));path=out/table['name'];tmp=path.with_suffix('.tmp')
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
        tmp.replace(path);output_records.append(dict(record(path),rows=len(rows)))
    result={'status':'COMPLETE_WITH_TECHNICAL_LIMITATIONS' if technical else 'COMPLETE','inputs':inputs,'gates':gates,'outputs':output_records,'technical_limitations':technical,'no_cross_task_metric_pooling':True,'source_values_unchanged':True}
    atomic(out/'MERGE_STATUS.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=merge(a.spec,a.out);print(json.dumps({'status':result['status'],'outputs':len(result.get('outputs',[]))}))
