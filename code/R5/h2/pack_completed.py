"""Incremental immutable recovery packs, one completed task/model at a time."""
import json,time,tarfile,datetime
from pathlib import Path
from r4.common import ARMS,STREAMS,file_record,atomic_json

def main():
    root=Path('/workspace/r5/H2');out=root/'recovery_archives';out.mkdir(exist_ok=True)
    pending={(t,a) for t in ['reacher','pusht'] for a in ARMS}
    while pending:
      for task,arm in sorted(pending):
        stem=f'{task}_{arm}_RECOVERY_V1';receipt_path=out/(stem+'_ARCHIVE.json')
        if receipt_path.exists():pending.remove((task,arm));continue
        audit=json.loads((root/task/'CONTROL_REUSE_AUDIT.json').read_text())
        controls=[r for r in audit['controls'] if r['arm']==arm];assert len(controls)==300
        folders=[root/task/'FORMAL'/r['stream']/r['case_id']/arm for r in controls]
        if not all((p/'COMPLETE.json').exists() for p in folders):continue
        members=[]
        for folder in folders:
            complete=json.loads((folder/'COMPLETE.json').read_text())
            for name,r in complete['files'].items():assert file_record(folder/name)==r
            for p in sorted(folder.rglob('*')):
                if not p.is_file() or p.name=='case.lock':continue
                members.append({'path':str(p),'relative_path':str(p.relative_to(root)),**file_record(p)})
        manifest=out/(stem+'_MANIFEST.json');archive=out/(stem+'.tar.gz')
        if manifest.exists() or archive.exists():raise RuntimeError('Interrupted archive requires separate version; never overwrite immutable pack')
        d={'module':'R5_H2','task':task,'arm':arm,'case_streams':300,'members':members,'world_model_updates':0,
            'scope':'All original new-H2 trajectories/results/attempt receipts for this model and task; fixed R4/R3 dependencies referenced separately, no new simulation.'}
        atomic_json(manifest,d)
        tmp=out/(stem+'.tar.gz.partial')
        with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
            for member in members:tar.add(member['path'],arcname=member['relative_path'],recursive=False)
        tmp.replace(archive)
        receipt={'status':'COMPLETE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':task,'arm':arm,'case_streams':300,'member_count':len(members),
            'manifest':{'path':str(manifest),**file_record(manifest)},'archive':{'path':str(archive),**file_record(archive)}}
        atomic_json(receipt_path,receipt);print(json.dumps(receipt),flush=True);pending.remove((task,arm))
      if pending:time.sleep(10)
    atomic_json(out/'PACKING_COMPLETE.json',{'status':'COMPLETE','archives':8,'new_trajectories':2400,'world_model_updates':0})

if __name__=='__main__':main()
