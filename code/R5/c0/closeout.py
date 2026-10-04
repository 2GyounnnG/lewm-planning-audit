"""Post-run packaging only; cannot execute/reset an environment or alter a score."""
from pathlib import Path
import datetime, json, os, shutil, tarfile
from .common import ROOT,R3,read,record,verify,atomic,sha

def main():
    gate=read(ROOT/'C0_GATE.json');contract=read(ROOT/'C0_CONTRACT.json')
    for name,entry in contract['source_map'].items():
        src=verify(entry['file']);dst=ROOT/'source_audit'/name
        if dst.suffix=='' and src.suffix:dst=dst.with_suffix(src.suffix)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for relative,expected in contract['official_configuration']['source_sha256'].items():
        src=R3/'source'/relative
        if sha(src)!=expected:raise RuntimeError('Official inherited source changed')
        dst=ROOT/'source_audit'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    files={}
    for p in ROOT.rglob('*'):
        if p.is_file() and 'recovery' not in p.relative_to(ROOT).parts and p.name!='RECOVERY_MANIFEST.json':
            files['C0/'+p.relative_to(ROOT).as_posix()]=record(p)
    code=Path(__file__).parent
    for p in code.iterdir():
        if p.is_file() and p.suffix in ('.py','.sh'):files['code/c0/'+p.name]=record(p)
    manifest={'status':'COMPLETE','module':'R5_C0','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gate':record(ROOT/'C0_GATE.json'),'contract':record(ROOT/'C0_CONTRACT.json'),'files':files,
        'inventory':{'TECH_cases':4,'independent_trials':8,'paired_case_timepoints':104,'raw_state_render_snapshots':208,'source_endpoint_pairs':4,'technical_raw_steps':200},
        'source_H5':'Not duplicated; earlier fully hash-verified pinned R4/X1 source referenced in contract. All C0 used raw actions, source endpoints and observed state/render arrays included.',
        'derivation':'No compression/removal of scientific arrays; all states.npz and every source/action array included byte-for-byte.',
        'scientific_code':'Only files listed in the frozen C0 contract generated gate results; closeout.py packages completed artifacts only.'}
    atomic(ROOT/'RECOVERY_MANIFEST.json',manifest)
    directory=ROOT/'recovery';directory.mkdir(exist_ok=True);archive=directory/'C0_v1.tar.gz';temp=directory/'C0_v1.tar.gz.tmp'
    with tarfile.open(temp,'w:gz',compresslevel=6) as t:
        for relative,r in files.items():t.add(verify(r),arcname=relative,recursive=False)
        t.add(ROOT/'RECOVERY_MANIFEST.json',arcname='C0/RECOVERY_MANIFEST.json')
    temp.replace(archive)
    receipt={'status':'COMPLETE','archive':record(archive),'manifest':record(ROOT/'RECOVERY_MANIFEST.json'),'member_count':len(files)+1,'inventory':manifest['inventory']}
    atomic(directory/'ARCHIVE_RECEIPT.json',receipt)
    print(json.dumps({'status':'COMPLETE','archive_bytes':archive.stat().st_size,'members':len(files)+1,'sha256':sha(archive)}),flush=True)

if __name__=='__main__':main()
