"""Background primary closeout after all100 initial-fork coverage records exist."""
import fcntl,json,os,subprocess,tarfile,time,hashlib,csv
from pathlib import Path

def main():
    line=Path('/workspace/r4_reacher');lock=(line/'FINALIZER.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while len(list((line/'s2').glob('*/COMPLETE.json')))<100:time.sleep(15)
    base=os.environ.copy();base.update(PYTHONPATH='/workspace/r4_v23_execution',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    commands=[['r4.module_status','--line',str(line),'--task','reacher'],['analysis.aggregate','s2','--input',str(line/'s2'),'--r3-root','/workspace/shared_data/r3','--task','reacher','--out',str(line/'reports/s2')]]
    for args in commands:subprocess.run(['taskset','-c','24-47','/workspace/env/bin/python','-m']+args,env=base,check=True)
    status=json.loads((line/'MAIN_MODULE_STATUS.json').read_text())
    if status['status']!='COMPLETE_WITH_TECHNICAL_LIMITATIONS':raise RuntimeError('Primary closeout not actually complete')
    paths=[line/'MAIN_MODULE_STATUS.json',line/'SIM_TECHNICAL_LIMITATION.json',line/'SIM_CEM_ESTIMATE.json']
    for name in ('all_learning','s3_final','s2','s1_offline','s1_cross'):
        paths.extend(p for p in (line/'reports'/name).glob('*') if p.is_file())
    records={str(p.relative_to(line)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':sum(1 for _ in csv.DictReader(p.open())) if p.suffix=='.csv' else None} for p in paths}
    manifest={'status':'COMPLETE_PRIMARY_WITH_TECHNICAL_LIMITATIONS','files':records,'frozen_analysis':{'aggregate.py':'f17eb75fc036986963b675c4281f19bc938728fb583773277ccd640f19b98d8a','statistics.py':'e110a936b19622a659812f6b1b1d01a82245fc83ac2381434762dd3a5ae3320b'},'original_result_and_trajectory_hashes':'reports/all_learning/CLOSED_LOOP_RAW.csv','all100_initial_forks_preserved':True}
    path=line/'FINAL_PRIMARY_MANIFEST.json';path.write_text(json.dumps(manifest,indent=2)+'\n');paths.append(path)
    with tarfile.open('/workspace/r4_reacher_primary_tables.tar.gz','w:gz') as archive:
        for path in paths:archive.add(path,arcname=str(path.relative_to(line)))
    for folder in sorted((line/'s2').iterdir()):
        if (folder/'COMPLETE.json').exists():paths.extend(folder/n for n in ('MENU_LOCK.json','menu.npz','candidate_raw.csv','result.json','COMPLETE.json','INITIAL_CEM_REPRODUCTION.json'))
    with tarfile.open('/workspace/r4_reacher_primary_values.tar.gz','w:gz') as archive:
        for path in paths:archive.add(path,arcname=str(path.relative_to(line)))
    print(json.dumps({'status':status['status'],'primary_tables_bytes':Path('/workspace/r4_reacher_primary_tables.tar.gz').stat().st_size,'primary_values_bytes':Path('/workspace/r4_reacher_primary_values.tar.gz').stat().st_size}),flush=True)

if __name__=='__main__':main()
