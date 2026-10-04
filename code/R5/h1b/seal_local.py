"""Append-only module receipt; execution completion is not a new numerical gate."""
import argparse,hashlib,json,os
from pathlib import Path
from common import *

def main(a):
    root=Path(a.root).resolve();dest=root/'reports/H1B_RECOVERY_SEAL.json'
    if dest.exists():raise RuntimeError('Seal exists; never overwrite a completed module seal')
    expected={'pusht':36,'reacher':34,'tworoom':32,'cube':34};recoveries={};archives={};remote=[];collections=[];models={'RIDGE':0,'MLP':0}
    for task,n in expected.items():
        r=root/'recovery'/task;cp=r/'CPU_RECOVERY_CHECK.json';d=read(cp);assert d['probes_reevaluated']==n and d['case_model_cells']==100*n and len(d['checks'])==n
        assert all(c['cases']==100 and c['optimizer_updates']==0 and c['world_model_updates']==0 for c in d['checks'])
        complete=list((r/'fits'/task).glob('**/COMPLETE.json'));assert len(complete)==n
        for p in complete:
            job=read(p);assert job['status']=='COMPLETE' and job['kind']=='FORMAL';models[job['model']]+=1
            assert (p.parent/'LOCAL_CPU_REEVALUATION.npz').is_file()
        m=r/'recovery'/f'{task}_RECOVERY_MANIFEST.json';manifest=read(m)
        for rec in manifest['members']:
            p=r/rec['relative_path'];assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256']
        remote.extend(manifest['excluded_remote_only']);remote.extend({**rec,'reason':'Large all-TRAIN matrix retained remotely; exact saved EVAL inputs are recovered separately'} for rec in read(r/'matrices'/task/'manifest.json')['files'].values())
        ar=root/'archives'/f'{task}_probe_recovery_v1.tar.gz';arrec=read(root/'archives'/f'{task}_ARCHIVE.json');assert sha(ar)==arrec['archive']['sha256']
        archives[task]={'archive':file(ar),'manifest':file(m)}
        recoveries[task]={'receipt':file(cp),'probes_reevaluated':n,'case_model_cells':100*n,'scope':d['scope'],'max_prediction_abs_difference':d['max_prediction_abs_difference'],'max_error_abs_difference':d['max_error_abs_difference']}
    assert models=={'RIDGE':40,'MLP':96}
    main=root/'reports/main';report=read(main/'COMPLETE.json')
    for f in report['files']:
        p=main/Path(f['path']).name;assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    assert report['rows']==445 and report['raw_rows']==59200
    ledger=read(main/'MEASUREMENT_UPDATE_LEDGER.json');assert ledger['formal_MLP_updates']==187200 and ledger['CPU_TECH_MLP_updates']==200 and ledger['total_measurement_updates']==187400 and ledger['world_model_updates']==0
    trigger=read(main/'H3_MEASUREMENT_ONLY.json');assert trigger['H3_started'] is False
    for task in ['pusht','reacher']:
        m=root/'recovery/source_metadata/cache'/task/'manifest.json'
        assert m.exists();collections.append({'task':task,'local_identity_manifest':file(m),'remote_root':'/workspace/r5/H1b/cache/'+task,'scope':'Cached frozen TRAIN/TECH/EVAL episode latent files remain remotely; exact cache/part identities recovered. No claim of local TRAIN cache recovery.'})
    scientific_code=root/'recovery/code_bundle/code/h1b';freeze=read(scientific_code/'CODE_FREEZE.json')
    for name,digest in freeze['files'].items():assert sha(scientific_code/name)==digest
    paths=set()
    for folder in [root/'recovery',root/'archives',root/'reports/main',root/'technical']:
        for p in folder.rglob('*'):
            if p.is_file() and not p.is_symlink() and not p.name.endswith('.log'):paths.add(p)
    for name in ['INPUT_AUDIT.json','CPU_TECH_RAW.csv','CPU_TECH_CONCLUSION_ZH.txt','MAIN_LOCAL_DELIVERY_ACCEPTANCE.json','CODE_RECOVERY_CHECK.json','LOCAL_RECOVERY_PROGRESS.json']:
        p=root/'reports'/name
        if p.exists():paths.add(p)
    for name in ['README.md','watch_recovery.py','seal_local.py']:paths.add(root/name)
    audit=root.parent/'validation_c0_review/FOUR_TASK_MAIN_ACTUAL_REPORT_REVIEW.json'
    if audit.exists():paths.add(audit)
    result={'status':'COMPLETE_LOCAL_RECOVERY_AND_CPU_REEVALUATION','utc':now(),'local_files':[file(p) for p in sorted(paths)],'archives':archives,'recoveries':recoveries,'expected_fit_counts':expected,'total_probes':136,'saved_EVAL_case_model_cells':13600,'selected_checkpoint_definition':'Every seed included; best TRAIN-holdout checkpoint within each of96 MLP runs, plus all40 final TRAIN-only ridge fits. Never selecting favorable EVAL/seed.','numerical_scope':'All recovered source bytes are SHA-verified. All136 selected probes re-evaluated on exact100 saved EVAL latent/action inputs each. Cross-platform prediction/error differences reported without a newly invented tolerance gate; original scientific values untouched. No encoder rerun, simulator step or training.','measurement_updates':{k:v for k,v in ledger.items() if k not in ('technical_receipts','formal_fit_receipts')},'remote_only_files':remote,'remote_only_collections':collections,'scientific_main':file(main/'COMPLETE.json'),'H3_measurement':file(main/'H3_MEASUREMENT_ONLY.json'),'H3_started':False,'current_scope':'C0/H1a/H1b only; H2/H3/C1/C2 not started by this module','world_model_updates':0,'scientific_limitations':report['limitations'],'seal_code_sha256':sha(__file__)}
    atomic(dest,result);print(json.dumps({'seal':file(dest),'local_files':len(result['local_files']),'remote_only_file_records':len(remote),'probes':136,'saved_EVAL_case_model_cells':13600},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args())
