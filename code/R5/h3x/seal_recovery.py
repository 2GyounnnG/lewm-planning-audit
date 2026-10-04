"""Seal exact local H3X recovery; does not train, simulate, or rewrite science."""
import json,sys,time
from pathlib import Path
from . import common
from .local_recovery import HERE,R3,SEEDS,file,read,sha,verify,write

def main():
    target=HERE/'reports/H3X_RECOVERY_SEAL.json'
    if target.exists():raise RuntimeError('Seal already exists; refusing replacement')
    common.source_hashes()
    for freeze in ('CODE_FREEZE.json','EVALUATION_CODE_FREEZE.json','REPORT_CODE_FREEZE.json'):
        for n,h in read(HERE/freeze)['files'].items():assert sha(HERE/n)==h,(freeze,n)
    complete=read(HERE/'reports/main/COMPLETE.json')
    assert complete['status']=='COMPLETE' and complete['new_openloop_rows']==1800 and complete['new_closedloop_rows']==900 and complete['summary_rows']==90
    for r in complete['files']:verify(HERE/'reports/main'/Path(r['path']).name,r)
    audit=read(HERE/'reports/main/INDEPENDENT_TABLE_CHECK.json')
    assert audit['status']=='PASS' and audit['module_complete_sha256']==sha(HERE/'reports/main/COMPLETE.json')
    metadata=HERE/'module_metadata';md=read(metadata/'METADATA_MANIFEST.json')
    for r in md['files']:verify(metadata/r['relative_path'],r['copied'])
    gate=read(metadata/'run/FORMAL_AUTHORIZATION.json')
    authorization=HERE.parent/'ops/CONTINUATION_AUTHORIZATION_V1.json'
    verify(authorization,gate['continuation_authorization'])
    training=read(metadata/'reports/training/COMPLETE.json')
    assert training['raw_rows']==90000 and training['formal_updates']==90000 and training['technical_updates']==101
    ledger=read(metadata/'run/state/TECHNICAL_LEDGER.json')
    assert sum(x['actual_updates'] for x in ledger['runs'].values())==101
    tech=read(HERE/'technical/EVALUATION_TECH_CHECK.json')
    assert tech['status']=='PASS' and tech['complete_trajectories']==1 and tech['optimizer_updates']==0
    for r in training['files']:verify(metadata/'reports/training'/Path(r['path']).name,r)
    copied_source_paths={x['original']['path'] for x in md['files']}
    recoveries={};archives=[];remote_only=[x for x in md['remote_only_originals'] if x['path'] not in copied_source_paths];cpu_ok=True
    for s in SEEDS:
        wr=read(HERE/f'recovery_validation/WEIGHTS_{s}.json');nr=read(HERE/f'recovery_validation/NUMERIC_{s}.json')
        assert wr['CPU_GPU_all_model_state_sha_identical'] and nr['derived_trajectories']==300
        for obj in (wr,nr):
            for key in ('archive','manifest','cpu_receipt'):verify(Path(obj[key]['path']),obj[key])
        cpu_path=HERE/f'weight_recovery/{s}/CPU_RECOVERY_CHECK.json';cpu=read(cpu_path)
        assert cpu['seed']==s and cpu['cases']==100 and cpu['histories']==2 and cpu['all5_forecast_steps_recomputed']
        assert cpu['optimizer_updates']==cpu['new_simulator_steps']==cpu['new_encoder_inference']==0
        checks=cpu['inherited_R3_replay_checks'];assert len(checks)==2
        assert all(c['atol']==c['rtol']==1e-5 and c['finite_coordinates']==96000 for c in checks)
        cpu_ok &= cpu['status']=='PASS_INHERITED_R3_CPU_REPLAY_TOLERANCE'
        b=HERE/f'recovery/{s}';manifest=read(b/'RECOVERY_MANIFEST.json');result=read(b/f'train/{s}/result.json')
        assert result['actual_updates']==30000 and not result['technical']
        assert result['frozen_before']==result['frozen_after'] and result['new_encoder_updates']==0
        assert result['final_primary_step']==30000 and not result['checkpoint_selection'] and not result['formal_initialization_inherits_technical_updates']
        checks=[json.loads(line) for line in (b/f'train/{s}/freeze_checks.jsonl').read_text().splitlines()]
        assert [x['step'] for x in checks]==list(range(100,30001,100))
        assert all(x['passed'] and x['frozen_sha256']==result['frozen_before'] for x in checks)
        assert len(manifest['derived_trajectories'])==manifest['closed_trajectories']==300
        for r in manifest['files']:verify(b/r['relative_path'],r)
        saved=read(b/f'evaluation/open_loop/{s}/OPEN_LOOP_COMPLETE.json')
        closed=read(b/f'evaluation/closed_loop/CLOSED_COMPLETE_{s}.json')
        assert cpu['model_all_state_before']==cpu['model_all_state_after']==saved['model_all_state_before']==saved['model_all_state_after']==closed['model_all_state_before']==closed['model_all_state_after']
        recoveries[str(s)]={'cpu_receipt':file(cpu_path),'numeric_receipt':file(HERE/f'recovery_validation/NUMERIC_{s}.json'),'full_bundle_manifest':file(b/'RECOVERY_MANIFEST.json'),'final_checkpoint':file(b/f'train/{s}/checkpoint_30000.pt'),'offline_cases':100,'histories':2,'all5_latent_prediction_coordinates':192000,'closed_numeric_trajectories':300,'new_simulator_steps':0,'status':cpu['status'],'model_all_state_sha256':cpu['model_all_state_before'],'scope':'CPU recomputed every saved offline prediction with inherited R3 tolerance; every retained closed-loop tensor verified exactly, no local simulator rerun.'}
        desc=read(HERE/f'recovery_metadata/H3X_WEIGHTS_{s}.tar.json');p=HERE/f'archives/H3X_WEIGHTS_{s}.tar.gz';verify(p,desc['archive']);archives.append(file(p))
        p=Path(nr['archive']['path']);verify(p,nr['archive']);archives.append(file(p))
    metaarchive=HERE/'archives/H3X_METADATA.tar.gz';verify(metaarchive,read(HERE/'recovery_metadata/H3X_METADATA.tar.json')['archive']);archives.append(file(metaarchive))
    # Pin inherited local assets needed to reconstruct official model and input tensors.
    inherited=[R3/n for n in ('official/reacher/config.json','official/reacher/weights.pt','manifests/reacher_model_assets.json','manifests/reacher_data_roles.json','manifests/NUMERICAL_TOLERANCES.json','state/lewm_source_manifest.json','state/spt_source_manifest.json','source/lewm/jepa.py','source/lewm/module.py','source/lewm/train.py','source/lewm/utils.py','source/spt/stable_pretraining/backbone/utils.py','artifacts/open_loop/reacher/inputs.npz','artifacts/open_loop/reacher/targets.npz','r3/model.py','r3/common.py','r3/data.py')]
    inherited.append(authorization)
    r4local=HERE.parents[1]/'r4_v23_execution'
    for n,h in read(HERE/'EVALUATION_CODE_FREEZE.json')['inherited_r4'].items():
        p=r4local/n;assert sha(p)==h;inherited.append(p)
    lewm=HERE.parent/'h1a/reports/HISTORY_CONDITION_4TASK_RAW.csv'
    reference=next(x for x in complete['sources'] if Path(x['path']).name==lewm.name)
    verify(lewm,reference);inherited.append(lewm)
    roots=('archives','reports','technical','weight_recovery','recovery','recovery_metadata','recovery_validation','module_metadata')
    local_files=[]
    for root in roots:
        for p in sorted((HERE/root).rglob('*')):
            if p.is_file() and p!=target and not p.name.endswith('.tmp'):local_files.append(file(p))
    for p in sorted(HERE.iterdir()):
        if p.is_file() and (p.suffix=='.py' or p.name in ('README.md','TRAIN_PROTOCOL.json','CODE_FREEZE.json','EVALUATION_CODE_FREEZE.json','REPORT_CODE_FREEZE.json','RECOVERY_CODE_FREEZE.json','PRESTART_LAUNCH_AMENDMENT_V2.json','CPU_RECOVERY_TOLERANCE_AMENDMENT_V2.json','DELIVERY_CPU_ALLOCATION_V2.json')):local_files.append(file(p))
    d={'schema':'H3X_RECOVERY_SEAL_V1','status':'COMPLETE_WITH_EXPLICIT_REMOTE_ONLY_ORIGINALS' if cpu_ok else 'HOLD_CPU_REPLAY_TOLERANCE_FAILURE','label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','created_unix':time.time(),'local_root':str(HERE),'primary_root':'/workspace/r5/H3X','scientific_main_complete':file(HERE/'reports/main/COMPLETE.json'),'independent_table_check':file(HERE/'reports/main/INDEPENDENT_TABLE_CHECK.json'),'coverage':{'formal_seeds':list(SEEDS),'final_checkpoint_steps':[30000]*3,'formal_predictor_updates':90000,'technical_predictor_updates':101,'total_new_predictor_updates':90101,'encoder_projector_action_encoder_BN_updates':0,'new_open_loop_raw_rows':1800,'new_closed_loop_trajectories':900,'paired_old_closed_loop_control_rows':1200,'main_summary_rows':90,'independent_EVAL_cases':100,'planning_streams':3,'CPU_forecast_coordinates':576000,'closed_numeric_tensors_verified_trajectories':900},'recoveries':recoveries,'archives':archives,'local_files':local_files,'inherited_local_readonly_dependencies':[file(p) for p in inherited],'remote_only_files':remote_only,'remote_only_TRAIN_cache':{'source':file(metadata/'run/inputs/CACHE_AUDIT.json'),'scope':'Original frozen H1b TRAIN cache reused read-only. Full TRAIN cache is not copied into H3X local recovery; no claim of local training rerun.'},'limitations':['Exploratory additional training after trigger defect; preset H3 remains NOT_TRIGGERED.','CPU reconstructs saved100-anchor offline forecasts only; local closed-loop simulator was not rerun.','Full original RGB trajectory files, optimizer/resume states, intermediate checkpoints and frozen TRAIN/MONITOR caches remain on primary host; original-to-derived SHA mappings are retained.','Bootstrap uses100case units with3fixedseeds and3streams averaged within case; confidence intervals are conditional and unadjusted.'],'code':file(Path(__file__))}
    d.update(scientific_complete=True,formal_updates=90000,technical_updates=101,
        formal_closed_trajectories=900,technical_closed_trajectories=1,open_raw_rows=1800,
        CPU_outlier_count=sum(c['elements_exceeding_tolerance'] for s in SEEDS
            for c in read(HERE/f'weight_recovery/{s}/CPU_RECOVERY_CHECK.json')['inherited_R3_replay_checks']),
        automatic_destroy=False,remote_originals_retained=True)
    descriptions=[read(HERE/f'recovery_metadata/H3X_NUMERIC_DEDUP_V2_{s}.tar.json') for s in SEEDS]
    v1=sum(x['original_V1_numeric_archive_retained']['bytes'] for x in descriptions)
    v2=sum(x['archive']['bytes'] for x in descriptions)
    d['transport_deduplication']={'version':'EXACT_DUPLICATE_OMISSION_TRANSPORT_V2',
        'V1_numeric_bytes':v1,'V2_numeric_bytes':v2,'saved_bytes':v1-v2,
        'reused_already_recovered_files':sum(len(x['reused_files']) for x in descriptions),
        'V1_archives_retained_primary':[x['original_V1_numeric_archive_retained'] for x in descriptions],
        'descriptors':[file(HERE/f'recovery_metadata/H3X_NUMERIC_DEDUP_V2_{s}.tar.json') for s in SEEDS],
        'assembly_verified_against_unchanged_original_full_manifests':True}
    write(target,d);print(json.dumps(file(target)),flush=True)

if __name__=='__main__':main()
