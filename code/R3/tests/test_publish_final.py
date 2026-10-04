"""Synthetic metadata/bytes only; no models, real arrays or GPU operations."""
import importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/publish_final.py'
spec=importlib.util.spec_from_file_location('publisher',SCRIPT); p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def fixture(root):
    def put(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else json.dumps(value,sort_keys=True).encode())
        return {'sha256':p.sha(path),'bytes':path.stat().st_size}
    def rec(name):return {'sha256':p.sha(root/name),'bytes':(root/name).stat().st_size}
    put('state/INSTANCE_REUSE.json',{'hostname':'fixture'})
    put('state/FINAL_GPU_EXIT.json',{'status':'ALL_R3_GPU_JOBS_EXITED','hostname':'fixture','active_compute_pids':[],'R3_processes':[],'checked_at':'fixed'})
    for name in ('scripts/final_acceptance.py','scripts/publish_final.py','r3/model.py','source/lewm/jepa.py'):
        put(name,b'# synthetic source\n')
    for name in ('manifests/PRIOR_EVIDENCE.json','manifests/NUMERICAL_TOLERANCES.json','manifests/OPEN_LOOP_ROUTING.json','state/TECHNICAL_LEDGER.json'):put(name,{})
    jobs=[];donejobs={};restorable={};lockfiles={};cases_by_task={}
    for task in p.TASKS:
        archive=f'data/source/{task}/source.tar.gz';h5=f'data/unpacked/{task}/source.h5'
        for name in (archive,h5):
            # Deliberately absent binary sources: only the inherited exact
            # identity is available to this metadata-only publisher.
            r={'sha256':('1' if name==archive else '2')*64,'bytes':123,'path':name}
            restorable[name]={**r,'policy':'Exact pinned original source'};lockfiles[name]=r
        assets=f'manifests/{task}_data_assets.json';ar=put(assets,{'files':{'source.tar.gz':restorable[archive]}})
        proof=f'state/unpack_members/{task}.json';pr=put(proof,{'status':'COMPLETE','sha256':restorable[h5]['sha256'],'bytes':123,'identity':{'archive':restorable[archive]}})
        member={**restorable[h5],'member_receipt':proof,'member_receipt_sha256':pr['sha256']}
        put(f'manifests/{task}_source_map.json',{'assets':{member['sha256']:member}})
        put(f'manifests/{task}_data_unpacked.json',{'source_receipt_sha256':ar['sha256'],'files':[member]})
        cases=[{'case_id':f'{task}_case{i}','episode_id':f'{task}_ep{i}'} for i in range(20)];cases_by_task[task]=cases
        role_path=f'manifests/{task}_data_roles.json';lockfiles[role_path]=put(role_path,{'cases':{'EVAL':cases},'episodes':[{'episode_id':c['episode_id'],'role':'EVAL'} for c in cases]})
        for suffix in ('weights.pt','config.json'):lockfiles[f'official/{task}/{suffix}']=put(f'official/{task}/{suffix}',b'synthetic only')
        cache=f'data/cache/{task}/ep.npz';lockfiles[cache]=put(cache,b'not numpy: metadata fixture')
        for seed in p.SEEDS:
            job={'task':task,'refit_seed':seed,'updates':30000,'job_id':f'R3_{task}_PRED_REFIT_s{seed}'};jobs.append(job)
            prefix='artifacts/train/'+job['job_id']
            result={'status':'REFIT_TRAINING_COMPLETE_UNSCORED','actual_updates':30000,'technical':False,'new_encoder_updates':0,'state_labels_read':0,'formal_initialization_inherits_technical_updates':False,'frozen_before':'frozen','frozen_after':'frozen'}
            put(prefix+'/result.json',result)
            raw=''.join(json.dumps({'step':i,'technical':False,'sampled_windows':128})+'\n' for i in range(1,30001))
            put(prefix+'/updates.jsonl',raw.encode())
            for name in ('resume.pt','resume.json','last.json','checkpoint_3000.pt','checkpoint_10000.pt','checkpoint_30000.pt','RUN_IDENTITY.json','PARAMETER_WHITELIST.json','FROZEN_INITIAL.json','freeze_checks.jsonl','monitor.jsonl','sampling_counts.npz'):
                put(prefix+'/'+name,b'{}')
            files={str(q.relative_to(root)):rec(str(q.relative_to(root))) for q in (root/prefix).iterdir()}
            publication=prefix+'/JOB_SHA256.json';put(publication,{'files':files})
            queue='state/recovery_queue/'+job['job_id']+'.json';put(queue,{'files':files})
            donejobs[job['job_id']]={key:{'path':path,**rec(path)} for key,path in (('result',prefix+'/result.json'),('publication',publication),('recovery_queue',queue))}
    put('manifests/JOBS.json',{'jobs':jobs});put('state/FORMAL_TRAINING_COMPLETE.json',{'status':'COMPLETE','complete_jobs':6,'actual_updates':180000,'jobs':donejobs})
    lockfiles[p.ASSETS]=put(p.ASSETS,{'version':'R3_OFFICIAL_ASSET_MANIFEST_V1','status':'COMPLETE_METADATA_ASSET_MAPPING','tasks':{t:{} for t in p.TASKS},'files':{'r3/model.py':rec('r3/model.py')}})
    put(p.LOCK,{'status':'MODELS_AND_SELECTION_LOCKED','files':lockfiles,'restorable_input_files':restorable})
    locksha=rec(p.LOCK)['sha256'];plan={'phase':'FORMAL','models_lock_sha256':locksha,'cases':[{'task':t,'case':c} for t,cc in cases_by_task.items() for c in cc],'files':{p.LOCK:rec(p.LOCK)},'expected_complete_trajectories':160}
    put('state/planning/FORMAL/EXECUTION_PLAN.json',plan);ledger={};completefiles={}
    for task,cases in cases_by_task.items():
        for case in cases:
            cp=f'artifacts/planning/FORMAL/{task}/{case["case_id"]}';casefiles={}
            for arm in p.ARMS:
                prefix=cp+'/'+arm+'/attempt_0';result={'status':'COMPLETE','evaluation_complete':True,'task':task,'case_id':case['case_id'],'arm':arm,'phase':'FORMAL','failure_category':None,'optimizer_updates':0}
                put(prefix+'/result.json',result);put(prefix+'/trajectory.npz',b'raw synthetic bytes')
                files={n:rec(prefix+'/'+n) for n in ('result.json','trajectory.npz')};put(prefix+'/TRAJECTORY_SHA256.json',{'files':files})
                for n in ('result.json','trajectory.npz','TRAJECTORY_SHA256.json'):casefiles[prefix+'/'+n]=rec(prefix+'/'+n)
                ledger[f'R3/FORMAL/{task}/{case["case_id"]}/{arm}/attempt_0']={'status':'COMPLETE','result_path':prefix+'/result.json','result_sha256':rec(prefix+'/result.json')['sha256']}
            put(cp+'/CASE_SHA256.json',{'status':'COMPLETE','complete_trajectories':4,'execution_plan_sha256':p.digest(plan),'files':casefiles})
            completefiles.update(casefiles);completefiles[cp+'/CASE_SHA256.json']=rec(cp+'/CASE_SHA256.json')
        prefix='artifacts/open_loop/'+task;openedfiles={};arms=[]
        for arm in ['H0']+[f'REFIT_{s}_{k}' for s in p.SEEDS for k in (3000,10000,30000)]:
            arms.append({'route':{'arm':arm}})
            for suffix in ('.npz','.json'):
                name=prefix+'/predictions/'+arm+suffix;openedfiles[name]=put(name,b'{}')
        for filename in ('inputs.npz','targets.npz','per_case.json','TRAIN_coordinates.npz','TRAIN_variance.json'):
            name=prefix+'/'+filename;openedfiles[name]=put(name,b'{}')
        put(prefix+'/OPEN_LOOP_RECEIPT.json',{'status':'COMPLETE','evaluation_kind':'OPEN_LOOP','task':task,'cases':20,'arms':10,'optimizer_updates':0,'all_future_targets_scoring_only':True,'arm_receipts':arms,'files':openedfiles})
        statdir='artifacts/statistics/'+task;statfiles={}
        for filename in p.STAT_FILES[:-1]:statfiles[filename]=put(statdir+'/'+filename,b'{}')
        put(statdir+'/STATISTICS_RECEIPT.json',{'status':'COMPLETE','task':task,'cases':20,'arms':list(p.ARMS),'files':statfiles})
    put('state/FORMAL_TRAJECTORY_LEDGER.json',{'trajectories':ledger});put('state/PLANNING_FORMAL_COMPLETE.json',{'status':'COMPLETE','phase':'FORMAL','execution_plan_sha256':p.digest(plan),'complete_trajectories':160,'files':completefiles})
    reports={}
    for base in p.REPORTS:reports['reports/'+base]=put('reports/'+base,b'Synthetic report, not a scientific result.\n')
    put('manifests/ANALYSIS_OUTPUTS.json',{'status':'COMPLETE','models_lock_sha256':locksha,'formal_training_jobs':6,'formal_training_updates':180000,'bootstrap_replicates':5000,
        'tasks':{t:{'statistics_dir':'artifacts/statistics/'+t,'open_loop_dir':'artifacts/open_loop/'+t,'planning_phase':'artifacts/planning/FORMAL','cases':20,'CEM_arms':4,'open_loop_arms':10} for t in p.TASKS},'inputs':{p.LOCK:rec(p.LOCK)},'files':reports})
    put('logs/active.log',b'excluded');put('.venv/environment.bin',b'excluded');put('state/transient.log',b'excluded');put('state/download.tmp',b'excluded')
    return put,rec


class Tests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.put,self.rec=fixture(self.root)
    def tearDown(self):self.temp.cleanup()
    def test_complete_matrix_create_only_and_bulk_omission(self):
        out,inv=p.build(self.root)
        self.assertEqual(out['completion']['updates'],180000);self.assertEqual(out['completion']['complete_formal_trajectories'],160)
        self.assertEqual(len(out['restorable_input_files']),4)
        self.assertFalse(any(x.startswith(('.venv/','logs/','data/source/','data/unpacked/')) for x in out['files']))
        self.assertNotIn('state/transient.log',out['files']);self.assertNotIn(p.TARGET,out['files']);self.assertNotIn('RELEASE_GATE.json',out['files'])
        value,created=p.publish(self.root,out,inv);self.assertTrue(created)
        before=(self.root/p.TARGET).read_bytes();again,inv2=p.build(self.root);_,created=p.publish(self.root,again,inv2)
        self.assertFalse(created);self.assertEqual(before,(self.root/p.TARGET).read_bytes());self.assertFalse((self.root/'RELEASE_GATE.json').exists())
    def test_missing_or_changed_fixed_predictions_rejected(self):
        (self.root/'artifacts/open_loop/pusht/predictions/H0.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA/bytes'):p.build(self.root)
    def test_short_training_journal_rejected(self):
        job='R3_pusht_PRED_REFIT_s103201';(self.root/f'artifacts/train/{job}/updates.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'journal'):p.build(self.root)
    def test_extra_formal_job_is_not_hidden_by_six_completed_jobs(self):
        self.put('artifacts/train/unplanned/updates.jsonl',b'{"step":1}\n')
        with self.assertRaisesRegex(ValueError,'extra formal-job'):p.build(self.root)
    def test_missing_member_of_CEM_matrix_rejected(self):
        (self.root/'artifacts/planning/FORMAL/pusht/pusht_case0/H0/attempt_0/trajectory.npz').unlink()
        with self.assertRaises(FileNotFoundError):p.build(self.root)
    def test_actual_exit_receipt_required(self):
        self.put('state/FINAL_GPU_EXIT.json',{'status':'ALL_R3_GPU_JOBS_EXITED','hostname':'fixture','active_compute_pids':[123],'R3_processes':[],'checked_at':'fixed'})
        with self.assertRaisesRegex(ValueError,'exit receipt'):p.build(self.root)
    def test_unlisted_bulk_cannot_be_silently_excluded(self):
        d=p.read(self.root/p.LOCK);d['restorable_input_files'].pop(next(iter(d['restorable_input_files'])));self.put(p.LOCK,d)
        with self.assertRaisesRegex(ValueError,'explicitly'):p.build(self.root)
    def test_different_existing_publication_and_mutation_preserved(self):
        out,inv=p.build(self.root);p.publish(self.root,out,inv);before=(self.root/p.TARGET).read_bytes()
        with self.assertRaisesRegex(RuntimeError,'Different existing'):p.publish(self.root,{**out,'optimizer_updates':1},inv)
        self.assertEqual(before,(self.root/p.TARGET).read_bytes())
        (self.root/'reports/FINAL_SCIENTIFIC_REPORT_ZH.md').write_text('mutation')
        with self.assertRaisesRegex(RuntimeError,'changed'):p.publish(self.root,out,inv)
    def test_symlink_and_circular_metadata_rejected(self):
        (self.root/'reports/link.txt').symlink_to(self.root/'r3/model.py')
        with self.assertRaisesRegex(ValueError,'Symlink'):p.build(self.root)
        with self.assertRaises(ValueError):p.Inventory(self.root).add(p.TARGET)
        with self.assertRaises(ValueError):p.local(self.root,'../escape')
    def test_required_asset_mapping_must_precede_lock(self):
        d=p.read(self.root/p.LOCK);d['files'].pop(p.ASSETS);self.put(p.LOCK,d)
        with self.assertRaisesRegex(ValueError,'official-asset'):p.build(self.root)


class AssetLockTests(unittest.TestCase):
    def test_model_lock_binds_only_root_relative_asset_evidence(self):
        source=SCRIPT.with_name('lock_models.py');spec=importlib.util.spec_from_file_location('metadata_lock',source)
        lock=importlib.util.module_from_spec(spec);spec.loader.exec_module(lock)
        class ReachedExistingLogic(Exception):pass
        class FakeInventory:
            def __init__(self,status='COMPLETE_METADATA_ASSET_MAPPING'):self.added=[];self.status=status
            def json(self,name,**kwargs):
                if name!=p.ASSETS:raise ReachedExistingLogic
                return {'version':'R3_OFFICIAL_ASSET_MANIFEST_V1','status':self.status,'tasks':{t:{} for t in p.TASKS},
                        'files':{'state/exact.json':{'sha256':'a'*64,'bytes':2}},
                        'sources':{'lewm':{'files':{'repo_relative.py':'b'*64}}}}
            def add(self,name,expected=None):self.added.append((name,expected))
        with tempfile.TemporaryDirectory() as td,patch.object(lock,'ROOT',Path(td)):
            good=FakeInventory()
            with self.assertRaises(ReachedExistingLogic):lock.static_inventory(good,{})
            self.assertEqual(good.added,[('state/exact.json',{'sha256':'a'*64,'bytes':2})])
            with self.assertRaisesRegex(RuntimeError,'official-asset'):lock.static_inventory(FakeInventory('PENDING'),{})


if __name__=='__main__':unittest.main(verbosity=2)
