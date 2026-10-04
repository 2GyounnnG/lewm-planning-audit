"""Synthetic acceptance evidence only; no real training, EVAL or GPU calls."""
import os, sys, json, hashlib, tempfile, unittest, importlib.util, copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('r3_acceptance_helper',ROOT/'scripts/final_acceptance.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
from r3 import statistics
from r3.train_worker import WindowAdapter, WindowSampler, digest

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='R3_SYNTHETIC_ACCEPTANCE_',dir='/private/tmp');self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def put(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,bytes):p.write_bytes(value)
        else:p.write_text(json.dumps(value))
        return p
    def audit(self):
        obj=a.Audit(self.root)
        for p in self.root.rglob('*'):
            if p.is_file():
                rel=str(p.relative_to(self.root));obj.files[rel]={'sha256':a.sha(p),'bytes':p.stat().st_size};obj.verified[rel]=a.signature(p)
        return obj
    def sample(self):
        cache={e:{'z':np.zeros((40,192),np.float32),'actions':np.zeros((40,2),np.float32),'raw_indices':np.arange(40),'stride':5,'legal_starts':np.arange(21)} for e in ('a','中文')}
        adapter=WindowAdapter(cache,{'history_size':3,'latent_dim':192,'macro_action_dim':10},allowed_ids=list(cache),role='REFIT_TRAIN');s=WindowSampler(adapter,19)
        for _ in range(3):s.count(s.next()[0])
        p=self.root/'counts.npz';s.save_counts(p)
        with np.load(p) as f:counts={k:f[k] for k in f.files}
        identity={'episode_ids':adapter.ids,'legal_starts_sha256':digest({e:st.tolist() for e,st in zip(adapter.ids,adapter.starts)})}
        return counts,identity
    def trajectory(self,task='pusht'):
        d=7 if task=='pusht' else 2; physical=np.arange(3*d,dtype=np.float64).reshape(3,d)*.03;goal=np.arange(d,dtype=np.float64)*.01
        arr={'raw_actions':np.zeros((3,2)),'rewards':np.zeros(3),'step_success':np.array([False,False,True]),'step_truncated':np.zeros(3,bool),'physical_state':physical,'goal_state':goal,'goal_error':np.array([float(np.linalg.norm(row-goal)) for row in physical]),'environment_step_seconds':np.array([.1,.2,.3])}
        row={'task':task,'executed_raw_steps':3,'replan_calls':1,'replans':[{'synchronized_wall_seconds':.12}],'failure_category':None,'any_step_success':1,'terminal_success':1,'success':1,'final_goal_error':float(arr['goal_error'][-1]),'environment_step_seconds':sum(arr['environment_step_seconds'].tolist()),'planning_synchronized_wall_seconds':.12}
        return row,arr
    def statistical(self,unknown=False):
        obj=self.audit();obj.analysis={'tasks':{t:{'statistics_dir':f'artifacts/statistics/{t}'} for t in a.TASKS}}
        for t in a.TASKS:
            rows=[]
            for i in range(23):
                for j,arm in enumerate(statistics.ARMS):rows.append({'task':t,'case_id':f'c{i:03}','episode_id':f'ep{i}','family_id':None if unknown else f'f{i//3}','arm':arm,'success':int((i+j)%3==0),'phase':'FORMAL','evaluation_kind':'CLOSED_LOOP_CEM','status':'COMPLETE','evaluation_complete':True,'failure_category':None,'trajectory_wall_seconds':float(i+j)})
            obj.formal_rows[t]=rows;d=self.root/f'artifacts/statistics/{t}';d.mkdir(parents=True);statistics.analyze_task(t,rows,d)
        evidence=self.audit();obj.files=evidence.files;obj.verified=evidence.verified
        return obj
    def test_safe_paths(self):
        for p in ('../x','/tmp/x','a//b','a/./b','a\n'):
            with self.assertRaises(ValueError):a.safe_path(self.root,p)
        (self.root/'jump').symlink_to('/private/tmp')
        with self.assertRaises(ValueError):a.safe_path(self.root,'jump/x')
    def test_external_device_no_fallback(self):
        with self.assertRaises(RuntimeError):a.external_guard(self.root)
    def test_uncertain_budget_conservative(self):
        x=a.budget_bounds({'runs':{'a':{'reserved_updates':20,'actual_updates':3},'b':{'reserved_updates':10,'actual_updates':None,'detail':{'known_successful_updates':4}}},'trajectories':{'c':{'status':'RESERVED'},'d':{'status':'INFRASTRUCTURE_FAILURE_INCOMPLETE'}}})
        self.assertEqual((x['optimizer_known_lower_bound'],x['optimizer_charged_upper_bound'],x['conservative_full_technical_trajectories']),(7,13,1))
    def test_budget_overrun(self):
        with self.assertRaises(ValueError):a.budget_bounds({'runs':{'a':{'reserved_updates':1024,'actual_updates':None},'b':{'reserved_updates':1,'actual_updates':1}}})
        with self.assertRaises(ValueError):a.budget_bounds({'trajectories':{str(i):{'status':'RESERVED'} for i in range(17)}})
    def test_sampling_exact_and_unicode_digest(self):
        c,i=self.sample();self.assertEqual(a.sampling_accounting(c,3,i)['sampled_windows'],384);self.assertEqual(a.digest([['中文',3]]),digest([['中文',3]]))
    def test_sampling_equal_total_wrong_transition_rejected(self):
        c,i=self.sample();c['raw_action_counts'][0]+=1;c['raw_action_counts'][1]-=1
        with self.assertRaisesRegex(ValueError,'reconstruction'):a.sampling_accounting(c,3,i)
    def test_sampling_nonofficial_span_rejected(self):
        c,i=self.sample();c['legal_starts'][0]=21
        with self.assertRaises(ValueError):a.sampling_accounting(c,3,i)
    def test_trajectory_two_tasks(self):
        for task in a.TASKS:
            row,arr=self.trajectory(task);self.assertEqual(a.trajectory_metrics(row,arr)['success'],1)
    def test_trajectory_goal_corruption_rejected(self):
        row,arr=self.trajectory();arr['goal_error'][1]+=.000001
        with self.assertRaisesRegex(ValueError,'Goal errors'):a.trajectory_metrics(row,arr)
    def test_trajectory_success_corruption_rejected(self):
        row,arr=self.trajectory();row['success']=0
        with self.assertRaisesRegex(ValueError,'success'):a.trajectory_metrics(row,arr)
    def test_method_failure_zero_steps_allowed(self):
        row,arr=self.trajectory();row.update(executed_raw_steps=0,replan_calls=0,replans=[],failure_category='METHOD',any_step_success=0,terminal_success=0,success=0,final_goal_error=None,environment_step_seconds=0.,planning_synchronized_wall_seconds=0.)
        arr={k:np.empty((0,2)) if k=='raw_actions' else np.empty((0,0)) if k=='physical_state' else np.empty(0) for k in arr}
        self.assertTrue(a.trajectory_metrics(row,arr)['method_failure'])
    def test_paired_stats_family_exact(self):
        obj=self.statistical();self.assertEqual(obj.statistics()['pusht']['cases'],23)
    def test_paired_stats_unknown_family_exact(self):
        obj=self.statistical(True);self.assertEqual(obj.statistics()['reacher']['family_status'],'UNAVAILABLE_FAMILY_UNKNOWN')
    def test_changed_raw_stat_rows_rejected(self):
        obj=self.statistical();obj.formal_rows['pusht'][0]['success']=1-obj.formal_rows['pusht'][0]['success']
        with self.assertRaisesRegex(ValueError,'exact completed'):obj.statistics()
    def test_changed_bootstrap_distribution_even_after_resealed_rejected(self):
        obj=self.statistical();p=self.root/'artifacts/statistics/pusht/BOOTSTRAP_DISTRIBUTIONS.npz'
        with np.load(p) as f:values={k:f[k] for k in f.files}
        values['case'][0,0]+=1;np.savez_compressed(p,**values)
        rec=self.root/'artifacts/statistics/pusht/STATISTICS_RECEIPT.json';r=a.read(rec);r['files'][p.name]={'sha256':a.sha(p),'bytes':p.stat().st_size};rec.write_text(json.dumps(r));o=self.audit();obj.files=o.files;obj.verified=o.verified
        with self.assertRaisesRegex(ValueError,'draws differ'):obj.statistics()
    def test_reset_fallback_exact_saved_matrix_and_effective_case(self):
        from scripts import validate_reset_fallback as reset
        helper=self.put('scripts/validate_reset_fallback.py',(ROOT/'scripts/validate_reset_fallback.py').read_bytes())
        cases=[{'case_id':f'tech{i}','episode_id':f'ep{i}','source_asset_sha256':'b'*64,'reset_seed':None} for i in range(2)]
        evaluation=[{'case_id':'eval','episode_id':'ep_eval','source_asset_sha256':'b'*64,'reset_seed':None}]
        roles={'cases':{'TECH':cases,'EVAL':evaluation}};role_path=self.put('manifests/pusht_data_roles.json',roles)
        prefix='artifacts/technical/reset_fixture';out=self.root/prefix;out.mkdir(parents=True)
        def produce(case,seed,repeat,path):
            phase={'configuration':{'SYNTHETIC':True},'variations_not_overridden':{'physics':'fixed'}}
            row={'case_id':case['case_id'],'seed':seed,'repeat':repeat,'path':path,'status':'PASS','close_called':True,'reset_attempts':1,'resets':1,'step_attempts':1,'raw_steps':1,'snapshots':{'before_zero_action':copy.deepcopy(phase),'after_zero_action':copy.deepcopy(phase)}}
            return row,{'before_zero_action__state7':np.arange(7,dtype=np.float64),'source_state_minus_reset_state':np.ones(7)*(seed+1)}
        matrix,arrays=reset.evaluate_matrix(cases,produce)
        for row in matrix['trials']:
            np.savez_compressed(out/f'trial_{row["trial_id"]}.npz',**arrays[row['trial_id']]);self.put(prefix+f'/trial_{row["trial_id"]}.json',row)
        for case in cases:np.savez_compressed(out/(case['case_id']+'_source.npz'),state=np.arange(7))
        self.put(prefix+'/PROGRESS.json',{'status':'SYNTHETIC_COMPLETE'})
        files={p.name:{'sha256':a.sha(p),'bytes':p.stat().st_size} for p in out.iterdir()}
        source={'path':'data/unpacked/source.h5','sha256':'b'*64,'bytes':99}
        receipt={**matrix,'task':'pusht','roles_sha256':a.sha(role_path),'cases':cases,'source_files':{'b'*64:source},'input_hashes':{'scripts/validate_reset_fallback.py':a.sha(helper)},'files':files}
        rp=self.put(prefix+'/RESET_VALIDATION_RECEIPT.json',receipt)
        record=lambda p:{'path':str(p.relative_to(self.root)),'sha256':a.sha(p),'bytes':p.stat().st_size}
        mf=self.put('manifests/RESET_FALLBACK_MANIFEST.json',{'status':'VALIDATED_FIXED_RESET_FALLBACK_FROZEN','tasks':{'pusht':{'data_roles':record(role_path),'validation_receipt':record(rp),'fallback_seed':0,'affected_case_ids':sorted(c['case_id'] for c in cases+evaluation),'provenance':'SOURCE_SEED_UNKNOWN_VALIDATED_FIXED_SEED0_NOT_RECOVERED'}}})
        self.put('manifests/MODELS_AND_SELECTION_LOCK.json',{'files':{str(p.relative_to(self.root)):record(p) for p in (mf,rp)}})
        obj=self.audit();obj.roles={'pusht':roles,'reacher':{'cases':{'TECH':[],'EVAL':[]}}};obj.restorable={source['path']:source}
        self.assertEqual(obj.reset_fallback_evidence()['pusht']['affected_cases'],3)
        changed=obj.effective_case('pusht',evaluation[0]);self.assertEqual(changed['reset_seed'],0);self.assertIsNone(evaluation[0]['reset_seed']);self.assertEqual(changed['reset_seed_validation_sha256'],a.sha(rp))

    def test_reacher_reset_uses_frozen_task_specific_comparison_graph(self):
        from scripts import validate_reacher_reset as reset
        helper=self.put('scripts/validate_reacher_reset.py',(ROOT/'scripts/validate_reacher_reset.py').read_bytes())
        cases=[{'case_id':f'tech{i}','episode_id':f'ep{i}','source_asset_sha256':'b'*64,'reset_seed':None} for i in range(2)]
        evaluation=[{'case_id':'eval','episode_id':'ep_eval','source_asset_sha256':'b'*64,'reset_seed':None}]
        roles={'cases':{'TECH':cases,'EVAL':evaluation}};role_path=self.put('manifests/reacher_data_roles.json',roles)
        prefix='artifacts/technical/reset_fixture';out=self.root/prefix;out.mkdir(parents=True)
        def produce(case,seed,repeat,path):
            phase={'configuration':{'SYNTHETIC':True},'variations_not_overridden':{'physics':'fixed'},'variations_all':{'physics':'fixed'}}
            row={'case_id':case['case_id'],'seed':seed,'repeat':repeat,'path':path,'status':'PASS','close_called':True,'reset_attempts':1,'resets':1,'step_attempts':1,'raw_steps':1,'snapshots':{'before_zero_action':copy.deepcopy(phase),'after_zero_action':copy.deepcopy(phase)}}
            return row,{'before_zero_action__state7':np.arange(7,dtype=np.float64),'point_task_diagnostic__native_reward':np.ones(7)*(seed+1)}
        matrix,arrays=reset.evaluate_matrix(cases,produce)
        for row in matrix['trials']:
            np.savez_compressed(out/f'trial_{row["trial_id"]}.npz',**arrays[row['trial_id']]);self.put(prefix+f'/trial_{row["trial_id"]}.json',row)
        for case in cases:np.savez_compressed(out/(case['case_id']+'_source.npz'),state=np.arange(7))
        self.put(prefix+'/PROGRESS.json',{'status':'SYNTHETIC_COMPLETE'})
        files={p.name:{'sha256':a.sha(p),'bytes':p.stat().st_size} for p in out.iterdir()}
        source={'path':'data/unpacked/source.h5','sha256':'b'*64,'bytes':99}
        receipt={**matrix,'task':'reacher','roles_sha256':a.sha(role_path),'cases':cases,'source_files':{'b'*64:source},'input_hashes':{'scripts/validate_reacher_reset.py':a.sha(helper)},'files':files}
        rp=self.put(prefix+'/RESET_VALIDATION_RECEIPT.json',receipt)
        record=lambda p:{'path':str(p.relative_to(self.root)),'sha256':a.sha(p),'bytes':p.stat().st_size}
        mf=self.put('manifests/RESET_FALLBACK_MANIFEST.json',{'status':'VALIDATED_FIXED_RESET_FALLBACK_FROZEN','tasks':{'reacher':{'data_roles':record(role_path),'validation_receipt':record(rp),'fallback_seed':0,'affected_case_ids':sorted(c['case_id'] for c in cases+evaluation),'provenance':'SOURCE_SEED_UNKNOWN_VALIDATED_FIXED_SEED0_NOT_RECOVERED'}}})
        self.put('manifests/MODELS_AND_SELECTION_LOCK.json',{'files':{str(p.relative_to(self.root)):record(p) for p in (mf,rp)}})
        obj=self.audit();obj.roles={'reacher':roles,'pusht':{'cases':{'TECH':[],'EVAL':[]}}};obj.restorable={source['path']:source}
        self.assertEqual(obj.reset_fallback_evidence()['reacher']['affected_cases'],3)
        changed=obj.effective_case('reacher',evaluation[0]);self.assertEqual(changed['reset_seed'],0);self.assertIsNone(evaluation[0]['reset_seed']);self.assertEqual(changed['reset_seed_validation_sha256'],a.sha(rp))

    def test_saved_cache_inputs_and_training_moments(self):
        rng=np.random.default_rng(551);frozen={'sha256':'a'*64,'tensors':{}}
        role_docs={};analysis={'tasks':{t:{'open_loop_dir':f'artifacts/open_loop/{t}'} for t in a.TASKS}}
        for task in a.TASKS:
            eps=[{'episode_id':f'{task}_a','role':'REFIT_TRAIN','length':45},{'episode_id':f'{task}_b','role':'EVAL','length':45}];case={'episode_id':eps[1]['episode_id'],'open_loop_window_start_raw':1};role_docs[task]={'episodes':eps,'cases':{'EVAL':[case]}}
            records={};train=None;ev=None
            for ep in eps:
                z=rng.normal(size=(45,192)).astype(np.float32);actions=rng.normal(size=(45,2)).astype(np.float32)
                name=f'data/cache/{task}/{ep["episode_id"]}.npz';p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
                np.savez(p,z=z,actions=actions,raw_indices=np.arange(45,dtype=np.int64),stride=np.array(5),legal_starts=np.arange(26,dtype=np.int64))
                item={'path':name,'sha256':a.sha(p),'bytes':p.stat().st_size,'role':ep['role'],'length':45,'legal_windows':26};records[ep['episode_id']]=item
                self.put(str(Path(name).with_suffix('.json')),{'identity':{'episode':ep},'file':item,'source_state_or_reward_read':False})
                if ep['role']=='REFIT_TRAIN':train=z
                else:ev=(z,actions)
            source=self.put(f'manifests/{task}_data_roles.json',role_docs[task])
            self.put(f'manifests/{task}_cache.json',{'status':'FROZEN_OBSERVED_CACHE_COMPLETE','episodes':records,'optimizer_updates':0,'state_labels_read':0,'input_hashes':{str(source.relative_to(self.root)):a.sha(source)},'frozen_hashes':frozen})
            self.put(f'artifacts/train/R3_{task}_PRED_REFIT_s103201/FROZEN_INITIAL.json',frozen)
            folder=self.root/analysis['tasks'][task]['open_loop_dir'];folder.mkdir(parents=True)
            z,act=ev;np.savez(folder/'inputs.npz',initial_z=z[np.array([1,6,11])][None],macro_actions=act[1:36].reshape(1,7,10));np.savez(folder/'targets.npz',target_z=z[np.array([16,21,26,31,36])][None])
            d=train.astype(np.float64);mean=d.sum(0)/45;var=np.square(d).sum(0)/45-mean**2
            np.savez(folder/'TRAIN_coordinates.npz',coordinate_mean=mean,coordinate_variance=var)
            from r3.open_loop import VARIANCE_FORMULA
            self.put(str((folder/'TRAIN_variance.json').relative_to(self.root)),{'scalar':float(var.mean()),'frames':45,'role':'REFIT_TRAIN','formula':VARIANCE_FORMULA})
        obj=self.audit();obj.roles=role_docs;obj.analysis=analysis
        self.assertEqual(obj.cached_input_evidence()['pusht']['TRAIN_frames'],45)
        p=self.root/'artifacts/open_loop/pusht/targets.npz'
        with np.load(p) as f:target=f['target_z'].copy()
        target[0,0,0]+=.1;np.savez(p,target_z=target)
        updated=self.audit();obj.files=updated.files;obj.verified=updated.verified
        with self.assertRaisesRegex(ValueError,'exact original cache'):obj.cached_input_evidence()

    def test_exact_manifest_and_mutation_guard(self):
        self.put('scripts/final_acceptance.py',Path(a.__file__).read_bytes());self.put('data.bin',b'abc')
        obj=self.audit();self.put('manifests/RECOVERY_FINAL.json',{'status':'FINAL_SNAPSHOT_READY_FOR_RECOVERY_AND_ACCEPTANCE','files':obj.files})
        obj=a.Audit(self.root)
        with patch.object(a,'DEVICE',self.root.stat().st_dev):self.assertEqual(obj.verify_manifest()['files'],2)
        self.put('data.bin',b'abd')
        with self.assertRaisesRegex(ValueError,'changed'):obj.require('data.bin')
    def test_manifest_wrong_bytes_rejected(self):
        self.put('scripts/final_acceptance.py',Path(a.__file__).read_bytes());self.put('data.bin',b'abc');obj=self.audit();obj.files['data.bin']['bytes']=4
        self.put('manifests/RECOVERY_FINAL.json',{'status':'FINAL_SNAPSHOT_READY_FOR_RECOVERY_AND_ACCEPTANCE','files':obj.files})
        with patch.object(a,'DEVICE',self.root.stat().st_dev),self.assertRaisesRegex(ValueError,'mismatch'):a.Audit(self.root).verify_manifest()
    def test_failed_acceptance_retains_attempt_no_gate(self):
        self.put('scripts/final_acceptance.py',Path(a.__file__).read_bytes());obj=self.audit();self.put('manifests/RECOVERY_FINAL.json',{'status':'FINAL_SNAPSHOT_READY_FOR_RECOVERY_AND_ACCEPTANCE','files':obj.files})
        with patch.object(a,'DEVICE',self.root.stat().st_dev),patch.object(a,'external_guard',return_value={'SYNTHETIC':True}),patch.object(a.Audit,'runtime_and_prior',side_effect=RuntimeError('synthetic dependency missing')):
            out=a.run(self.root)
        self.assertEqual(out['status'],'ACCEPTANCE_PENDING_OR_FAILED');self.assertFalse((self.root/'RELEASE_GATE.json').exists());self.assertEqual(len(list((self.root/'reports/acceptance_attempts').glob('*.json'))),1)

if __name__=='__main__':unittest.main(verbosity=2)
