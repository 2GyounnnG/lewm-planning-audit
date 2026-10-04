"""Pure synthetic metadata fixtures. No real receipts, tensors, GPU or optimizer."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r3_authorize_tested',ROOT/'scripts/authorize_formal.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class Tests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory(prefix='R3_SYNTHETIC_AUTH_');self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def put(self,name,x):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(x if isinstance(x,bytes) else json.dumps(x).encode());return p
    def rec(self,name):
        p=self.root/name;return {'path':name,'sha256':a.sha(p),'bytes':p.stat().st_size}
    def data(self,name):return json.loads((self.root/name).read_text())
    def fixture(self):
        for name in a.FIXED_INPUTS:self.put(name,{} if name.endswith('.json') else b'# synthetic code\n')
        self.put('protocol/marker.md',b'SYNTHETIC ONLY')
        self.put('manifests/JOBS.json',{'jobs':[{'job_id':f'R3_{t}_PRED_REFIT_s{s}','task':t,'refit_seed':s,'updates':30000,'effective_batch':128,'microbatch':128} for t in a.TASKS for s in a.SEEDS]})
        self.put('manifests/EXPERIMENT_LOCK.json',{'status':'EXPERIMENT_PROTOCOL_FROZEN_BEFORE_OPTIMIZER_AND_OUTCOME','formal_updates':180000,'protocol_files':{'protocol/marker.md':self.rec('protocol/marker.md')['sha256']}})
        self.put('state/OFFICIAL_STRICT_LOAD_CPU.json',{'status':'STRICT_LOAD_AND_SYNTHETIC_WRAPPER_PASS','tasks':{t:{'strict_load':{'missing_keys':[],'unexpected_keys':[]}} for t in a.TASKS}})
        for tree in a.SOURCE_TREES:
            path=f'source/{tree}/source.py';self.put(path,b'# pinned fake source');self.put(f'state/{tree}_source_manifest.json',{'files':{'source.py':self.rec(path)}})
        roles={};data_paths={};cost_paths={}
        for t in a.TASKS:
            episodes=[{'episode_id':f'{t}_train','role':'REFIT_TRAIN','length':40},{'episode_id':f'{t}_monitor','role':'MONITOR','length':40}]+[{'episode_id':f'{t}_tech{i}','role':'TECH','length':40} for i in range(2)]+[{'episode_id':f'{t}_eval{i}','role':'EVAL','length':40} for i in range(20)]
            cases={role:[{'task':t,'case_id':ep['episode_id'],'episode_id':ep['episode_id'],'reset_seed':0,'role':role} for ep in episodes if ep['role']==role] for role in ('TECH','EVAL')}
            doc={'task':t,'status':'METADATA_ROLES_FROZEN','blocked_reasons':[],'selection_used_model_outputs':False,'episodes':episodes,'cases':cases,'monitor_windows':[[t+'_monitor',0]],'technical_monitor_windows':[[t+'_tech0',0]]};roles[t]=doc;rolefile=f'manifests/{t}_data_roles.json';self.put(rolefile,doc)
            records={}
            for ep in episodes:
                name=f'data/cache/{t}/{ep["episode_id"]}.npz';self.put(name,b'not an actual tensor; metadata fixture')
                item={**self.rec(name),'role':ep['role'],'length':40};records[ep['episode_id']]=item
                self.put(str(Path(name).with_suffix('.json')),{'file':item,'identity':{'episode':ep},'source_state_or_reward_read':False})
            self.put(f'manifests/{t}_cache.json',{'task':t,'status':'FROZEN_OBSERVED_CACHE_COMPLETE','episodes':records,'optimizer_updates':0,'state_labels_read':0,'input_hashes':{rolefile:self.rec(rolefile)['sha256']}})
            model_files={};tree=[]
            for filename in ('weights.pt','config.json'):
                name=f'official/{t}/{filename}';self.put(name,b'not a real weight; fixture only');r=self.rec(name);model_files[filename]=r;tree.append({'path':filename,'size':r['bytes'],'lfs':{'oid':r['sha256']}})
            self.put(f'manifests/{t}_model_assets.json',{'repo':'quentinll/lewm-'+t,'revision':'modelrev','status':'PINNED_ASSETS_DOWNLOADED_VERIFIED','files':model_files})
            self.put(f'state/{t}_model_api.json',{'sha':'modelrev'});self.put(f'state/{t}_model_tree.json',tree)
            archive_name=f'data/source/{t}/original.zst';self.put(archive_name,b'fake original archive');archive=self.rec(archive_name)
            h5_name=f'data/unpacked/{t}/source.h5';self.put(h5_name,b'fake HDF5 never opened as HDF5');h5=self.rec(h5_name)
            self.put(f'manifests/{t}_source_map.json',{'official_revision':'datarev','selection_based_on_model_outputs':False,'assets':{h5['sha256']:h5}})
            self.put(f'manifests/{t}_normalization.json',{'source_assets':{h5['sha256']:h5}})
            asset_name=f'manifests/{t}_data_assets.json';self.put(asset_name,{'repo':'quentinll/lewm-'+t,'revision':'datarev','status':'PINNED_ASSETS_DOWNLOADED_VERIFIED','files':{'original.zst':archive}})
            self.put(f'state/{t}_data_api.json',{'sha':'datarev'});self.put(f'state/{t}_data_tree.json',[{'path':'original.zst','size':archive['bytes'],'lfs':{'oid':archive['sha256']}}])
            member=f'state/unpack_members/{t}.json';self.put(member,{'status':'COMPLETE','bytes':h5['bytes'],'sha256':h5['sha256'],'identity':{'archive':archive}})
            self.put(f'manifests/{t}_data_unpacked.json',{'status':'SOURCE_UNPACKED_SHA_VERIFIED','source_receipt_sha256':self.rec(asset_name)['sha256'],'files':[{**h5,'member_receipt':member,'member_receipt_sha256':self.rec(member)['sha256']}]})
            for kind,basename in [('data','REAL_DATA_MODEL_CHECK.json'),('cost','COST_WRAPPER_CHECK.json')]:
                prefix=f'artifacts/technical/{t}_{kind}';name=prefix+'/'+basename
                d={'task':t,'status':'PASS','optimizer_updates':0,'input_hashes':{'r3/model.py':self.rec('r3/model.py')['sha256']}}
                if kind=='data':d.update(loaded_model_tensors_unchanged=True,optimizer_objects_created=0,full_policy_trajectories=0,EVAL_arrays_read=0,source_state_reward_goal_labels_read=0);data_paths[t]=name
                else:d.update(complete_CEM_trajectories=0,environment_steps=0,source_future_actions_read=False,rows=[{'case_id':c['case_id'],**{k:True for k in ('zero_update_wrapped_cost_exact','official_goal_encoding_exact','terminal_squared_sum_cost_matches','all_finite')}} for c in cases['TECH']]);cost_paths[t]=name
                self.put(name,d);self.put(prefix+'/SHA256.json',{'status':'PASS','files':{basename:self.rec(name)}})
        artifact='artifacts/technical/profile/result.pt';self.put(artifact,b'not real weights')
        historical='state/technical/split4.json';self.put(historical,{'status':'PASS','files':[{'path':'superseded/resume4.pt','sha256':'a'*64,'bytes':1}]})
        ident='a'*64;selection={'selected_workers':4,'workers_per_gpu':2,'microbatch':128,'scientific_outcomes_used':False}
        A={'status':'PASS','end_to_end_updates_per_second':2.,'steady_update_span':{'status':'BOUNDED_OBSERVED_INTERVAL','aggregate_updates_per_second_upper':2.1},'telemetry_errors':[],'files':[self.rec(artifact)]}
        B={'status':'PASS','end_to_end_updates_per_second':3.,'steady_update_span':{'status':'BOUNDED_OBSERVED_INTERVAL','aggregate_updates_per_second_lower':2.8},'telemetry_errors':[],'files':[self.rec(artifact)]}
        self.put('state/GPU_SMOKE.json',{'status':'PASS','identity_sha256':ident,'optimizer_updates':32,'genuine_process_exit_resume':True,'files':[self.rec(artifact),self.rec(historical)]})
        self.put('state/GPU_PROFILE.json',{'status':'PROFILE_COMPLETE','identity_sha256':ident,**selection,'A':A,'B':B})
        self.put('reports/GPU_PROFILE.csv',b'synthetic fixture')
        self.put('state/TRAINING_TECHNICAL_GATE.json',{'status':'TRAINING_TECHNICAL_GATES_PASS','identity_sha256':ident,**selection,'source_files':{'r3/model.py':self.rec('r3/model.py')['sha256']},'smoke_receipt':self.rec('state/GPU_SMOKE.json'),'profile_receipt':self.rec('state/GPU_PROFILE.json'),'profile_table':self.rec('reports/GPU_PROFILE.csv')})
        specs=[{'task':t,'case':c} for t in a.TASKS for c in roles[t]['cases']['TECH']];plan={'phase':'TECH','cases':specs,'files':{'r3/planning.py':self.rec('r3/planning.py')}}
        self.put('state/planning/TECH/EXECUTION_PLAN.json',plan);sealed={};trajectories={}
        for spec in specs:
            t=spec['task'];cid=spec['case']['case_id']
            for arm in ('H0','H0_CLONE'):
                path=f'artifacts/planning/TECH/{t}/{cid}/{arm}/attempt_0/result.json';self.put(path,{'status':'COMPLETE','evaluation_complete':True,'failure_category':None,'task':t,'case_id':cid,'arm':arm,'phase':'TECH'});sealed[path]=self.rec(path)
                trajectories[f'R3/TECH/{t}/{cid}/{arm}/attempt_0']={'status':'COMPLETE','result_path':path,'result_sha256':self.rec(path)['sha256']}
        self.put('state/PLANNING_TECH_GATE.json',{'status':'PASS','complete_trajectories':8,'clone_exact':True,'execution_plan_sha256':a.digest(plan),'files':sealed,'clone_comparisons':[{'status':'PASS'} for _ in range(4)]})
        self.put('state/TECHNICAL_LEDGER.json',{'runs':{'smoke':{'reserved_updates':32,'actual_updates':32},'A':{'reserved_updates':96,'actual_updates':96},'B':{'reserved_updates':192,'actual_updates':192}},'trajectories':trajectories})
        return data_paths,cost_paths
    def rebuild_gate(self):
        d=self.data('state/TRAINING_TECHNICAL_GATE.json');d['profile_receipt']=self.rec('state/GPU_PROFILE.json');self.put('state/TRAINING_TECHNICAL_GATE.json',d)
    def test_complete_metadata_authorizes_without_training(self):
        dc,cc=self.fixture();out,e=a.build_authorization(self.root,dc,cc)
        self.assertEqual(out['status'],'TECHNICAL_GATES_PASS');self.assertEqual(len(out['job_ids']),6);self.assertFalse(out['training_launched'])
        for name in ('data/unpacked/pusht/source.h5','artifacts/technical/profile/result.pt','data/cache/pusht/pusht_train.npz'):
            self.assertIn(name,out['verified_files']);self.assertNotIn(name,out['source_hashes'])
        self.assertIn(dc['pusht'],out['source_hashes']);self.assertIn('official/pusht/weights.pt',out['source_hashes'])
        published,new=a.publish(self.root,out);self.assertTrue(new);before=(self.root/a.AUTH).read_bytes();again,new=a.publish(self.root,out);self.assertFalse(new);self.assertEqual(before,(self.root/a.AUTH).read_bytes())
    def test_existing_different_authority_never_overwritten(self):
        self.fixture();self.put(a.AUTH,{'status':'OTHER'});before=(self.root/a.AUTH).read_bytes()
        with self.assertRaises(RuntimeError):a.publish(self.root,{'status':'TECHNICAL_GATES_PASS'})
        self.assertEqual(before,(self.root/a.AUTH).read_bytes())
    def test_corrupted_H5_initial_authorization_rejected(self):
        dc,cc=self.fixture();self.put('data/unpacked/pusht/source.h5',b'changed raw source')
        with self.assertRaisesRegex(RuntimeError,'SHA/bytes'):a.build_authorization(self.root,dc,cc)
    def test_wrong_cache_identity_rejected(self):
        dc,cc=self.fixture();name='data/cache/pusht/pusht_train.npz';self.put(name,b'wrong input')
        with self.assertRaisesRegex(RuntimeError,'SHA/bytes'):a.build_authorization(self.root,dc,cc)
    def test_explicit_wrong_task_receipt_rejected(self):
        dc,cc=self.fixture();dc['pusht'],dc['reacher']=dc['reacher'],dc['pusht']
        with self.assertRaisesRegex(RuntimeError,'Selected actual task'):a.build_authorization(self.root,dc,cc)
    def test_formal_prior_attempt_rejected(self):
        dc,cc=self.fixture();self.put('artifacts/train/job/updates.jsonl',b'{"step":1}\n')
        with self.assertRaisesRegex(RuntimeError,'prior formal attempt'):a.build_authorization(self.root,dc,cc)
    def test_profile_ambiguous_defaults_two(self):
        dc,cc=self.fixture();p=self.data('state/GPU_PROFILE.json');p.update(selected_workers=2,workers_per_gpu=1);p['B']['steady_update_span']={'status':'AMBIGUOUS_OBSERVATION_RESOLUTION'};self.put('state/GPU_PROFILE.json',p)
        g=self.data('state/TRAINING_TECHNICAL_GATE.json');g.update(selected_workers=2,workers_per_gpu=1);self.put('state/TRAINING_TECHNICAL_GATE.json',g);self.rebuild_gate()
        out,_=a.build_authorization(self.root,dc,cc);self.assertEqual(out['selection']['selected_workers'],2)
    def test_profile_forged_four_despite_low_gain_rejected(self):
        dc,cc=self.fixture();p=self.data('state/GPU_PROFILE.json');p['B']['end_to_end_updates_per_second']=2.01;self.put('state/GPU_PROFILE.json',p);self.rebuild_gate()
        with self.assertRaisesRegex(RuntimeError,'fixed comparable10%'):a.build_authorization(self.root,dc,cc)
    def test_audited_higher_concurrency_OOM_does_not_require320(self):
        dc,cc=self.fixture();p=self.data('state/GPU_PROFILE.json');p.update(selected_workers=2,workers_per_gpu=1);p['B'].update(status='FAILED',higher_concurrency_rejected='AUDITED_OUT_OF_MEMORY');self.put('state/GPU_PROFILE.json',p)
        g=self.data('state/TRAINING_TECHNICAL_GATE.json');g.update(selected_workers=2,workers_per_gpu=1);self.put('state/TRAINING_TECHNICAL_GATE.json',g);self.rebuild_gate()
        ledger=self.data('state/TECHNICAL_LEDGER.json');ledger['runs']['B']={'reserved_updates':48,'actual_updates':None,'status':'UNCERTAIN_OPTIMIZER_ATTEMPT','detail':{'known_successful_updates':0}};self.put('state/TECHNICAL_LEDGER.json',ledger)
        out,_=a.build_authorization(self.root,dc,cc);self.assertEqual(out['technical_budget']['known_optimizer_updates'],128);self.assertEqual(out['technical_budget']['charged_optimizer_upper_bound'],176)
    def test_overbudget_rejected(self):
        with self.assertRaises(ValueError):a.validate_budget({'runs':{'a':{'reserved_updates':1024,'actual_updates':1024},'b':{'reserved_updates':1,'actual_updates':1}}})
    def test_unfinished_technical_active_reservation_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'active technical'):a.validate_budget({'runs':{'a':{'reserved_updates':48,'actual_updates':None}}})
    def test_hash_mutation_before_publication_rejected(self):
        self.put('r3/test.py',b'a');e=a.Evidence(self.root);e.bind('r3/test.py');self.put('r3/test.py',b'b')
        with self.assertRaises(RuntimeError):e.unchanged()
    def test_paths_explicit_no_auto_selection(self):
        self.assertEqual(a.parse_receipts(['pusht=a.json','reacher=b.json']),{'pusht':'a.json','reacher':'b.json'})
        with self.assertRaises(ValueError):a.parse_receipts(['pusht=a','pusht=b'])
        with self.assertRaises(ValueError):a.local(self.root,'../escape')
        with self.assertRaises(PermissionError):a.Evidence(self.root).bind('artifacts/open_loop/pusht/per_case.json')

if __name__=='__main__':unittest.main(verbosity=2)
