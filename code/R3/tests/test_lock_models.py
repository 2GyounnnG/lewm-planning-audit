"""Synthetic CPU tensors/metadata; no actual checkpoint, cache, environment or GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from r3.model import DELTA_FORMAT
from r3.open_loop import validate_routes
from r3.train_worker import scheduler_record

spec=importlib.util.spec_from_file_location('lock_models_fixture',Path(__file__).resolve().parents[1]/'scripts/lock_models.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def jobs():
    return [{'job_id':f'R3_{task}_PRED_REFIT_s{seed}','task':task,'refit_seed':seed,'updates':30000,'effective_batch':128,'microbatch':128}
            for task in ('pusht','reacher') for seed in (103201,103202,103203)]


def payload(step=30000,full=False):
    identity={'version':'synthetic','base_identity':{'task':'synthetic','weights_sha256':'a'*64},'contract':{'latent_dim':2}}
    frozen={'sha256':'f'*64,'tensors':{}}
    whitelist={'parameters':[{'name':'predictor.weight','trainable':True,'shared_parameter_identity':'predictor.weight','shape':[2],'dtype':'torch.float32'}]}
    p={'version':'synthetic','step':step,'identity_sha256':m.digest(identity),'frozen':frozen,
       'delta':{'format':DELTA_FORMAT,'base_identity':identity['base_identity'],'contract':identity['contract'],
                'frozen_sha256':frozen['sha256'],'replacement_parameters':{'predictor.weight':torch.ones(2)}},
       'sampling_summary':{'sampled_windows':step*128,'sampled_episode_draws':step*128,'predicted_target_tokens':step*384,'raw_action_exposures':step*1920}}
    if full:
        n=np.random.RandomState(1).get_state()
        p.update(scheduler=scheduler_record(30000),optimizer_state_is_new_post_refit=True,new_encoder_updates=0,state_labels_read=0,
                 rng={'python':random.Random(2).getstate(),'numpy':{'bit_generator':n[0],'keys':n[1].tolist(),'pos':n[2],'has_gauss':n[3],'cached_gaussian':n[4]},
                      'torch_cpu':torch.Generator(device='cpu').manual_seed(3).get_state(),'torch_cuda':[torch.zeros(16,dtype=torch.uint8)]},
                 optimizer={'param_groups':[{'params':[0],'lr':scheduler_record(30000)['lr_last'],'weight_decay':.001,'betas':(.9,.999),'eps':1e-8}],
                            'state':{0:{'step':torch.tensor(30000.),'exp_avg':torch.zeros(2),'exp_avg_sq':torch.ones(2)}}},
                 next_batch_sha256='b'*64,sampler={'rng':np.random.default_rng(4).bit_generator.state})
    return p,identity,frozen,whitelist


class LockChecks(unittest.TestCase):
    def test_fixed_exact_routing_no_chosen_winner(self):
        files={f"artifacts/train/{j['job_id']}/checkpoint_{step}.pt":{'sha256':f'{i:064x}','bytes':100}
               for i,j in enumerate(jobs()) for step in (3000,10000,30000)}
        routes=m.route_matrix(list(reversed(jobs())),files)
        for task in ('pusht','reacher'):
            arms=validate_routes(routes,task);self.assertEqual(len(arms),10)
            self.assertEqual(arms[0],{'arm':'H0','seed':None,'step':0})
            self.assertEqual([(a['seed'],a['step']) for a in arms[1:]],[(s,n) for s in (103201,103202,103203) for n in (3000,10000,30000)])
        self.assertEqual(routes['tasks']['pusht']['planning_steps'],[0,30000])

    def test_inventory_root_relative_and_same_directory_receipts(self):
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)):
            root=Path(td);folder=root/'state/check';folder.mkdir(parents=True)
            (folder/'raw.npz').write_bytes(b'no actual values')
            h=m.common.sha256(folder/'raw.npz')
            (folder/'receipt.json').write_text(json.dumps({'files':{'raw.npz':{'sha256':h,'bytes':16}}}))
            inv=m.Inventory();inv.json('state/check/receipt.json',expand=True)
            self.assertIn('state/check/raw.npz',inv.files);inv.unchanged()
            (folder/'raw.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'changed'):inv.unchanged()

    def test_actual_technical_gate_profile_table_root_path_and_basename_seal(self):
        # run_training_technical writes reports/GPU_PROFILE.csv and embeds its
        # ROOT-relative record in state/TRAINING_TECHNICAL_GATE.profile_table.
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)):
            root=Path(td);(root/'reports').mkdir();(root/'state').mkdir()
            table=root/'reports/GPU_PROFILE.csv';table.write_bytes(b'stage,workers\nA,2\nB,4\n')
            record={'path':'reports/GPU_PROFILE.csv','sha256':m.common.sha256(table),'bytes':table.stat().st_size}
            gate=root/'state/TRAINING_TECHNICAL_GATE.json'
            gate.write_text(json.dumps({'status':'TRAINING_TECHNICAL_GATES_PASS','profile_table':record}))
            inv=m.Inventory();inv.json(str(gate.relative_to(root)),expand=True)
            self.assertIn('reports/GPU_PROFILE.csv',inv.files)
            self.assertNotIn('state/reports/GPU_PROFILE.csv',inv.files)
            same_dir=root/'reports/SHA256.json'
            same_dir.write_text(json.dumps({'files':{'GPU_PROFILE.csv':{k:record[k] for k in ('sha256','bytes')}}}))
            inv.json('reports/SHA256.json',expand=True)
            self.assertEqual(inv.files['reports/GPU_PROFILE.csv'],{k:record[k] for k in ('sha256','bytes')})
            table.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'SHA/bytes differ: reports/GPU_PROFILE.csv'):
                m.Inventory().json(str(gate.relative_to(root)),expand=True)
            table.unlink()
            with self.assertRaisesRegex(FileNotFoundError,'missing: reports/GPU_PROFILE.csv'):
                m.Inventory().json(str(gate.relative_to(root)),expand=True)

    def test_source_manifest_closure_resolves_official_tree(self):
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)):
            root=Path(td);(root/'source/lewm').mkdir(parents=True);(root/'state').mkdir()
            p=root/'source/lewm/LICENSE';p.write_bytes(b'license')
            (root/'state/lewm_source_manifest.json').write_text(json.dumps({'files':{'LICENSE':{'sha256':m.common.sha256(p),'bytes':7}}}))
            inv=m.Inventory();inv.json('state/lewm_source_manifest.json',expand=True)
            self.assertIn('source/lewm/LICENSE',inv.files)
            self.assertEqual(inv.reference_path(str(p),'state/check.json'),'source/lewm/LICENSE')

    def test_nested_planning_source_map_has_exact_pinned_source_root(self):
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)):
            root=Path(td);p=root/'source/swm_compat/stable_worldmodel/world/world.py';p.parent.mkdir(parents=True);p.write_bytes(b'official')
            (root/'state').mkdir();(root/'state/swm_compat_source_manifest.json').write_text(json.dumps({'revision':'pinned','files':{}}))
            (root/'state/preflight.json').write_text(json.dumps({'source':{'revision':'pinned','files':{'stable_worldmodel/world/world.py':m.common.sha256(p)}}}))
            inv=m.Inventory();inv.json('state/preflight.json',expand=True)
            self.assertIn(str(p.relative_to(root)),inv.files)

    def test_inventory_rejects_eval_effects_and_escape_and_self(self):
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)):
            inv=m.Inventory()
            for name in ('artifacts/open_loop/pusht/per_case.json','artifacts/planning/FORMAL/a.json','tables/results.csv'):
                with self.assertRaises(PermissionError):inv.add(name)
            for name in ('../x',m.LOCK,m.ROUTING):
                with self.assertRaises(ValueError):inv.add(name)
            with self.assertRaises(ValueError):inv.reference_path('/outside/root/file','state/x.json')

    def test_immutable_publish_same_reuse_different_reject(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'lock.json';m.immutable_write(p,{'a':1});before=p.read_bytes();stamp=p.stat().st_mtime_ns
            m.immutable_write(p,{'a':1});self.assertEqual(p.stat().st_mtime_ns,stamp)
            with self.assertRaises(RuntimeError):m.immutable_write(p,{'a':2})
            self.assertEqual(p.read_bytes(),before)

    def test_compact_all_milestones_validate_and_wrong_identity_blocks(self):
        for step in (3000,10000,30000):
            p,i,f,w=payload(step);m.audit_payload(p,step,i,f,w)
            p['step']+=1
            with self.assertRaisesRegex(RuntimeError,'step/identity'):m.audit_payload(p,step,i,f,w)

    def test_whitelist_nonfinite_and_sampling_not_accepted(self):
        p,i,f,w=payload();p['delta']['replacement_parameters']['predictor.weight'][0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'replacement'):m.audit_payload(p,30000,i,f,w)
        p,i,f,w=payload();p['sampling_summary']['predicted_target_tokens']-=1
        with self.assertRaisesRegex(RuntimeError,'sampling/token'):m.audit_payload(p,30000,i,f,w)

    def test_full_resume_checks_rng_optimizer_lr_and_step(self):
        p,i,f,w=payload(full=True);cpu_before=torch.random.get_rng_state().clone();python_before=random.getstate()
        m.audit_payload(p,30000,i,f,w,full=True)
        self.assertTrue(torch.equal(cpu_before,torch.random.get_rng_state()));self.assertEqual(python_before,random.getstate())
        for key,value in [('lr',1.),('weight_decay',0.)]:
            wrong=copy.deepcopy(p);wrong['optimizer']['param_groups'][0][key]=value
            with self.assertRaisesRegex(RuntimeError,'AdamW'):m.audit_payload(wrong,30000,i,f,w,full=True)
        p['rng']['torch_cuda']=[]
        with self.assertRaisesRegex(RuntimeError,'CUDA RNG'):m.audit_payload(p,30000,i,f,w,full=True)

    def test_incomplete_training_never_publishes_route_or_lock(self):
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)):
            root=Path(td);(root/'state').mkdir()
            (root/'state/FORMAL_TRAINING_COMPLETE.json').write_text(json.dumps({'status':'RUNNING','complete_jobs':5,'actual_updates':150000}))
            with patch.object(m.controller,'build_plan',return_value={'jobs':jobs()}):
                with self.assertRaisesRegex(RuntimeError,'six-job'):m.run()
            self.assertFalse((root/m.ROUTING).exists());self.assertFalse((root/m.LOCK).exists())


if __name__=='__main__':unittest.main(verbosity=2)

class SmokeHistoryChecks(unittest.TestCase):
    def fixture(self, root):
        from scripts import run_training_technical as t
        folder=root/'state/technical_training/synthetic';folder.mkdir(parents=True)
        specs={};checks=[]
        for task in ('pusht','reacher'):
            base={'job':{'task':task,'job_id':task},'run_id':task+'_split8','gpu':0 if task=='pusht' else 1,'slot':0,'endpoint':8,'stop_after':None}
            specs[task]=base
            checks.append({'task':task,'status':'PASS','split_process_ids':[11,22] if task=='pusht' else [33,44],'files':[]})
        def stage(name,count,ss,pids):
            return {'status':'PASS','stage':name,'identity_sha256':'x'*64,'total_successful_updates':count,'specs':ss,
                    'processes':[{'run_id':s['run_id'],'pid':pid} for s,pid in zip(ss,pids)],'files':[]}
        partial=stage('smoke_split4',8,[dict(x,stop_after=4) for x in specs.values()],[11,33])
        # The historical endpoint really differs. Only its exact bound stage may skip it.
        terminal=root/'artifacts/technical/state.json';terminal.parent.mkdir(parents=True);terminal.write_text('{"step":8}')
        partial['files']=[{'path':'artifacts/technical/state.json','sha256':'0'*64,'bytes':10}]
        continuous=stage('smoke_continuous8',16,[dict(x,run_id=x['job']['task']+'_continuous8') for x in specs.values()],[55,66])
        resume=stage('smoke_resume4',8,list(specs.values()),[22,44])
        for x in (partial,continuous,resume):(folder/(x['stage']+'.json')).write_text(json.dumps(x))
        p=folder/'smoke_split4.json'
        smoke={'status':'PASS','optimizer_updates':32,'genuine_process_exit_resume':True,'identity_sha256':'x'*64,'tasks':checks,
               'files':[{'path':str(p.relative_to(root)),'sha256':m.common.sha256(p),'bytes':p.stat().st_size}]}
        (root/'state/GPU_SMOKE.json').write_text(json.dumps(smoke))
        return t,checks,p

    def test_only_bound_history_exemption_and_exact_final_recheck(self):
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)):
            root=Path(td);t,checks,p=self.fixture(root)
            with patch.object(t,'ROOT',root),patch.object(t,'compare_smoke',side_effect=checks) as compare:
                inv=m.Inventory();out=m.audit_smoke_history(inv)
                self.assertEqual(out['status'],'PASS');self.assertEqual(compare.call_count,2)
                self.assertIn(str(p.relative_to(root)),inv.files)
                self.assertNotIn('artifacts/technical/state.json',inv.files)
            # No wildcard skip of arbitrary old receipts.
            other=root/'state/technical_training/synthetic/other.json';other.write_text(p.read_text())
            with self.assertRaisesRegex(RuntimeError,'differ'):inv.json(str(other.relative_to(root)),expand=True)

    def test_history_pid_chain_and_gate_bytes_cannot_change(self):
        with tempfile.TemporaryDirectory() as td,patch.object(m,'ROOT',Path(td)):
            root=Path(td);t,checks,p=self.fixture(root)
            wrong=copy.deepcopy(checks);wrong[0]['split_process_ids']=[11,99]
            with patch.object(t,'ROOT',root),patch.object(t,'compare_smoke',side_effect=wrong):
                with self.assertRaisesRegex(RuntimeError,'differs'):m.audit_smoke_history(m.Inventory())
            p.write_text(p.read_text()+' ')
            with patch.object(t,'ROOT',root):
                with self.assertRaisesRegex(RuntimeError,'differ'):m.audit_smoke_history(m.Inventory())
