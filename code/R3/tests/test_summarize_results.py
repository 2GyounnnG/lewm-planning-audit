"""Synthetic post-processing/plot metadata checks; no actual experiment input."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from r3 import statistics

spec=importlib.util.spec_from_file_location('r3_test_summary',Path(__file__).resolve().parents[1]/'scripts/summarize_results.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def row(task='pusht',arm='H0',seed=None,step=0,h=5,case='a',risk=1.,finite=True):
    return {'task':task,'arm':arm,'refit_seed':seed,'checkpoint_step':step,'horizon_macro':h,'case_id':case,
            'latent_raw_MSE':risk,'latent_TRAIN_variance_normalized_MSE':float(risk)/2,
            'prediction_finite':finite,'metric_finite':finite,'prediction_norm_l2':1.,'target_norm_l2':1.,
            'prediction_norm2_over_D':1.,'target_norm2_over_D':1.,'prediction_displacement_from_initial_MSE':.5}


class SummaryChecks(unittest.TestCase):
    def test_equal_case_mean_and_tasks_never_pooled(self):
        out=s.aggregate_open_rows([row(case='a',risk=1),row(case='b',risk=9),row('reacher',case='a',risk=100)])
        self.assertEqual(len(out),2)
        self.assertEqual(next(r for r in out if r['task']=='pusht')['mean_latent_raw_MSE'],5)
        self.assertEqual(next(r for r in out if r['task']=='reacher')['mean_latent_raw_MSE'],100)

    def test_nonfinite_case_preserved_in_extended_mean(self):
        out=s.aggregate_open_rows([row(case='a',risk=1),row(case='b',risk=float('inf'),finite=False)])[0]
        self.assertEqual(out['cases'],2);self.assertEqual(out['mean_latent_raw_MSE'],'Infinity')
        self.assertEqual(out['nonfinite_prediction_cases'],1);json.dumps(out,allow_nan=False)

    def test_duplicate_arm_case_not_silently_reweighted(self):
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):s.aggregate_open_rows([row(),row()])

    def test_updates_contiguous_and_token_budget(self):
        rows=[{'step':i,'technical':False,'sampled_windows':128,'predicted_tokens':384,'raw_action_exposures':1920,
               'loss_raw_MSE':1.,'lr':1e-5,'preclip_gradient_norm':1.,'compute_seconds':.1} for i in (1,2,3)]
        s.validate_updates(rows,expected=3)
        with self.assertRaises(RuntimeError):s.validate_updates(rows)
        rows[1]['predicted_tokens']=128
        with self.assertRaisesRegex(RuntimeError,'exposure'):s.validate_updates(rows,expected=3)

    def test_worker_sampling_summary_real_token_field(self):
        totals={'sampled_windows':128,'predicted_tokens':384,'raw_action_exposures':1920}
        s.validate_sampling_totals(totals,{'sampled_windows':128,'predicted_target_tokens':384,'raw_action_exposures':1920})
        with self.assertRaisesRegex(RuntimeError,'sampler totals'):
            s.validate_sampling_totals(totals,{'sampled_windows':128,'predicted_target_tokens':128,'raw_action_exposures':1920})

    def test_csv_gzip_keeps_raw_tagged_nonfinite_and_full_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'raw.csv.gz';s.write_csv(path,[{'i':1,'risk':float('inf'),'values':[1,2]},{'i':2,'risk':3.,'values':[3,4]}])
            with gzip.open(path,'rt') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2);self.assertEqual(rows[0]['risk'],'Infinity');self.assertEqual(rows[1]['values'],'[3, 4]')

    def test_input_receipt_hash_tamper_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'value.json').write_text('{"x":1}')
            (root/'receipt.json').write_text(json.dumps({'files':{'value.json':s.record(root/'value.json')}}))
            with patch.object(s,'local_path',lambda rel:root/rel):
                reader=s.Inputs();reader.receipt('receipt.json');(root/'value.json').write_text('{"x":2}')
                with self.assertRaisesRegex(RuntimeError,'changed'):reader.unchanged()

    def test_planning_fallback_overlay_compared_without_changing_raw_case_ids(self):
        raw={'case_id':'case','episode_id':'ep','family_id':'family','reset_seed':None}
        effective={**raw,'reset_seed':0,'reset_seed_provenance':'VALIDATED_TEST_ONLY'}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); results={}; entries={}; files={}
            for arm in statistics.ARMS:
                rel=arm+'.json';value={'task':'pusht','case_id':'case','episode_id':'ep','family_id':'family',
                    'arm':arm,'phase':'FORMAL','identity':{'case':effective}}
                (root/rel).write_text(json.dumps(value));results[rel]=value;files[rel]=s.record(root/rel)
                entries[arm]={'status':'COMPLETE','result_path':rel,'result_sha256':files[rel]['sha256']}
            payload={**results,'state/FORMAL_TRAJECTORY_LEDGER.json':{'trajectories':entries},
                     'manifests/pusht_data_roles.json':{'cases':{'EVAL':[raw]}}}
            class Reader:
                def __init__(self):self.bound=[]
                def receipt(self,rel):return {'status':'COMPLETE','files':files,'complete_trajectories':4}
                def json(self,rel):return payload[rel]
                def bind(self,rel):self.bound.append(rel)
            reader=Reader()
            with patch.object(s.common,'TASKS',('pusht',)),patch.object(s,'local_path',lambda rel:root/rel), \
                 patch.object(s,'evidence_paths',return_value=['fallback_evidence.json']),patch.object(s,'effective_case',return_value=effective), \
                 patch.object(s.statistics,'normalize_rows'):
                by,attempts,completion=s.planning_data(reader)
            self.assertEqual(len(by['pusht']),4);self.assertIsNone(raw['reset_seed'])
            self.assertEqual(reader.bound,['fallback_evidence.json'])

    def test_reset_failures_and_diagnostic_steps_are_separate_from_CEM(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'state').mkdir();(root/'artifacts').mkdir()
            (root/'state/TECHNICAL_LEDGER.json').write_text(json.dumps({'runs':{},'trajectories':{}}))
            for i,status,resets in [(1,'BLOCKED_RESET_FALLBACK',0),(2,'PASS_TECH_RESET_FALLBACK',16)]:
                folder=root/f'state/reset/attempt{i}';folder.mkdir(parents=True)
                (folder/'RESET_VALIDATION_RECEIPT.json').write_text(json.dumps({'status':status,'wall_seconds':float(i),
                    'counts':{'reset_attempts':16,'resets':resets,'raw_steps':resets,'optimizer_updates':0,'CEM_calls':0}}))
            with patch.object(s,'ROOT',root),patch.object(s,'local_path',lambda rel:root/rel):
                result=s.compute_data(s.Inputs(),{'summary':[],'failures':[]},[],{'complete_trajectories':0})
            rows=result['zero_update_check_attempts'];self.assertEqual(len(rows),2)
            self.assertEqual(sum(r['counts']['reset_attempts'] for r in rows),32)
            self.assertEqual(sum(r['counts']['resets'] for r in rows),16)
            self.assertEqual(result['technical_complete_trajectories'],0)
            self.assertEqual(result['technical_optimizer_updates_exact'],0)

    def test_synthetic_publication_plots_generate_png_and_svg(self):
        updates=[];monitors=[];opened=[];stats={}
        for task in ('pusht','reacher'):
            for seed in (103201,103202,103203):
                for step in (1,2,3):updates.append({'task':task,'refit_seed':seed,'step':step,'loss_raw_MSE':1./step})
                for step in (0,1000,2000):monitors.append({'task':task,'refit_seed':seed,'step':step,
                    'teacher_forced_one_step_raw_MSE':1.,'free_H5_mean_raw_MSE':2.,'free_H5_terminal_raw_MSE':3.})
            for h in (1,2,5):
                opened.append(row(task,h=h))
                for seed in (103201,103202,103203):
                    for step in (3000,10000,30000):opened.append(row(task,f'REFIT_{seed}_{step}',seed,step,h,risk=1.))
            raw=[{'task':task,'case_id':str(i),'family_id':str(i//4),'episode_id':str(i),'arm':arm,'success':int((i+j)%3==0)}
                 for i in range(20) for j,arm in enumerate(statistics.ARMS)]
            stats[task]=statistics.analyze_task(task,raw)
        with tempfile.TemporaryDirectory() as td,patch.object(s,'ROOT',Path(td)):
            paths=s.figures({'updates':updates,'monitors':monitors},{'summary':s.aggregate_open_rows(opened)},stats)
            self.assertEqual(len(paths),16)
            for rel in paths:
                data=(Path(td)/rel).read_bytes();self.assertGreater(len(data),100)
                if rel.endswith('.png'):self.assertTrue(data.startswith(b'\x89PNG'))
                else:self.assertIn(b'<svg',data)


if __name__=='__main__':unittest.main(verbosity=2)
