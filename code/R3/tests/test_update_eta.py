"""Pure metadata/arithmetic checks; no GPU, model, network or true episode reads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('eta',Path(__file__).resolve().parents[1]/'scripts/update_eta.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def jobs(n=6,remaining=30000):
    return [{'job_id':str(i),'task':'pusht' if i<3 else 'reacher','remaining_updates':remaining} for i in range(n)]


class ETATests(unittest.TestCase):
    def test_quantile_sparse_four_times(self):
        self.assertEqual(m.quantile([10,20,30,40],.5),25)
        self.assertEqual(m.quantile([10,20,30,40],.9),37)

    def test_fifo_not_worker_hour_sum(self):
        self.assertEqual(m.fifo_wall(jobs(),2,1.)['wall_seconds'],90000)
        self.assertEqual(m.fifo_wall(jobs(),4,1.)['wall_seconds'],60000)
        with self.assertRaises(ValueError):m.fifo_wall(jobs(),6,1.)

    def test_active_slot_and_remainder_kept(self):
        x=jobs(3,10);x[0]['remaining_updates']=2
        out=m.fifo_wall(x,2,1.,[{'job_id':'0','gpu':1,'slot':0}])
        self.assertEqual(out['wall_seconds'],12)
        self.assertEqual(out['assignment'][2]['slot_key'],(1,0))

    def test_cem_actual_sha_assignment_four_arms_and_completed(self):
        cases={'pusht':[{'case_id':'p1'},{'case_id':'p2'}],'reacher':[{'case_id':'r1'}]}
        full=m.cem_wall(cases,{'pusht':10,'reacher':20})
        self.assertEqual(sum(full['per_gpu_seconds']),160)
        less=m.cem_wall(cases,{'pusht':10,'reacher':20},[('pusht','p1','H0')])
        self.assertEqual(sum(less['per_gpu_seconds']),150)
        self.assertEqual(less['wall_seconds'],max(less['per_gpu_seconds']))

    def test_snapshot_partial_tail_and_holes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j.jsonl';p.write_text('{"step":1}\n{"step":')
            s=m.Snapshot(td);self.assertEqual(len(s.journal('j.jsonl',True)),1);self.assertEqual(len(s.warnings),1)
            p.write_text('{"step":2}\n')
            with self.assertRaises(RuntimeError):m.Snapshot(td).journal('j.jsonl',True)

    def test_pending_does_not_invent_eta_or_failed_cost_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'manifests').mkdir();(root/'state/reset/attempt1').mkdir(parents=True)
            (root/'manifests/JOBS.json').write_text(json.dumps({'jobs':[dict(x,updates=30000) for x in jobs()]}))
            (root/'state/reset/attempt1/RESET_VALIDATION_RECEIPT.json').write_text(json.dumps({'status':'BLOCKED_RESET_FALLBACK','wall_seconds':2.5,'counts':{'reset_attempts':16,'resets':0,'raw_steps':0}}))
            out=m.run(root)
            self.assertIsNone(out['total_delivery_eta_seconds']);self.assertIsNone(out['measured_phase_projection_seconds_range'])
            record=out['actual_costs']['records'][0]
            self.assertEqual(record['actual_recorded_fields']['wall_seconds'],2.5)
            self.assertEqual(record['actual_recorded_fields']['counts']['reset_attempts'],16)
            self.assertIn('未知', (root/m.REPORT).read_text())

    def test_partial_technical_original_rows_reported_without_eta(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'state').mkdir()
            rel='artifacts/planning/TECH/pusht/c/H0/attempt_0/result.json'
            p=root/rel;p.parent.mkdir(parents=True);p.write_text(json.dumps({'status':'COMPLETE','phase':'TECH','task':'pusht','case_id':'c','arm':'H0','trajectory_wall_seconds':2.5}))
            h=m.hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'state/TECHNICAL_LEDGER.json').write_text(json.dumps({'trajectories':{'one':{'status':'COMPLETE','result_path':rel,'result_sha256':h}}}))
            out=m.planning(m.Snapshot(root))
            self.assertEqual(out['tasks']['pusht']['median_seconds'],2.5)
            self.assertEqual(out['tasks']['pusht']['timing_trajectories'],1)
            self.assertNotIn('remaining_critical_path_seconds_median_to_p90',out)
            self.assertNotIn('reacher',out['tasks'])

    def test_final_report_seal_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'manifests').mkdir()
            (root/'manifests/ANALYSIS_OUTPUTS.json').write_text(json.dumps({'status':'COMPLETE','files':{m.REPORT:{'sha256':'x'}}}))
            with self.assertRaisesRegex(RuntimeError,'already seals'):m.run(root)
            self.assertFalse((root/'state/ETA_PROFILE.json').exists())

    def test_completed_profile_parallel_critical_path(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'manifests').mkdir();(root/'state').mkdir()
            (root/'manifests/JOBS.json').write_text(json.dumps({'jobs':[dict(x,updates=30000) for x in jobs()]}))
            phase={'status':'PASS','stage':'profile_B','specs':[{}]*4,'launch_wall_seconds':10,'total_successful_updates':192,'end_to_end_updates_per_second':19.2,'steady_update_span':{'status':'BOUNDED_OBSERVED_INTERVAL','aggregate_updates_per_second_lower':20,'aggregate_updates_per_second_upper':40}}
            (root/'state/GPU_PROFILE.json').write_text(json.dumps({'status':'PROFILE_COMPLETE','selected_workers':4,'B':phase}))
            (root/'state/TRAINING_TECHNICAL_GATE.json').write_text(json.dumps({'status':'TRAINING_TECHNICAL_GATES_PASS','selected_workers':4,'microbatch':128}))
            out=m.training(m.Snapshot(root))
            self.assertEqual(out['remaining_critical_path_seconds_range'],[6000,12000])
            self.assertIsNone(out['formal_full_cache_startup_seconds'])

if __name__=='__main__':unittest.main()
