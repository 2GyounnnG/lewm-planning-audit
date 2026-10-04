"""Metadata-only controller checks: no actual model/environment/GPU execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from r3.roles import build_roles

PATH=Path(__file__).resolve().parents[1]/'scripts/run_planning.py'
spec=importlib.util.spec_from_file_location('r3_test_planning_controller',PATH)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def roles(task='pusht'):
    records=[{'episode_id':f'e{i}','source_episode_idx':i,'source_asset_sha256':'a'*64,
              'episode_sha256':f'{i:064x}','length':80,'family_id':f'f{i//20}',
              'planning_starts':[0,13],'open_loop_starts':[10,13],'source_seed':42} for i in range(200)]
    return build_roles(task,records,history_size=3,frameskip=5)


def fixture(folder,action_offset=0.,method_failure=False):
    folder.mkdir(parents=True)
    result={'status':'COMPLETE','evaluation_complete':True,'identity_sha256':'a'*64,
            'task':'pusht','case_id':'synthetic','success':0,'any_step_success':0,'terminal_success':0,
            'executed_raw_steps':2,'replan_calls':1,'final_goal_error':1.,'failure_category':'METHOD' if method_failure else None}
    (folder/'result.json').write_text(json.dumps(result));(folder/'STARTED.json').write_text('{}')
    np.savez(folder/'trajectory.npz',raw_actions=np.zeros((2,2))+action_offset,step_success=np.zeros(2,dtype=bool),
             step_truncated=np.zeros(2,dtype=bool),physical_state=np.zeros((2,5)),goal_error=np.ones(2),
             goal_state=np.ones(5),rewards=np.zeros(2),environment_step_seconds=np.array([.01,.02]))
    (folder/'TRAJECTORY_SHA256.json').write_text(json.dumps({'status':'COMPLETE','identity_sha256':'a'*64,
        'files':{name:p.record(folder/name) for name in ('STARTED.json','result.json','trajectory.npz')}}))


class PlanningControllerTests(unittest.TestCase):
    def test_fixed_first_two_tech_cases_per_task_total_eight(self):
        total=0
        for task in ('pusht','reacher'):
            r=roles(task);selected=p.select_cases(r,task,'TECH')
            self.assertEqual(selected,r['cases']['TECH'][:2]);self.assertEqual(len(selected),2)
            total+=len(selected)*2
        self.assertEqual(total,8)

    def test_formal_is_all_eval_not_tech_and_not_success_selected(self):
        r=roles();selected=p.select_cases(r,'pusht','FORMAL')
        self.assertEqual(selected,r['cases']['EVAL']);self.assertEqual(len(selected),36)
        self.assertFalse({c['episode_id'] for c in selected}&{c['episode_id'] for c in r['cases']['TECH']})
        r['cases']['EVAL'][0]['role']='TECH'
        with self.assertRaises(ValueError):p.select_cases(r,'pusht','FORMAL')

    def test_sha_assignment_identical_for_all_arms_and_schedule_order(self):
        ids=['a','b','synthetic-case'];expected={s:int(hashlib.sha256(s.encode()).hexdigest(),16)%2 for s in ids}
        self.assertEqual({s:p.gpu_for_case(s) for s in reversed(ids)},expected)
        self.assertEqual(p.public_arm({'arm':'REFIT_103201_30000','seed':103201,'step':30000}),'REFIT_103201')
        self.assertEqual(p.public_arm({'arm':'H0_CLONE','seed':None,'step':0}),'H0_CLONE')

    def test_no_guess_missing_reset_seed_and_bad_path(self):
        r=roles();r['cases']['TECH'][0]['reset_seed']=None
        with self.assertRaisesRegex(ValueError,'reset_seed'):p.select_cases(r,'pusht','TECH')
        for s in ('../x','a/b','', '..'):
            with self.assertRaises(ValueError):p.safe_component(s)

    def test_clone_exact_arrays_independent_of_timing(self):
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'a',Path(td)/'b';fixture(a);fixture(b)
            result=p.compare_clone(a,b);self.assertEqual(result['status'],'PASS')
            self.assertTrue(result['timing_fields_excluded_from_equality'])

    def test_clone_wrong_action_and_method_failure_block(self):
        for action_offset,failure in ((.1,False),(0,True)):
            with tempfile.TemporaryDirectory() as td:
                a,b=Path(td)/'a',Path(td)/'b';fixture(a);fixture(b,action_offset,failure)
                with self.assertRaises(RuntimeError):p.compare_clone(a,b)

    def test_complete_seal_detects_tamper_and_unfinished_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'case';folder.mkdir();(folder/'STARTED.json').write_text('{}')
            self.assertIsNone(p.complete_trajectory(folder))
            other=Path(td)/'done';fixture(other);self.assertIsNotNone(p.complete_trajectory(other))
            (other/'result.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'differs'):p.complete_trajectory(other)

    def test_expected_identity_is_checked_even_if_artifact_hashes_are_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'case';fixture(folder)
            with self.assertRaises((KeyError,RuntimeError)):p.complete_trajectory(folder,{'a':'different'})


if __name__=='__main__':unittest.main(verbosity=2)
