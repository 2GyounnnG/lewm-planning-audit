#!/usr/bin/env python3
"""Package-only checks. No SSH, data access, optimizer or scientific evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent

def read_csv(name):
    with (ROOT / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def case_seed(task, case_id, replan=0):
    s=f'R3_CEM_CASE_20261002/{task}/{case_id}/{replan}'
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], 'big')

class Contracts(unittest.TestCase):
    def test_train_budget(self):
        jobs=read_csv('05_TRAIN_JOBS.csv')
        self.assertEqual(len(jobs),6)
        self.assertEqual(sum(int(j['updates']) for j in jobs),180000)
        self.assertEqual(len({j['job_id'] for j in jobs}),6)
    def test_official_origins(self):
        jobs=read_csv('05_TRAIN_JOBS.csv')
        for task in ['pusht','reacher']:
            part=[j for j in jobs if j['task']==task]
            self.assertEqual(len(part),3)
            self.assertEqual({j['official_model_repo'] for j in part},{f'quentinll/lewm-{task}'})
    def test_eval_budget(self):
        arms=read_csv('06_EVAL_ARMS.csv')
        self.assertEqual(len(arms),8)
        self.assertEqual(sum(int(a['max_cases']) for a in arms),800)
    def test_h0_once_per_task(self):
        arms=read_csv('06_EVAL_ARMS.csv')
        for task in ['pusht','reacher']:
            self.assertEqual(sum(a['task']==task and a['arm_id'].endswith('_H0') for a in arms),1)
    def test_seed_separation_and_pairing(self):
        self.assertEqual(case_seed('pusht','e12',2),case_seed('pusht','e12',2))
        self.assertNotEqual(case_seed('pusht','e12',2),case_seed('pusht','e12',3))
        self.assertNotEqual(case_seed('pusht','e12',2),case_seed('reacher','e12',2))
    def test_cem_budget_same_across_arms(self):
        arms=read_csv('06_EVAL_ARMS.csv')
        keys=['goal_offset_raw','eval_budget_raw','cem_num_samples','cem_iterations','cem_topk']
        for task in ['pusht','reacher']:
            part=[a for a in arms if a['task']==task]
            self.assertEqual(len({tuple(a[k] for k in keys) for a in part}),1)
        self.assertEqual(5*5,25)
    def test_paired_mean_not_ensemble_or_pseudoreplication(self):
        h0=[0.,1.,0.,1.]
        refits=[[1.,1.,0.,1.],[0.,1.,1.,1.],[1.,1.,1.,0.]]
        per_case=[sum(r[i] for r in refits)/3-h0[i] for i in range(4)]
        per_seed=[sum(r[i]-h0[i] for i in range(4))/4 for r in refits]
        self.assertAlmostEqual(sum(per_case)/4,sum(per_seed)/3)
        self.assertEqual(len(per_case),4)
    def test_required_files_and_planned_status(self):
        for name in ['01_CODEX_PROMPT.md','02_EXPERIMENT_SPEC.md','03_PRIMARY_SOURCES.md','04_START_MESSAGE.txt']:
            self.assertTrue((ROOT/name).is_file())
            self.assertGreater((ROOT/name).stat().st_size,200)
        self.assertTrue(all(j['status']=='PLANNED_NOT_EXECUTED' for j in read_csv('05_TRAIN_JOBS.csv')))
        self.assertTrue(all(j['status']=='PLANNED_NOT_EXECUTED' for j in read_csv('06_EVAL_ARMS.csv')))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    res=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'scope':'PACKAGE_CONTRACTS_ONLY_NOT_REMOTE_OR_SCIENTIFIC_TESTS','tests':res.testsRun,'failures':len(res.failures),'errors':len(res.errors),'passed':res.wasSuccessful()}
    (ROOT/'08_PACKAGE_CHECKS_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if res.wasSuccessful() else 1)
