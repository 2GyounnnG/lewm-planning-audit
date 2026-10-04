"""Synthetic metadata only: no GPU query, models, real cache, or optimizer."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_formal_training.py'
spec = importlib.util.spec_from_file_location('test_formal_controller_module', SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def jobs():
    return [{'job_id': f'R3_{task}_PRED_REFIT_s{seed}', 'task': task, 'refit_seed': seed,
             'updates': 30000, 'effective_batch': 128, 'microbatch': 128}
            for task in ('pusht', 'reacher') for seed in (103201, 103202, 103203)]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def plan_fixture(root):
    for name in m.REQUIRED_HASHES: write(root/name, {})
    write(root/'manifests/JOBS.json', {'jobs': jobs()})
    write(root/m.GATE, {'status': 'TRAINING_TECHNICAL_GATES_PASS', 'selected_workers': 4, 'workers_per_gpu': 2, 'microbatch': 128})
    write(root/'state/GPU_SMOKE.json', {'status': 'PASS', 'genuine_process_exit_resume': True, 'optimizer_updates': 32})
    write(root/'state/PLANNING_TECH_GATE.json', {'status': 'PASS', 'complete_trajectories': 8, 'clone_exact': True})
    names = set(m.REQUIRED_HASHES)
    for task in ('pusht', 'reacher'):
        name = f'state/synthetic/{task}/REAL_DATA_MODEL_CHECK.json'; names.add(name)
        write(root/name, {'task': task, 'status': 'PASS'})
        name = f'state/synthetic/{task}/COST_WRAPPER_CHECK.json'; names.add(name)
        write(root/name, {'task': task, 'status': 'PASS'})
    hashes = {name: m.common.sha256(root/name) for name in names}
    write(root/m.AUTH, {'status': 'TECHNICAL_GATES_PASS', 'job_ids': [j['job_id'] for j in jobs()], 'source_hashes': hashes})


def publish_fixture(root, job, steps=30000):
    folder = root/'artifacts/train'/job['job_id']; folder.mkdir(parents=True)
    identity = {'job': job, 'technical': False, 'endpoint': 30000, 'microbatch': 128,
                'inputs': {'fixture': 'identity'}, 'source': {}, 'base_identity': {'official': 'synthetic'}}
    write(folder/'RUN_IDENTITY.json', identity)
    for name in ('resume.pt', 'checkpoint_3000.pt', 'checkpoint_10000.pt', 'checkpoint_30000.pt', 'sampling_counts.npz'):
        (folder/name).write_bytes(b'synthetic metadata fixture, never deserialize')
    for name in ('FROZEN_INITIAL.json', 'PARAMETER_WHITELIST.json'): write(folder/name, {})
    pointers = {}
    for name, target in [('last.json', 'checkpoint_30000.pt'), ('resume.json', 'resume.pt')]:
        pointers[name] = {'path': target, 'step': 30000, 'sha256': m.common.sha256(folder/target)}; write(folder/name, pointers[name])
    result = {'status': 'REFIT_TRAINING_COMPLETE_UNSCORED', 'actual_updates': 30000, 'job_id': job['job_id'],
              'task': job['task'], 'technical': False, 'identity_sha256': m.digest(identity), 'frozen_before': 'same', 'frozen_after': 'same',
              'base_identity': identity['base_identity'], 'microbatch': 128, 'last_checkpoint': pointers['last.json'], 'resume_checkpoint': pointers['resume.json']}
    write(folder/'result.json', result)
    (folder/'updates.jsonl').write_text(''.join(json.dumps({'step': i, 'technical': False, 'sampled_windows': 128})+'\n' for i in range(1, steps+1)))
    (folder/'monitor.jsonl').write_text(''.join(json.dumps({'step': i})+'\n' for i in range(0, 30001, 1000)))
    seal_fixture(root, folder, job)
    return folder


def seal_fixture(root, folder, job):
    receipt = {'job_id': job['job_id'], 'files': {str(p.relative_to(root)): m.record(p) for p in folder.rglob('*') if p.is_file() and p.name != 'JOB_SHA256.json'}}
    write(folder/'JOB_SHA256.json', receipt); write(root/'state/recovery_queue'/f"{job['job_id']}.json", receipt)


class FormalControllerTests(unittest.TestCase):
    def root(self, root):
        return patch.multiple(m, ROOT=root)

    def test_fixed_fifo_slots_batch_and_command(self):
        self.assertEqual(m.slots_for(2), [(0, 0), (1, 0)])
        self.assertEqual(m.slots_for(4), [(0, 0), (1, 0), (0, 1), (1, 1)])
        for x in (0, 1, 3, 5, True, 2., '2'):
            with self.assertRaises(ValueError): m.slots_for(x)
        command = m.worker_command(jobs()[0], 1)
        self.assertEqual(command[-4:], ['--microbatch', '128', '--slot', '1'])
        self.assertNotIn('--technical-steps', command); self.assertNotIn('--stop-after', command)
        env = m.worker_environment(1)
        self.assertEqual([env[x] for x in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG')], ['1', '8', '4', ':4096:8'])

    def test_six_jobs_no_drop_seed_or_budget_change(self):
        self.assertEqual(m.validate_jobs(jobs()), jobs())
        for key, value in [('refit_seed', 999), ('updates', 30001), ('microbatch', 64), ('effective_batch', 64)]:
            altered = jobs(); altered[0][key] = value
            with self.assertRaises(ValueError): m.validate_jobs(altered)
        with self.assertRaises(ValueError): m.validate_jobs(jobs()[:-1])

    def test_plan_requires_actual_all_gates_and_self_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); plan_fixture(root)
            with self.root(root), patch.object(m.common, 'ROOT', root):
                plan = m.build_plan(); self.assertEqual(plan['job_order'], [j['job_id'] for j in jobs()])
                self.assertEqual(plan['selected_workers'], 4); m.assert_plan(plan)
                auth = m.read(m.AUTH); del auth['source_hashes']['scripts/run_formal_training.py']; write(root/m.AUTH, auth)
                with self.assertRaisesRegex(RuntimeError, 'omits'): m.build_plan()

    def test_failed_gate_even_if_rehashed_never_allows_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); plan_fixture(root)
            with self.root(root), patch.object(m.common, 'ROOT', root):
                path = root/'state/PLANNING_TECH_GATE.json'; write(path, {'status': 'PASS', 'complete_trajectories': 7, 'clone_exact': True})
                auth = m.read(m.AUTH); auth['source_hashes'][str(path.relative_to(root))] = m.common.sha256(path); write(root/m.AUTH, auth)
                with self.assertRaisesRegex(RuntimeError, 'Eight'): m.build_plan()

    def test_missing_actual_cost_wrapper_gate_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); plan_fixture(root)
            with self.root(root), patch.object(m.common, 'ROOT', root):
                auth = m.read(m.AUTH)
                del auth['source_hashes']['state/synthetic/reacher/COST_WRAPPER_CHECK.json']
                write(root/m.AUTH, auth)
                with self.assertRaisesRegex(RuntimeError, 'cost-wrapper'): m.build_plan()

    def test_unstarted_and_partial_are_distinct_no_automatic_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.root(root):
                self.assertIsNone(m.verify_complete(jobs()[0]))
                write(root/'artifacts/train'/jobs()[0]['job_id']/'RUN_IDENTITY.json', {})
                with self.assertRaisesRegex(RuntimeError, 'explicit review'): m.verify_complete(jobs()[0])
                with self.assertRaises(ValueError): m.local_path('../escape')

    def test_all_published_bytes_and_full_journal_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); folder = publish_fixture(root, jobs()[0])
            with self.root(root), patch.object(m.common, 'identity_hashes', return_value={'fixture': 'identity'}):
                complete = m.verify_complete(jobs()[0]); self.assertEqual(complete['actual_updates'], 30000)
                (folder/'checkpoint_30000.pt').write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'bytes differ'): m.verify_complete(jobs()[0])

    def test_resealed_missing_update_and_queue_drift_block(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); folder = publish_fixture(root, jobs()[0], steps=29999)
            with self.root(root), patch.object(m.common, 'identity_hashes', return_value={'fixture': 'identity'}):
                with self.assertRaisesRegex(RuntimeError, 'exact steps'): m.verify_complete(jobs()[0])
                write(root/'state/recovery_queue'/f"{jobs()[0]['job_id']}.json", {})
                with self.assertRaisesRegex(RuntimeError, 'publication/queue'): m.verify_complete(jobs()[0])

    def loop(self, root, fail_first=False, workers=2):
        launched = []; processes = {}; completed = {}
        class Process:
            def __init__(self, job_id): self.pid = 100+len(launched); self.job_id = job_id; self.polls = 0
            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else (1 if fail_first and self.job_id == jobs()[0]['job_id'] else 0)
        def spawn(command, **kwargs):
            jid = command[command.index('--job')+1]; launched.append(jid); p = Process(jid); processes[jid] = p; return p
        def complete(job):
            proc = processes.get(job['job_id'])
            return {'job_id': job['job_id'], 'actual_updates': 30000} if proc and proc.polls >= 2 else None
        plan = {'jobs': jobs(), 'slots': [list(s) for s in m.slots_for(workers)]}
        with self.root(root), patch.object(m, 'assert_plan'), patch.object(m, 'verify_complete', side_effect=complete), patch.object(m.common, 'require_authorization'), patch.object(m.common, 'ensure_space'):
            result = m.control_loop(plan, root, (), completed, popen=spawn, sample=lambda: {'synthetic': True}, sleep=lambda _: None)
        return launched, result

    def test_failure_stops_new_launches_and_allows_other_worker_completion(self):
        with tempfile.TemporaryDirectory() as td:
            launched, result = self.loop(Path(td), fail_first=True)
            self.assertEqual(launched, [j['job_id'] for j in jobs()[:2]])
            self.assertEqual(list(result['completed']), [jobs()[1]['job_id']]); self.assertTrue(result['errors'])
            self.assertEqual(len(result['pending']), 4)

    def test_successful_four_slot_fifo_exactly_six_no_retries(self):
        with tempfile.TemporaryDirectory() as td:
            launched, result = self.loop(Path(td), workers=4)
            self.assertEqual(launched, [j['job_id'] for j in jobs()]); self.assertEqual(len(result['completed']), 6)
            self.assertFalse(result['errors']); self.assertEqual(result['pending'], [])


if __name__ == '__main__': unittest.main(verbosity=2)
