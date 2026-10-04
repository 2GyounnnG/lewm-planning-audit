"""Incremental transport of completed R3 publications to the guarded external disk."""
from pathlib import Path
import argparse, fcntl, json, os, subprocess, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recover_r3 as recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-verified', required=True, help='Exact preceding publication SHA')
    args = parser.parse_args()
    if len(args.after_verified) != 64 or any(c not in '0123456789abcdef' for c in args.after_verified):
        raise ValueError('Exact preceding publication SHA required')
    recovery.guard()
    lock_path = recovery.target('recovery_receipts/watcher.lock', True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = 'recovery_receipts/WATCHER_STATUS.json'
        def status(phase, **extra):
            recovery.write(status_path, (json.dumps({'phase': phase, 'pid': os.getpid(), 'updated_at_unix': time.time(), **extra}, indent=2)+'\n').encode())
        def stopping():
            return recovery.target('recovery_receipts/WATCHER_STOP_AFTER_CURRENT.json').exists()
        status('WAITING_FOR_PRECEDING_VERIFIED_PUBLICATION', preceding_sha256=args.after_verified)
        prior = recovery.target(f'recovery_receipts/{args.after_verified}/VERIFIED.json')
        while not prior.exists():
            recovery.guard()
            if stopping():
                status('STOPPED_BEFORE_FIRST_PUBLICATION', completed_publications=0)
                return
            time.sleep(30)
        if json.loads(prior.read_text())['manifest_sha256'] != args.after_verified:
            raise RuntimeError('Preceding receipt identity differs')
        done = set()
        while True:
            recovery.guard()
            code = """from pathlib import Path
import json,os
r=Path('/workspace/r3_official_lewm_predictor_refit')
assert os.uname().nodename=='6494ba5e1b1a'
paths=['manifests/RECOVERY_REACHER_INPUTS_V1.json']
paths += [str(p.relative_to(r)) for p in sorted((r/'state/recovery_queue').glob('*.json'))]
print(json.dumps([p for p in paths if (r/p).is_file()]))
"""
            for manifest in recovery.remote_python(code):
                if stopping():
                    status('STOPPED_AFTER_COMPLETED_PUBLICATIONS', completed_publications=len(done))
                    return
                if manifest in done:
                    continue
                status('RECOVERING_COMPLETED_PUBLICATION', manifest=manifest, completed_publications=len(done))
                # The shared recovery lock prevents overlap with manual/final recovery.
                command = [sys.executable, '-B', str(Path(recovery.__file__)), '--manifest', manifest]
                result = subprocess.run(command, text=True, capture_output=True)
                if result.returncode:
                    status('FAILED_RETAINED_EXTERNAL_PARTIAL', manifest=manifest, returncode=result.returncode,
                           stderr_tail=result.stderr[-3000:], stdout_tail=result.stdout[-1000:])
                    raise RuntimeError('Published-unit recovery failed; raw evidence retained')
                done.add(manifest)
                status('VERIFIED_PUBLICATION', manifest=manifest, completed_publications=len(done), receipt_summary=result.stdout[-1500:])
            if stopping():
                status('STOPPED_AFTER_COMPLETED_PUBLICATIONS', completed_publications=len(done))
                return
            status('WAITING_FOR_NEW_COMPLETED_PUBLICATIONS', completed_publications=len(done))
            time.sleep(60)


if __name__ == '__main__':
    main()
