#!/usr/bin/env bash
set -euo pipefail
cd '/Users/richwang/Documents/ChatGPT/热'
task_recovery='r5_execution/c2/recovery'
task_python='/Users/richwang/Documents/ChatGPT/热/jepa_low_label_bridge_gpu_v2/g1/.venv/bin/python'
task_ssh='ssh -S /private/tmp/r5_control -p 46720 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/Users/richwang/Documents/ChatGPT/热/r4_v23_execution/ops/known_hosts'
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
while [ ! -f "$task_recovery/BULK_SLOT_RELEASED_BY_H3X.json" ]; do sleep 20; done
"$task_python" -B -c 'import json,pathlib; r=json.loads(pathlib.Path("r5_execution/c2/recovery/BULK_SLOT_RELEASED_BY_H3X.json").read_text()); assert r["remaining_bulk_downloads"]==0 and r["C2_may_start"] is True'
date -u '+C2_DOWNLOAD_STARTED %Y-%m-%dT%H:%M:%SZ'
task_download_ok=false
for task_attempt in 1 2 3; do
  if rsync -az --partial -e "$task_ssh" root@211.72.13.202:/workspace/r5/C2/recovery/C2_partial_minimal_v1.tar.gz "$task_recovery/"; then task_download_ok=true; break; fi
  sleep 10
done
if [ "$task_download_ok" != true ]; then exit 1; fi
date -u '+C2_DOWNLOAD_FINISHED %Y-%m-%dT%H:%M:%SZ'
"$task_python" -B r5_execution/c2_recovery/partial_verify_local.py --recovery "$task_recovery"
rsync -az -e "$task_ssh" "$task_recovery/LOCAL_RECOVERY_CHECK.json" root@211.72.13.202:/workspace/r5/C2/recovery/
ssh -S /private/tmp/r5_control -p 46720 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/Users/richwang/Documents/ChatGPT/热/r4_v23_execution/ops/known_hosts root@211.72.13.202 'env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/r5/code:/workspace/r4_v23_execution OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 80-95 /workspace/env/bin/python -B -m c2_recovery.partial_seal'
rsync -az -e "$task_ssh" root@211.72.13.202:/workspace/r5/C2/RECOVERY_SEAL.json "$task_recovery/"
"$task_python" -B - <<'PY'
from pathlib import Path
import json,hashlib,datetime
root=Path('r5_execution/c2/recovery');s=json.loads((root/'RECOVERY_SEAL.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert s['scientific_status']=='TECHNICALLY_UNEVALUABLE' and s['completed_observed_trajectories']==280
for key,p in [('local_numeric_recovery',root/'LOCAL_RECOVERY_CHECK.json'),('CPU_recovery',root/'cpu_reference/CPU_RECOVERY_CHECK.json'),('independent_degradation_check',root.parent/'reports/INDEPENDENT_DEGRADATION_CHECK.json'),('module_status',root.parent/'reports/MODULE_STATUS.json'),('archive',root/'C2_partial_minimal_v1.tar.gz')]:
    assert sha(p)==s[key]['sha256'],key
r={'status':'PASS_PARTIAL_RECOVERY_SEAL_MIRROR','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seal_sha256':sha(root/'RECOVERY_SEAL.json'),'remote_seal_path':'/workspace/r5/C2/RECOVERY_SEAL.json','local_seal_path':str((root/'RECOVERY_SEAL.json').resolve()),'scope':'280 completed observed/800 planned, C2 TECHNICALLY_UNEVALUABLE; no completed800 claim','bound_local_receipts_and_archive_sha_match':True}
(root/'SEAL_MIRROR_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
PY
date -u '+C2_PARTIAL_RECOVERY_SEALED %Y-%m-%dT%H:%M:%SZ'
