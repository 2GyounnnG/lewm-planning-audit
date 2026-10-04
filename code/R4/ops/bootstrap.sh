#!/usr/bin/env bash
set -euo pipefail
umask 022
mkdir -p /workspace/{r4_pusht,r4_reacher,x1_tworoom,x1_cube,x2,shared_data,r4_v23_execution/ops}
cd /workspace/r4_v23_execution/ops
trap 'printf "{\"status\":\"FAILED\",\"exit\":%s}\n" "$?" > bootstrap_status.json' ERR
export UV_CONCURRENT_DOWNLOADS=4 UV_HTTP_TIMEOUT=1200 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv venv --python /venv/main/bin/python /workspace/env
uv pip install --python /workspace/env/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /workspace/env/bin/python -r r3_requirements.lock.txt
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libosmesa6 libgl1 libegl1 zstd tmux
/workspace/env/bin/python host_audit.py
printf '{"status":"ENVIRONMENT_READY"}\n' > bootstrap_status.json
