#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/r5/code:/workspace/r4_v23_execution:/workspace/x1_cube/ogbench_only
export R3_ROOT=/workspace/shared_data/r3 X1_R3_ROOT=/workspace/shared_data/r3 X1_ROOT=/workspace/x1_cube
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
taskset -c 88-99 /workspace/env/bin/python -B -m unittest c0.test_contract -v >/workspace/r5/C0/logs/cpu_contract.log 2>&1
taskset -c 88-99 /workspace/env/bin/python -B -c 'from c0.common import *; atomic(ROOT/"CPU_CONTRACT_CHECK.json",{"status":"PASS","tests":3,"actual_raw_simulation_steps":0,"log":record(ROOT/"logs/cpu_contract.log"),"contract":record(ROOT/"C0_CONTRACT.json")})'
if test -e /workspace/r5/C0/RUN_PID; then cat /workspace/r5/C0/RUN_PID; exit 0; fi
nohup taskset -c 88-99 /workspace/env/bin/python -B -m c0.runner >/workspace/r5/C0/logs/caa.log 2>&1 </dev/null &
echo $! >/workspace/r5/C0/RUN_PID
cat /workspace/r5/C0/RUN_PID
