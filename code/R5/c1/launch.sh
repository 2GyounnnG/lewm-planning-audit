#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/r5/code:/workspace/r4_v23_execution:/workspace/x1_cube/ogbench_only
export R3_ROOT=/workspace/shared_data/r3 X1_R3_ROOT=/workspace/shared_data/r3 X1_ROOT=/workspace/x1_cube
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 X1_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p /workspace/r5/C1/logs
exec 9>/workspace/r5/C1/SUPERVISOR.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=6
taskset -c 80-95 /workspace/env/bin/python -B -m c1.prepare > /workspace/r5/C1/logs/prepare.log 2>&1
taskset -c 80-95 /workspace/env/bin/python -B -m unittest c1.test_contract > /workspace/r5/C1/logs/cpu_contract.log 2>&1
taskset -c 80-95 /workspace/env/bin/python -B -m c1.runner --phase TECH > /workspace/r5/C1/logs/tech.log 2>&1
CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=6 taskset -c 80-95 /workspace/env/bin/python -B -m c1.runner --phase EVAL --shard 0 > /workspace/r5/C1/logs/eval_0.log 2>&1 &
a=$!;echo "$a" > /workspace/r5/C1/EVAL_0_PID
CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 taskset -c 96-111 /workspace/env/bin/python -B -m c1.runner --phase EVAL --shard 1 > /workspace/r5/C1/logs/eval_1.log 2>&1 &
b=$!;echo "$b" > /workspace/r5/C1/EVAL_1_PID
wait "$a"
wait "$b"
taskset -c 80-95 /workspace/env/bin/python -B -m c1.report > /workspace/r5/C1/logs/report.log 2>&1
