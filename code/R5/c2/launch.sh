#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/r5/code:/workspace/r4_v23_execution:/workspace/x1_cube/ogbench_only
export R3_ROOT=/workspace/shared_data/r3 X1_R3_ROOT=/workspace/shared_data/r3 X1_ROOT=/workspace/x1_cube
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 X1_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p /workspace/r5/C2/logs
exec 9>/workspace/r5/C2/SUPERVISOR.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
taskset -c 80-95 /workspace/env/bin/python -B -m c2.prepare > /workspace/r5/C2/logs/prepare.log 2>&1
for stream in R4_ALT_CEM_1 R4_ALT_CEM_2; do
 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 taskset -c 80-95 /workspace/env/bin/python -B -m c2.runner --stream "$stream" --shard 0 > "/workspace/r5/C2/logs/${stream}_0.log" 2>&1 &
 a=$!;echo "$a" > "/workspace/r5/C2/${stream}_0_PID"
 CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 taskset -c 96-111 /workspace/env/bin/python -B -m c2.runner --stream "$stream" --shard 1 > "/workspace/r5/C2/logs/${stream}_1.log" 2>&1 &
 b=$!;echo "$b" > "/workspace/r5/C2/${stream}_1_PID"
 wait "$a";wait "$b"
done
taskset -c 80-95 /workspace/env/bin/python -B -m c2.report > /workspace/r5/C2/logs/report.log 2>&1
