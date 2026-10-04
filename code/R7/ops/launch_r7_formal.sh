#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/r7
R3=/workspace/shared_data/r3
CODE=/workspace/r7/code
PY=/workspace/env/bin/python
mkdir -p "$ROOT/raw/reacher" "$ROOT/raw/pusht" "$ROOT/ops/logs"
for i in 0 1 2 3; do
  nohup env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$CODE:/workspace/r4_v23_execution" "$PY" -B "$CODE/r7_long_runner.py" \
    --r3-root "$R3" --assets "$ROOT/assets" --selection "$ROOT/ops/R7_REACHER_CASE_SELECTION.json" \
    --output "$ROOT/raw/reacher" --task reacher --phase FORMAL --device "cuda:$i" \
    --worker-index "$i" --workers 4 --threads 1 --all-streams > "$ROOT/ops/logs/formal_reacher_$i.log" 2>&1 &
  echo $! > "$ROOT/ops/formal_reacher_$i.pid"
done
for i in 0 1 2 3; do
  gpu=$((i+4))
  nohup env PYTHONPATH="$CODE:/workspace/r4_v23_execution" "$PY" -B "$CODE/r7_long_runner.py" \
    --r3-root "$R3" --assets "$ROOT/assets" --selection "$ROOT/ops/R7_PUSHT_CASE_SELECTION.json" \
    --output "$ROOT/raw/pusht" --task pusht --phase FORMAL --device "cuda:$gpu" \
    --worker-index "$i" --workers 4 --threads 1 --all-streams > "$ROOT/ops/logs/formal_pusht_$i.log" 2>&1 &
  echo $! > "$ROOT/ops/formal_pusht_$i.pid"
done
date -u +%FT%TZ > "$ROOT/ops/formal_started_utc.txt"
