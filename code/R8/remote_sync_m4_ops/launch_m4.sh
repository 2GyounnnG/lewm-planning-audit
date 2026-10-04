#!/usr/bin/env bash
set -u
BASE=/workspace/r8/m4
PY=/workspace/env/bin/python
COMMON=(--r3-root /workspace/shared_data/r3 --assets /workspace/shared_data/r4_assets --task pusht --output "$BASE/trajectories" --s0-bcd /workspace/r4_pusht/s0 --reuse-root /workspace/r4_pusht/trajectories --workers 2)
ROLES=(H0_MENU_RERANK SIM_LAT_RERANK SIM_TASK_RERANK)
STREAMS=(R3_ORIGINAL R4_ALT_CEM_1 R4_ALT_CEM_2)
mkdir -p "$BASE/trajectories/FORMAL" "$BASE/ops/logs"
: > "$BASE/ops/M4_FORMAL_PIDS"
: > "$BASE/ops/M4_FORMAL_COMMANDS"
i=0
for role in "${ROLES[@]}"; do
  for stream in "${STREAMS[@]}"; do
    gpu=$((i % 8))
    for wi in 0 1; do
      log="$BASE/ops/logs/formal_${role}_${stream}_w${wi}.log"
      cmd=(env PYTHONPATH=/workspace/r8/m4:/workspace/r4_v23_execution:/workspace/shared_data/r3 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m r4ext.s3 "${COMMON[@]}" --role "$role" --stream "$stream" --device cuda --worker-index "$wi")
      printf '%s\n' "${cmd[*]}" >> "$BASE/ops/M4_FORMAL_COMMANDS"
      nohup "${cmd[@]}" > "$log" 2>&1 &
      pid=$!
      printf '%s\t%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$role" "$stream" "$wi" >> "$BASE/ops/M4_FORMAL_PIDS"
    done
    i=$((i+1))
  done
done
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/ops/M4_FORMAL_STARTED_UTC"
cat "$BASE/ops/M4_FORMAL_PIDS"
