#!/usr/bin/env bash
set -euo pipefail

# Root of the repo
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Configuration (override via env vars when calling the script)
# -----------------------------------------------------------------------------
DATASETS_ROOT="${DATASETS_ROOT:-$ROOT/depth/datasets}"
CPU_DATASETS="${CPU_DATASETS:-$DATASETS_ROOT/*.txt}"
UVM_DATASETS="${UVM_DATASETS:-$ROOT/baselines/uvm/g*.txt}"
GPU_WITH_DATASETS="${GPU_WITH_DATASETS:-$ROOT/gpu/with_filter/g*.txt}"
GPU_WITHOUT_DATASETS="${GPU_WITHOUT_DATASETS:-$ROOT/gpu/without_filter/g*.txt}"
EXT_STREAMS_DATASETS="${EXT_STREAMS_DATASETS:-$ROOT/external/streams/input.txt}"
EXT_NOSTREAMS_DATASETS="${EXT_NOSTREAMS_DATASETS:-$ROOT/external/without_streams/input.txt}"
WK2017_DATASETS="${WK2017_DATASETS:-$ROOT/baselines/wk-bcc-2017/datasets/*.txt}"
WK2018_DATASETS="${WK2018_DATASETS:-$ROOT/baselines/wk-bcc-2018/datasets/*.txt}"
DEPTH_DATASETS="${DEPTH_DATASETS:-$DATASETS_ROOT/*.txt}"

CPU_ROUNDS="${CPU_ROUNDS:-3}"
GPU_SHARE="${GPU_SHARE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-1048576}"
WK2018_K="${WK2018_K:-2}"
WK2018_VERBOSE="${WK2018_VERBOSE:-0}"
DEPTH_SOURCE="${DEPTH_SOURCE:-0}"

# -----------------------------------------------------------------------------
log() { printf "\n[%s] %s\n" "$1" "$2"; }

die() { echo "Error: $*" >&2; exit 1; }

run_task() {
  local name="$1" dir="$2" cmd_template="$3" datasets="$4"
  local ran=false
  for path in ${datasets}; do
    if [[ -f "$path" ]]; then
      ran=true
      local cmd="$cmd_template"
      cmd="${cmd//\{file\}/\"$path\"}"
      cmd="${cmd//\{rounds\}/$CPU_ROUNDS}"
      cmd="${cmd//\{gpu_share\}/$GPU_SHARE}"
      cmd="${cmd//\{batch\}/$BATCH_SIZE}"
      cmd="${cmd//\{k_out\}/$WK2018_K}"
      cmd="${cmd//\{verbose\}/$WK2018_VERBOSE}"
      cmd="${cmd//\{src\}/$DEPTH_SOURCE}"
      log "$name" "cd $dir && $cmd"
      (cd "$dir" && eval "$cmd")
    fi
  done
  if [[ "$ran" == false ]]; then
    log "$name" "skipped (no dataset files matched: $datasets)"
  fi
}

# -----------------------------------------------------------------------------
# Build everything first using the root Makefile
# -----------------------------------------------------------------------------
log "build" "make -j -C $ROOT"
make -j -C "$ROOT"

# -----------------------------------------------------------------------------
# Runs (edit dataset globs above as needed)
# -----------------------------------------------------------------------------
run_task "cpu-baseline" "$ROOT/baselines/cpu/src" "./FAST_BCC {file} {rounds}" "$CPU_DATASETS"
run_task "uvm" "$ROOT/baselines/uvm" "./main {file}" "$UVM_DATASETS"
run_task "gpu-with-filter" "$ROOT/gpu/with_filter" "./main {file}" "$GPU_WITH_DATASETS"
run_task "gpu-without-filter" "$ROOT/gpu/without_filter" "./main {file}" "$GPU_WITHOUT_DATASETS"
run_task "external-streams" "$ROOT/external/streams" "./ext-bcc {file} {gpu_share} {batch}" "$EXT_STREAMS_DATASETS"
run_task "external-no-streams" "$ROOT/external/without_streams" "./ext-bcc {file} {gpu_share} {batch}" "$EXT_NOSTREAMS_DATASETS"
run_task "wk-bcc-2017" "$ROOT/baselines/wk-bcc-2017" "./bin/cuda_bcc -i {file} -a ebcc" "$WK2017_DATASETS"
run_task "wk-bcc-2018" "$ROOT/baselines/wk-bcc-2018" "./bin/main {file} {k_out} {verbose}" "$WK2018_DATASETS"
run_task "depth-bfs" "$ROOT/depth" "./bfs {file} {src}" "$DEPTH_DATASETS"

log "done" "all tasks completed"
