#!/usr/bin/env bash
#
# Launch the experiments of the clogging application, detached (survives logout).
#
#   ./run.sh full                        full-data run, both SGs          -> results/
#   ./run.sh sweep                       ESTICOL first-k sweep, both SGs  -> results_sweep/
#   ./run.sh kinetics [--only NAME]      SG B kinetics experiment         -> results_kinetics/
#   ./run.sh all                         the three above, one after the other
#
# Extra arguments are passed to the python script (single experiments only), e.g.
#   ./run.sh kinetics --only exponential
#
# Environment: PYTHON (default: python), OMP_NUM_THREADS (default 4),
#   for `all`:  PARALLEL=1 runs the three jobs at once;  ARCHIVE=1 first moves non-empty
#   result folders to <dir>_previous_<timestamp>;  FORCE=1 writes into them anyway.
#
# Logs and pid files are written next to the results: <outdir>/<experiment>_<timestamp>.{log,pid}
# Always run from (or through) this folder: DATA/, configs/ and results*/ are relative to it.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python}"
TS="$(date +%Y%m%d_%H%M%S)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

usage() { sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# experiment -> "script config outdir"
spec() {
  case "$1" in
    full)     echo "scripts/full_run.py configs/rul_tables_config.json results" ;;
    sweep)    echo "scripts/esticol_sweep.py configs/esticol_sweep_config.json results_sweep" ;;
    kinetics) echo "scripts/esticol_kinetics.py configs/esticol_kinetics_config.json results_kinetics" ;;
    *) echo "unknown experiment: $1" >&2; usage 1 ;;
  esac
}

launch() {   # name [extra python args...]
  local name="$1"; shift
  local script config outdir
  read -r script config outdir <<< "$(spec "$name")"
  mkdir -p "$outdir"
  local log="$outdir/${name}_${TS}.log"
  nohup "$PYTHON" "$script" --config "$config" --outdir "$outdir" "$@" > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$outdir/${name}_${TS}.pid"
  disown "$pid" 2>/dev/null || true
  echo "launched $name  pid $pid  log $log"
}

run_all() {
  local names=(full sweep kinetics)
  for name in "${names[@]}"; do
    local script config outdir
    read -r script config outdir <<< "$(spec "$name")"
    if [ -d "$outdir" ] && [ -n "$(ls -A "$outdir" 2>/dev/null)" ]; then
      if [ "${ARCHIVE:-0}" = "1" ]; then
        mv "$outdir" "${outdir}_previous_${TS}"; echo "archived $outdir -> ${outdir}_previous_${TS}"
      elif [ "${FORCE:-0}" != "1" ]; then
        echo "ERROR: $outdir is not empty. Re-run with ARCHIVE=1 (move it aside) or FORCE=1." >&2
        exit 1
      fi
    fi
  done
  if [ "${PARALLEL:-0}" = "1" ]; then
    for name in "${names[@]}"; do launch "$name"; done
  else
    # one detached shell runs the three jobs back to back
    local chain=""
    for name in "${names[@]}"; do
      local script config outdir
      read -r script config outdir <<< "$(spec "$name")"
      mkdir -p "$outdir"
      chain+="'$PYTHON' '$script' --config '$config' --outdir '$outdir' > '$outdir/${name}_${TS}.log' 2>&1; "
    done
    nohup bash -c "$chain" > "run_all_${TS}.log" 2>&1 &
    disown $! 2>/dev/null || true
    echo "launched full -> sweep -> kinetics sequentially (pid $!, master log run_all_${TS}.log)"
  fi
}

[ $# -ge 1 ] || usage 1
case "$1" in
  -h|--help) usage 0 ;;
  all)       run_all ;;
  *)         name="$1"; shift; spec "$name" >/dev/null; launch "$name" "$@" ;;
esac
