#!/usr/bin/env bash
#
# Run every experiment from scratch, with the current likelihood
# (log-sum-exp aggregation, fixed scrambled-Sobol weight design, nb_weights = 1024):
#
#   1. full-data run        run_rul_tables.py            -> results/
#   2. ESTICOL sweep        run_esticol_sweep.py         -> results_sweep/
#   3. ESTICOL kinetics     run_esticol_kinetics.py      -> results_kinetics/
#
#   ./run_all_experiments.sh                  # jobs run one after the other, detached
#   PARALLEL=1 ./run_all_experiments.sh       # the three jobs at once (needs ~3x the cores)
#   ARCHIVE=1  ./run_all_experiments.sh       # first move existing result dirs to <dir>_previous_<ts>
#
# Refuses to start if a result directory already holds files, unless ARCHIVE=1 (rename it
# aside) or FORCE=1 (write into it anyway). Survives logout (nohup + disown).
# Each job logs to <outdir>/run_<ts>.log; follow with `tail -f`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
PARALLEL="${PARALLEL:-0}"
ARCHIVE="${ARCHIVE:-0}"
FORCE="${FORCE:-0}"
# source "$HOME/.conda/etc/profile.d/conda.sh"; conda activate clogging

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

TS="$(date +%Y%m%d_%H%M%S)"

# name | script | config | outdir
JOBS=(
  "rul_tables|run_rul_tables.py|rul_tables_config.json|results"
  "esticol_sweep|run_esticol_sweep.py|esticol_sweep_config.json|results_sweep"
  "esticol_kinetics|run_esticol_kinetics.py|esticol_kinetics_config.json|results_kinetics"
)

# --- make sure we never mix old and new results ------------------------------
for job in "${JOBS[@]}"; do
  IFS='|' read -r name script config outdir <<< "$job"
  if [ -d "$outdir" ] && [ -n "$(ls -A "$outdir" 2>/dev/null)" ]; then
    if [ "$ARCHIVE" = "1" ]; then
      mv "$outdir" "${outdir}_previous_${TS}"
      echo "archived $outdir -> ${outdir}_previous_${TS}"
    elif [ "$FORCE" != "1" ]; then
      echo "ERROR: $outdir is not empty. Re-run with ARCHIVE=1 (move it aside) or FORCE=1." >&2
      exit 1
    fi
  fi
done

run_job() {   # name script config outdir
  local name="$1" script="$2" config="$3" outdir="$4"
  mkdir -p "$outdir"
  echo "[$(date +%H:%M:%S)] starting $name (log: $outdir/run_${TS}.log)"
  "$PYTHON" "$script" --config "$config" --outdir "$outdir" > "$outdir/run_${TS}.log" 2>&1
  echo "[$(date +%H:%M:%S)] finished $name"
}
export -f run_job
export PYTHON TS

if [ "$PARALLEL" = "1" ]; then
  for job in "${JOBS[@]}"; do
    IFS='|' read -r name script config outdir <<< "$job"
    nohup bash -c "run_job '$name' '$script' '$config' '$outdir'" > "/dev/null" 2>&1 &
    disown $! 2>/dev/null || true
    echo "launched $name in parallel (pid $!)"
  done
else
  SEQ=""
  for job in "${JOBS[@]}"; do
    IFS='|' read -r name script config outdir <<< "$job"
    SEQ+="run_job '$name' '$script' '$config' '$outdir'; "
  done
  MASTER_LOG="run_all_${TS}.log"
  nohup bash -c "$SEQ" > "$MASTER_LOG" 2>&1 &
  disown $! 2>/dev/null || true
  echo "launched all jobs sequentially (pid $!, master log: $MASTER_LOG)"
fi

echo
echo "follow with:  tail -f results/run_${TS}.log results_sweep/run_${TS}.log results_kinetics/run_${TS}.log"
