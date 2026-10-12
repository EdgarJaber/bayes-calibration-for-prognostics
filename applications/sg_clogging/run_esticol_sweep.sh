#!/usr/bin/env bash
#
# Run the ESTICOL observation-size sweep in the background on a remote machine.
#
#   ./run_esticol_sweep.sh
#   tail -f results_sweep/sweep_YYYYmmdd_HHMMSS.log
#
# Survives logout (nohup + disown). On a SLURM cluster, use
# submit_esticol_sweep.slurm with sbatch instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-esticol_sweep_config.json}"
OUTDIR="${OUTDIR:-results_sweep}"
PYTHON="${PYTHON:-python}"
# source "$HOME/.conda/etc/profile.d/conda.sh"; conda activate clogging

mkdir -p "$OUTDIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$OUTDIR/sweep_${TS}.log"
PIDFILE="$OUTDIR/sweep_${TS}.pid"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

echo "launching ESTICOL sweep"
echo "  config : $CONFIG"
echo "  outdir : $OUTDIR"
echo "  log    : $LOG"

nohup "$PYTHON" run_esticol_sweep.py --config "$CONFIG" --outdir "$OUTDIR" \
      > "$LOG" 2>&1 &
JOB_PID=$!
echo "$JOB_PID" > "$PIDFILE"
disown "$JOB_PID" 2>/dev/null || true

echo "  pid    : $JOB_PID  (saved to $PIDFILE)"
echo
echo "follow with:  tail -f $LOG"
echo "stop with:    kill \$(cat $PIDFILE)"
