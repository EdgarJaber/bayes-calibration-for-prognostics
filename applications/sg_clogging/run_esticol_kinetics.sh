#!/usr/bin/env bash
#
# Run the ESTICOL kinetics experiment (SG B: quasi-linear vs exponential
# sub-populations of the last cycle's ESTICOL points) in the background.
#
#   ./run_esticol_kinetics.sh
#   tail -f results_kinetics/kinetics_YYYYmmdd_HHMMSS.log
#
# Survives logout (nohup + disown). Override with CONFIG=, OUTDIR=, PYTHON=.
# Extra arguments are passed to the script, e.g.  ./run_esticol_kinetics.sh --only exponential

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-esticol_kinetics_config.json}"
OUTDIR="${OUTDIR:-results_kinetics}"
PYTHON="${PYTHON:-python}"
# source "$HOME/.conda/etc/profile.d/conda.sh"; conda activate clogging

mkdir -p "$OUTDIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$OUTDIR/kinetics_${TS}.log"
PIDFILE="$OUTDIR/kinetics_${TS}.pid"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

echo "launching ESTICOL kinetics experiment"
echo "  config : $CONFIG"
echo "  outdir : $OUTDIR"
echo "  log    : $LOG"

nohup "$PYTHON" run_esticol_kinetics.py --config "$CONFIG" --outdir "$OUTDIR" "$@" \
      > "$LOG" 2>&1 &
JOB_PID=$!
echo "$JOB_PID" > "$PIDFILE"
disown "$JOB_PID" 2>/dev/null || true

echo "  pid    : $JOB_PID  (saved to $PIDFILE)"
echo
echo "follow with:  tail -f $LOG"
echo "stop with:    kill \$(cat $PIDFILE)"
