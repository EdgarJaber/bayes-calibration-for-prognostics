#!/usr/bin/env bash
#
# Run the two-SG RUL table computation in the background on a remote machine.
#
#   ./run_rul_tables.sh              # launches, detaches, returns immediately
#   tail -f results/run_YYYYmmdd_HHMMSS.log   # follow progress
#
# The job keeps running after you log out (nohup + disown). On completion,
# results/rul_tables.tex and results/rul_stats.json are written.

set -euo pipefail

# --- resolve script directory so paths work regardless of CWD ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- configuration ----------------------------------------------------------
CONFIG="${CONFIG:-rul_tables_config.json}"
OUTDIR="${OUTDIR:-results}"
PYTHON="${PYTHON:-python}"          # override, e.g. PYTHON=python3.11
# Uncomment / edit to activate an environment before running:
# source "$HOME/.conda/etc/profile.d/conda.sh"; conda activate clogging

mkdir -p "$OUTDIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$OUTDIR/run_${TS}.log"
PIDFILE="$OUTDIR/run_${TS}.pid"

# keep BLAS from oversubscribing cores on a shared node
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

echo "launching RUL table run"
echo "  config : $CONFIG"
echo "  outdir : $OUTDIR"
echo "  log    : $LOG"

# --- launch detached --------------------------------------------------------
nohup "$PYTHON" run_rul_tables.py --config "$CONFIG" --outdir "$OUTDIR" \
      > "$LOG" 2>&1 &
JOB_PID=$!
echo "$JOB_PID" > "$PIDFILE"
disown "$JOB_PID" 2>/dev/null || true

echo "  pid    : $JOB_PID  (saved to $PIDFILE)"
echo
echo "follow with:  tail -f $LOG"
echo "stop with:    kill \$(cat $PIDFILE)"
