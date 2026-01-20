#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-${RUN_DIR:-}}"
if [ -z "$RUN_DIR" ]; then
  echo "Usage: RUN_DIR=/path/to/run ./slurm/monitor.sh [run_dir]" >&2
  exit 1
fi

echo "RUN_DIR: $RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
CKPT_DIR="$RUN_DIR/checkpoints"

if [ -d "$LOG_DIR" ]; then
  latest_log=$(ls -1t "$LOG_DIR" 2>/dev/null | head -n1 || true)
  if [ -n "$latest_log" ]; then
    echo "Latest log: $LOG_DIR/$latest_log"
    tail -n 20 "$LOG_DIR/$latest_log"
  else
    echo "No logs found in $LOG_DIR"
  fi
else
  echo "Log directory missing: $LOG_DIR"
fi

if [ -d "$CKPT_DIR" ]; then
  latest_ckpt=$(ls -1t "$CKPT_DIR"/*.pt 2>/dev/null | head -n1 || true)
  if [ -n "$latest_ckpt" ]; then
    echo "Latest checkpoint: $latest_ckpt"
  else
    echo "No checkpoints found in $CKPT_DIR"
  fi
else
  echo "Checkpoint directory missing: $CKPT_DIR"
fi
