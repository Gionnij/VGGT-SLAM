#!/bin/bash
#SBATCH -J film_m2f_train
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Load shared env + venv (vggt_m2f_env) if available.
if [ -f "${REPO_ROOT}/shell/bashrc_shared" ]; then
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/shell/bashrc_shared"
  if type vggt_m2f_env >/dev/null 2>&1; then
    vggt_m2f_env || echo "[warn] vggt_m2f_env failed; continuing"
  fi
fi

RUN_DIR="${RUN_DIR:-"$PWD/runs/film_m2f_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to your packed dataset root (with chunks/labels).}"
LABELS_ROOT="${LABELS_ROOT:-}"
EXTRA_ARGS=("$@")

mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/checkpoints"

LOG_PATH="$RUN_DIR/logs/train_${SLURM_JOB_ID:-manual}.log"
touch "$LOG_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1

on_usr1() {
  echo "[signal] USR1 received at $(date -Iseconds); requesting checkpoint." | tee -a "$LOG_PATH"
  touch "$RUN_DIR/REQUEST_SAVE"
}
trap on_usr1 USR1

echo "[env] RUN_DIR=$RUN_DIR"
echo "[env] DATASET_ROOT=$DATASET_ROOT"
if [ -n "$LABELS_ROOT" ]; then
  echo "[env] LABELS_ROOT=$LABELS_ROOT"
fi

TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"

CMD=("$TRAIN_PYTHON" -u scripts/train_film_m2f_optimized.py \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "$RUN_DIR" \
  --ckpt-dir "$RUN_DIR/checkpoints" \
  --resume auto)

if [ -n "$LABELS_ROOT" ]; then
  CMD+=(--labels-root "$LABELS_ROOT")
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[cmd] ${CMD[*]}"
srun "${CMD[@]}"
