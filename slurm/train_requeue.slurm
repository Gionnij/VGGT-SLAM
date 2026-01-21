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
    echo "[env] activating vggt_m2f_env"
    if ! vggt_m2f_env; then
      echo "[warn] vggt_m2f_env failed; continuing with current PATH"
    fi
  else
    echo "[warn] vggt_m2f_env not found after sourcing bashrc_shared"
  fi
else
  echo "[warn] bashrc_shared not found at ${REPO_ROOT}/shell/bashrc_shared"
fi

# Ensure local repo paths are ahead on PYTHONPATH (covers mask2former package layout).
# Include both the submodule root and the package dir to avoid namespace shadowing from site-packages.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/salad:${REPO_ROOT}/mask2former:${REPO_ROOT}/mask2former/mask2former:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

RUN_DIR="${RUN_DIR:-"$PWD/runs/film_m2f_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to your packed dataset root (with chunks/labels).}"
LABELS_ROOT="${LABELS_ROOT:-}"
EXTRA_ARGS=("$@")
if [ "${#EXTRA_ARGS[@]}" -gt 0 ] && [ "${EXTRA_ARGS[0]}" = "--" ]; then
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi

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

if [ -z "${TRAIN_PYTHON:-}" ]; then
  if [ -x "${HOME}/.venvs/vggt-m2f/bin/python" ]; then
    TRAIN_PYTHON="${HOME}/.venvs/vggt-m2f/bin/python"
    echo "[env] using TRAIN_PYTHON=${TRAIN_PYTHON} (venv)"
  else
    TRAIN_PYTHON="python3"
    echo "[env] using TRAIN_PYTHON=${TRAIN_PYTHON} (fallback)"
  fi
else
  echo "[env] using TRAIN_PYTHON=${TRAIN_PYTHON} (provided)"
fi

echo "[envcheck] verifying detectron2/mask2former imports with ${TRAIN_PYTHON}"
"${TRAIN_PYTHON}" - <<'PY'
import importlib, sys
mods = ["detectron2", "mask2former"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        loc = getattr(mod, "__file__", None)
        if loc is None and hasattr(mod, "__path__"):
            loc = list(mod.__path__)[0]
        print(f"[envcheck] {m} import OK from {loc}")
    except Exception as e:
        print(f"[envcheck] {m} import FAILED: {e}")
        sys.exit(1)
try:
    from mask2former import add_maskformer2_config as cfg_fn
    print(f"[envcheck] mask2former.add_maskformer2_config OK -> {cfg_fn}")
except Exception as e:
    print(f"[envcheck] mask2former.add_maskformer2_config FAILED: {e}")
    try:
        from mask2former.config import add_maskformer2_config as cfg_fn
        print(f"[envcheck] mask2former.config.add_maskformer2_config OK -> {cfg_fn}")
    except Exception as e2:
        print(f"[envcheck] mask2former.config.add_maskformer2_config FAILED: {e2}")
        sys.exit(1)
print("[envcheck] sys.path:")
for p in sys.path:
    print(" ", p)
PY

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
