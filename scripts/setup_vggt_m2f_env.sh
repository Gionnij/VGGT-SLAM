#!/usr/bin/env bash
set -euo pipefail

# Helper to build a single venv that can run both VGGT and Mask2Former.
# It assumes CUDA 11.8 + Python 3.10.7 and installs Torch/Detectron2/Mask2Former
# on top of the VGGT requirements.

REPO_ROOT="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
: "${VGGT_VENV_DIR:=${HOME}/.venvs}"
: "${VGGT_COMBINED_ENV:=vggt-m2f}"

ENV_PATH="${VGGT_VENV_DIR}/${VGGT_COMBINED_ENV}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.3.1 torchvision==0.18.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
DETECTRON2_SRC="${DETECTRON2_SRC:-git+https://github.com/facebookresearch/detectron2.git}"

mkdir -p "${VGGT_VENV_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load python/3.10.7 nvidia/cuda-11.8 2>/dev/null || true
fi

echo "[setup_vggt_m2f_env] Using python=${PYTHON_BIN}, env=${ENV_PATH}"
"${PYTHON_BIN}" -m venv "${ENV_PATH}"
# shellcheck disable=SC1090
source "${ENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --extra-index-url "${TORCH_INDEX_URL}" ${TORCH_SPEC}

# Install VGGT requirements (skip torch/torchvision duplicates and gtsam placeholder).
grep -Ev '^(torch|torchvision|gtsam)' "${REPO_ROOT}/requirements.txt" | python -m pip install -r /dev/stdin

# Detectron2 built against the installed torch.
python -m pip install "${DETECTRON2_SRC}"

# Mask2Former editable install (so local changes are picked up).
python -m pip install -e "${REPO_ROOT}/mask2former"

echo "[setup_vggt_m2f_env] Done. Activate with: source ${REPO_ROOT}/shell/bashrc_shared && vggt_mask2former_env"
