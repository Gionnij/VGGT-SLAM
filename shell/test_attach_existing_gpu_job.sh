#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

JOB_ID=""
BASHRC_SHARED="${REPO_ROOT}/shell/bashrc_shared"
STATE_FILE="${HOME}/.vggt_active_gpu_attach_test.env"
RUN_ID="attach_test_$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<USAGE
Usage: $(basename "$0") --job-id JOB_ID [options]

Options:
  --job-id ID          Existing Slurm GPU allocation/job id to reuse.
  --bashrc PATH        bashrc_shared path to source on the compute node.
  --state-file PATH    State file written by hpc_record_gpu_state.
  --run-id ID          Run id stored in the state file.
  -h, --help           Show this help.

This script does not start the full pipeline. It only validates that:
  1. the job id exists and is running,
  2. a new srun step can be launched inside that allocation,
  3. bashrc_shared can be sourced there,
  4. hpc_record_gpu_state writes the expected state file.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)
      JOB_ID="${2:-}"; shift 2 ;;
    --bashrc)
      BASHRC_SHARED="${2:-}"; shift 2 ;;
    --state-file)
      STATE_FILE="${2:-}"; shift 2 ;;
    --run-id)
      RUN_ID="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[test_attach_existing_gpu_job] Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "${JOB_ID}" ]]; then
  echo "[test_attach_existing_gpu_job] Missing --job-id." >&2
  usage
  exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "[test_attach_existing_gpu_job] srun not found in PATH." >&2
  exit 1
fi

if ! command -v scontrol >/dev/null 2>&1; then
  echo "[test_attach_existing_gpu_job] scontrol not found in PATH." >&2
  exit 1
fi

JOB_INFO="$(scontrol show job "${JOB_ID}" 2>/dev/null || true)"
if [[ -z "${JOB_INFO}" ]]; then
  echo "[test_attach_existing_gpu_job] Job ${JOB_ID} not found." >&2
  exit 1
fi

JOB_STATE="$(sed -n 's/.*JobState=\([^ ]*\).*/\1/p' <<<"${JOB_INFO}" | head -n 1)"
NODE_LIST="$(sed -n 's/.*NodeList=\([^ ]*\).*/\1/p' <<<"${JOB_INFO}" | head -n 1)"
PARTITION="$(sed -n 's/.*Partition=\([^ ]*\).*/\1/p' <<<"${JOB_INFO}" | head -n 1)"

echo "[attach-test] job_id=${JOB_ID}"
echo "[attach-test] job_state=${JOB_STATE:-unknown}"
echo "[attach-test] partition=${PARTITION:-unknown}"
echo "[attach-test] nodelist=${NODE_LIST:-unknown}"
echo "[attach-test] state_file=${STATE_FILE}"
echo "[attach-test] run_id=${RUN_ID}"

if [[ "${JOB_STATE}" != "RUNNING" ]]; then
  echo "[test_attach_existing_gpu_job] Job ${JOB_ID} is not RUNNING." >&2
  exit 1
fi

run_step() {
  local label="$1"
  local script="$2"
  echo
  echo "[attach-test] step=${label}"
  srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 bash -lc "${script}"
}

run_step "basic" '
  set -euo pipefail
  echo "hostname=$(hostname -s)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_step_id=${SLURM_STEP_ID:-}"
  echo "slurm_job_gpus=${SLURM_JOB_GPUS:-<unset>}"
  echo "slurm_step_gpus=${SLURM_STEP_GPUS:-<unset>}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
  else
    echo "nvidia-smi not found"
  fi
'

run_step "bashrc" "
  set -euo pipefail
  source \"${BASHRC_SHARED}\"
  echo \"vggt_root=\${VGGT_SLAM_ROOT:-<unset>}\"
  type hpc_record_gpu_state >/dev/null 2>&1
  type run_pipeline >/dev/null 2>&1
  echo \"pre_bind_cuda_visible_devices=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
  if type _vggt_bind_gpu >/dev/null 2>&1; then
    _vggt_bind_gpu
    echo \"post_bind_cuda_visible_devices=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
  fi
  echo \"bashrc_shared_ok=1\"
"

run_step "state-file" "
  set -euo pipefail
  source \"${BASHRC_SHARED}\"
  export VGGT_GPU_STATE_FILE=\"${STATE_FILE}\"
  export HPC_GPU_RUN_ID=\"${RUN_ID}\"
  hpc_record_gpu_state
  echo \"state_file_contents_begin\"
  cat \"${STATE_FILE}\"
  echo \"state_file_contents_end\"
"

echo
echo "[attach-test] success"
echo "[attach-test] Existing allocation ${JOB_ID} can run new steps via srun --jobid --overlap."
