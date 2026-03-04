#!/usr/bin/env bash
set -euo pipefail

# Launch robot->HPC pipeline with automatic Slurm interactive allocations in tmux.
# Windows:
#  1) cpu-tunnel   (sinteractive CPU -> hpc_robot_stream_tunnel)
#  2) cpu-bridge   (sinteractive CPU -> hpc_robot_bridge)
#  3) gpu-pipeline (sinteractive GPU -> run_pipeline)

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
BASHRC_SHARED="${REPO_ROOT}/shell/bashrc_shared"

SESSION_NAME="${SESSION_NAME:-robot_pipeline}"
LOCAL_PORT="${LOCAL_PORT:-5001}"
HEAD_PORT="${HEAD_PORT:-5001}"
FPS="${FPS:-2.0}"
ATTACH="1"
FORCE_KILL="0"

CHECKPOINT="${VGGT_FINETUNE_CKPT:-}"
DEMO_ROOT="${VGGT_DEMO_ROOT:-$HOME/src/VGGT-SLAM/demo}"
LOG_RESULTS="${VGGT_LOG_RESULTS:-0}"
MAX_LIVE_STEPS="${VGGT_MAX_LIVE_STEPS:-0}"
STOP_TOPIC="${VGGT_STOP_TOPIC:-/stream/stop}"
STATS_INTERVAL="${HPC_ROBOT_STATS_INTERVAL:-5}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  -s, --session NAME        tmux session name (default: ${SESSION_NAME})
  -l, --local-port PORT     local tunnel port on compute nodes (default: ${LOCAL_PORT})
  -p, --head-port PORT      port opened on HPC head by robot reverse tunnel (default: ${HEAD_PORT})
  -f, --fps FPS             bridge publish rate (default: ${FPS})
  -c, --checkpoint PATH     fine-tuned checkpoint for run_pipeline
  -d, --demo-root PATH      demo output root (default: ${DEMO_ROOT})
      --log-results 0|1     enable legacy log_results path (default: ${LOG_RESULTS})
      --max-live-steps N    auto-stop after N submaps (0 = no auto-stop)
      --stop-topic TOPIC    stop topic for ingest (default: ${STOP_TOPIC})
      --stats-interval SEC  bridge stats interval seconds (default: ${STATS_INTERVAL})
      --no-attach           do not attach automatically
  -k, --kill-existing       kill existing session with same name
  -h, --help                show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--session)
      SESSION_NAME="$2"; shift 2 ;;
    -l|--local-port)
      LOCAL_PORT="$2"; shift 2 ;;
    -p|--head-port)
      HEAD_PORT="$2"; shift 2 ;;
    -f|--fps)
      FPS="$2"; shift 2 ;;
    -c|--checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    -d|--demo-root)
      DEMO_ROOT="$2"; shift 2 ;;
    --log-results)
      LOG_RESULTS="$2"; shift 2 ;;
    --max-live-steps)
      MAX_LIVE_STEPS="$2"; shift 2 ;;
    --stop-topic)
      STOP_TOPIC="$2"; shift 2 ;;
    --stats-interval)
      STATS_INTERVAL="$2"; shift 2 ;;
    --no-attach)
      ATTACH="0"; shift ;;
    -k|--kill-existing)
      FORCE_KILL="1"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "[start_hpc_robot_pipeline_alloc_tmux] tmux not found in PATH." >&2
  exit 1
fi

if ! command -v sinteractive >/dev/null 2>&1; then
  echo "[start_hpc_robot_pipeline_alloc_tmux] sinteractive not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${BASHRC_SHARED}" ]]; then
  echo "[start_hpc_robot_pipeline_alloc_tmux] Missing ${BASHRC_SHARED}" >&2
  exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[start_hpc_robot_pipeline_alloc_tmux] Missing checkpoint. Set VGGT_FINETUNE_CKPT or pass --checkpoint." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [[ "${FORCE_KILL}" == "1" ]]; then
    tmux kill-session -t "${SESSION_NAME}"
  else
    echo "[start_hpc_robot_pipeline_alloc_tmux] Session '${SESSION_NAME}' already exists." >&2
    echo "Use -k to replace it, or attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
  fi
fi

RUN_DIR="${TMPDIR:-/tmp}/vggt_alloc_tmux_${SESSION_NAME}_$$"
mkdir -p "${RUN_DIR}"

cat > "${RUN_DIR}/cpu_tunnel.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
source "${BASHRC_SHARED}"
echo "[cpu-tunnel] requesting CPU node..."
sinteractive --partition=main --mem=40G --time=24:00:00 bash -lc '
  set -euo pipefail
  source "${BASHRC_SHARED}"
  echo "[cpu-tunnel] allocation granted on: \\$(hostname)"
  hpc_robot_stream_tunnel "${LOCAL_PORT}" "${HEAD_PORT}"
'
SCRIPT

cat > "${RUN_DIR}/cpu_bridge.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
source "${BASHRC_SHARED}"
echo "[cpu-bridge] requesting CPU node..."
sinteractive --partition=main --mem=40G --time=24:00:00 bash -lc '
  set -euo pipefail
  source "${BASHRC_SHARED}"
  echo "[cpu-bridge] allocation granted on: \\$(hostname)"
  export HPC_ROBOT_STATS_INTERVAL="${STATS_INTERVAL}"
  hpc_robot_bridge 127.0.0.1 "${LOCAL_PORT}" "${FPS}"
'
SCRIPT

cat > "${RUN_DIR}/gpu_pipeline.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
source "${BASHRC_SHARED}"
echo "[gpu-pipeline] requesting GPU node..."
sinteractive --partition=main --gres=gpu:ampere:1 --mem=40G --time=24:00:00 bash -lc '
  set -euo pipefail
  source "${BASHRC_SHARED}"
  echo "[gpu-pipeline] allocation granted on: \\$(hostname)"
  export VGGT_FINETUNE_CKPT="${CHECKPOINT}"
  export VGGT_DEMO_ROOT="${DEMO_ROOT}"
  export VGGT_LOG_RESULTS="${LOG_RESULTS}"
  export VGGT_MAX_LIVE_STEPS="${MAX_LIVE_STEPS}"
  export VGGT_STOP_TOPIC="${STOP_TOPIC}"
  run_pipeline
'
SCRIPT

chmod +x "${RUN_DIR}/cpu_tunnel.sh" "${RUN_DIR}/cpu_bridge.sh" "${RUN_DIR}/gpu_pipeline.sh"

tmux new-session -d -s "${SESSION_NAME}" -n cpu-tunnel "bash '${RUN_DIR}/cpu_tunnel.sh'"
tmux new-window  -t "${SESSION_NAME}" -n cpu-bridge "bash '${RUN_DIR}/cpu_bridge.sh'"
tmux new-window  -t "${SESSION_NAME}" -n gpu-pipeline "bash '${RUN_DIR}/gpu_pipeline.sh'"

tmux select-window -t "${SESSION_NAME}:gpu-pipeline"

echo "[start_hpc_robot_pipeline_alloc_tmux] Session '${SESSION_NAME}' started."
echo "  windows: cpu-tunnel | cpu-bridge | gpu-pipeline"
echo "  attach : tmux attach -t ${SESSION_NAME}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  demo root : ${DEMO_ROOT}"

if [[ "${ATTACH}" == "1" ]]; then
  exec tmux attach -t "${SESSION_NAME}"
fi
