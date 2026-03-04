#!/usr/bin/env bash
set -euo pipefail

# Start a tmux session with 3 windows for robot->HPC streaming + VGGT:
# 1) hpc_robot_stream_tunnel
# 2) hpc_robot_bridge
# 3) run_vggt
#
# Prereq on robot: reverse tunnel must be active, e.g.
#   ssh -N -R 15001:127.0.0.1:5001 s2984792@hpc-head1.ewi.utwente.nl

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
BASHRC_SHARED="${REPO_ROOT}/shell/bashrc_shared"

SESSION_NAME="${SESSION_NAME:-vggt_robot}"
LOCAL_PORT="${LOCAL_PORT:-15001}"
HEAD_PORT="${HEAD_PORT:-15001}"
FPS="${FPS:-2.0}"
ATTACH="1"
FORCE_KILL="0"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  -s, --session NAME      tmux session name (default: ${SESSION_NAME})
  -l, --local-port PORT   local tunnel port on compute/gpu node (default: ${LOCAL_PORT})
  -p, --head-port PORT    port opened on HPC head by robot reverse tunnel (default: ${HEAD_PORT})
  -f, --fps FPS           bridge publish rate (default: ${FPS})
      --no-attach         do not attach automatically
  -k, --kill-existing     kill existing session with same name
  -h, --help              show this help
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
  echo "[start_hpc_robot_vggt_tmux] tmux not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${BASHRC_SHARED}" ]]; then
  echo "[start_hpc_robot_vggt_tmux] Missing ${BASHRC_SHARED}" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [[ "${FORCE_KILL}" == "1" ]]; then
    tmux kill-session -t "${SESSION_NAME}"
  else
    echo "[start_hpc_robot_vggt_tmux] Session '${SESSION_NAME}' already exists." >&2
    echo "Use -k to replace it, or attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
  fi
fi

# Window 1: tunnel from compute/gpu node to head loopback
CMD_TUNNEL="source '${BASHRC_SHARED}'; hpc_robot_stream_tunnel '${LOCAL_PORT}' '${HEAD_PORT}'"
# Window 2: TCP->ROS2 bridge for /camera/image_color + /camera/camera_info
CMD_BRIDGE="source '${BASHRC_SHARED}'; hpc_robot_bridge 127.0.0.1 '${LOCAL_PORT}' '${FPS}'"
# Window 3: main VGGT live pipeline subscriber
CMD_VGGT="source '${BASHRC_SHARED}'; run_vggt"

tmux new-session -d -s "${SESSION_NAME}" -n tunnel "bash -lc \"${CMD_TUNNEL}\""
tmux new-window  -t "${SESSION_NAME}" -n bridge "bash -lc \"${CMD_BRIDGE}\""
tmux new-window  -t "${SESSION_NAME}" -n vggt   "bash -lc \"${CMD_VGGT}\""

tmux select-window -t "${SESSION_NAME}:vggt"

echo "[start_hpc_robot_vggt_tmux] Session '${SESSION_NAME}' started."
echo "  windows: tunnel | bridge | vggt"
echo "  attach : tmux attach -t ${SESSION_NAME}"

if [[ "${ATTACH}" == "1" ]]; then
  exec tmux attach -t "${SESSION_NAME}"
fi
