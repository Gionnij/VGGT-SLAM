#!/usr/bin/env bash
set -euo pipefail

###############################################
# ScanNet++ per-scene rasterize + semantics
# This runs the heavy rasterization + 2D label
# projection for one scene at a time, keeps
# only the final PNG masks, and deletes bulky
# intermediates (obj_id .pth, crops, etc.).
#
# Usage:
#   run_semantics_per_scene.sh --scene-list exported_scenes.txt
# or
#   run_semantics_per_scene.sh --scenes 0a7...,0b0...,...
#
# Env overrides (same defaults as scannetpp_run.sh):
#   SCANNETPP_ROOT     (default: $HOME/data/scannetpp)
#   SCANNETPP_OUT      (default: $HOME/outputs/scannetpp)
#   SCANNETPP_TOOLBOX  (default: $HOME/src/scannetpp-toolbox)
###############################################

usage() {
  cat <<'EOF'
Usage:
  run_semantics_per_scene.sh --scene-list FILE
  run_semantics_per_scene.sh --scenes SCENE1,SCENE2,...

Options:
  --scene-list   Path to file with one scene id per line.
  --scenes       Comma-separated list of scene ids (no spaces).
  -h, --help     Show this help message and exit.

Environment (can be overridden):
  SCANNETPP_ROOT     (default: $HOME/data/scannetpp)
  SCANNETPP_OUT      (default: $HOME/outputs/scannetpp)
  SCANNETPP_TOOLBOX  (default: $HOME/src/scannetpp-toolbox)
EOF
}

SCANNETPP_ROOT="${SCANNETPP_ROOT:-$HOME/data/scannetpp}"
SCANNETPP_OUT="${SCANNETPP_OUT:-$HOME/outputs/scannetpp}"
SCANNETPP_TOOLBOX="${SCANNETPP_TOOLBOX:-$HOME/src/scannetpp-toolbox}"

DATA_ROOT="$SCANNETPP_ROOT/data"
METADATA_DIR="$SCANNETPP_ROOT/metadata"
SPLITS_DIR="$SCANNETPP_ROOT/splits"
mkdir -p "$SPLITS_DIR" "$SCANNETPP_OUT" "$METADATA_DIR"

SCENE_FILE=""
SCENE_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-list)
      SCENE_FILE="$2"
      shift 2
      ;;
    --scenes)
      SCENE_ARG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SCENE_FILE" && -z "$SCENE_ARG" ]]; then
  echo "[ERR] Provide --scene-list or --scenes." >&2
  usage
  exit 1
fi

if [[ -n "$SCENE_FILE" && ! -f "$SCENE_FILE" ]]; then
  echo "[ERR] Scene list file not found: $SCENE_FILE" >&2
  exit 1
fi

if [[ ! -d "$SCANNETPP_ROOT" ]]; then
  echo "[ERR] SCANNETPP_ROOT does not exist: $SCANNETPP_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERR] DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SCANNETPP_TOOLBOX" ]]; then
  echo "[ERR] SCANNETPP_TOOLBOX missing: $SCANNETPP_TOOLBOX" >&2
  exit 1
fi

# Activate toolbox env if helper exists.
if command -v scannetpp_env >/dev/null 2>&1; then
  scannetpp_env
else
  echo "[WARN] scannetpp_env not found; assuming environment already active." >&2
fi
export PYTHONPATH="$SCANNETPP_TOOLBOX:${HOME}/pydeps:${PYTHONPATH:-}"

CLASSES_TXT="$METADATA_DIR/semantic_classes.txt"
PALETTE_TXT="$METADATA_DIR/semantic_palette.txt"
if [[ ! -f "$CLASSES_TXT" ]]; then
  echo "[ERR] semantic_classes.txt missing at $CLASSES_TXT" >&2
  exit 1
fi
if [[ ! -f "$PALETTE_TXT" ]]; then
  echo "[INFO] semantic_palette.txt not found — generating one..."
  SCANNETPP_CLASSES="$CLASSES_TXT" \
  SCANNETPP_PALETTE="$PALETTE_TXT" \
  python - <<'PY'
import os, numpy as np
cls_path = os.environ["SCANNETPP_CLASSES"]
out_path = os.environ["SCANNETPP_PALETTE"]
with open(cls_path) as f:
    classes = [ln.strip() for ln in f if ln.strip()]
n = len(classes)
golden = 0.61803398875
h = 0.0
colors = []
def hsv2rgb(h, s, v):
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    r, g, b = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][i]
    return [int(255 * r), int(255 * g), int(255 * b)]
for _ in range(n):
    h = (h + golden) % 1.0
    colors.append(hsv2rgb(h, 0.95, 1.0))
palette = np.zeros((max(n, 256), 3), np.uint8)
palette[:n] = np.array(colors, np.uint8)
palette[0] = [0, 0, 0]
palette[255] = [255, 255, 255]
np.savetxt(out_path, palette[:n], fmt="%d")
print(f"[palette] wrote {out_path} rows={len(palette[:n])}")
PY
fi

read_scene_list() {
  if [[ -n "$SCENE_FILE" ]]; then
    grep -v '^\s*$' "$SCENE_FILE" | tr -d '\r'
  else
    echo "$SCENE_ARG" | tr ',' '\n' | grep -v '^\s*$'
  fi
}

cleanup_scene_artifacts() {
  local scene="$1"
  local raster_dir="$SCANNETPP_OUT/dslr/$scene"
  if [[ -d "$raster_dir" ]]; then
    rm -rf "$raster_dir"
  fi
  local extras=(img_crops img_crops_nobg img_bbox viz_obj_ids viz_obj_ids_txt obj_ids undistorted obj_pcs semantics_viz)
  for d in "${extras[@]}"; do
    local target="$SCANNETPP_OUT/semantics_2d/$d/$scene"
    if [[ -d "$target" ]]; then
      rm -rf "$target"
    fi
  done
}

scene_has_prereqs() {
  local scene="$1"
  local scene_root="$DATA_ROOT/$scene"
  local mesh_ply="$scene_root/scans/mesh_aligned_0.05.ply"
  local cameras_txt="$scene_root/dslr/colmap/cameras.txt"
  local anno_json="$scene_root/scans/segments_anno.json"

  local missing=0
  if [[ ! -f "$mesh_ply" ]]; then
    echo "[warn $scene] missing mesh file: $mesh_ply"
    missing=1
  fi
  if [[ ! -f "$cameras_txt" ]]; then
    echo "[warn $scene] missing COLMAP cameras: $cameras_txt"
    missing=1
  fi
  if [[ ! -f "$anno_json" ]]; then
    echo "[warn $scene] missing semantic annotation: $anno_json"
    missing=1
  fi

  return $missing
}

process_scene() {
  local scene="$1"
  if ! scene_has_prereqs "$scene"; then
    echo "[skip $scene] prerequisites missing, moving to next scene."
    return
  fi

  local tmp_list
  tmp_list="$(mktemp "$SPLITS_DIR/scene_${scene}_XXXX.txt")"
  echo "$scene" > "$tmp_list"

  echo
  echo "=== [$scene] Rasterization ==="
  if ! python -m semantic.prep.rasterize \
    ++data_root="$DATA_ROOT" \
    ++scene_list_file="$tmp_list" \
    ++rasterout_dir="$SCANNETPP_OUT" \
    ++image_type=dslr \
    ++undistort_dslr=true \
    ++image_downsample_factor=1 \
    ++subsample_factor=1 \
    ++batch_size=6; then
    echo "[warn $scene] rasterization failed, skipping scene."
    rm -f "$tmp_list"
    cleanup_scene_artifacts "$scene"
    return
  fi

  echo
  echo "=== [$scene] Semantics 2D ==="
  if ! python -m semantic.prep.semantics_2d \
    ++data_root="$DATA_ROOT" \
    ++dataset_root="$SCANNETPP_ROOT" \
    ++scene_list_file="$tmp_list" \
    ++rasterout_dir="$SCANNETPP_OUT" \
    ++visiblity_cache_dir="$SCANNETPP_OUT" \
    ++save_dir_root="$SCANNETPP_OUT" \
    ++save_dir="semantics_2d" \
    ++image_type=dslr \
    ++undistort_dslr=true \
    ++subsample_factor=1 \
    ++save_semantic_gt_2d=true \
    ++save_objid_gt_2d=false \
    ++viz_semantic_gt_2d=false \
    ++dbg.viz_obj_ids=false \
    ++viz_obj_ids_txt=false \
    ++save_undistorted_images=false \
    ++skip_existing_semantic_gt_2d=false \
    ++semantic_classes_file="$CLASSES_TXT" \
    ++semantic_2d_palette_path="$PALETTE_TXT"; then
    echo "[warn $scene] semantics projection failed, skipping scene."
    rm -f "$tmp_list"
    cleanup_scene_artifacts "$scene"
    return
  fi

  rm -f "$tmp_list"
  cleanup_scene_artifacts "$scene"

  echo "[done $scene] PNG masks at $SCANNETPP_OUT/semantics_2d/semantics/$scene"
}

while read -r scene_id; do
  [[ -z "$scene_id" ]] && continue
  process_scene "$scene_id"
done < <(read_scene_list)

echo
echo "[all scenes completed] PNG masks stored in $SCANNETPP_OUT/semantics_2d/semantics/"
