#!/usr/bin/env bash
set -euo pipefail

########################################
# ScanNet++ rasterize + semantics driver
#
# Usage:
#   scannetpp_run.sh --scenes 0a7cc12c0e,0b031f3119,...
#
# Assumes:
#   - scannetpp_env is defined (sets the conda env + CUDA etc.)
#   - ScanNet++ data lives under:  $HOME/data/scannetpp
#   - Toolbox repo lives under:    $HOME/src/scannetpp-toolbox
########################################

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --scenes SCENE1,SCENE2,...

Options:
  --scenes   Comma-separated list of scene IDs (no spaces)
  -h, --help Show this help

Environment (can be overridden):
  SCANNETPP_ROOT     (default: \$HOME/data/scannetpp)
  SCANNETPP_OUT      (default: \$HOME/outputs/scannetpp)
  SCANNETPP_TOOLBOX  (default: \$HOME/src/scannetpp-toolbox)
EOF
}

# ---------- config & arguments ----------

SCANNETPP_ROOT="${SCANNETPP_ROOT:-$HOME/data/scannetpp}"
SCANNETPP_OUT="${SCANNETPP_OUT:-$HOME/outputs/scannetpp}"
SCANNETPP_TOOLBOX="${SCANNETPP_TOOLBOX:-$HOME/src/scannetpp-toolbox}"

DATA_ROOT="$SCANNETPP_ROOT/data"
METADATA_DIR="$SCANNETPP_ROOT/metadata"
SPLITS_DIR="$SCANNETPP_ROOT/splits"
SCENE_LIST_FILE="$SPLITS_DIR/scene_list_auto.txt"

SCENES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenes)
      SCENES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SCENES" ]]; then
  echo "[ERR] You must pass --scenes SCENE1,SCENE2,..." >&2
  usage
  exit 1
fi

# ---------- sanity checks ----------

if [[ ! -d "$SCANNETPP_ROOT" ]]; then
  echo "[ERR] SCANNETPP_ROOT does not exist: $SCANNETPP_ROOT" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERR] DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
fi

if [[ ! -d "$SCANNETPP_TOOLBOX" ]]; then
  echo "[ERR] Toolbox not found at: $SCANNETPP_TOOLBOX" >&2
  exit 1
fi

mkdir -p "$SPLITS_DIR" "$SCANNETPP_OUT" "$METADATA_DIR"

# ---------- create scene list file ----------

echo "$SCENES" | tr ',' '\n' > "$SCENE_LIST_FILE"
N_SCENES=$(wc -l < "$SCENE_LIST_FILE" | tr -d ' ')
echo "Scene list: $SCENE_LIST_FILE"
echo "$N_SCENES $(cat "$SCENE_LIST_FILE" | tr '\n' ' ')"

# ---------- activate env & PYTHONPATH ----------

if command -v scannetpp_env >/dev/null 2>&1; then
  scannetpp_env
else
  echo "[WARN] scannetpp_env not found in PATH; assuming env is already active." >&2
fi

export PYTHONPATH="$SCANNETPP_TOOLBOX:${HOME}/pydeps:${PYTHONPATH:-}"

# ---------- ensure semantic class & palette files ----------

CLASSES_TXT="$METADATA_DIR/semantic_classes.txt"
PALETTE_TXT="$METADATA_DIR/semantic_palette.txt"

if [[ ! -f "$CLASSES_TXT" ]]; then
  echo "[ERR] semantic_classes.txt not found at: $CLASSES_TXT" >&2
  echo "      (You already had this file on ctit086, so copy/symlink it here.)" >&2
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

golden = 0.61803398875
h = 0.0
colors = []
for _ in range(n):
    h = (h + golden) % 1.0
    colors.append(hsv2rgb(h, 0.95, 1.0))

palette = np.zeros((max(n, 256), 3), np.uint8)
palette[:n] = np.array(colors, np.uint8)
palette[0] = [0, 0, 0]        # background
palette[255] = [255, 255, 255]  # reserved/unknown

np.savetxt(out_path, palette[:256], fmt="%d")
print(f"Wrote palette: {out_path} rows: {len(palette[:256])}")
PY
fi

# ---------- step 1: rasterization ----------

echo
echo "=== [1/2] Rasterization ==="
python -m semantic.prep.rasterize \
  ++data_root="$DATA_ROOT" \
  ++scene_list_file="$SCENE_LIST_FILE" \
  ++rasterout_dir="$SCANNETPP_OUT" \
  ++image_type=dslr \
  ++undistort_dslr=true \
  ++image_downsample_factor=1 \
  ++subsample_factor=1 \
  ++batch_size=6

# ---------- step 2: semantics 2D ----------

echo
echo "=== [2/2] Semantics 2D ==="
python -m semantic.prep.semantics_2d \
  ++data_root="$DATA_ROOT" \
  ++dataset_root="$SCANNETPP_ROOT" \
  ++scene_list_file="$SCENE_LIST_FILE" \
  ++rasterout_dir="$SCANNETPP_OUT" \
  ++visiblity_cache_dir="$SCANNETPP_OUT" \
  ++save_dir_root="$SCANNETPP_OUT" \
  ++save_dir="semantics_2d" \
  ++image_type=dslr \
  ++undistort_dslr=true \
  ++subsample_factor=1 \
  ++save_semantic_gt_2d=true \
  ++save_objid_gt_2d=true \
  ++viz_semantic_gt_2d=false \
  ++skip_existing_semantic_gt_2d=true \
  ++semantic_classes_file="$CLASSES_TXT" \
  ++semantic_2d_palette_path="$PALETTE_TXT"

echo
echo "=== Done ==="
echo "Raster + semantics written under: $SCANNETPP_OUT"
echo "Scenes: $(tr '\n' ' ' < "$SCENE_LIST_FILE")"