#!/usr/bin/env bash
# shellcheck shell=bash

# Canonical dashboard launch profile for the live HPC pipeline.
# The dashboard exports VGGT_FINETUNE_CKPT / VGGT_DEMO_ROOT before sourcing this
# file, so those values can still be overridden from the UI. We keep working
# defaults here as a fallback for manual sourcing.

RUN_TAG="$(date +%Y%m%d_%H%M%S)"

: "${VGGT_FINETUNE_CKPT:=/home/s2984792/src/VGGT-SLAM/runs/train_selected60_mean_ce_fusioninit/checkpoints/film_m2f_epoch0020_step00782260_20260305_074546.pt}"
: "${VGGT_DEMO_ROOT:=/home/s2984792/src/VGGT-SLAM/demo}"

export VGGT_FINETUNE_CKPT
export VGGT_DEMO_ROOT

export VGGT_M2F_CFG=/home/s2984792/src/VGGT-SLAM/mask2former/configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml
export VGGT_M2F_WEIGHTS="${VGGT_FINETUNE_CKPT}"
export VGGT_FUSION_SCRIPT=/home/s2984792/src/VGGT-SLAM/scripts/train_film_m2f_optimized_png.py

export VGGT_SEMANTIC_HEAD=1
export VGGT_SEM_BACKEND=fusion
export VGGT_SEM_DPT_SOURCE=raw
export VGGT_SEM_CLASSES=60
export VGGT_SEM_QUERIES=100
export VGGT_SEM_FRAME_MODE=all
export VGGT_SEM_DINO_SCALE=1.0
export VGGT_SEM_TARGET_HW=original
export VGGT_SEM_STRIDE_MODE=s4

export VGGT_FUSE_FILM=0
export VGGT_SEM_LOG_EVERY=1
export VGGT_FILM_DELTA_LOG_EVERY=50

export VGGT_SEM_DEBUG=1
export VGGT_SEM_DEBUG_EVERY=5
export VGGT_SEM_DEBUG_SAVE=1
export VGGT_SEM_DEBUG_SAVE_MAX=20
export VGGT_SEM_DEBUG_DIR=/home/s2984792/src/VGGT-SLAM/debug/semantic_debug_live

export VGGT_DINO_PROJ_WEIGHTS=/home/s2984792/src/VGGT-SLAM/runs/proj_0271889ec0.pt
export VGGT_DINO_ALIGN_PHASE=auto

unset VGGT_SEM_INJECT_CHUNK_DIR
unset VGGT_SEM_INJECT_CHUNK_FORMAT
unset VGGT_SEM_INJECT_START_INDEX
unset VGGT_SEM_INJECT_CHUNK_INDEX

export VGGT_WINDOW_SIZE=15
export VGGT_OVERLAP_SIZE=1
export VGGT_TARGET_FPS=1.0
export VGGT_MAX_LOOPS=0
export VGGT_MAX_LIVE_STEPS=50

export VGGT_LOG_RESULTS=0

export VGGT_LIVE_TEMP_IMAGE_FORMAT=png
export VGGT_LIVE_TEMP_PNG_COMPRESSION=1
export VGGT_LIVE_TEMP_JPEG_QUALITY=95
export VGGT_MODEL_AUTOCAST=off

export HPC_ROBOT_MAX_WIDTH=0
export HPC_ROBOT_MAX_HEIGHT=0
export VGGT_STRICT_FORWARD=1

export VGGT_RUNTIME_METRICS=1
export VGGT_RUNTIME_METRICS_PATH="/home/s2984792/src/VGGT-SLAM/tap_logs/runtime_metrics_${RUN_TAG}.jsonl"

mkdir -p /home/s2984792/src/VGGT-SLAM/tap_logs
rm -f "${VGGT_RUNTIME_METRICS_PATH}"
