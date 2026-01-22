#!/usr/bin/env python3
"""
prepare_dataset.py
==================

Copies images, labels, and exported embeddings into a unified
dataset layout:

dataset_ready/
  <scene_id>/
    images/*.JPG
    labels/*.JPG.png
    chunks/*.pt or chunks/*.safetensors (+ sidecar .json)
    meta.json

This avoids juggling multiple roots during training.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack images/labels/embeddings into a unified dataset folder.")
    p.add_argument("--image-root", required=True, help="Root containing scene subfolders with RGB frames.")
    p.add_argument(
        "--image-subdir",
        default="dslr/resized_undistorted_images",
        help="Subdirectory under each scene where RGB frames live.",
    )
    p.add_argument("--label-root", required=True, help="Root containing label PNGs per scene.")
    p.add_argument("--feature-root", required=True, help="Root containing exported embeddings per scene.")
    p.add_argument(
        "--output-root",
        default="dataset_ready",
        help="Where to write the unified dataset (relative to CWD or absolute).",
    )
    p.add_argument("--scenes", help="Comma-separated list of scene ids (default: all in feature-root).")
    p.add_argument("--skip-existing", action="store_true", help="Skip scenes already present in output-root.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files inside a scene directory.")
    return p.parse_args()


def list_scenes(feature_root: Path, scenes_arg: Optional[str]) -> List[str]:
    if scenes_arg:
        return [s.strip() for s in scenes_arg.split(",") if s.strip()]
    return sorted([p.name for p in feature_root.iterdir() if p.is_dir()])


def copy_many(srcs: Iterable[Path], dst_dir: Path, overwrite: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in tqdm(list(srcs), desc=f"copy -> {dst_dir}", leave=False):
        dst = dst_dir / src.name
        if dst.exists():
            if not overwrite:
                continue
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    label_root = Path(args.label_root)
    feature_root = Path(args.feature_root)
    output_root = Path(args.output_root)
    scenes = list_scenes(feature_root, args.scenes)

    for scene in scenes:
        src_feat = feature_root / scene
        src_chunks = src_feat / "chunks"
        src_meta = src_feat / "meta.json"
        if not src_chunks.is_dir() or not src_meta.is_file():
            print(f"[skip {scene}] embeddings missing at {src_feat}")
            continue

        dest_scene = output_root / scene
        if args.skip_existing and dest_scene.is_dir():
            print(f"[skip {scene}] already present in {dest_scene}")
            continue

        src_images = image_root / scene / args.image_subdir
        src_labels = label_root / scene
        if not src_images.is_dir():
            print(f"[warn {scene}] images not found at {src_images}")
        if not src_labels.is_dir():
            print(f"[warn {scene}] labels not found at {src_labels}")

        # Copy images
        if src_images.is_dir():
            imgs = sorted(p for p in src_images.iterdir() if p.is_file())
            copy_many(imgs, dest_scene / "images", overwrite=args.overwrite)

        # Copy labels
        if src_labels.is_dir():
            lbls = sorted(p for p in src_labels.iterdir() if p.is_file())
            copy_many(lbls, dest_scene / "labels", overwrite=args.overwrite)

        # Copy chunks + meta
        chunk_files = (
            list(src_chunks.glob("*.pt"))
            + list(src_chunks.glob("*.safetensors"))
            + list(src_chunks.glob("*.json"))
        )
        copy_many(sorted(chunk_files), dest_scene / "chunks", overwrite=args.overwrite)
        dest_meta = dest_scene / "meta.json"
        dest_meta.parent.mkdir(parents=True, exist_ok=True)
        if dest_meta.exists() and args.overwrite:
            dest_meta.unlink()
        shutil.copy2(src_meta, dest_meta)
        print(f"[done] packed scene {scene} into {dest_scene}")


if __name__ == "__main__":
    main()
