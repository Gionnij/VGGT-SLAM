#!/usr/bin/env python3
"""
remap_labels_dense.py
=====================

Build a new dataset root with labels remapped to a dense class index.
Original `dataset_ready` is left untouched.

For each selected scene, this script:
  - Reads all label PNGs under <in_root>/<scene>/labels
  - Collects unique label ids (excluding ignore_value)
  - Builds a mapping orig_id -> [0..K-1]
  - Writes new masks under <out_root>/<scene>/labels with remapped ids
  - Symlinks images/ and chunks/ from the input root

Usage (single scene overfit):
  python scripts/remap_labels_dense.py \
    --in-root /home/s2984792/src/VGGT-SLAM/dataset_ready \
    --out-root /home/s2984792/src/VGGT-SLAM/dataset_ready_dense \
    --scenes 0a7cc12c0e \
    --ignore-value 65535

Then train with:
  --dataset-root /home/s2984792/src/VGGT-SLAM/dataset_ready_dense
  --num-classes <K> (printed by this script)
  --ignore-index 65535
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remap ScanNet++ labels to a dense id space.")
    p.add_argument(
        "--in-root",
        required=True,
        help="Existing dataset_ready root (with <scene>/images, <scene>/labels, <scene>/chunks).",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="New root where remapped dataset will be written.",
    )
    p.add_argument(
        "--scenes",
        help="Comma-separated list of scene ids to process. Default: all scenes under in-root.",
    )
    p.add_argument(
        "--ignore-value",
        type=int,
        default=65535,
        help="Label value used as ignore (kept unchanged in remapped masks).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute and print mapping, do not write any files.",
    )
    return p.parse_args()


def collect_scenes(in_root: Path, scenes_arg: Optional[str]) -> List[Path]:
    if scenes_arg:
        names = [s.strip() for s in scenes_arg.split(",") if s.strip()]
        dirs = [in_root / n for n in names]
    else:
        dirs = [p for p in in_root.iterdir() if p.is_dir()]
    # keep only those that look like dataset_ready scenes
    return sorted([d for d in dirs if (d / "labels").is_dir()])


def find_unique_labels(scene_dirs: List[Path], ignore_val: int) -> Dict[str, Set[int]]:
    per_scene: Dict[str, Set[int]] = {}
    for scene_dir in scene_dirs:
        sid = scene_dir.name
        lbl_dir = scene_dir / "labels"
        seen: Set[int] = set()
        for p in lbl_dir.glob("*.png"):
            arr = np.array(Image.open(p), copy=False)
            vals = np.unique(arr)
            seen.update(vals.tolist())
        if ignore_val in seen:
            seen.remove(ignore_val)
        per_scene[sid] = seen
    return per_scene


def build_global_mapping(per_scene: Dict[str, Set[int]]) -> Dict[int, int]:
    all_ids: Set[int] = set()
    for ids in per_scene.values():
        all_ids.update(ids)
    sorted_ids = sorted(all_ids)
    return {orig: new for new, orig in enumerate(sorted_ids)}


def remap_mask(arr: np.ndarray, mapping: Dict[int, int], ignore_val: int) -> np.ndarray:
    """
    arr: original label array (uint16)
    mapping: orig_id -> new_id
    ignore_val: keep as-is
    """
    out = np.full_like(arr, fill_value=ignore_val, dtype=np.uint16)
    # For each original id, assign its new id
    for orig, new in mapping.items():
        out[arr == orig] = np.uint16(new)
    # ignore_val remains ignore_val
    return out


def ensure_symlink(src: Path, dst: Path) -> None:
    """
    Create a symlink dst -> src if dst does not exist.
    If dst exists and is already the correct symlink, do nothing.
    """
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Use relative symlink if possible
    try:
        rel = src.relative_to(dst.parent)
        dst.symlink_to(rel)
    except ValueError:
        dst.symlink_to(src)


def main() -> None:
    args = parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = collect_scenes(in_root, args.scenes)
    if not scene_dirs:
        raise RuntimeError("No scenes found under in-root with labels/ present.")

    per_scene = find_unique_labels(scene_dirs, ignore_val=args.ignore_value)
    mapping = build_global_mapping(per_scene)

    print("=== Label remapping summary ===")
    print(f"in-root:  {in_root}")
    print(f"out-root: {out_root}")
    print(f"ignore_value: {args.ignore_value}")
    print(f"scenes: {[d.name for d in scene_dirs]}")
    print(f"total distinct non-ignore ids: {len(mapping)}")
    if mapping:
        orig_ids = sorted(mapping.keys())
        print(f"min orig id: {orig_ids[0]}, max orig id: {orig_ids[-1]}")
        print("sample mapping (orig -> new):")
        for orig in orig_ids[:10]:
            print(f"  {orig} -> {mapping[orig]}")
    else:
        print("WARNING: no non-ignore labels found; nothing to remap.")

    # Save mapping to JSON for reference
    mapping_path = out_root / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ignore_value": args.ignore_value,
                "mapping": {str(k): int(v) for k, v in mapping.items()},
                "scenes": [d.name for d in scene_dirs],
            },
            f,
            indent=2,
        )
    print(f"Saved mapping to {mapping_path}")

    if args.dry_run:
        print("Dry run requested, skipping file writes.")
        return

    # Remap labels and mirror structure
    for scene_dir in scene_dirs:
        sid = scene_dir.name
        print(f"[scene {sid}] remapping labels and linking images/chunks...")
        out_scene = out_root / sid

        # Symlink images and chunks from original root
        for sub in ("images", "chunks"):
            src = scene_dir / sub
            if src.is_dir():
                dst = out_scene / sub
                dst.mkdir(parents=True, exist_ok=True)
                for p in src.iterdir():
                    ensure_symlink(p, dst / p.name)

        # Remap labels
        lbl_in = scene_dir / "labels"
        lbl_out = out_scene / "labels"
        lbl_out.mkdir(parents=True, exist_ok=True)

        for p in lbl_in.glob("*.png"):
            arr = np.array(Image.open(p), copy=False)
            arr_remap = remap_mask(arr, mapping, ignore_val=args.ignore_value)
            out_path = lbl_out / p.name
            Image.fromarray(arr_remap).save(out_path)

        # Copy meta.json if present
        meta_in = scene_dir / "meta.json"
        if meta_in.is_file():
            meta_out = out_scene / "meta.json"
            meta_out.parent.mkdir(parents=True, exist_ok=True)
            meta_out.write_bytes(meta_in.read_bytes())

    print("Done. Use num-classes =", len(mapping), "for training on this remapped dataset.")


if __name__ == "__main__":
    main()
