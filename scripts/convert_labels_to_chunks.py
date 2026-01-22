#!/usr/bin/env python3
"""
convert_labels_to_chunks.py
===========================

Offline utility to pack per-frame label PNGs into chunk-aligned tensor files.
This avoids decoding hundreds of small PNGs during training and keeps label
I/O aligned with the pre-exported embedding chunks.

Default layout (mirrors export_embeddings.py output):
  <dataset_root>/<scene>/chunks/*.pt or *.safetensors (+ .json)  # embeddings (input)
  <dataset_root>/<scene>/labels/*.png        # label PNGs (input)
  <dataset_root>/<scene>/label_chunks/*.pt   # label tensors (output)

Each output file stores the labels for exactly one embedding chunk in the same
frame order as chunk["frame_paths"].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert label PNGs into chunked tensor files for faster training.")
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Root containing per-scene folders with chunks/ and labels/.",
    )
    p.add_argument(
        "--labels-root",
        help="Optional separate root containing labels/<scene>/*.png. Defaults to <dataset-root>.",
    )
    p.add_argument(
        "--output-root",
        help="Where to write label chunks. Defaults to <dataset-root> (creates <scene>/label_chunks).",
    )
    p.add_argument(
        "--chunk-subdir",
        default="chunks",
        help="Subdirectory under each scene where embedding chunks live.",
    )
    p.add_argument(
        "--label-chunk-subdir",
        default="label_chunks",
        help="Subdirectory to create under each scene for label chunk tensors.",
    )
    p.add_argument(
        "--scenes",
        help="Comma-separated list of scene ids to process (default: all scenes under dataset-root).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label chunk files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the work without writing any files.",
    )
    return p.parse_args()


def list_scenes(root: Path, scenes_arg: Optional[str]) -> List[str]:
    if scenes_arg:
        return [s.strip() for s in scenes_arg.split(",") if s.strip()]
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def label_path_for_frame(scene_dir: Path, frame_path: str, labels_root: Optional[Path]) -> Path:
    fname = Path(frame_path).name + ".png"
    if labels_root:
        return labels_root / scene_dir.name / fname
    return scene_dir / "labels" / fname


def load_label_png(path: Path) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    arr = np.array(Image.open(path), copy=True)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Expected single-channel label, got shape {arr.shape} at {path}")
    return torch.from_numpy(arr).long()


def iter_chunks(scene_dir: Path, chunk_subdir: str) -> Iterable[Path]:
    chunk_dir = scene_dir / chunk_subdir
    if not chunk_dir.is_dir():
        return []
    pt_files = sorted(p for p in chunk_dir.glob("*.pt") if p.is_file())
    st_files = sorted(p for p in chunk_dir.glob("*.safetensors") if p.is_file())
    st_stems = {p.stem for p in st_files}
    pt_files = [p for p in pt_files if p.stem not in st_stems]
    return sorted(st_files + pt_files)


def load_chunk_metadata(chunk_path: Path) -> Dict:
    if chunk_path.suffix == ".safetensors":
        json_path = chunk_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing chunk metadata JSON for {chunk_path}")
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return torch.load(chunk_path, map_location="cpu")


def convert_chunk(
    *,
    chunk_path: Path,
    scene_dir: Path,
    labels_root: Optional[Path],
    output_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> Optional[Path]:
    out_path = output_dir / chunk_path.name
    if out_path.exists() and not overwrite:
        return out_path

    chk = load_chunk_metadata(chunk_path)
    frame_paths: Sequence[str] = chk.get("frame_paths") or []
    image_size = chk.get("image_size")
    if not frame_paths:
        print(f"[warn] chunk {chunk_path} missing frame_paths; skipping.")
        return None

    labels: List[torch.Tensor] = []
    missing = []
    for fpath in frame_paths:
        lbl_path = label_path_for_frame(scene_dir, fpath, labels_root)
        if not lbl_path.is_file():
            missing.append(lbl_path)
            continue
        labels.append(load_label_png(lbl_path))

    if missing:
        print(f"[warn] {chunk_path.name}: missing {len(missing)} labels, first missing: {missing[0]}")
        return None

    stacked = torch.stack(labels, dim=0).long()
    payload: Dict = {
        "scene_id": chk.get("scene_id", scene_dir.name),
        "chunk_name": chunk_path.name,
        "chunk_path": str(chunk_path),
        "frame_paths": list(frame_paths),
        "image_size": image_size,
        "labels": stacked,
        "label_dtype": str(stacked.dtype),
    }

    if dry_run:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    labels_root = Path(args.labels_root).expanduser() if args.labels_root else None
    output_root = Path(args.output_root).expanduser() if args.output_root else dataset_root

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset-root not found: {dataset_root}")
    if labels_root and not labels_root.is_dir():
        raise FileNotFoundError(f"labels-root not found: {labels_root}")
    if not output_root.is_dir() and not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    scenes = list_scenes(dataset_root, args.scenes)
    if not scenes:
        raise RuntimeError(f"No scenes found under {dataset_root}")

    for scene in scenes:
        scene_dir = dataset_root / scene
        out_dir = output_root / scene / args.label_chunk_subdir
        chunks = list(iter_chunks(scene_dir, args.chunk_subdir))
        if not chunks:
            print(f"[skip {scene}] no chunks found under {scene_dir / args.chunk_subdir}")
            continue

        for chk_path in tqdm(chunks, desc=f"{scene} chunks", leave=False):
            dst = convert_chunk(
                chunk_path=chk_path,
                scene_dir=scene_dir,
                labels_root=labels_root,
                output_dir=out_dir,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            if dst is None:
                continue
            if args.dry_run:
                print(f"[dry-run] would write {dst}")
        print(f"[done {scene}] processed {len(chunks)} chunks -> {out_dir}")


if __name__ == "__main__":
    main()
