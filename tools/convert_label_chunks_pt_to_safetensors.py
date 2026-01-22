#!/usr/bin/env python3
"""
convert_label_chunks_pt_to_safetensors.py
========================================

Convert label chunk .pt files to .safetensors with a JSON sidecar.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert label chunk .pt files to .safetensors + JSON metadata.")
    p.add_argument("--src-dir", required=True, help="Root directory containing <scene>/label_chunks/*.pt files.")
    p.add_argument("--dst-dir", required=True, help="Output root for converted label chunks.")
    p.add_argument(
        "--label-chunk-subdir",
        default="label_chunks",
        help="Subdirectory under each scene where label chunks live.",
    )
    p.add_argument("--scenes", help="Comma-separated list of scene ids to process.")
    p.add_argument("--scenes-file", help="Optional file with one scene id per line to process.")
    p.add_argument("--dtype", choices=["int32", "int64"], default="int32", help="Tensor dtype to store.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .safetensors outputs.")
    p.add_argument("--workers", type=int, default=1, help="Number of worker threads.")
    p.add_argument("--verify", action="store_true", help="Verify a few converted files by comparing shapes/dtypes.")
    return p.parse_args()


def _target_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "int64":
        return torch.int64
    return torch.int32


def _output_paths(src_path: Path, src_root: Path, dst_root: Path) -> Tuple[Path, Path]:
    rel = src_path.relative_to(src_root)
    out_path = (dst_root / rel).with_suffix(".safetensors")
    meta_path = out_path.with_suffix(".json")
    return out_path, meta_path


def _load_scenes(args: argparse.Namespace, src_root: Path) -> List[str]:
    if args.scenes_file:
        path = Path(args.scenes_file).expanduser()
        scenes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            scenes.append(name)
        return scenes
    if args.scenes:
        return [s.strip() for s in args.scenes.split(",") if s.strip()]
    return sorted([p.name for p in src_root.iterdir() if p.is_dir()])


def iter_label_chunks_by_scene(
    src_root: Path, scenes: Sequence[str], label_chunk_subdir: str
) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for scene in scenes:
        base = src_root / scene / label_chunk_subdir
        if not base.is_dir():
            continue
        files = sorted(p for p in base.glob("*.pt") if p.is_file())
        if files:
            out[scene] = files
    return out


def convert_one(
    src_path: Path,
    src_root: Path,
    dst_root: Path,
    dtype: str,
    overwrite: bool,
) -> str:
    out_path, meta_path = _output_paths(src_path, src_root, dst_root)
    if out_path.exists() and not overwrite:
        return "skip"
    payload = torch.load(src_path, map_location="cpu")
    if "labels" not in payload:
        raise KeyError(f"Label chunk missing 'labels' tensor: {src_path}")
    labels = payload["labels"]
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"labels is not a tensor in {src_path}")
    target = _target_dtype(dtype)
    labels = labels.to(dtype=target).contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"labels": labels}, str(out_path))

    meta: Dict = {
        "format": "label_safetensors_v1",
        "scene_id": payload.get("scene_id"),
        "chunk_name": payload.get("chunk_name", src_path.name),
        "chunk_path": payload.get("chunk_path"),
        "frame_paths": payload.get("frame_paths"),
        "image_size": payload.get("image_size"),
        "labels_shape": list(labels.shape),
        "tensor_dtype": dtype,
        "source_chunk": str(src_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return "ok"


def verify_one(src_path: Path, src_root: Path, dst_root: Path) -> None:
    out_path, meta_path = _output_paths(src_path, src_root, dst_root)
    payload = torch.load(src_path, map_location="cpu")
    tensors = load_file(str(out_path))
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    expected_dtype = meta.get("tensor_dtype")
    if expected_dtype == "int64":
        target = torch.int64
    else:
        target = torch.int32
    labels = tensors.get("labels")
    if labels is None:
        raise KeyError(f"Missing labels in {out_path}")
    if labels.shape != payload["labels"].shape:
        raise ValueError(f"labels shape mismatch for {src_path}")
    if labels.dtype != target:
        raise ValueError(f"labels dtype mismatch for {src_path}")


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_dir).expanduser()
    dst_root = Path(args.dst_dir).expanduser()
    if not src_root.is_dir():
        raise FileNotFoundError(f"src-dir not found: {src_root}")
    scenes = _load_scenes(args, src_root)
    scene_chunks = iter_label_chunks_by_scene(src_root, scenes, args.label_chunk_subdir)
    if not scene_chunks:
        raise FileNotFoundError(f"No label chunk .pt files found under {src_root}")
    all_chunks = [p for files in scene_chunks.values() for p in files]

    workers = max(1, int(args.workers))
    results = {"ok": 0, "skip": 0, "fail": 0}

    scenes_bar = tqdm(total=len(scene_chunks), desc="scenes", unit="scene")
    for scene, chunk_files in scene_chunks.items():
        if workers == 1:
            for src_path in tqdm(chunk_files, desc=scene, leave=False):
                try:
                    status = convert_one(src_path, src_root, dst_root, args.dtype, args.overwrite)
                    results[status] += 1
                except Exception as exc:
                    results["fail"] += 1
                    print(f"[fail] {src_path}: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(convert_one, src, src_root, dst_root, args.dtype, args.overwrite): src
                    for src in chunk_files
                }
                for fut in tqdm(as_completed(futures), total=len(futures), desc=scene, leave=False):
                    src_path = futures[fut]
                    try:
                        status = fut.result()
                        results[status] += 1
                    except Exception as exc:
                        results["fail"] += 1
                        print(f"[fail] {src_path}: {exc}")
        scenes_bar.update(1)
    scenes_bar.close()

    print(
        f"[done] converted {len(all_chunks)} files: "
        f"ok={results['ok']} skip={results['skip']} fail={results['fail']}"
    )

    if args.verify:
        sample = all_chunks[: min(3, len(all_chunks))]
        for src_path in sample:
            verify_one(src_path, src_root, dst_root)
            print(f"[verify] ok: {src_path}")


if __name__ == "__main__":
    main()
