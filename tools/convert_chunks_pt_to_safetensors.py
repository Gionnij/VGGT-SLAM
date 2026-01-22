#!/usr/bin/env python3
"""
convert_chunks_pt_to_safetensors.py
==================================

Convert exported chunk .pt files to .safetensors with a JSON sidecar for metadata.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert VGGT chunk .pt files to .safetensors + JSON metadata.")
    p.add_argument("--src-dir", required=True, help="Root directory containing <scene>/chunks/*.pt files.")
    p.add_argument("--dst-dir", required=True, help="Output root for converted chunks.")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="Tensor dtype to store.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .safetensors outputs.")
    p.add_argument("--workers", type=int, default=1, help="Number of worker threads.")
    p.add_argument("--verify", action="store_true", help="Verify a few converted files by comparing shapes/dtypes.")
    return p.parse_args()


def discover_chunks(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("chunks/*.pt") if p.parent.name == "chunks"])


def _target_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def _cast_tensor(tensor: torch.Tensor, target: torch.dtype) -> torch.Tensor:
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=target)
    return tensor.contiguous()


def _output_paths(src_path: Path, src_root: Path, dst_root: Path) -> Tuple[Path, Path]:
    rel = src_path.relative_to(src_root)
    out_path = (dst_root / rel).with_suffix(".safetensors")
    meta_path = out_path.with_suffix(".json")
    return out_path, meta_path


def convert_one(src_path: Path, src_root: Path, dst_root: Path, dtype: str, overwrite: bool) -> str:
    out_path, meta_path = _output_paths(src_path, src_root, dst_root)
    if out_path.exists() and not overwrite:
        return "skip"
    payload = torch.load(src_path, map_location="cpu")
    if "dino_features" not in payload or "dpt_pyramid" not in payload:
        raise KeyError(f"Missing required keys in {src_path}")
    frame_paths = payload.get("frame_paths")
    if frame_paths is None:
        raise KeyError(f"Missing frame_paths in {src_path}")

    target = _target_dtype(dtype)
    dino = _cast_tensor(payload["dino_features"], target)
    dpt_pyramid = payload["dpt_pyramid"]
    if not isinstance(dpt_pyramid, (list, tuple)):
        raise TypeError(f"dpt_pyramid has unexpected type {type(dpt_pyramid)} in {src_path}")
    tensors: Dict[str, torch.Tensor] = {"dino_features": dino}
    dpt_keys = []
    for i, level in enumerate(dpt_pyramid):
        key = f"dpt_pyramid_{i}"
        tensors[key] = _cast_tensor(level, target)
        dpt_keys.append(key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))

    meta = {
        "format": "safetensors_v1",
        "scene_id": payload.get("scene_id"),
        "frame_start": int(payload.get("frame_start", 0)),
        "frame_end": int(payload.get("frame_end", len(frame_paths))),
        "num_frames": len(frame_paths),
        "frame_paths": frame_paths,
        "image_size": payload.get("image_size"),
        "dpt_pyramid_keys": dpt_keys,
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
    if expected_dtype == "fp16":
        target = torch.float16
    elif expected_dtype == "fp32":
        target = torch.float32
    else:
        target = None

    if payload.get("frame_paths") is None:
        raise KeyError(f"Missing frame_paths in {src_path}")
    if len(payload["frame_paths"]) != int(meta.get("num_frames", 0)):
        raise ValueError(f"Frame count mismatch for {src_path}")

    if tensors["dino_features"].shape != payload["dino_features"].shape:
        raise ValueError(f"dino_features shape mismatch for {src_path}")
    if target is not None and tensors["dino_features"].dtype != target:
        raise ValueError(f"dino_features dtype mismatch for {src_path}")

    dpt_pyramid = payload["dpt_pyramid"]
    for i, level in enumerate(dpt_pyramid):
        key = f"dpt_pyramid_{i}"
        if key not in tensors:
            raise KeyError(f"Missing {key} in {out_path}")
        if tensors[key].shape != level.shape:
            raise ValueError(f"{key} shape mismatch for {src_path}")
        if target is not None and tensors[key].dtype != target:
            raise ValueError(f"{key} dtype mismatch for {src_path}")


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_dir).expanduser()
    dst_root = Path(args.dst_dir).expanduser()
    chunk_files = discover_chunks(src_root)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk .pt files found under {src_root}")

    workers = max(1, int(args.workers))
    results = {"ok": 0, "skip": 0, "fail": 0}

    if workers == 1:
        for src_path in tqdm(chunk_files, desc="convert"):
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
            for fut in tqdm(as_completed(futures), total=len(futures), desc="convert"):
                src_path = futures[fut]
                try:
                    status = fut.result()
                    results[status] += 1
                except Exception as exc:
                    results["fail"] += 1
                    print(f"[fail] {src_path}: {exc}")

    print(
        f"[done] converted {len(chunk_files)} files: "
        f"ok={results['ok']} skip={results['skip']} fail={results['fail']}"
    )

    if args.verify:
        sample = chunk_files[: min(3, len(chunk_files))]
        for src_path in sample:
            verify_one(src_path, src_root, dst_root)
            print(f"[verify] ok: {src_path}")


if __name__ == "__main__":
    main()
