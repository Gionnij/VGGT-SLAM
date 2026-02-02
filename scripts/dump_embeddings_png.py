#!/usr/bin/env python3
"""
dump_embeddings_png.py
======================

Dump a few DINO/DPT embedding visualizations and the corresponding label PNGs
to visually check alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump embedding visualizations + labels.")
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Root with per-scene folders containing chunks/ (from prepare_dataset.py).",
    )
    p.add_argument(
        "--labels-root",
        help="Root containing labels/<scene_id>/*.png (optional).",
    )
    p.add_argument("--scene", help="Single scene id to process.")
    p.add_argument(
        "--scenes",
        help="Comma-separated list of scenes to include (overrides --scene).",
    )
    p.add_argument(
        "--scenes-file",
        help="Text file with scene ids (one per line); overrides --scene/--scenes.",
    )
    p.add_argument(
        "--output-dir",
        default="./debug_embeddings",
        help="Directory to write PNGs.",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="How many frames to dump (across chunks).",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional cap on chunks for quick tests.",
    )
    p.add_argument(
        "--chunk-format",
        choices=["auto", "pt", "safetensors"],
        default="auto",
        help="Chunk storage format to load (auto prefers safetensors when present).",
    )
    return p.parse_args()


def _load_list_from_file(path_str: Optional[str]) -> Optional[List[str]]:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"List file not found: {path}")
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        val = line.strip()
        if not val or val.startswith("#"):
            continue
        out.append(val)
    return out or None


def list_chunk_files(
    dataset_root: Path,
    *,
    scenes: Optional[Sequence[str]] = None,
    chunk_format: str = "auto",
) -> List[Path]:
    chunk_files: List[Path] = []
    scene_filter = set(scenes) if scenes else None
    for scene_dir in sorted(dataset_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        chunk_dir = scene_dir / "chunks"
        if not chunk_dir.is_dir():
            continue
        pt_files = sorted(chunk_dir.glob("*.pt"))
        st_files = sorted(chunk_dir.glob("*.safetensors"))
        if chunk_format == "pt":
            chosen = pt_files
        elif chunk_format == "safetensors":
            chosen = st_files
        else:
            st_stems = {p.stem for p in st_files}
            pt_files = [p for p in pt_files if p.stem not in st_stems]
            chosen = st_files + pt_files
        chunk_files.extend(chosen)
    if not chunk_files:
        raise RuntimeError("No chunk files found with given filters.")
    return chunk_files


def _chunk_json_path(chunk_path: Path) -> Path:
    return chunk_path.with_suffix(".json")


def _load_chunk_metadata(chunk_path: Path) -> Dict:
    if chunk_path.suffix == ".safetensors":
        json_path = _chunk_json_path(chunk_path)
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing chunk metadata JSON for {chunk_path}")
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta
    return torch.load(chunk_path, map_location="cpu")


def _load_safetensors_chunk(chunk_path: Path) -> Dict:
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise ImportError("safetensors is required to load .safetensors chunks") from exc

    tensors = load_file(str(chunk_path))
    meta = _load_chunk_metadata(chunk_path)
    keys = meta.get("dpt_pyramid_keys")
    if not keys:
        keys = sorted(
            [k for k in tensors.keys() if k.startswith("dpt_pyramid_")],
            key=lambda x: int(x.split("_")[-1]),
        )
    if not keys:
        raise KeyError(f"No dpt_pyramid_* tensors found in {chunk_path}")
    dpt_pyramid = [tensors[k] for k in keys]
    dino = tensors["dino_features"]
    return {
        "dino_features": dino,
        "dpt_pyramid": dpt_pyramid,
        "frame_paths": meta.get("frame_paths"),
        "scene_id": meta.get("scene_id"),
    }


def _load_chunk(chunk_path: Path) -> Dict:
    if chunk_path.suffix == ".safetensors":
        return _load_safetensors_chunk(chunk_path)
    return torch.load(chunk_path, map_location="cpu")


def label_path_for_frame(scene_id: str, frame_path: str, labels_root: Optional[Path]) -> Optional[Path]:
    if labels_root is None:
        return None
    fname = Path(frame_path).name + ".png"
    return labels_root / scene_id / fname


def _to_uint8(img: torch.Tensor) -> Image.Image:
    arr = img.float()
    arr = arr - arr.min()
    denom = arr.max().clamp(min=1e-6)
    arr = (arr / denom * 255.0).clamp(0, 255).byte()
    return Image.fromarray(arr.cpu().numpy(), mode="L")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    labels_root = Path(args.labels_root).expanduser() if args.labels_root else None
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes_from_file = _load_list_from_file(args.scenes_file)
    if scenes_from_file:
        scenes = scenes_from_file
    elif args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    elif args.scene:
        scenes = [args.scene]
    else:
        scenes = None

    chunk_files = list_chunk_files(dataset_root, scenes=scenes, chunk_format=args.chunk_format)
    if args.max_chunks:
        chunk_files = chunk_files[: args.max_chunks]

    remaining = max(1, int(args.num_frames))
    for chk_path in chunk_files:
        chk = _load_chunk(chk_path)
        frame_paths = chk.get("frame_paths") or []
        scene_id = chk.get("scene_id") or chk_path.parent.parent.name
        dino = chk["dino_features"]  # [1,S,C,h,w]
        dpt_list = chk["dpt_pyramid"]  # list of 4 [1,S,C,h,w]
        for idx, fpath in enumerate(frame_paths):
            if remaining <= 0:
                break
            base = Path(fpath).name
            stem = Path(base).stem
            out_scene = output_dir / scene_id
            out_scene.mkdir(parents=True, exist_ok=True)

            dino_map = dino[0, idx].mean(dim=0)
            _to_uint8(dino_map).save(out_scene / f"{stem}_dino.png")

            for lvl_idx, lvl in enumerate(dpt_list):
                lvl_map = lvl[0, idx].mean(dim=0)
                _to_uint8(lvl_map).save(out_scene / f"{stem}_dpt{lvl_idx}.png")

            label_path = label_path_for_frame(scene_id, fpath, labels_root)
            if label_path and label_path.is_file():
                lbl = Image.open(label_path)
                lbl.save(out_scene / f"{stem}_label.png")
                lbl_np = np.array(lbl)
                lbl_vis = Image.fromarray((lbl_np % 256).astype(np.uint8), mode="L")
                lbl_vis.save(out_scene / f"{stem}_label_vis.png")
            remaining -= 1
        if remaining <= 0:
            break

    print(f"[done] wrote debug PNGs to {output_dir}")


if __name__ == "__main__":
    main()
