#!/usr/bin/env python3
"""
Batch exporter for VGGT-SLAM embeddings.

This script walks over ScanNet++ scene folders (or a single folder),
runs VGGT in manageable windows (same preprocessing used by VGGT-SLAM),
and stores the raw depth DPT pyramid plus DINOv2 token features so we can
train FiLM + Mask2Former offline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from hiding_folder.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DINO/DPT embeddings via VGGT-SLAM windows.")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--image-dir",
        help="Single directory with RGB frames (requires --scene-id).",
    )
    src_group.add_argument(
        "--data-root",
        help="Root directory containing multiple scene sub-folders (one level deep).",
    )
    parser.add_argument("--scene-id", help="Scene identifier when using --image-dir.")
    parser.add_argument(
        "--scenes",
        help="Optional comma-separated subset when using --data-root. Defaults to all subfolders.",
    )
    parser.add_argument("--output-dir", required=True, help="Where to store exported tensors.")
    parser.add_argument(
        "--image-subdir",
        default="",
        help="Optional subdirectory under each scene (e.g., dslr/resized_undistorted_images).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Number of frames per VGGT window (like SLAM submap size).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Overlap between consecutive windows. Set >0 to get denser coverage.",
    )
    parser.add_argument(
        "--image-ext",
        default=".JPG",
        help="Image extension to glob (case-sensitive).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Store tensors as float16 instead of float32 to save disk space.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenes that already have a meta.json in the output directory.",
    )
    return parser.parse_args()


def sorted_images(image_dir: Path, ext: str) -> List[Path]:
    images = sorted(p for p in image_dir.glob(f"*{ext}") if p.is_file())
    if not images:
        raise FileNotFoundError(f"No images ending with {ext} under {image_dir}")
    return images


def chunk_indices(total: int, size: int, overlap: int = 0) -> Iterable[Tuple[int, int]]:
    if size <= 0:
        raise ValueError("window-size must be > 0")
    step = max(1, size - overlap)
    start = 0
    while start < total:
        end = min(total, start + size)
        yield start, end
        if end == total:
            break
        start += step


class DinoFeatureTap:
    """
    Registers a forward hook in VGGT patch-embed blocks (same as FiLM taps).
    After each forward pass call .extract(...) to retrieve reshaped feature maps.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        tap_candidates: Optional[Sequence[str]] = None,
        out_channels: int = 256,
        patch_size: int = 14,
    ) -> None:
        self.out_ch = out_channels
        self.patch_size = patch_size
        self._cache: Optional[torch.Tensor] = None
        self._proj: Optional[nn.Conv2d] = None

        tap_candidates = tap_candidates or [
            "aggregator.patch_embed.blocks.23",
            "aggregator.patch_embed.blocks.21",
            "aggregator.patch_embed.blocks.15",
        ]
        modmap = {name: module for name, module in model.named_modules()}
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        for name in tap_candidates:
            if name in modmap:
                self._handle = modmap[name].register_forward_hook(self._hook(name))
                break
        if self._handle is None:
            raise KeyError(f"None of the DINO tap candidates exist: {tap_candidates}")

    def _hook(self, _name: str):
        def fn(_module, _inputs, output):
            if torch.is_tensor(output):
                self._cache = output
            elif isinstance(output, (list, tuple)):
                for item in output:
                    if torch.is_tensor(item):
                        self._cache = item
                        break

        return fn

    def extract(self, batch_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError("DINO hook cache is empty. Did you run VGGT forward first?")
        B, S, H, W = batch_shape
        tokens = self._cache  # shape (B*S, N, Cin)
        self._cache = None

        if tokens.dim() != 3:
            raise ValueError(f"Unexpected DINO token shape: {tokens.shape}")

        BS, N, Cin = tokens.shape
        if BS != B * S:
            raise ValueError(f"DINO token batch mismatch: expected {B*S}, got {BS}")

        Htok = H // self.patch_size
        Wtok = W // self.patch_size
        expected = Htok * Wtok
        if expected <= 0 or H % self.patch_size or W % self.patch_size:
            raise ValueError(f"Input size {H}x{W} not divisible by patch size {self.patch_size}")
        if N < expected:
            raise ValueError(f"Not enough tokens: have {N}, expected >= {expected}")

        patch_tokens = tokens[:, N - expected :, :]
        fmap = patch_tokens.transpose(1, 2).reshape(B * S, Cin, Htok, Wtok)
        if self._proj is None:
            self._proj = nn.Conv2d(Cin, self.out_ch, kernel_size=1).to(fmap.device)
        fmap = self._proj(fmap)
        return fmap.view(B, S, self.out_ch, Htok, Wtok)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def load_model(device: torch.device) -> VGGT:
    os.environ.setdefault("VGGT_FUSE_FILM", "0")
    model = VGGT().to(device)
    checkpoint_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def tensor_to_disk(tensor: torch.Tensor, *, use_half: bool) -> torch.Tensor:
    if use_half:
        tensor = tensor.to(dtype=torch.float16)
    return tensor.cpu().contiguous()


def export_scene(
    *,
    scene_id: str,
    image_dir: Path,
    output_root: Path,
    image_ext: str,
    window_size: int,
    chunk_overlap: int,
    use_half: bool,
    model: VGGT,
    dino_tap: DinoFeatureTap,
) -> None:
    chunk_dir = output_root / scene_id / "chunks"
    meta_path = output_root / scene_id / "meta.json"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted_images(image_dir, image_ext)
    manifest: List[Dict] = []
    chunk_idx = 0

    with torch.no_grad():
        for start, end in chunk_indices(len(frames), window_size, chunk_overlap):
            window_paths = frames[start:end]
            imgs = load_and_preprocess_images([str(p) for p in window_paths])
            # load_and_preprocess_images returns [S,3,H,W]; add batch dim for consistency
            if imgs.ndim == 4:
                imgs = imgs.unsqueeze(0)
            imgs = imgs.to(next(model.parameters()).device)
            B, S = imgs.shape[:2]
            H, W = imgs.shape[-2:]

            preds = model(imgs)
            pyramid = preds.get("pyramid")
            if pyramid is None:
                raise RuntimeError("VGGT depth head did not emit pyramid outputs.")

            dpt_levels = [
                tensor_to_disk(level, use_half=use_half)  # already shaped [B,S,C,H,W]
                for level in pyramid
            ]
            dino_feats = tensor_to_disk(dino_tap.extract((B, S, H, W)), use_half=use_half)

            chunk_data = {
                "scene_id": scene_id,
                "frame_start": start,
                "frame_end": end,
                "frame_paths": [str(p) for p in window_paths],
                "image_size": [H, W],
                "dpt_pyramid": dpt_levels,
                "dino_features": dino_feats,
            }
            chunk_file = chunk_dir / f"{scene_id}_chunk_{chunk_idx:05d}.pt"
            torch.save(chunk_data, chunk_file)

            manifest.append(
                {
                    "chunk_path": str(chunk_file),
                    "frame_start": start,
                    "frame_end": end,
                    "num_frames": len(window_paths),
                    "image_size": [H, W],
                }
            )
            chunk_idx += 1

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scene_id": scene_id,
                "num_chunks": chunk_idx,
                "window_size": window_size,
                "chunk_overlap": chunk_overlap,
                "half_precision": bool(use_half),
                "image_dir": str(image_dir),
                "chunks": manifest,
            },
            f,
            indent=2,
        )
    print(f"[scene {scene_id}] wrote {chunk_idx} chunks to {chunk_dir}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    image_subdir = args.image_subdir.strip().strip("/")

    if args.image_dir:
        if not args.scene_id:
            raise ValueError("--scene-id is required when using --image-dir")
        base_dir = Path(args.image_dir)
        img_dir = base_dir / image_subdir if image_subdir else base_dir
        scene_specs = [(args.scene_id, img_dir)]
    else:
        root = Path(args.data_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Data root not found: {root}")
        wanted = None
        if args.scenes:
            wanted = set(s.strip() for s in args.scenes.split(",") if s.strip())
        scene_specs = []
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            if wanted and sub.name not in wanted:
                continue
            img_dir = sub / image_subdir if image_subdir else sub
            scene_specs.append((sub.name, img_dir))
        if not scene_specs:
            raise RuntimeError("No scenes found under data root with the given filters.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dino_tap = DinoFeatureTap(model)

    try:
        for scene_id, img_dir in scene_specs:
            meta_path = output_root / scene_id / "meta.json"
            if args.skip_existing and meta_path.is_file():
                print(f"[scene {scene_id}] meta exists, skipping (--skip-existing).")
                continue
            print(f"[scene {scene_id}] processing images from {img_dir}")
            export_scene(
                scene_id=scene_id,
                image_dir=img_dir,
                output_root=output_root,
                image_ext=args.image_ext,
                window_size=args.window_size,
                chunk_overlap=args.chunk_overlap,
                use_half=args.half,
                model=model,
                dino_tap=dino_tap,
            )
    finally:
        dino_tap.remove()


if __name__ == "__main__":
    main()
