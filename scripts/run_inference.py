#!/usr/bin/env python3
"""
run_inference.py
================

Run FiLM+Mask2Former on pre-exported embeddings and save masks + overlays.

Inputs:
  - dataset_ready/<scene>/chunks/*.pt (from prepare_dataset.py)
  - dataset_ready/<scene>/images/*.JPG (copied RGBs)

Outputs (per run):
  complete_outputs/<date>_run_<nn>/
    semantics/<scene>/masks/*.png          (uint16 labels)
    semantics/<scene>/overlays/*.png       (RGB overlay)

Usage:
  python scripts/run_inference.py \
    --dataset-root /home/s2984792/src/VGGT-SLAM/dataset_ready \
    --checkpoint /home/s2984792/src/VGGT-SLAM/checkpoints/film_m2f_20251126_run_02.pt \
    --num-classes 2878 \
    --palette /home/s2984792/data/scannetpp/metadata/semantic_palette.txt
"""

from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from scripts.train_film_m2f import FusionMask2Former


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FiLM+Mask2Former inference and save masks/overlays.")
    p.add_argument("--dataset-root", required=True, help="dataset_ready root with scenes/images/labels/chunks.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path to load (state_dict).")
    p.add_argument("--scenes", help="Comma-separated scene ids (default: all scenes under dataset-root).")
    p.add_argument("--num-classes", type=int, required=True, help="Number of semantic classes.")
    p.add_argument("--ignore-index", type=int, default=65535, help="Ignore label value (not used in inference).")
    p.add_argument("--config-path", help="Mask2Former config path if different from default.")
    p.add_argument("--weights-path", help="Mask2Former weights path if different from default.")
    p.add_argument("--output-root", default="/home/s2984792/src/VGGT-SLAM/complete_outputs", help="Root to save outputs.")
    p.add_argument("--palette", help="Optional palette txt (256x3) to colorize overlays.")
    p.add_argument("--alpha", type=float, default=0.5, help="Overlay blend factor (0..1).")
    p.add_argument("--max-chunks", type=int, default=None, help="Optional cap for quick tests.")
    return p.parse_args()


def _next_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    today = datetime.datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf"{today}_run_(\d+)")
    existing = [p.name for p in output_root.iterdir() if p.is_dir() and pattern.match(p.name)]
    runs: List[int] = []
    for name in existing:
        m = pattern.match(name)
        if m:
            try:
                runs.append(int(m.group(1)))
            except ValueError:
                continue
    next_run = (max(runs) + 1) if runs else 1
    run_dir = output_root / f"{today}_run_{next_run:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_palette(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    pal = np.loadtxt(path, dtype=np.uint8)
    if pal.ndim != 2 or pal.shape[1] != 3:
        raise ValueError(f"Palette must be Nx3, got {pal.shape}")
    return pal


def colorize(mask: np.ndarray, palette: Optional[np.ndarray]) -> np.ndarray:
    if palette is None:
        # simple grayscale colorization
        return np.stack([mask % 256, mask % 256, mask % 256], axis=-1).astype(np.uint8)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    max_idx = min(len(palette), mask.max() + 1)
    for idx in range(max_idx):
        out[mask == idx] = palette[idx % len(palette)]
    return out


def collect_scenes(dataset_root: Path, scenes_arg: Optional[str]) -> List[Path]:
    if scenes_arg:
        names = [s.strip() for s in scenes_arg.split(',') if s.strip()]
        dirs = [dataset_root / n for n in names]
    else:
        dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    return sorted([d for d in dirs if (d / "chunks").is_dir()])


def load_chunks(scene_dir: Path, max_chunks: Optional[int]) -> List[Path]:
    chunks = sorted((scene_dir / "chunks").glob("*.pt"))
    if max_chunks:
        chunks = chunks[:max_chunks]
    return chunks


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_root)
    run_dir = _next_run_dir(Path(args.output_root))
    print(f"Saving outputs to {run_dir}")

    palette = load_palette(args.palette)
    model = FusionMask2Former(device=device, num_classes=args.num_classes, config_path=args.config_path, weights_path=args.weights_path)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    scenes = collect_scenes(dataset_root, args.scenes)
    if not scenes:
        raise RuntimeError("No scenes found for inference.")

    with torch.no_grad():
        for scene_dir in scenes:
            scene_id = scene_dir.name
            print(f"[scene {scene_id}] loading chunks...")
            out_masks = run_dir / "semantics" / scene_id / "masks"
            out_ov = run_dir / "semantics" / scene_id / "overlays"
            out_masks.mkdir(parents=True, exist_ok=True)
            out_ov.mkdir(parents=True, exist_ok=True)

            chunk_files = load_chunks(scene_dir, args.max_chunks)
            for chk_file in tqdm(chunk_files, desc=f"chunks {scene_id}"):
                chk = torch.load(chk_file, map_location="cpu")
                frame_paths = chk["frame_paths"]
                dino = chk["dino_features"]  # [1,S,C,h,w]
                dpt_list = chk["dpt_pyramid"]  # list of 4 [1,S,C,h,w]
                H_img, W_img = chk["image_size"]

                for idx, fpath in enumerate(frame_paths):
                    basename = Path(fpath).name
                    img_path = scene_dir / "images" / basename
                    if not img_path.is_file():
                        continue

                    dino_t = dino[0, idx].unsqueeze(0).to(device)  # [1,C,h,w]
                    dpt_levels = [lvl[0, idx].unsqueeze(0).to(device) for lvl in dpt_list]

                    cls_logits, mask_logits = model(dino_t, dpt_levels, label_shape=(H_img, W_img))
                    S_frames = cls_logits.shape[1] if cls_logits.dim() == 4 else 1
                    seg_logits = cls_logits.softmax(dim=-1)[..., :-1]  # [B,S,Q,C]
                    mask_probs = mask_logits.sigmoid()
                    if seg_logits.dim() == 4:
                        seg_logits = seg_logits.reshape(1 * S_frames, *seg_logits.shape[2:])
                        mask_probs = mask_probs.reshape(1 * S_frames, *mask_probs.shape[2:])
                    seg_dense = torch.einsum("bqc,bqhw->bchw", seg_logits, mask_probs)
                    seg_dense = F.interpolate(seg_dense, size=(H_img, W_img), mode="bilinear", align_corners=False)
                    seg_pred = seg_dense.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint16)

                    # Save mask
                    mask_out = out_masks / f"{basename}.png"
                    Image.fromarray(seg_pred).save(mask_out)

                    # Overlay
                    img = np.array(Image.open(img_path).convert("RGB"))
                    colored = colorize(seg_pred, palette)
                    overlay = (args.alpha * colored + (1 - args.alpha) * img).astype(np.uint8)
                    ov_out = out_ov / f"{basename}.png"
                    Image.fromarray(overlay).save(ov_out)

    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
