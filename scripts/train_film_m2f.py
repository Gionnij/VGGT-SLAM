#!/usr/bin/env python3
"""
train_film_m2f.py
=================

Minimal offline trainer that consumes pre-exported VGGT embeddings
(DPT pyramid + DINO fmap) and ScanNet++ 2D labels to fine-tune a
FiLM fusion module plus the Mask2Former semantic head.

It expects embeddings exported by scripts/export_embeddings.py and
labels produced by the ScanNet++ rasterizer. Overlapping frames
across chunks are deduplicated by keeping the first occurrence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hiding_folder.semantic_head import SemanticHead


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FiLM + Mask2Former on saved VGGT embeddings.")
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Root with per-scene folders: images/, labels/, chunks/, meta.json (output of prepare_dataset.py).",
    )
    p.add_argument("--scenes", help="Comma-separated list of scene ids to include (default: all under dataset-root).")
    p.add_argument("--batch-size", type=int, default=1, help="Frames per batch (keep small for memory).")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-chunks", type=int, default=None, help="Optional cap on chunks for quick tests.")
    p.add_argument("--num-classes", type=int, default=200, help="Number of semantic classes.")
    p.add_argument("--ignore-index", type=int, default=65535, help="Label value to ignore in loss.")
    p.add_argument("--config-path", help="Mask2Former config path (defaults to COCO R50).")
    p.add_argument("--weights-path", help="Optional Mask2Former checkpoint to init from.")
    p.add_argument("--use-half", action="store_true", help="Use mixed precision training.")
    return p.parse_args()


def list_chunk_files(dataset_root: Path, scenes: Optional[Sequence[str]] = None) -> List[Path]:
    chunk_files: List[Path] = []
    for scene_dir in sorted(dataset_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if scenes and scene_dir.name not in scenes:
            continue
        chunk_dir = scene_dir / "chunks"
        if not chunk_dir.is_dir():
            continue
        chunk_files.extend(sorted(chunk_dir.glob("*.pt")))
    if not chunk_files:
        raise RuntimeError("No chunk files found with given filters.")
    return chunk_files


def label_path_for_frame(scene_dir: Path, frame_path: str) -> Path:
    fname = Path(frame_path).name + ".png"
    return scene_dir / "labels" / fname


class ChunkDataset(Dataset):
    """
    Streams per-frame samples from exported chunks, deduplicating overlapping frames.
    """

    def __init__(self, chunk_files: Sequence[Path]) -> None:
        self.samples: List[Dict] = []
        seen = set()
        for chk_file in chunk_files:
            chk = torch.load(chk_file, map_location="cpu")
            scene_id = chk["scene_id"]
            scene_dir = chk_file.parent.parent  # .../<scene>/
            frame_paths = chk["frame_paths"]
            dino = chk["dino_features"]  # [1,S,C,h,w]
            dpt_list = chk["dpt_pyramid"]  # list of 4 [1,S,C,h,w]
            for idx, fpath in enumerate(frame_paths):
                if fpath in seen:
                    continue  # drop overlaps, keep first occurrence
                seen.add(fpath)
                label_path = label_path_for_frame(scene_dir, fpath)
                self.samples.append(
                    {
                        "scene_id": scene_id,
                        "frame_path": fpath,
                        "label_path": label_path,
                        "dino": dino[0, idx],  # [C,h,w]
                        "dpt_levels": [lvl[0, idx] for lvl in dpt_list],  # list of tensors
                        "image_size": chk["image_size"],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        label = load_label_png(sample["label_path"])
        return {
            "scene_id": sample["scene_id"],
            "frame_path": sample["frame_path"],
            "dino": sample["dino"],
            "dpt_levels": sample["dpt_levels"],
            "label": label,
            "image_size": sample["image_size"],
        }


def load_label_png(path: Path) -> torch.Tensor:
    # PIL supports 16-bit PNGs; torchvision.io.read_image does not.
    from PIL import Image
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    arr = np.array(Image.open(path), copy=True)  # make writable
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Expected single-channel label, got shape {arr.shape} at {path}")
    return torch.from_numpy(arr).long()


class FiLMFusion(nn.Module):
    """
    Simple FiLM-style fusion: condition on global-pooled DINO fmap to modulate each DPT level.
    """

    def __init__(self, dino_ch: int = 256, dpt_channels: Sequence[int] = (256, 512, 1024, 1024)):
        super().__init__()
        self.mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(dino_ch, c * 2)) for c in dpt_channels]
        )

    def forward(self, dino: torch.Tensor, dpt_levels: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        # dino: [B,C,h,w] -> global pool
        cond = dino.mean(dim=(-2, -1))  # [B,C]
        fused = []
        for lvl, mlp in zip(dpt_levels, self.mlps):
            gb = mlp(cond)  # [B, 2*C]
            C = lvl.shape[1]
            gamma, beta = gb[:, :C], gb[:, C:]
            gamma = gamma.view(-1, C, 1, 1)
            beta = beta.view(-1, C, 1, 1)
            fused.append(gamma * lvl + beta)
        return fused


class FusionMask2Former(nn.Module):
    """
    Wrapper: FiLM fuse DINO+DPT, then run Mask2Former head to get semantic logits.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        num_classes: int,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        self.fusion = FiLMFusion()
        self.sem_head = SemanticHead(
            device=device,
            config_path=config_path,
            weights_path=weights_path,
            num_classes=num_classes,
            num_queries=None,
        )
        self.to(device)

    def forward(self, dino: torch.Tensor, dpt_levels: Sequence[torch.Tensor], label_shape: Tuple[int, int]):
        # dino: [B,256,h,w], dpt_levels: list of 4 [B,C,h,w]
        fused_levels = self.fusion(dino, dpt_levels)
        # SemanticHead expects images (for H,W) plus frame_indices and pyramid
        B = dino.shape[0]
        H, W = label_shape
        dummy_images = torch.zeros(B, 1, 3, H, W, device=dino.device)
        cls_logits, mask_logits, _ = self.sem_head(
            dummy_images,
            frame_indices=[0],
            film_pyramid=fused_levels,
        )
        return cls_logits, mask_logits


def dense_logits_from_queries(cls_logits: torch.Tensor, mask_logits: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """
    Turn query logits into dense per-class logits, restoring [B,S,C,H,W].
    cls_logits: [B,S,Q,C] or [B,Q,C]
    mask_logits: [B,S,Q,H,W] or [B,Q,H,W]
    """
    # Flatten frames into batch for einsum, then reshape back
    if cls_logits.dim() == 4:
        cls_flat = cls_logits.reshape(B * S, *cls_logits.shape[-2:])
        mask_flat = mask_logits.reshape(B * S, *mask_logits.shape[-3:])
    else:
        cls_flat = cls_logits
        mask_flat = mask_logits
    class_probs = cls_flat.softmax(dim=-1)[..., :-1]  # drop no-object
    mask_probs = mask_flat.sigmoid()
    seg_flat = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    if cls_logits.dim() == 4:
        seg_flat = seg_flat.reshape(B, S, *seg_flat.shape[1:])
    return seg_flat


def collate_fn(batch: List[Dict]) -> Dict:
    # batch size small; dino/dpt have varying spatial sizes per level but consistent within a scene
    dino = torch.stack([b["dino"] for b in batch], dim=0)
    dpt_levels = []
    for lvl_idx in range(len(batch[0]["dpt_levels"])):
        dpt_levels.append(torch.stack([b["dpt_levels"][lvl_idx] for b in batch], dim=0))
    labels = [b["label"] for b in batch]
    return {
        "dino": dino,
        "dpt_levels": dpt_levels,
        "labels": labels,
        "image_sizes": [b["image_size"] for b in batch],
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scenes = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    chunk_files = list_chunk_files(Path(args.dataset_root), scenes=scenes)
    if args.max_chunks:
        chunk_files = chunk_files[: args.max_chunks]

    ds = ChunkDataset(chunk_files)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = FusionMask2Former(
        device=device,
        num_classes=args.num_classes,
        config_path=args.config_path,
        weights_path=args.weights_path,
    )
    model.train()

    # Train FiLM + Mask2Former head
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_half and device.type == "cuda")

    for epoch in range(args.epochs):
        running = 0.0
        for step, batch in enumerate(tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")):
            dino = batch["dino"].to(device)
            dpt_levels = [lvl.to(device) for lvl in batch["dpt_levels"]]
            # labels list; assume all same H,W within batch
            labels = batch["labels"]
            H, W = labels[0].shape[-2:]
            label_tensor = torch.stack(labels, dim=0).to(device)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_half and device.type == "cuda"):
                cls_logits, mask_logits = model(dino, dpt_levels, label_shape=(H, W))
                S_frames = cls_logits.shape[1] if cls_logits.dim() == 4 else 1
                seg_logits = dense_logits_from_queries(cls_logits, mask_logits, B=dino.shape[0], S=S_frames)  # [B, S, C, H', W'] or [B,C,H',W']
                if seg_logits.dim() == 5:
                    seg_logits = seg_logits.reshape(dino.shape[0] * S_frames, *seg_logits.shape[2:])
                seg_logits = F.interpolate(seg_logits, size=(H, W), mode="bilinear", align_corners=False)
                if seg_logits.dim() == 4 and S_frames > 1:
                    seg_logits = seg_logits.view(dino.shape[0], S_frames, *seg_logits.shape[1:])
                elif seg_logits.dim() == 4:
                    seg_logits = seg_logits.unsqueeze(1)  # [B,1,C,H,W]
                seg_logits = seg_logits.squeeze(1)
                loss = F.cross_entropy(seg_logits, label_tensor, ignore_index=args.ignore_index)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if (step + 1) % 10 == 0:
                avg = running / 10
                print(f"[epoch {epoch+1}] step {step+1} loss {avg:.4f}")
                running = 0.0

    print("Training finished.")


if __name__ == "__main__":
    main()
