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
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import OrderedDict
import datetime
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
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
    p.add_argument(
        "--labels-root",
        help="Optional separate root containing labels/<scene_id>/*.png. "
        "If omitted, labels are expected under <dataset-root>/<scene>/labels.",
    )
    p.add_argument(
        "--index-cache",
        help="Optional path to a cache file (.pt) storing the precomputed sample index. "
        "If present and up-to-date, avoids rescanning all chunk files. "
        "Updated after each scan to include new/changed chunks.",
    )
    p.add_argument("--batch-size", type=int, default=1, help="Frames per batch (keep small for memory).")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-chunks", type=int, default=None, help="Optional cap on chunks for quick tests.")
    p.add_argument("--num-classes", type=int, default=200, help="Number of semantic classes.")
    p.add_argument("--ignore-index", type=int, default=65535, help="Label value to ignore in loss.")
    p.add_argument(
        "--ignore-classes",
        help="Comma-separated list of class ids to remap to ignore-index before training.",
    )
    p.add_argument(
        "--ignore-classes-file",
        help="Text file with one class id per line to ignore.",
    )
    p.add_argument(
        "--remap-classes-file",
        help=(
            "Optional text file with one original class id per line. "
            "Masks are remapped to a dense id space [0..K-1] using this list; "
            "values not in the list are set to ignore-index. "
            "num-classes defaults to len(list) when provided."
        ),
    )
    p.add_argument("--config-path", help="Mask2Former config path (defaults to COCO R50).")
    p.add_argument("--weights-path", help="Optional Detectron2-style Mask2Former checkpoint to init from.")
    p.add_argument(
        "--resume-checkpoint",
        help="Full FusionMask2Former checkpoint (state_dict) to resume from. "
        "If provided, overrides weights-path after model construction.",
    )
    p.add_argument("--use-half", action="store_true", help="Use mixed precision training.")
    p.add_argument("--focal-alpha", type=float, default=0.25, help="Alpha weighting for focal loss.")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma exponent for focal loss.")
    p.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory to save checkpoints.")
    p.add_argument(
        "--checkpoint-name",
        default=None,
        help="Checkpoint filename. If omitted, auto-named as film_m2f_<date>_run_<n>.pt",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save intermediate checkpoints every N epochs (0 disables intermediate saves).",
    )
    p.add_argument(
        "--scenes-file",
        help="Optional text file with scene ids (one per line) to train on. Overrides --scenes if provided.",
    )
    p.add_argument(
        "--chunk-cache-size",
        type=int,
        default=4,
        help="How many chunks to keep in the in-memory LRU cache for faster reuse.",
    )
    p.add_argument(
        "--label-cache-size",
        type=int,
        default=0,
        help="Optional LRU cache size for decoded label PNG tensors (0 disables).",
    )
    p.add_argument(
        "--label-chunks-root",
        help="Optional root containing per-chunk label tensors under <scene>/<label_chunk_subdir>/*.pt. Defaults to dataset-root.",
    )
    p.add_argument(
        "--label-chunk-subdir",
        default="label_chunks",
        help="Subdirectory under each scene containing chunked label tensors.",
    )
    p.add_argument(
        "--label-chunk-ext",
        default=".pt",
        help="File extension for chunked label tensors (default: .pt).",
    )
    p.add_argument(
        "--label-chunk-cache-size",
        type=int,
        default=2,
        help="LRU cache size for loaded label chunk tensors (per worker).",
    )
    p.add_argument(
        "--no-label-chunks",
        dest="use_label_chunks",
        action="store_false",
        help="Disable binary label chunks even if present; always decode PNGs.",
    )
    p.set_defaults(use_label_chunks=True)
    p.add_argument(
        "--shuffle-mode",
        choices=["global", "chunk", "none"],
        default="chunk",
        help=(
            "How to shuffle samples. 'global' is default PyTorch shuffle across all samples (worst locality). "
            "'chunk' shuffles chunk order each epoch and shuffles within each chunk (much better I/O locality). "
            "'none' preserves index order (fastest but least stochastic)."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for shuffling (0 -> use PyTorch default/random).",
    )
    p.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Enable persistent workers in DataLoader to reduce worker startup overhead.",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Prefetch factor per worker for DataLoader (when num_workers > 0).",
    )
    p.add_argument(
        "--no-debug-print",
        action="store_true",
        help="Disable per-batch debug prints to reduce overhead.",
    )
    return p.parse_args()


def list_chunk_files(dataset_root: Path, scenes: Optional[Sequence[str]] = None) -> List[Path]:
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
        chunk_files.extend(sorted(chunk_dir.glob("*.pt")))
    if not chunk_files:
        raise RuntimeError("No chunk files found with given filters.")
    return chunk_files


def _chunk_stat(path: Path) -> Dict[str, int]:
    st = path.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def label_chunk_path_for_chunk(
    chunk_path: Path,
    *,
    label_chunk_root: Optional[Path],
    label_chunk_subdir: str,
    label_chunk_ext: str,
) -> Path:
    """
    Resolve where a label chunk tensor should live for a given embedding chunk.
    Defaults to <dataset_root>/<scene>/<label_chunk_subdir>/<chunk_name>.pt
    """
    scene_dir = chunk_path.parent.parent
    dataset_root = scene_dir.parent
    base_root = label_chunk_root if label_chunk_root else dataset_root
    chunk_name = chunk_path.with_suffix(label_chunk_ext).name if label_chunk_ext else chunk_path.name
    return base_root / scene_dir.name / label_chunk_subdir / chunk_name


def build_or_load_index(
    *,
    chunk_files: Sequence[Path],
    labels_root: Optional[Path],
    cache_path: Optional[Path],
    label_chunk_root: Optional[Path],
    label_chunk_subdir: str,
    label_chunk_ext: str,
    use_label_chunks: bool,
) -> List[Dict]:
    """
    Build per-frame sample index. If cache_path is provided and valid, reuse it and
    only process new/changed chunks.
    """
    cached: Dict = {"chunk_meta": {}, "samples": []}
    if cache_path and cache_path.is_file():
        try:
            cached = torch.load(cache_path, map_location="cpu")
            print(f"[info] loaded index cache from {cache_path}")
        except Exception as e:
            print(f"[warn] failed to load index cache {cache_path}: {e}")
            cached = {"chunk_meta": {}, "samples": []}

    cached_meta: Dict[str, Dict[str, int]] = cached.get("chunk_meta", {})
    cached_samples: List[Dict] = cached.get("samples", [])

    # Build map from chunk -> list of cached samples
    samples_by_chunk: Dict[str, List[Dict]] = {}
    for samp in cached_samples:
        samp.setdefault("label_chunk_path", None)
        samples_by_chunk.setdefault(samp["chunk_path"], []).append(samp)

    # Precompute label chunk paths for all chunks (if requested).
    label_chunk_lookup: Dict[str, Optional[str]] = {}
    if use_label_chunks:
        for chk_path in chunk_files:
            lc_path = label_chunk_path_for_chunk(
                chk_path,
                label_chunk_root=label_chunk_root,
                label_chunk_subdir=label_chunk_subdir,
                label_chunk_ext=label_chunk_ext,
            )
            if lc_path.is_file():
                label_chunk_lookup[str(chk_path)] = str(lc_path)

    new_samples: List[Dict] = []
    seen_frames = set()
    labels_root = labels_root if labels_root else None
    skipped_missing = 0

    for idx, chk_path in enumerate(chunk_files, start=1):
        chk_key = str(chk_path)
        meta_now = _chunk_stat(chk_path)
        meta_cached = cached_meta.get(chk_key)
        if meta_cached == meta_now and chk_key in samples_by_chunk:
            # reuse cached samples, respecting dedup of overlapping frames
            for samp in samples_by_chunk[chk_key]:
                samp["label_chunk_path"] = label_chunk_lookup.get(chk_key)
                has_chunk = samp.get("label_chunk_path") and Path(str(samp["label_chunk_path"])).is_file()
                has_png = Path(samp.get("label_path", "")).is_file()
                if not has_chunk and not has_png:
                    skipped_missing += 1
                    continue
                if samp["frame_path"] in seen_frames:
                    continue
                seen_frames.add(samp["frame_path"])
                new_samples.append(samp)
            continue

        # need to read chunk
        chk = torch.load(chk_path, map_location="cpu")
        scene_id = chk["scene_id"]
        scene_dir = chk_path.parent.parent  # .../<scene>/
        frame_paths = chk["frame_paths"]
        image_size = chk["image_size"]
        label_chunk_path = label_chunk_lookup.get(chk_key)
        for fidx, fpath in enumerate(frame_paths):
            if fpath in seen_frames:
                continue
            seen_frames.add(fpath)
            label_path = label_path_for_frame(scene_dir, fpath, labels_root=labels_root)
            if not label_path.is_file() and not label_chunk_path:
                skipped_missing += 1
                continue
            new_samples.append(
                {
                    "scene_id": scene_id,
                    "frame_path": fpath,
                    "label_path": str(label_path),
                    "label_chunk_path": label_chunk_path,
                    "chunk_path": chk_key,
                    "frame_idx": fidx,
                    "image_size": image_size,
                }
            )
        cached_meta[chk_key] = meta_now
        if idx % 50 == 0:
            print(f"[info] indexed {idx}/{len(chunk_files)} chunks ({len(new_samples)} samples)")
        del chk

    if skipped_missing:
        print(f"[warn] skipped {skipped_missing} samples due to missing labels")

    if cache_path:
        try:
            torch.save({"chunk_meta": cached_meta, "samples": new_samples}, cache_path)
            print(f"[info] saved index cache to {cache_path} ({len(new_samples)} samples)")
        except Exception as e:
            print(f"[warn] failed to save index cache {cache_path}: {e}")
    return new_samples


def label_path_for_frame(scene_dir: Path, frame_path: str, labels_root: Optional[Path] = None) -> Path:
    fname = Path(frame_path).name + ".png"
    if labels_root:
        return labels_root / scene_dir.name / fname
    return scene_dir / "labels" / fname


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


def _parse_ignore_classes(arg: Optional[str]) -> Set[int]:
    ignore: Set[int] = set()
    if not arg:
        return ignore
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        ignore.add(int(item))
    return ignore


class ChunkDataset(Dataset):
    """
    Streams per-frame samples from exported chunks, deduplicating overlaps.
    Uses an LRU cache of loaded chunks to avoid holding everything in memory.
    """

    def __init__(
        self,
        samples: Sequence[Dict],
        *,
        ignore_classes: Optional[Sequence[int]] = None,
        ignore_value: Optional[int] = None,
        remap_dict: Optional[Dict[int, int]] = None,
        labels_root: Optional[Path] = None,
        cache_size: int = 2,
        label_cache_size: int = 0,
        label_chunk_cache_size: int = 2,
        use_label_chunks: bool = True,
    ) -> None:
        self.samples: List[Dict] = list(samples)
        self.ignore_classes: Set[int] = set(ignore_classes or [])
        self.ignore_value = ignore_value
        self.remap_dict = remap_dict or {}
        self.labels_root = labels_root
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[Path, Dict] = OrderedDict()
        self.use_label_chunks = use_label_chunks

        # Optional cache of decoded label PNGs (per worker process).
        self.label_cache_size = max(0, int(label_cache_size))
        self._label_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        # Optional cache of loaded label chunks (per worker process).
        self.label_chunk_cache_size = max(1, int(label_chunk_cache_size))
        self._label_chunk_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

        # Fast label remap via LUT (vectorized). This avoids Python loops per sample.
        # We keep a reasonably sized LUT for 16-bit label PNGs.
        lut_size = 65536
        self._lut = torch.arange(lut_size, dtype=torch.long)
        if self.ignore_value is not None and self.ignore_classes:
            for cls in self.ignore_classes:
                if 0 <= cls < lut_size:
                    self._lut[cls] = int(self.ignore_value)
        if self.remap_dict and self.ignore_value is not None:
            # Default all to ignore, then set known ids.
            self._lut.fill_(int(self.ignore_value))
            for orig, new in self.remap_dict.items():
                if 0 <= orig < lut_size:
                    self._lut[orig] = int(new)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        label_chunk_path = sample.get("label_chunk_path")
        label: Optional[torch.Tensor] = None
        if self.use_label_chunks and label_chunk_path:
            try:
                label = self._get_label_from_chunk(Path(label_chunk_path), sample["frame_idx"])
            except (FileNotFoundError, KeyError, IndexError, ValueError):
                label = None
        if label is None:
            label_path = Path(sample["label_path"])
            label = self._get_label(label_path)
        # Vectorized remap/ignore.
        if self._lut is not None:
            # Clamp for safety in case of unexpected values.
            label = self._lut[label.clamp(0, self._lut.numel() - 1)]
        chk = self._get_chunk(sample["chunk_path"])
        dino = chk["dino_features"][0, sample["frame_idx"]]
        dpt_levels = [lvl[0, sample["frame_idx"]] for lvl in chk["dpt_pyramid"]]
        return {
            "scene_id": sample["scene_id"],
            "frame_path": sample["frame_path"],
            "dino": dino,
            "dpt_levels": dpt_levels,
            "label": label,
            "image_size": sample["image_size"],
        }

    def _get_label(self, path: Path) -> torch.Tensor:
        if self.label_cache_size <= 0:
            return load_label_png(path)
        if path in self._label_cache:
            t = self._label_cache.pop(path)
            self._label_cache[path] = t
            return t
        t = load_label_png(path)
        self._label_cache[path] = t
        if len(self._label_cache) > self.label_cache_size:
            self._label_cache.popitem(last=False)
        return t

    def _get_label_from_chunk(self, path: Path, frame_idx: int) -> torch.Tensor:
        path = Path(path)
        if path in self._label_chunk_cache:
            labels = self._label_chunk_cache.pop(path)
            self._label_chunk_cache[path] = labels
        else:
            payload = torch.load(path, map_location="cpu")
            if "labels" not in payload:
                raise KeyError(f"Label chunk missing 'labels' tensor: {path}")
            labels = payload["labels"]
            if labels.dim() < 3:
                raise ValueError(f"Label chunk tensor has unexpected shape {labels.shape} in {path}")
            self._label_chunk_cache[path] = labels
            if len(self._label_chunk_cache) > self.label_chunk_cache_size:
                self._label_chunk_cache.popitem(last=False)
        if frame_idx >= labels.shape[0]:
            raise IndexError(f"frame_idx {frame_idx} out of bounds for label chunk {path}")
        return labels[frame_idx].long()

    def _get_chunk(self, path: Path) -> Dict:
        path = Path(path)
        # Simple LRU cache
        if path in self._cache:
            chk = self._cache.pop(path)
            self._cache[path] = chk
            return chk
        chk = torch.load(path, map_location="cpu")
        self._cache[path] = chk
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return chk


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
        if isinstance(device, str):
            device = torch.device(device)
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
        target_dtype = next(self.parameters()).dtype
        dino = dino.to(dtype=target_dtype)
        dpt_levels = [lvl.to(dtype=target_dtype) for lvl in dpt_levels]
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


class ChunkShuffleSampler(Sampler[int]):
    """Shuffle with better disk locality.

    We group indices by chunk_path. Each epoch we shuffle the order of chunks,
    and shuffle indices within each chunk.
    """

    def __init__(self, samples: Sequence[Dict], *, seed: int = 0) -> None:
        self.seed = int(seed)
        self._chunk_to_indices: Dict[str, List[int]] = {}
        for i, s in enumerate(samples):
            self._chunk_to_indices.setdefault(s["chunk_path"], []).append(i)
        self._chunks: List[str] = list(self._chunk_to_indices.keys())
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(len(v) for v in self._chunk_to_indices.values())

    def __iter__(self):
        g = torch.Generator()
        # seed==0 => let PyTorch default randomness vary; otherwise stable per epoch
        if self.seed:
            g.manual_seed(self.seed + self.epoch)
        chunks = self._chunks
        if self.seed:
            perm = torch.randperm(len(chunks), generator=g).tolist()
            chunks = [chunks[i] for i in perm]
        else:
            chunks = [chunks[i] for i in torch.randperm(len(chunks)).tolist()]

        for chk in chunks:
            idxs = self._chunk_to_indices[chk]
            if len(idxs) <= 1:
                for i in idxs:
                    yield i
                continue
            if self.seed:
                local_perm = torch.randperm(len(idxs), generator=g).tolist()
            else:
                local_perm = torch.randperm(len(idxs)).tolist()
            for j in local_perm:
                yield idxs[j]


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=ignore_index)
    valid = targets != ignore_index
    if not valid.any():
        return ce.sum() * 0.0
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    if alpha is not None and alpha > 0:
        focal = alpha * focal
    return focal[valid].mean()


def _auto_checkpoint_name(ckpt_dir: Path) -> str:
    today = datetime.datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf"film_m2f_{today}_run_(\d+)\.pt")
    existing = [p.name for p in ckpt_dir.glob(f"film_m2f_{today}_run_*.pt")]
    runs = []
    for name in existing:
        m = pattern.match(name)
        if m:
            try:
                runs.append(int(m.group(1)))
            except ValueError:
                continue
    next_run = (max(runs) + 1) if runs else 1
    return f"film_m2f_{today}_run_{next_run:02d}.pt"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reasonable defaults for speed on modern NVIDIA GPUs.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    scenes_from_file = _load_list_from_file(args.scenes_file)
    if scenes_from_file and args.scenes:
        print("[info] scenes-file provided; overriding --scenes.")
    scenes_cli = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    scenes = scenes_from_file or scenes_cli

    ignore_classes = _parse_ignore_classes(args.ignore_classes)
    ignore_from_file = _load_list_from_file(args.ignore_classes_file)
    if ignore_from_file:
        ignore_classes.update(int(s) for s in ignore_from_file)
    if ignore_classes:
        print(f"[info] remapping {len(ignore_classes)} classes to ignore_index={args.ignore_index}")

    remap_classes = _load_list_from_file(args.remap_classes_file)
    remap_dict: Dict[int, int] = {}
    if remap_classes:
        remap_classes = [int(x) for x in remap_classes]
        remap_classes = sorted(set(remap_classes))
        remap_dict = {orig: new for new, orig in enumerate(remap_classes)}
        if args.num_classes == 200:  # default value implies not set explicitly
            args.num_classes = len(remap_classes)
            print(f"[info] remap_classes_file provided; setting num_classes={args.num_classes}")
        else:
            print(f"[info] remap_classes_file provided; keeping user num_classes={args.num_classes}")

    chunk_files = list_chunk_files(Path(args.dataset_root), scenes=scenes)
    if args.max_chunks:
        chunk_files = chunk_files[: args.max_chunks]

    samples = build_or_load_index(
        chunk_files=chunk_files,
        labels_root=Path(args.labels_root).expanduser() if args.labels_root else None,
        cache_path=Path(args.index_cache).expanduser() if args.index_cache else None,
        label_chunk_root=Path(args.label_chunks_root).expanduser() if args.label_chunks_root else None,
        label_chunk_subdir=args.label_chunk_subdir,
        label_chunk_ext=args.label_chunk_ext,
        use_label_chunks=args.use_label_chunks,
    )

    ds = ChunkDataset(
        samples,
        ignore_classes=sorted(ignore_classes),
        ignore_value=args.ignore_index,
        remap_dict=remap_dict,
        labels_root=Path(args.labels_root).expanduser() if args.labels_root else None,
        cache_size=max(1, args.chunk_cache_size),
        label_cache_size=args.label_cache_size,
        label_chunk_cache_size=args.label_chunk_cache_size,
        use_label_chunks=args.use_label_chunks,
    )
    sampler: Optional[Sampler[int]] = None
    shuffle = False
    if args.shuffle_mode == "global":
        shuffle = True
    elif args.shuffle_mode == "chunk":
        sampler = ChunkShuffleSampler(samples, seed=args.seed)
    elif args.shuffle_mode == "none":
        shuffle = False
    else:
        raise ValueError(f"Unknown shuffle mode: {args.shuffle_mode}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model = FusionMask2Former(
        device=device,
        num_classes=args.num_classes,
        config_path=args.config_path,
        weights_path=args.weights_path,
    )
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"[resume] loading {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.train()

    # Train FiLM + Mask2Former head
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_half and device.type == "cuda")

    # ############ DEBUG: count how many trainable params we actually update ############
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dbg] trainable parameters: {num_params}")
    # ############ END DEBUG ###########################################################
    print(f"[info] dataset samples: {len(ds)} from {len(chunk_files)} chunks")

    ckpt_dir = Path(args.checkpoint_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.checkpoint_name if args.checkpoint_name else _auto_checkpoint_name(ckpt_dir)
    base_path = ckpt_dir / base_name
    base_stem = base_path.stem
    suffix = base_path.suffix or ".pt"

    def _epoch_ckpt_path(epoch_idx: int) -> Path:
        return ckpt_dir / f"{base_stem}_epoch{epoch_idx:02d}{suffix}"

    for epoch in range(args.epochs):
        # Make shuffling deterministic per epoch when using our sampler.
        if isinstance(sampler, ChunkShuffleSampler):
            sampler.set_epoch(epoch)
        running = 0.0
        for step, batch in enumerate(tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")):
            dino = batch["dino"].to(device, non_blocking=True)
            dpt_levels = [lvl.to(device, non_blocking=True) for lvl in batch["dpt_levels"]]
            # labels list; assume all same H,W within batch
            labels = batch["labels"]
            H, W = labels[0].shape[-2:]
            label_tensor = torch.stack(labels, dim=0).to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_half and device.type == "cuda"):
                cls_logits, mask_logits = model(dino, dpt_levels, label_shape=(H, W))
                S_frames = cls_logits.shape[1] if cls_logits.dim() == 4 else 1
                seg_logits = dense_logits_from_queries(cls_logits, mask_logits, B=dino.shape[0], S=S_frames)  # [B, S, C, H', W'] or [B,C,H',W']
                if seg_logits.dim() == 5:
                    seg_logits = seg_logits.reshape(dino.shape[0] * S_frames, *seg_logits.shape[2:])
                # Compute loss at the native mask resolution to save memory; downsample labels instead of upsampling logits.
                target_size = seg_logits.shape[-2:]
                label_down = F.interpolate(label_tensor.unsqueeze(1).float(), size=target_size, mode="nearest").squeeze(1).long()
            loss = focal_loss(
                seg_logits,
                label_down,
                ignore_index=args.ignore_index,
                alpha=args.focal_alpha,
                gamma=args.focal_gamma,
            )

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if (step + 1) % 10 == 0:
                avg = running / 10
                print(f"[epoch {epoch+1}] step {step+1} loss {avg:.4f}")
                running = 0.0

        if args.save_every and (epoch + 1) % args.save_every == 0:
            ckpt_path = _epoch_ckpt_path(epoch + 1)
            torch.save(model.state_dict(), ckpt_path)
            print(f"[checkpoint] saved weights at epoch {epoch+1} -> {ckpt_path}")

    torch.save(model.state_dict(), base_path)
    print(f"Training finished. Saved final checkpoint to {base_path}")


if __name__ == "__main__":
    main()
