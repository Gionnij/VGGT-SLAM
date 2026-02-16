#!/usr/bin/env python3
"""
train_film_m2f_optimized_png.py
===============================

Offline trainer that consumes pre-exported VGGT embeddings (DPT pyramid + DINO fmap)
and ScanNet++ 2D label PNGs to fine-tune a FiLM fusion module plus the Mask2Former
semantic head.

It expects embeddings exported by scripts/export_embeddings.py and label PNGs from
the ScanNet++ rasterizer. Overlapping frames across chunks are deduplicated by
keeping the first occurrence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import OrderedDict, deque
import datetime
import time
import itertools
import os
import statistics
import math
import sys
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor

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
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: auto).",
    )
    p.add_argument(
        "--num-workers-auto",
        action="store_true",
        help="Use automatic num_workers selection (overrides --num-workers if set).",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-fusion", type=float, default=None, help="Optional LR override for FiLM fusion params.")
    p.add_argument("--lr-head", type=float, default=None, help="Optional LR override for Mask2Former head params.")
    p.add_argument(
        "--init-fusion-from",
        help="Optional checkpoint path to initialize FiLM fusion weights from (e.g., a known-good run).",
    )
    p.add_argument(
        "--freeze-fusion",
        action="store_true",
        help="Freeze FiLM fusion parameters (no grads, lr=0).",
    )
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
        "--resume",
        default="none",
        help="Resume from 'auto' (latest in ckpt-dir), 'none', or a checkpoint path.",
    )
    p.add_argument("--use-half", action="store_true", help="Use mixed precision training.")
    p.add_argument("--focal-alpha", type=float, default=0.25, help="Alpha weighting for focal loss.")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma exponent for focal loss.")
    p.add_argument(
        "--loss-input",
        choices=["probs", "logprobs", "loglse"],
        default="probs",
        help=(
            "Interpret seg scores as probs (legacy), logprobs (log of dense probs), "
            "or logprobs via logsumexp over queries."
        ),
    )
    p.add_argument(
        "--loss-eps",
        type=float,
        default=1e-6,
        help="Epsilon for log-prob computation when --loss-input=logprobs.",
    )
    p.add_argument(
        "--loss-scale",
        type=float,
        default=1.0,
        help="Optional scalar to multiply the loss (useful for tiny gradients).",
    )
    p.add_argument(
        "--loss-reduction",
        choices=["mean", "sum", "batch"],
        default="mean",
        help="How to reduce per-pixel loss: mean, sum, or sum divided by batch size.",
    )
    p.add_argument(
        "--freeze-head-epochs",
        type=int,
        default=0,
        help="Freeze Mask2Former head for the first N epochs (LR=0, grads off).",
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
        "--prefetch-next-chunk",
        dest="prefetch_next_chunk",
        action="store_true",
        help="Prefetch the next chunk asynchronously when num_workers=0.",
    )
    p.add_argument(
        "--no-prefetch-next-chunk",
        dest="prefetch_next_chunk",
        action="store_false",
        help="Disable next-chunk prefetch even when num_workers=0.",
    )
    p.set_defaults(prefetch_next_chunk=None)
    p.add_argument(
        "--chunk-format",
        choices=["auto", "pt", "safetensors"],
        default="auto",
        help="Chunk storage format to load (auto prefers safetensors when present).",
    )
    p.add_argument(
        "--diagnose-dataloader",
        type=int,
        default=0,
        help="Log fetch/step timings for the first N training iterations (0 disables).",
    )
    p.add_argument(
        "--check-chunk-shuffle",
        type=int,
        default=0,
        help="Iterate M batches and report chunk locality without training (0 disables).",
    )
    p.add_argument(
        "--debug-labels",
        type=int,
        default=0,
        help="Print label stats for the first N steps (0 disables).",
    )
    p.add_argument(
        "--debug-grads",
        type=int,
        default=0,
        help="Print gradient norms for the first N steps (0 disables).",
    )
    p.add_argument(
        "--debug-remap",
        type=int,
        default=0,
        help="Print raw vs remapped label stats for the first N steps (0 disables).",
    )
    p.add_argument(
        "--debug-probs",
        type=int,
        default=0,
        help="Print log-prob normalization stats for the first N steps (0 disables).",
    )
    p.add_argument(
        "--debug-preds",
        type=int,
        default=0,
        help="Print top-k predicted/label class histograms for the first N steps (0 disables).",
    )
    p.add_argument(
        "--debug-masks",
        type=int,
        default=0,
        help="Print mask logits stats for the first N steps (0 disables).",
    )
    p.add_argument(
        "--debug-classes",
        type=int,
        default=0,
        help="Print class-logit stats for the first N steps (0 disables).",
    )
    p.add_argument(
        "--debug-per-epoch",
        action="store_true",
        help="Reset debug counters each epoch (prints N steps per epoch instead of per run).",
    )
    p.add_argument(
        "--mask-logit-temp",
        type=float,
        default=1.0,
        help="Divide mask logits by this temperature before loss (1.0 disables).",
    )
    p.add_argument(
        "--mask-logit-clamp",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Clamp mask logits to [MIN, MAX] before loss (disabled if unset).",
    )
    p.add_argument(
        "--no-debug-print",
        action="store_true",
        help="Disable per-batch debug prints to reduce overhead.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log loss/grad stats every N steps.",
    )
    p.add_argument(
        "--suppress-warnings",
        action="store_true",
        help="Suppress Python warnings for cleaner output.",
    )
    p.add_argument(
        "--run-dir",
        default="./runs/film_m2f",
        help="Run directory for checkpoints/logs (checkpoints are written to <run-dir>/checkpoints).",
    )
    p.add_argument(
        "--ckpt-dir",
        help="Optional override for checkpoint directory (defaults to <run-dir>/checkpoints).",
    )
    p.add_argument(
        "--save-every-minutes",
        type=int,
        default=10,
        help="Minutes between checkpoint saves (0 disables interval-based saves).",
    )
    return p.parse_args()


def _get_nproc() -> Optional[int]:
    try:
        return int(os.sysconf("SC_NPROCESSORS_ONLN"))
    except (AttributeError, ValueError, OSError):
        return None


def _read_env_int(name: str) -> Optional[int]:
    val = os.getenv(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _is_deepstore_path(path: Path) -> bool:
    path_str = str(path)
    return path_str.startswith("/deepstore") or "/deepstore/" in path_str


def _auto_num_workers(dataset_root: Path) -> int:
    if _is_deepstore_path(dataset_root):
        return 0
    return 1


def _log_startup_info(
    args: argparse.Namespace,
    *,
    pin_memory: bool,
    prefetch_factor: Optional[int],
    persistent_workers: bool,
    num_workers: int,
) -> None:
    cpu_count = os.cpu_count()
    nproc = _get_nproc()
    nproc_str = str(nproc) if nproc is not None else "n/a"
    print(f"[info] cpu_count={cpu_count} nproc={nproc_str}")
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK") or "unset"
    slurm_job_cpus = os.getenv("SLURM_JOB_CPUS_PER_NODE") or "unset"
    omp_threads = os.getenv("OMP_NUM_THREADS") or "unset"
    print(
        f"[info] env SLURM_CPUS_PER_TASK={slurm_cpus} "
        f"SLURM_JOB_CPUS_PER_NODE={slurm_job_cpus} "
        f"OMP_NUM_THREADS={omp_threads}"
    )
    print(
        f"[info] dataloader num_workers={num_workers} "
        f"prefetch_factor={prefetch_factor} "
        f"persistent_workers={persistent_workers} "
        f"pin_memory={pin_memory}"
    )
    slurm_cpus_int = _read_env_int("SLURM_CPUS_PER_TASK")
    if slurm_cpus_int is not None and slurm_cpus_int < num_workers:
        print(
            f"[warn] SLURM_CPUS_PER_TASK={slurm_cpus_int} < num_workers={num_workers}; "
            "worker oversubscription likely."
        )


def _short_chunk_id(chunk_path: str, dataset_root: Optional[Path]) -> str:
    path = Path(chunk_path)
    if dataset_root is not None:
        try:
            rel = path.relative_to(dataset_root)
            return str(rel)
        except ValueError:
            pass
    return path.name


def _format_histogram(values: Sequence[int]) -> str:
    if not values:
        return "n/a"
    max_val = max(values)
    counts = [0] * (max_val + 1)
    for v in values:
        if 0 <= v <= max_val:
            counts[v] += 1
    parts = [f"{i}:{counts[i]}" for i in range(1, max_val + 1)]
    return " ".join(parts)


def _format_eta_minutes(minutes: float) -> str:
    if minutes < 0:
        minutes = 0
    total = int(minutes + 0.5)
    hours = total // 60
    mins = total % 60
    return f"{hours}h {mins:02d}m"


def _chunk_order_from_samples(samples: Sequence[Dict]) -> List[str]:
    order: List[str] = []
    seen: Set[str] = set()
    for sample in samples:
        chk = sample.get("chunk_path")
        if chk and chk not in seen:
            seen.add(chk)
            order.append(chk)
    return order


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
    from safetensors.torch import load_file

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
    }


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


def _chunk_stat(path: Path) -> Dict[str, int]:
    st = path.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def build_or_load_index(
    *,
    chunk_files: Sequence[Path],
    labels_root: Optional[Path],
    cache_path: Optional[Path],
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
        samples_by_chunk.setdefault(samp["chunk_path"], []).append(samp)

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
                if not Path(samp["label_path"]).is_file():
                    skipped_missing += 1
                    continue
                if samp["frame_path"] in seen_frames:
                    continue
                seen_frames.add(samp["frame_path"])
                new_samples.append(samp)
            continue

        # need to read chunk metadata
        chk = _load_chunk_metadata(chk_path)
        scene_id = chk["scene_id"]
        scene_dir = chk_path.parent.parent  # .../<scene>/
        frame_paths = chk.get("frame_paths")
        if frame_paths is None:
            raise KeyError(f"Chunk metadata missing frame_paths: {chk_path}")
        image_size = chk["image_size"]
        for fidx, fpath in enumerate(frame_paths):
            if fpath in seen_frames:
                continue
            seen_frames.add(fpath)
            label_path = label_path_for_frame(scene_dir, fpath, labels_root=labels_root)
            if not label_path.is_file():
                skipped_missing += 1
                continue
            new_samples.append(
                {
                    "scene_id": scene_id,
                    "frame_path": fpath,
                    "label_path": str(label_path),
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


def _load_fusion_state(path_str: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt_path = Path(path_str).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Fusion init checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model_state") if isinstance(state, dict) else None
    if model_state is None and isinstance(state, dict):
        model_state = state
    if model_state is None or not isinstance(model_state, dict):
        raise RuntimeError(f"Unexpected checkpoint format for fusion init: {ckpt_path}")
    fusion_state = {}
    for k, v in model_state.items():
        if k.startswith("fusion."):
            fusion_state[k.replace("fusion.", "", 1)] = v
        elif k.startswith("mlps."):
            fusion_state[k] = v
    if not fusion_state:
        raise RuntimeError(f"No fusion weights found in checkpoint: {ckpt_path}")
    return fusion_state


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
        prefetch_next_chunk: bool = False,
    ) -> None:
        self.samples: List[Dict] = list(samples)
        self.ignore_classes: Set[int] = set(ignore_classes or [])
        self.ignore_value = ignore_value
        self.remap_dict = remap_dict or {}
        self.labels_root = labels_root
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[Path, Dict] = OrderedDict()

        # Optional cache of decoded label PNGs (per worker process).
        self.label_cache_size = max(0, int(label_cache_size))
        self._label_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._cache_lock = threading.Lock()

        self._prefetch_enabled = bool(prefetch_next_chunk)
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_inflight: Dict[Path, Future] = {}
        # Best-effort cache stats (only reliable when num_workers=0).
        self._chunk_cache_hits = 0
        self._chunk_cache_misses = 0
        self._label_cache_hits = 0
        self._label_cache_misses = 0

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
            "chunk_path": sample["chunk_path"],
            "dino": dino,
            "dpt_levels": dpt_levels,
            "label": label,
            "label_path": str(label_path),
            "image_size": sample["image_size"],
        }

    def reset_cache_stats(self) -> None:
        self._chunk_cache_hits = 0
        self._chunk_cache_misses = 0
        self._label_cache_hits = 0
        self._label_cache_misses = 0

    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        return {
            "chunk_cache": {
                "hits": self._chunk_cache_hits,
                "misses": self._chunk_cache_misses,
                "enabled": 1,
            },
            "label_cache": {
                "hits": self._label_cache_hits,
                "misses": self._label_cache_misses,
                "enabled": 1 if self.label_cache_size > 0 else 0,
            },
        }

    def _ensure_prefetch_executor(self) -> ThreadPoolExecutor:
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk_prefetch")
        return self._prefetch_executor

    def prefetch_chunk(self, path: Optional[str]) -> None:
        if not self._prefetch_enabled or not path:
            return
        chunk_path = Path(path)
        if not chunk_path.is_file():
            return
        with self._cache_lock:
            if chunk_path in self._cache or chunk_path in self._prefetch_inflight:
                return
            executor = self._ensure_prefetch_executor()
            fut = executor.submit(self._load_chunk_from_disk, chunk_path)
            self._prefetch_inflight[chunk_path] = fut

    def _get_label(self, path: Path) -> torch.Tensor:
        if self.label_cache_size <= 0:
            return load_label_png(path)
        if path in self._label_cache:
            t = self._label_cache.pop(path)
            self._label_cache[path] = t
            self._label_cache_hits += 1
            return t
        self._label_cache_misses += 1
        t = load_label_png(path)
        self._label_cache[path] = t
        if len(self._label_cache) > self.label_cache_size:
            self._label_cache.popitem(last=False)
        return t

    def _get_chunk(self, path: Path) -> Dict:
        path = Path(path)
        # Simple LRU cache with optional prefetch support.
        with self._cache_lock:
            if path in self._cache:
                chk = self._cache.pop(path)
                self._cache[path] = chk
                self._chunk_cache_hits += 1
                return chk
            fut = self._prefetch_inflight.pop(path, None)
        if fut is not None:
            try:
                chk = fut.result()
                if chk is not None:
                    self._chunk_cache_hits += 1
            except Exception:
                chk = None
        else:
            chk = None
        if chk is None:
            self._chunk_cache_misses += 1
            chk = self._load_chunk_from_disk(path)
        with self._cache_lock:
            self._cache[path] = chk
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return chk

    def _load_chunk_from_disk(self, path: Path) -> Dict:
        if path.suffix == ".safetensors":
            return _load_safetensors_chunk(path)
        return torch.load(path, map_location="cpu")


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


def dense_logprobs_from_queries(
    cls_logits: torch.Tensor, mask_logits: torch.Tensor, B: int, S: int
) -> torch.Tensor:
    """
    Compute per-class log-probabilities from query logits using logsumexp over queries.
    Returns [B,S,C,H,W] or [B,C,H,W] depending on inputs.
    """
    if cls_logits.dim() == 4:
        cls_flat = cls_logits.reshape(B * S, *cls_logits.shape[-2:])
        mask_flat = mask_logits.reshape(B * S, *mask_logits.shape[-3:])
    else:
        cls_flat = cls_logits
        mask_flat = mask_logits

    log_p_class = F.log_softmax(cls_flat, dim=-1)[..., :-1]  # drop no-object
    log_p_mask = F.logsigmoid(mask_flat)

    # [B*S, C, Q, 1, 1] + [B*S, 1, Q, H, W] -> logsumexp over Q
    log_p_class = log_p_class.transpose(1, 2)  # [B*S, C, Q]
    log_p = torch.logsumexp(
        log_p_class.unsqueeze(-1).unsqueeze(-1) + log_p_mask.unsqueeze(1),
        dim=2,
    )

    if cls_logits.dim() == 4:
        log_p = log_p.reshape(B, S, *log_p.shape[1:])
    return log_p


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
        "chunk_paths": [b["chunk_path"] for b in batch],
        "label_paths": [b["label_path"] for b in batch],
        "image_sizes": [b["image_size"] for b in batch],
    }


def _maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed_dl_iter(dl: DataLoader):
    it = iter(dl)
    while True:
        start = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            return
        yield time.perf_counter() - start, batch


def run_chunk_shuffle_check(
    *,
    dl: DataLoader,
    max_batches: int,
    dataset_root: Optional[Path],
) -> None:
    if max_batches <= 0:
        return
    unique_counts: List[int] = []
    mixed_batches: List[Tuple[int, int, List[str]]] = []
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= max_batches:
            break
        chunk_paths = batch.get("chunk_paths") or []
        chunk_ids = [_short_chunk_id(p, dataset_root) for p in chunk_paths]
        unique = len(set(chunk_ids))
        unique_counts.append(unique)
        if unique > 1:
            mixed_batches.append((unique, batch_idx, chunk_ids))
    if not unique_counts:
        print("[check] no batches produced by DataLoader.")
        return
    avg_unique = sum(unique_counts) / len(unique_counts)
    med_unique = statistics.median(unique_counts)
    hist = _format_histogram(unique_counts)
    mixed_ratio = sum(1 for v in unique_counts if v > 1) / len(unique_counts)
    print(
        f"[check] unique_chunks_per_batch avg={avg_unique:.2f} "
        f"median={med_unique:.1f} hist={hist} mixed_ratio={mixed_ratio:.2f}"
    )
    if mixed_batches:
        worst = sorted(mixed_batches, key=lambda x: x[0], reverse=True)[:5]
        for unique, batch_idx, chunk_ids in worst:
            uniq_ids = sorted(set(chunk_ids))
            print(
                f"[check] batch={batch_idx} unique_chunks={unique} chunks={uniq_ids}"
            )


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
        self.start_offset = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_offset(self, offset: int) -> None:
        self.start_offset = max(0, int(offset))

    def __len__(self) -> int:
        total = sum(len(v) for v in self._chunk_to_indices.values())
        if self.start_offset:
            return max(0, total - self.start_offset)
        return total

    def __iter__(self):
        g = self._get_epoch_generator()
        chunks = self._get_epoch_chunks(g)

        def _iter_indices():
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

        iterator = _iter_indices()
        if self.start_offset:
            iterator = itertools.islice(iterator, self.start_offset, None)
        yield from iterator

    def _get_epoch_generator(self) -> torch.Generator:
        g = torch.Generator()
        # seed==0 => let PyTorch default randomness vary; otherwise stable per epoch
        if self.seed:
            g.manual_seed(self.seed + self.epoch)
        return g

    def _get_epoch_chunks(self, g: Optional[torch.Generator] = None) -> List[str]:
        if g is None:
            g = self._get_epoch_generator()
        chunks = self._chunks
        if self.seed:
            perm = torch.randperm(len(chunks), generator=g).tolist()
            chunks = [chunks[i] for i in perm]
        else:
            chunks = [chunks[i] for i in torch.randperm(len(chunks)).tolist()]
        return chunks

    def get_epoch_chunk_order(self) -> List[str]:
        return self._get_epoch_chunks()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    alpha: float,
    gamma: float,
    reduction: str = "mean",
    batch_size: Optional[int] = None,
    input_type: str = "probs",
) -> torch.Tensor:
    valid = targets != ignore_index
    if input_type != "probs":
        logp = logits
        safe_targets = targets.clone()
        safe_targets[~valid] = 0
        logp_y = logp.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        ce = -logp_y
        pt = logp_y.exp()
        ce = ce.masked_fill(~valid, 0.0)
        pt = pt.masked_fill(~valid, 0.0)
    else:
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=ignore_index)
        pt = torch.exp(-ce)
    if not valid.any():
        return ce.sum() * 0.0
    focal = ((1 - pt) ** gamma) * ce
    if alpha is not None and alpha > 0:
        focal = alpha * focal
    if reduction == "sum":
        return focal[valid].sum()
    if reduction == "batch":
        denom = max(1, int(batch_size or 0))
        return focal[valid].sum() / float(denom)
    return focal[valid].mean()


def _summarize_labels(
    labels: torch.Tensor,
    *,
    name: str,
    ignore_index: int,
    num_classes: int,
) -> str:
    total = labels.numel()
    valid_mask = labels != ignore_index
    valid = int(valid_mask.sum().item())
    ignore = total - valid
    if valid > 0:
        valid_vals = labels[valid_mask]
        min_val = int(valid_vals.min().item())
        max_val = int(valid_vals.max().item())
    else:
        min_val = -1
        max_val = -1
    out_of_range = ((labels < 0) | (labels >= num_classes)) & valid_mask
    oor = int(out_of_range.sum().item())
    return (
        f"{name}: total={total} valid={valid} ignore={ignore} "
        f"min={min_val} max={max_val} oor={oor}"
    )


def _is_rank0() -> bool:
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    try:
        return torch.distributed.get_rank() == 0
    except Exception:
        return True


def _latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.is_dir():
        return None
    candidates = [p for p in ckpt_dir.glob("*.pt") if not p.name.endswith(".tmp")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _checkpoint_path(ckpt_dir: Path, epoch: int, global_step: int) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ckpt_dir / f"film_m2f_epoch{epoch:04d}_step{global_step:08d}_{ts}.pt"


def _atomic_save_checkpoint(path: Path, payload: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def main() -> None:
    args = parse_args()
    if args.suppress_warnings:
        warnings.filterwarnings("ignore")
    if args.mask_logit_temp <= 0:
        raise ValueError("--mask-logit-temp must be > 0")
    if args.mask_logit_clamp is not None:
        lo, hi = args.mask_logit_clamp
        if lo > hi:
            lo, hi = hi, lo
        args.mask_logit_clamp = (lo, hi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_root).expanduser()
    if args.num_workers_auto or args.num_workers is None:
        auto_workers = _auto_num_workers(dataset_root)
        if args.num_workers is not None and args.num_workers != auto_workers:
            print(f"[info] num_workers auto override {args.num_workers} -> {auto_workers}")
        else:
            print(f"[info] num_workers auto -> {auto_workers} (dataset_root={dataset_root})")
        args.num_workers = auto_workers
    if args.num_workers is None:
        args.num_workers = 0
    args.num_workers = max(0, int(args.num_workers))
    if _is_deepstore_path(dataset_root) and args.num_workers > 1:
        print(
            "[warn] num_workers>1 on deepstore can amplify I/O (per-worker caches duplicate chunk loads). "
            "Consider --num-workers 0 or 1."
        )
    if args.num_workers > 1:
        print(
            "[warn] num_workers>1 duplicates chunk/label caches per worker; can increase I/O on shared filesystems."
        )
    prefetch_next_chunk = args.prefetch_next_chunk
    if prefetch_next_chunk is None:
        prefetch_next_chunk = args.num_workers == 0
    if prefetch_next_chunk and args.num_workers != 0:
        print("[warn] next-chunk prefetch is only supported with num_workers=0; disabling.")
        prefetch_next_chunk = False
    pin_memory = True
    persistent_workers = args.persistent_workers and args.num_workers > 0
    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None
    _log_startup_info(
        args,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        num_workers=args.num_workers,
    )

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

    chunk_files = list_chunk_files(dataset_root, scenes=scenes, chunk_format=args.chunk_format)
    if args.max_chunks:
        chunk_files = chunk_files[: args.max_chunks]

    samples = build_or_load_index(
        chunk_files=chunk_files,
        labels_root=Path(args.labels_root).expanduser() if args.labels_root else None,
        cache_path=Path(args.index_cache).expanduser() if args.index_cache else None,
    )

    ds = ChunkDataset(
        samples,
        ignore_classes=sorted(ignore_classes),
        ignore_value=args.ignore_index,
        remap_dict=remap_dict,
        labels_root=Path(args.labels_root).expanduser() if args.labels_root else None,
        cache_size=max(1, args.chunk_cache_size),
        label_cache_size=args.label_cache_size,
        prefetch_next_chunk=prefetch_next_chunk,
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
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    diagnose_batches = max(0, int(args.diagnose_dataloader))
    check_batches = max(0, int(args.check_chunk_shuffle))
    if check_batches > 0:
        if isinstance(sampler, ChunkShuffleSampler):
            sampler.set_epoch(0)
            sampler.set_start_offset(0)
        run_chunk_shuffle_check(dl=dl, max_batches=check_batches, dataset_root=dataset_root)
        return

    model = FusionMask2Former(
        device=device,
        num_classes=args.num_classes,
        config_path=args.config_path,
        weights_path=args.weights_path,
    )
    model.train()

    # Train FiLM + Mask2Former head (optionally with separate LRs).
    fusion_params = list(model.fusion.parameters())
    head_params = list(model.sem_head.parameters())
    if args.init_fusion_from:
        fusion_state = _load_fusion_state(args.init_fusion_from, device=device)
        missing, unexpected = model.fusion.load_state_dict(fusion_state, strict=False)
        if missing:
            print(f"[info] fusion init missing keys: {missing}")
        if unexpected:
            print(f"[info] fusion init unexpected keys: {unexpected}")
        print(f"[info] initialized fusion from {args.init_fusion_from}")
    if args.freeze_fusion:
        for p in fusion_params:
            p.requires_grad = False
        print("[info] freezing fusion parameters (lr=0)")
    fusion_lr = float(args.lr_fusion) if args.lr_fusion is not None else float(args.lr)
    head_lr = float(args.lr_head) if args.lr_head is not None else float(args.lr)
    if args.freeze_fusion:
        fusion_lr = 0.0
    optim = torch.optim.AdamW(
        [
            {"params": fusion_params, "lr": fusion_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_half and device.type == "cuda")
    head_frozen = False
    head_lr_active = head_lr
    if int(args.freeze_head_epochs) > 0:
        for p in head_params:
            p.requires_grad = False
        head_frozen = True
        head_lr_active = 0.0
        optim.param_groups[1]["lr"] = head_lr_active
        print(f"[info] freezing head for {int(args.freeze_head_epochs)} epochs (head lr=0)")

    # ############ DEBUG: count how many trainable params we actually update ############
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dbg] trainable parameters: {num_params}")
    print(f"[info] lr fusion={fusion_lr:g} head={head_lr:g}")
    # ############ END DEBUG ###########################################################
    print(f"[info] dataset samples: {len(ds)} from {len(chunk_files)} chunks")

    run_dir = Path(args.run_dir).expanduser()
    ckpt_dir = Path(args.ckpt_dir).expanduser() if args.ckpt_dir else run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    request_save_file = run_dir / "REQUEST_SAVE"
    for d in (run_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    scheduler = None  # placeholder for future schedulers; still checkpointed for completeness

    global_step = 0
    start_epoch = 0
    start_step_in_epoch = 0

    resume_arg = (args.resume or "none").strip().lower()
    resume_path: Optional[Path] = None
    if resume_arg == "auto":
        resume_path = _latest_checkpoint(ckpt_dir)
        if resume_path:
            print(f"[resume] auto-selected latest checkpoint: {resume_path}")
    elif resume_arg not in ("", "none"):
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    if resume_path:
        state = torch.load(resume_path, map_location=device)
        model_state = state.get("model_state") if isinstance(state, dict) else None
        if model_state is None and isinstance(state, dict):
            model_state = state
        if not isinstance(model_state, dict):
            raise RuntimeError(f"Unexpected resume checkpoint format: {resume_path}")
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"[resume] missing model keys: {len(missing)}")
        if unexpected:
            print(f"[resume] unexpected model keys: {len(unexpected)}")
        if isinstance(state, dict) and "optimizer_state" in state:
            optim.load_state_dict(state["optimizer_state"])
        if scheduler and isinstance(state, dict) and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        if isinstance(state, dict) and "scaler_state" in state and state["scaler_state"] is not None:
            scaler.load_state_dict(state["scaler_state"])
        if isinstance(state, dict):
            start_epoch = int(state.get("epoch", 0))
            start_step_in_epoch = int(state.get("step_in_epoch", -1)) + 1
            if state.get("save_reason") == "epoch_end":
                start_epoch += 1
                start_step_in_epoch = 0
            start_step_in_epoch = max(0, start_step_in_epoch)
            global_step = int(state.get("global_step", 0))
        print(
            f"[resume] epoch={start_epoch+1} step_in_epoch={start_step_in_epoch} global_step={global_step}"
        )

    last_save_time = time.time()
    diagnose_active = diagnose_batches > 0
    diagnose_remaining = diagnose_batches
    diag_fetch_times: List[float] = []
    diag_step_times: List[float] = []
    diag_unique_counts: List[int] = []
    diag_summary_printed = False
    debug_labels_count = max(0, int(args.debug_labels))
    debug_grads_count = max(0, int(args.debug_grads))
    debug_remap_count = max(0, int(args.debug_remap))
    debug_probs_count = max(0, int(args.debug_probs))
    debug_preds_count = max(0, int(args.debug_preds))
    debug_masks_count = max(0, int(args.debug_masks))
    debug_classes_count = max(0, int(args.debug_classes))

    debug_labels_left = debug_labels_count
    debug_grads_left = debug_grads_count
    debug_remap_left = debug_remap_count
    debug_probs_left = debug_probs_count
    debug_preds_left = debug_preds_count
    debug_masks_left = debug_masks_count
    debug_classes_left = debug_classes_count

    def _print_diag_summary() -> None:
        nonlocal diag_summary_printed
        if diag_summary_printed or not diag_fetch_times:
            return
        avg_fetch = sum(diag_fetch_times) / len(diag_fetch_times)
        med_fetch = statistics.median(diag_fetch_times)
        avg_step = sum(diag_step_times) / len(diag_step_times)
        med_step = statistics.median(diag_step_times)
        avg_unique = sum(diag_unique_counts) / len(diag_unique_counts)
        hist = _format_histogram(diag_unique_counts)
        cache_info = "cache_stats=unavailable (num_workers>0)"
        if args.num_workers == 0:
            stats = ds.get_cache_stats()
            parts = []
            for key, info in stats.items():
                if not info.get("enabled"):
                    parts.append(f"{key}=disabled")
                    continue
                total = int(info.get("hits", 0)) + int(info.get("misses", 0))
                rate = (info.get("hits", 0) / total) if total else 0.0
                parts.append(f"{key}_hit_rate={rate:.2f}")
            cache_info = " ".join(parts)
        print(
            "[diag] summary "
            f"n={len(diag_fetch_times)} "
            f"t_fetch_ms_avg={avg_fetch * 1000:.1f} "
            f"t_fetch_ms_med={med_fetch * 1000:.1f} "
            f"t_step_ms_avg={avg_step * 1000:.1f} "
            f"t_step_ms_med={med_step * 1000:.1f} "
            f"unique_chunks_avg={avg_unique:.2f} "
            f"hist={hist} "
            f"{cache_info}"
        )
        diag_summary_printed = True

    def save_checkpoint(epoch_idx: int, step_in_epoch: int, reason: str) -> Optional[Path]:
        if not _is_rank0():
            return None
        ckpt_path = _checkpoint_path(ckpt_dir, epoch_idx + 1, global_step)
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch_idx,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "args": vars(args),
            "save_reason": reason,
        }
        out_path = _atomic_save_checkpoint(ckpt_path, payload)
        print(f"[checkpoint] saved ({reason}) -> {out_path}")
        return out_path

    steps_per_epoch = math.ceil(len(ds) / max(1, int(args.batch_size)))
    if start_epoch >= args.epochs:
        print(f"[info] start_epoch {start_epoch} >= total epochs {args.epochs}; nothing to train.")
        return

    resume_sample_offset = 0
    resume_step_in_epoch = start_step_in_epoch
    if start_step_in_epoch > 0:
        resume_sample_offset = start_step_in_epoch * int(args.batch_size)

    total_steps = steps_per_epoch * args.epochs
    completed_steps = start_epoch * steps_per_epoch
    use_tqdm = sys.stderr.isatty()
    log_every = max(1, int(args.log_every))
    eta_window = 1500
    train_start_time = time.perf_counter()
    steps_done = 0
    train_time_window: deque = deque(maxlen=eta_window)
    for epoch in range(start_epoch, args.epochs):
        # Make shuffling deterministic per epoch when using our sampler.
        if isinstance(sampler, ChunkShuffleSampler):
            sampler.set_epoch(epoch)
            if epoch == start_epoch and resume_sample_offset > 0:
                sampler.set_start_offset(resume_sample_offset)
                skip_steps = 0
            else:
                sampler.set_start_offset(0)
                skip_steps = 0
        else:
            skip_steps = start_step_in_epoch if epoch == start_epoch else 0
        if epoch > start_epoch:
            start_step_in_epoch = 0
        if head_frozen and epoch >= int(args.freeze_head_epochs):
            for p in head_params:
                p.requires_grad = True
            head_frozen = False
            head_lr_active = head_lr
            optim.param_groups[1]["lr"] = head_lr_active
            print(f"[info] unfreezing head at epoch {epoch+1} (head lr={head_lr_active:g})")
        next_chunk_map: Optional[Dict[str, str]] = None
        if prefetch_next_chunk and args.num_workers == 0:
            chunk_order: Optional[List[str]] = None
            if isinstance(sampler, ChunkShuffleSampler):
                chunk_order = sampler.get_epoch_chunk_order()
            elif args.shuffle_mode == "none":
                chunk_order = _chunk_order_from_samples(samples)
            if chunk_order:
                next_chunk_map = {
                    chunk_order[i]: chunk_order[i + 1] for i in range(len(chunk_order) - 1)
                }
        last_prefetched: Optional[str] = None
        step_in_epoch = -1
        running = 0.0
        epoch_start_time = time.perf_counter()
        epoch_steps_done = 0
        epoch_time_window: deque = deque(maxlen=eta_window)
        if args.debug_per_epoch:
            debug_labels_left = debug_labels_count
            debug_grads_left = debug_grads_count
            debug_remap_left = debug_remap_count
            debug_probs_left = debug_probs_count
            debug_preds_left = debug_preds_count
            debug_masks_left = debug_masks_count
            debug_classes_left = debug_classes_count
        diagnose_epoch = diagnose_active and epoch == start_epoch
        if diagnose_epoch and args.num_workers == 0:
            ds.reset_cache_stats()
        steps_this_epoch = steps_per_epoch
        resume_step = resume_step_in_epoch if epoch == start_epoch else 0
        resume_offset = resume_step if (resume_step > 0 and skip_steps == 0) else 0
        if resume_step > 0:
            completed_steps += resume_step
        data_iter = _timed_dl_iter(dl) if diagnose_epoch else dl
        epoch_bar = None
        if use_tqdm:
            epoch_bar = tqdm(
                total=steps_this_epoch,
                initial=resume_step,
                desc=f"epoch {epoch+1}/{args.epochs}",
                unit="step",
                leave=True,
                position=0,
                dynamic_ncols=True,
            )
        for step, data in enumerate(data_iter):
            if step < skip_steps:
                # Skip steps already completed before the checkpoint.
                continue
            if diagnose_epoch:
                t_fetch, batch = data
            else:
                t_fetch, batch = None, data
            step_abs = step + resume_offset
            diagnose_now = diagnose_epoch and diagnose_remaining > 0
            if diagnose_now:
                _maybe_synchronize(device)
                t_step_start = time.perf_counter()
            if next_chunk_map:
                chunk_paths = batch.get("chunk_paths") or []
                if chunk_paths:
                    unique_chunks = sorted(set(chunk_paths))
                    if len(unique_chunks) == 1:
                        next_chunk = next_chunk_map.get(unique_chunks[0])
                        if next_chunk and next_chunk != last_prefetched:
                            ds.prefetch_chunk(next_chunk)
                            last_prefetched = next_chunk
            dino = batch["dino"].to(device, non_blocking=True)
            dpt_levels = [lvl.to(device, non_blocking=True) for lvl in batch["dpt_levels"]]
            # labels list; assume all same H,W within batch
            labels = batch["labels"]
            H, W = labels[0].shape[-2:]
            label_tensor = torch.stack(labels, dim=0).to(device, non_blocking=True)
            if debug_remap_left > 0:
                label_paths = batch.get("label_paths") or []
                if label_paths:
                    raw_labels = [load_label_png(Path(p)) for p in label_paths]
                    raw_tensor = torch.stack(raw_labels, dim=0)
                    raw_valid = raw_tensor != args.ignore_index
                    remap_cpu = label_tensor.detach().cpu()
                    remap_valid = remap_cpu != args.ignore_index
                    raw_unique = torch.unique(raw_tensor[raw_valid]) if raw_valid.any() else torch.tensor([])
                    remap_unique = torch.unique(remap_cpu[remap_valid]) if remap_valid.any() else torch.tensor([])
                    raw_min = int(raw_unique.min().item()) if raw_unique.numel() else -1
                    raw_max = int(raw_unique.max().item()) if raw_unique.numel() else -1
                    remap_min = int(remap_unique.min().item()) if remap_unique.numel() else -1
                    remap_max = int(remap_unique.max().item()) if remap_unique.numel() else -1
                    print(
                        f"[dbg] remap raw_unique={raw_unique.numel()} raw_min={raw_min} raw_max={raw_max} "
                        f"remap_unique={remap_unique.numel()} remap_min={remap_min} remap_max={remap_max}"
                    )
                    debug_remap_left -= 1

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_half and device.type == "cuda"):
                cls_logits, mask_logits = model(dino, dpt_levels, label_shape=(H, W))
                if debug_classes_left > 0:
                    cls_stats = cls_logits.float()
                    print(
                        "[dbg] cls_logits stats "
                        f"min={cls_stats.min().item():.3f} "
                        f"max={cls_stats.max().item():.3f} "
                        f"mean={cls_stats.mean().item():.3f} "
                        f"std={cls_stats.std(unbiased=False).item():.3f}"
                    )
                    cls_probs = F.softmax(cls_stats, dim=-1)
                    if cls_probs.dim() == 4:
                        cls_probs_flat = cls_probs.reshape(-1, cls_probs.shape[-2], cls_probs.shape[-1])
                    else:
                        cls_probs_flat = cls_probs.reshape(-1, cls_probs.shape[-2], cls_probs.shape[-1])
                    top_vals, top_idx = cls_probs_flat[..., :-1].max(dim=-1)
                    noobj = cls_probs_flat[..., -1]
                    noobj_mean = float(noobj.mean().item())
                    top_mean = float(top_vals.mean().item())
                    noobj_gt = float((noobj > top_vals).float().mean().item())
                    top_hist = torch.bincount(
                        top_idx.reshape(-1).cpu(), minlength=args.num_classes
                    )
                    top = torch.topk(top_hist, k=min(5, args.num_classes))
                    top_pairs = list(zip(top.indices.tolist(), top.values.tolist()))
                    print(
                        "[dbg] cls_probs "
                        f"noobj_mean={noobj_mean:.3f} "
                        f"top_mean={top_mean:.3f} "
                        f"noobj_gt_top={noobj_gt:.3f} "
                        f"top5={top_pairs}"
                    )
                    debug_classes_left -= 1
                if args.mask_logit_temp != 1.0:
                    mask_logits = mask_logits / float(args.mask_logit_temp)
                if args.mask_logit_clamp is not None:
                    lo, hi = args.mask_logit_clamp
                    mask_logits = mask_logits.clamp(min=lo, max=hi)
                S_frames = cls_logits.shape[1] if cls_logits.dim() == 4 else 1
                if args.loss_input == "loglse":
                    seg_scores = dense_logprobs_from_queries(
                        cls_logits, mask_logits, B=dino.shape[0], S=S_frames
                    )
                else:
                    seg_scores = dense_logits_from_queries(
                        cls_logits, mask_logits, B=dino.shape[0], S=S_frames
                    )
                if seg_scores.dim() == 5:
                    seg_scores = seg_scores.reshape(dino.shape[0] * S_frames, *seg_scores.shape[2:])
                # Compute loss at the native mask resolution to save memory; downsample labels instead of upsampling logits.
                target_size = seg_scores.shape[-2:]
                label_down = F.interpolate(label_tensor.unsqueeze(1).float(), size=target_size, mode="nearest").squeeze(1).long()
                if debug_labels_left > 0:
                    msg_full = _summarize_labels(
                        label_tensor,
                        name="labels_full",
                        ignore_index=args.ignore_index,
                        num_classes=args.num_classes,
                    )
                    msg_down = _summarize_labels(
                        label_down,
                        name="labels_down",
                        ignore_index=args.ignore_index,
                        num_classes=args.num_classes,
                    )
                    print(
                        f"[dbg] {msg_full} | {msg_down} "
                        f"downsample_to={tuple(target_size)}"
                    )
                    debug_labels_left -= 1
                # Map any out-of-range labels to ignore_index to avoid NLL loss device asserts.
                invalid = (label_down != args.ignore_index) & (
                    (label_down < 0) | (label_down >= args.num_classes)
                )
                if invalid.any():
                    label_down = label_down.masked_fill(invalid, args.ignore_index)
                    if not args.no_debug_print and step == 0 and epoch == 0 and not args.suppress_warnings:
                        bad_count = int(invalid.sum().item())
                        print(
                            f"[warn] mapped {bad_count} labels outside [0,{args.num_classes - 1}] "
                            f"to ignore_index={args.ignore_index}"
                        )
                if args.loss_input == "probs":
                    seg_loss_input = seg_scores
                elif args.loss_input == "logprobs":
                    eps = float(args.loss_eps)
                    seg_log = (seg_scores + eps).log()
                    seg_loss_input = seg_log - torch.logsumexp(seg_log, dim=1, keepdim=True)
                else:  # loglse
                    seg_loss_input = seg_scores - torch.logsumexp(seg_scores, dim=1, keepdim=True)

                logp_debug = None
                if debug_probs_left > 0:
                    if args.loss_input == "probs":
                        logp_debug = F.log_softmax(seg_scores.float(), dim=1)
                    else:
                        logp_debug = seg_loss_input.float()
                    prob_sum = logp_debug.exp().sum(dim=1)
                    print(
                        "[dbg] logp stats "
                        f"logp_min={logp_debug.min().item():.3f} "
                        f"logp_max={logp_debug.max().item():.3f} "
                        f"sum_min={prob_sum.min().item():.3f} "
                        f"sum_mean={prob_sum.mean().item():.3f} "
                        f"sum_max={prob_sum.max().item():.3f}"
                    )
                    debug_probs_left -= 1

                if debug_preds_left > 0:
                    if logp_debug is None:
                        if args.loss_input == "probs":
                            logp_debug = F.log_softmax(seg_scores.float(), dim=1)
                        else:
                            logp_debug = seg_loss_input.float()
                    pred = logp_debug.argmax(dim=1)
                    pred_counts = torch.bincount(
                        pred.reshape(-1).cpu(), minlength=args.num_classes
                    )
                    valid_mask = label_down != args.ignore_index
                    if valid_mask.any():
                        label_counts = torch.bincount(
                            label_down[valid_mask].reshape(-1).cpu(),
                            minlength=args.num_classes,
                        )
                    else:
                        label_counts = torch.zeros(args.num_classes, dtype=torch.long)
                    pred_unique = int((pred_counts > 0).sum().item())
                    label_unique = int((label_counts > 0).sum().item())
                    pred_top = torch.topk(pred_counts, k=min(5, args.num_classes))
                    label_top = torch.topk(label_counts, k=min(5, args.num_classes))
                    pred_top_pairs = list(zip(pred_top.indices.tolist(), pred_top.values.tolist()))
                    label_top_pairs = list(zip(label_top.indices.tolist(), label_top.values.tolist()))
                    print(
                        "[dbg] preds "
                        f"pred_unique={pred_unique} label_unique={label_unique} "
                        f"pred_top5={pred_top_pairs} label_top5={label_top_pairs}"
                    )
                    debug_preds_left -= 1

                if debug_masks_left > 0:
                    mask_stats = mask_logits.float()
                    print(
                        "[dbg] mask_logits stats "
                        f"min={mask_stats.min().item():.3f} "
                        f"max={mask_stats.max().item():.3f} "
                        f"mean={mask_stats.mean().item():.3f} "
                        f"std={mask_stats.std(unbiased=False).item():.3f}"
                    )
                    debug_masks_left -= 1
            loss = focal_loss(
                seg_loss_input,
                label_down,
                ignore_index=args.ignore_index,
                alpha=args.focal_alpha,
                gamma=args.focal_gamma,
                reduction=args.loss_reduction,
                batch_size=label_down.shape[0],
                input_type=args.loss_input,
            )
            if args.loss_scale != 1.0:
                loss = loss * float(args.loss_scale)

            scaler.scale(loss).backward()
            unscaled = False
            if debug_grads_left > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                    unscaled = True
                fusion_w = model.fusion.mlps[0][0].weight
                head_param = None
                for p in model.sem_head.head.parameters():
                    head_param = p
                    break
                fusion_norm = float(fusion_w.grad.norm().item()) if fusion_w.grad is not None else 0.0
                head_norm = float(head_param.grad.norm().item()) if head_param is not None and head_param.grad is not None else 0.0
                print(
                    f"[dbg] grad_norm fusion={fusion_norm:.6f} head={head_norm:.6f} "
                    f"loss={float(loss.item()):.4f}"
                )
                debug_grads_left -= 1
            scaler.step(optim)
            scaler.update()
            if diagnose_now:
                _maybe_synchronize(device)
                t_step = time.perf_counter() - t_step_start
                diag_fetch_times.append(float(t_fetch))
                diag_step_times.append(t_step)
                chunk_paths = batch.get("chunk_paths") or []
                chunk_ids = [_short_chunk_id(p, dataset_root) for p in chunk_paths]
                unique_chunks = len(set(chunk_ids))
                diag_unique_counts.append(unique_chunks)
                if args.num_workers == 0:
                    stats = ds.get_cache_stats()
                    cache_parts = []
                    for key in ("chunk_cache", "label_cache"):
                        info = stats.get(key, {})
                        if not info.get("enabled"):
                            cache_parts.append(f"{key}=disabled")
                        else:
                            cache_parts.append(f"{key}={info.get('hits', 0)}/{info.get('misses', 0)}")
                    cache_msg = " cache=" + " ".join(cache_parts)
                else:
                    cache_msg = " cache=unavailable"
                print(
                    f"[diag] epoch={epoch+1} step={step_abs+1} "
                    f"t_fetch_ms={t_fetch * 1000:.1f} "
                    f"t_step_ms={t_step * 1000:.1f} "
                    f"unique_chunks={unique_chunks} "
                    f"chunks={chunk_ids}"
                    f"{cache_msg}"
                )
                diagnose_remaining -= 1
                if diagnose_remaining == 0:
                    _print_diag_summary()

            running += loss.item()
            global_step += 1
            step_in_epoch = step_abs
            completed_steps += 1
            steps_done += 1
            epoch_steps_done += 1
            now = time.perf_counter()
            epoch_time_window.append(now)
            train_time_window.append(now)
            overall_pct = 100.0 * completed_steps / max(1, total_steps)
            if epoch_bar is not None:
                epoch_bar.update(1)
                epoch_bar.set_postfix_str(f"overall={overall_pct:.1f}%")
            if (step_abs + 1) % log_every == 0:
                avg = running / log_every
                grad_msg = ""
                if scaler.is_enabled() and not unscaled:
                    scaler.unscale_(optim)
                    unscaled = True
                fusion_w = model.fusion.mlps[0][0].weight
                head_param = None
                for p in model.sem_head.head.parameters():
                    head_param = p
                    break
                fusion_norm = float(fusion_w.grad.norm().item()) if fusion_w.grad is not None else 0.0
                head_norm = float(head_param.grad.norm().item()) if head_param is not None and head_param.grad is not None else 0.0
                grad_msg = f" grad_norm fusion={fusion_norm:.2e} head={head_norm:.2e}"
                if use_tqdm:
                    tqdm.write(f"[epoch {epoch+1}] step {step_abs+1} loss {avg:.4f}{grad_msg}")
                else:
                    epoch_done = step_abs + 1
                    remaining_epoch_steps = max(0, steps_this_epoch - epoch_done)
                    if len(epoch_time_window) >= 2:
                        epoch_span = epoch_time_window[-1] - epoch_time_window[0]
                        epoch_rate = (len(epoch_time_window) - 1) / max(1e-6, epoch_span)
                        epoch_eta = remaining_epoch_steps / max(1e-6, epoch_rate)
                    else:
                        epoch_elapsed = time.perf_counter() - epoch_start_time
                        epoch_eta = epoch_elapsed / max(1, epoch_steps_done) * remaining_epoch_steps
                    remaining_steps = max(0, total_steps - completed_steps)
                    if len(train_time_window) >= 2:
                        train_span = train_time_window[-1] - train_time_window[0]
                        train_rate = (len(train_time_window) - 1) / max(1e-6, train_span)
                        train_eta = remaining_steps / max(1e-6, train_rate)
                    else:
                        train_elapsed = time.perf_counter() - train_start_time
                        train_eta = train_elapsed / max(1, steps_done) * remaining_steps
                    display_step = epoch_done
                    print(
                        f"[epoch {epoch+1}] step {display_step}/{steps_this_epoch} "
                        f"loss {avg:.4f} grad_norm fusion={fusion_norm:.2e} head={head_norm:.2e} "
                        f"overall={overall_pct:.1f}% "
                        f"epoch_eta={_format_eta_minutes(epoch_eta / 60.0)} "
                        f"train_eta={_format_eta_minutes(train_eta / 60.0)}"
                    )
                running = 0.0

            if args.save_every_minutes > 0:
                elapsed = time.time() - last_save_time
                if elapsed >= args.save_every_minutes * 60:
                    save_checkpoint(epoch, step_in_epoch, reason="interval")
                    last_save_time = time.time()

            if request_save_file.is_file():
                save_checkpoint(epoch, step_in_epoch, reason="REQUEST_SAVE")
                try:
                    request_save_file.unlink()
                except FileNotFoundError:
                    pass

        if diagnose_epoch:
            _print_diag_summary()
            diagnose_active = False
        if epoch_bar is not None:
            epoch_bar.close()

        # End-of-epoch checkpoint
        save_checkpoint(epoch, step_in_epoch, reason="epoch_end")
        last_save_time = time.time()

    print("Training finished.")


if __name__ == "__main__":
    main()
