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
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import OrderedDict
import datetime
import time
import itertools
import os
import statistics
import threading
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
        default="auto",
        help="Resume from 'auto' (latest in ckpt-dir), 'none', or a checkpoint path.",
    )
    p.add_argument("--use-half", action="store_true", help="Use mixed precision training.")
    p.add_argument("--focal-alpha", type=float, default=0.25, help="Alpha weighting for focal loss.")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma exponent for focal loss.")
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
        "--no-debug-print",
        action="store_true",
        help="Disable per-batch debug prints to reduce overhead.",
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

        # need to read chunk metadata
        chk = _load_chunk_metadata(chk_path)
        scene_id = chk["scene_id"]
        scene_dir = chk_path.parent.parent  # .../<scene>/
        frame_paths = chk.get("frame_paths")
        if frame_paths is None:
            raise KeyError(f"Chunk metadata missing frame_paths: {chk_path}")
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
        prefetch_next_chunk: bool = False,
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
        # Track label chunk files that failed to load to avoid repeated errors.
        self._bad_label_chunks: Set[Path] = set()
        self._cache_lock = threading.Lock()

        self._prefetch_enabled = bool(prefetch_next_chunk)
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_inflight: Dict[Path, Future] = {}
        # Best-effort cache stats (only reliable when num_workers=0).
        self._chunk_cache_hits = 0
        self._chunk_cache_misses = 0
        self._label_cache_hits = 0
        self._label_cache_misses = 0
        self._label_chunk_cache_hits = 0
        self._label_chunk_cache_misses = 0

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
            lc_path = Path(label_chunk_path)
            if lc_path not in self._bad_label_chunks:
                try:
                    label = self._get_label_from_chunk(lc_path, sample["frame_idx"])
                except (FileNotFoundError, KeyError, IndexError, ValueError, RuntimeError, OSError, EOFError) as exc:
                    # Corrupted/incomplete chunk -> fall back to PNGs for this chunk.
                    if lc_path not in self._bad_label_chunks:
                        print(f"[warn] failed to load label chunk {lc_path}: {exc}; falling back to PNG labels")
                        self._bad_label_chunks.add(lc_path)
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
            "chunk_path": sample["chunk_path"],
            "dino": dino,
            "dpt_levels": dpt_levels,
            "label": label,
            "image_size": sample["image_size"],
        }

    def reset_cache_stats(self) -> None:
        self._chunk_cache_hits = 0
        self._chunk_cache_misses = 0
        self._label_cache_hits = 0
        self._label_cache_misses = 0
        self._label_chunk_cache_hits = 0
        self._label_chunk_cache_misses = 0

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
            "label_chunk_cache": {
                "hits": self._label_chunk_cache_hits,
                "misses": self._label_chunk_cache_misses,
                "enabled": 1 if self.use_label_chunks else 0,
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

    def _get_label_from_chunk(self, path: Path, frame_idx: int) -> torch.Tensor:
        path = Path(path)
        if path in self._label_chunk_cache:
            labels = self._label_chunk_cache.pop(path)
            self._label_chunk_cache[path] = labels
            self._label_chunk_cache_hits += 1
        else:
            self._label_chunk_cache_misses += 1
            labels = self._load_label_chunk_tensor(path)
            if labels.dim() < 3:
                raise ValueError(f"Label chunk tensor has unexpected shape {labels.shape} in {path}")
            self._label_chunk_cache[path] = labels
            if len(self._label_chunk_cache) > self.label_chunk_cache_size:
                self._label_chunk_cache.popitem(last=False)
        if frame_idx >= labels.shape[0]:
            raise IndexError(f"frame_idx {frame_idx} out of bounds for label chunk {path}")
        return labels[frame_idx].long()

    def _load_label_chunk_tensor(self, path: Path) -> torch.Tensor:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            tensors = load_file(str(path))
            labels = tensors.get("labels")
            if labels is None:
                raise KeyError(f"Label chunk missing 'labels' tensor: {path}")
            return labels
        payload = torch.load(path, map_location="cpu")
        if "labels" not in payload:
            raise KeyError(f"Label chunk missing 'labels' tensor: {path}")
        return payload["labels"]

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
        chunks = self._get_epoch_chunks()

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

    def _get_epoch_chunks(self) -> List[str]:
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
        return chunks

    def get_epoch_chunk_order(self) -> List[str]:
        return self._get_epoch_chunks()


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
        prefetch_next_chunk=prefetch_next_chunk,
    )
    sampler: Optional[Sampler[int]] = None
    shuffle = False
    if args.shuffle_mode == "global":
        shuffle = True
        # TODO: add resume offset support for global shuffle mode to avoid skip-on-resume cost.
    elif args.shuffle_mode == "chunk":
        sampler = ChunkShuffleSampler(samples, seed=args.seed)
    elif args.shuffle_mode == "none":
        shuffle = False
        # TODO: add resume offset support for non-shuffled mode to avoid skip-on-resume cost.
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

    # Train FiLM + Mask2Former head
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_half and device.type == "cuda")

    # ############ DEBUG: count how many trainable params we actually update ############
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dbg] trainable parameters: {num_params}")
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
        model_state = state.get("model_state") or state
        model.load_state_dict(model_state, strict=False)
        if "optimizer_state" in state:
            optim.load_state_dict(state["optimizer_state"])
        if scheduler and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        if "scaler_state" in state:
            scaler.load_state_dict(state["scaler_state"])
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
    if start_epoch >= args.epochs:
        print(f"[info] start_epoch {start_epoch} >= total epochs {args.epochs}; nothing to train.")
        return

    resume_sample_offset = 0
    if start_step_in_epoch > 0:
        resume_sample_offset = start_step_in_epoch * int(args.batch_size)

    last_save_time = time.time()
    diagnose_active = diagnose_batches > 0
    diagnose_remaining = diagnose_batches
    diag_fetch_times: List[float] = []
    diag_step_times: List[float] = []
    diag_unique_counts: List[int] = []
    diag_summary_printed = False

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

    global_bar = tqdm(total=0, desc="train", unit="step")
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
        diagnose_epoch = diagnose_active and epoch == start_epoch
        if diagnose_epoch and args.num_workers == 0:
            ds.reset_cache_stats()
        steps_this_epoch = max(0, len(dl) - skip_steps)
        global_bar.total += steps_this_epoch
        global_bar.refresh()
        data_iter = _timed_dl_iter(dl) if diagnose_epoch else dl
        for step, data in enumerate(data_iter):
            if step < skip_steps:
                # Skip steps already completed before checkpoint.
                continue
            if diagnose_epoch:
                t_fetch, batch = data
            else:
                t_fetch, batch = None, data
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
                    for key in ("chunk_cache", "label_cache", "label_chunk_cache"):
                        info = stats.get(key, {})
                        if not info.get("enabled"):
                            cache_parts.append(f"{key}=disabled")
                        else:
                            cache_parts.append(f"{key}={info.get('hits', 0)}/{info.get('misses', 0)}")
                    cache_msg = " cache=" + " ".join(cache_parts)
                else:
                    cache_msg = " cache=unavailable"
                print(
                    f"[diag] epoch={epoch+1} step={step+1} "
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
            step_in_epoch = step
            global_bar.update(1)
            if (step + 1) % 10 == 0:
                avg = running / 10
                print(f"[epoch {epoch+1}] step {step+1} loss {avg:.4f}")
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

        # End-of-epoch checkpoint
        save_checkpoint(epoch, step_in_epoch, reason="epoch_end")
        last_save_time = time.time()

    print("Training finished.")
    global_bar.close()


if __name__ == "__main__":
    main()
