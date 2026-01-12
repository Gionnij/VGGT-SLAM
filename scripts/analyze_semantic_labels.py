#!/usr/bin/env python3
"""
analyze_semantic_labels.py
==========================

Scan a directory of ScanNet++ semantic masks and summarize how often each
class id appears. The script saves tabular data plus an optional plot of the
label distribution so we can choose which classes to keep for training.

Example:
    python scripts/analyze_semantic_labels.py \
        --mask-root ~/scannetpp_raster_dec2024/semantics_2d/semantics \
        --output-dir ./label_stats \
        --ignore-value 65535 \
        --top-k 50 \
        --min-image-frequency 200
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib optional
    plt = None  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize class ids that appear in ScanNet++ PNG masks.")
    p.add_argument(
        "--mask-root",
        required=True,
        help="Directory containing per-scene folders with *.png semantic masks.",
    )
    p.add_argument(
        "--output-dir",
        help="Directory to store stats. Defaults to <mask-root>/label_stats.",
    )
    p.add_argument("--ignore-value", type=int, default=65535, help="Label value reserved for ignore.")
    p.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of most frequent classes to include in the bar plot.",
    )
    p.add_argument(
        "--min-image-frequency",
        type=int,
        help="If set, classes seen in fewer masks than this are written to ignored_classes.txt.",
    )
    p.add_argument(
        "--min-pixel-frequency",
        type=int,
        help="Optional alternative threshold over pixel counts for inclusion.",
    )
    p.add_argument(
        "--scenes-list",
        default="scenes_with_labels.txt",
        help="Filename (under output-dir) that stores all scene ids discovered.",
    )
    return p.parse_args()


def iter_mask_files(scene_dir: Path) -> Iterable[Path]:
    for p in sorted(scene_dir.glob("*.png")):
        if p.is_file():
            yield p


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path), copy=False)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask at {path} is not single channel.")
    return arr


def update_counters(
    mask: np.ndarray,
    *,
    ignore_value: int,
    image_counter: Counter,
    pixel_counter: Counter,
) -> None:
    vals, counts = np.unique(mask, return_counts=True)
    for val, cnt in zip(vals.tolist(), counts.tolist()):
        if val == ignore_value:
            continue
        image_counter[val] += 1
        pixel_counter[val] += cnt


def summarize_counts(image_counter: Counter, pixel_counter: Counter) -> List[Dict]:
    classes = sorted(image_counter.keys())
    rows: List[Dict] = []
    for cid in classes:
        rows.append(
            {
                "class_id": int(cid),
                "image_frequency": int(image_counter[cid]),
                "pixel_count": int(pixel_counter[cid]),
            }
        )
    return rows


def save_table(rows: List[Dict], output_dir: Path) -> None:
    csv_path = output_dir / "class_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "image_frequency", "pixel_count"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "class_counts.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def plot_distribution(rows: List[Dict], output_dir: Path, top_k: int) -> Optional[Path]:
    if not rows or plt is None:
        return None
    sorted_rows = sorted(rows, key=lambda r: r["image_frequency"], reverse=True)
    if top_k > 0:
        sorted_rows = sorted_rows[:top_k]
    fig, ax = plt.subplots(figsize=(12, 6))
    class_ids = [r["class_id"] for r in sorted_rows]
    freqs = [r["image_frequency"] for r in sorted_rows]
    ax.bar(range(len(class_ids)), freqs)
    ax.set_title(f"Top {len(class_ids)} classes by image frequency")
    ax.set_xlabel("class id")
    ax.set_ylabel("number of masks containing class")
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels([str(cid) for cid in class_ids], rotation=90)
    fig.tight_layout()
    plot_path = output_dir / "label_frequency.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def decide_ignored_classes(
    rows: List[Dict],
    *,
    min_image_frequency: Optional[int],
    min_pixel_frequency: Optional[int],
) -> Tuple[List[int], List[int]]:
    if min_image_frequency is None and min_pixel_frequency is None:
        return sorted(r["class_id"] for r in rows), []
    keep: List[int] = []
    ignore: List[int] = []
    for row in rows:
        img_freq = row["image_frequency"]
        px_freq = row["pixel_count"]
        ok_img = min_image_frequency is None or img_freq >= min_image_frequency
        ok_px = min_pixel_frequency is None or px_freq >= min_pixel_frequency
        if ok_img and ok_px:
            keep.append(row["class_id"])
        else:
            ignore.append(row["class_id"])
    return keep, ignore


def write_list(items: Iterable[int], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def main() -> None:
    args = parse_args()
    mask_root = Path(args.mask_root).expanduser()
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else mask_root / "label_stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([p for p in mask_root.iterdir() if p.is_dir()])
    if not scene_dirs:
        raise RuntimeError(f"No scene folders found under {mask_root}")

    image_counter: Counter = Counter()
    pixel_counter: Counter = Counter()
    total_masks = 0
    scene_ids: List[str] = []

    for scene_dir in scene_dirs:
        masks = list(iter_mask_files(scene_dir))
        if not masks:
            continue
        scene_ids.append(scene_dir.name)
        for mask_path in masks:
            mask = load_mask(mask_path)
            update_counters(
                mask,
                ignore_value=args.ignore_value,
                image_counter=image_counter,
                pixel_counter=pixel_counter,
            )
            total_masks += 1

    rows = summarize_counts(image_counter, pixel_counter)
    save_table(rows, output_dir)
    plot_path = plot_distribution(rows, output_dir, top_k=args.top_k)

    keep_classes, ignored_classes = decide_ignored_classes(
        rows,
        min_image_frequency=args.min_image_frequency,
        min_pixel_frequency=args.min_pixel_frequency,
    )
    summary = {
        "mask_root": str(mask_root),
        "scenes_found": len(scene_ids),
        "total_masks": total_masks,
        "distinct_classes": len(rows),
        "ignore_value": args.ignore_value,
        "min_image_frequency": args.min_image_frequency,
        "min_pixel_frequency": args.min_pixel_frequency,
        "plot_path": str(plot_path) if plot_path else None,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if keep_classes:
        write_list(keep_classes, output_dir / "kept_classes.txt")
    if ignored_classes:
        write_list(ignored_classes, output_dir / "ignored_classes.txt")

    scenes_list_path = output_dir / args.scenes_list
    with scenes_list_path.open("w", encoding="utf-8") as f:
        for sid in scene_ids:
            f.write(f"{sid}\n")

    print("=== Label analysis ===")
    for key, val in summary.items():
        print(f"{key}: {val}")
    print(f"Saved {len(rows)} entries to {output_dir / 'class_counts.csv'}")
    if plot_path:
        print(f"Saved distribution plot to {plot_path}")
    if ignored_classes:
        print(
            f"Classes flagged for ignore (based on thresholds): {len(ignored_classes)} "
            f"(written to {(output_dir / 'ignored_classes.txt')})"
        )
    if keep_classes:
        print(f"Classes retained: {len(keep_classes)} (written to {(output_dir / 'kept_classes.txt')})")
    print(f"Scene ids written to {scenes_list_path}")


if __name__ == "__main__":
    main()
