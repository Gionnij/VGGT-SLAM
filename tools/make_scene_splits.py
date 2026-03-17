#!/usr/bin/env python3
"""
Create train/val/test scene splits from:
  - all available scene ids under a root directory, or
  - a fixed list of available scene ids
  - an existing train scene list

Typical use for your setup:
  python tools/make_scene_splits.py \
    --all-scenes-file /home/s2984792/scannetpp_raster_dec2024/label_stats/scenes_with_labels.txt \
    --train-scenes-file /home/s2984792/scannetpp_raster_dec2024/label_stats/scenes_selected_60.txt \
    --out-dir /home/s2984792/scannetpp_raster_dec2024/label_stats \
    --val-count 5 \
    --test-count 5
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Set, Tuple


def _read_scene_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Scene list not found: {path}")
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        scene = line.strip()
        if not scene or scene.startswith("#"):
            continue
        out.append(scene)
    return out


def _list_scenes_from_root(root: Path) -> List[str]:
    if not root.is_dir():
        raise NotADirectoryError(f"All-scenes root not found: {root}")
    scenes = [p.name for p in root.iterdir() if p.is_dir()]
    scenes.sort()
    if not scenes:
        raise RuntimeError(f"No scene folders found in: {root}")
    return scenes


def _choose_counts(remaining: int, val_count: int, test_count: int) -> Tuple[int, int]:
    if remaining < 0:
        raise ValueError("remaining must be >= 0")

    if val_count < 0 or test_count < 0:
        raise ValueError("val_count/test_count must be >= 0")

    if val_count == 0 and test_count == 0:
        val_count = remaining // 2
        test_count = remaining - val_count
    elif val_count > 0 and test_count == 0:
        if val_count > remaining:
            raise ValueError(f"val_count={val_count} exceeds remaining scenes={remaining}")
        test_count = remaining - val_count
    elif val_count == 0 and test_count > 0:
        if test_count > remaining:
            raise ValueError(f"test_count={test_count} exceeds remaining scenes={remaining}")
        val_count = remaining - test_count
    else:
        if (val_count + test_count) > remaining:
            raise ValueError(
                f"val_count + test_count ({val_count + test_count}) exceeds remaining scenes ({remaining})"
            )
    return val_count, test_count


def _write_list(path: Path, values: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(values)
    if body:
        body += "\n"
    path.write_text(body, encoding="utf-8")


def _load_available_scenes(*, all_scenes_root: str, all_scenes_file: str) -> Tuple[List[str], str]:
    root_arg = str(all_scenes_root or "").strip()
    file_arg = str(all_scenes_file or "").strip()
    if bool(root_arg) == bool(file_arg):
        raise ValueError("Provide exactly one of --all-scenes-root or --all-scenes-file.")

    if root_arg:
        root = Path(root_arg).expanduser()
        return _list_scenes_from_root(root), str(root)

    src = Path(file_arg).expanduser()
    scenes = sorted(set(_read_scene_list(src)))
    if not scenes:
        raise RuntimeError(f"No scenes found in available-scenes file: {src}")
    return scenes, str(src)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate deterministic val/test splits from all-scenes minus train-scenes.")
    p.add_argument("--all-scenes-root", help="Directory containing all scene folders.")
    p.add_argument("--all-scenes-file", help="Text file with the full available scene universe (one scene id per line).")
    p.add_argument("--train-scenes-file", required=True, help="Train scene list (one scene id per line).")
    p.add_argument("--out-dir", required=True, help="Output directory for split files.")
    p.add_argument("--train-out", default="scenes_train.txt")
    p.add_argument("--val-out", default="scenes_val.txt")
    p.add_argument("--test-out", default="scenes_test.txt")
    p.add_argument("--unused-out", default="scenes_unused.txt")
    p.add_argument("--summary-out", default="split_summary.json")
    p.add_argument(
        "--strategy",
        choices=["random", "alphabetical"],
        default="random",
        help="How to order non-train scenes before slicing val/test.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed used when --strategy=random.")
    p.add_argument("--val-count", type=int, default=0, help="0 means auto.")
    p.add_argument("--test-count", type=int, default=0, help="0 means auto.")
    args = p.parse_args()

    train_file = Path(args.train_scenes_file).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    all_scenes, all_source = _load_available_scenes(
        all_scenes_root=args.all_scenes_root,
        all_scenes_file=args.all_scenes_file,
    )
    train_scenes_raw = _read_scene_list(train_file)
    train_scenes: List[str] = sorted(set(train_scenes_raw))
    all_set: Set[str] = set(all_scenes)

    missing_train = sorted([s for s in train_scenes if s not in all_set])
    train_in_all = sorted([s for s in train_scenes if s in all_set])

    remaining = [s for s in all_scenes if s not in set(train_in_all)]
    if args.strategy == "random":
        rng = random.Random(int(args.seed))
        rng.shuffle(remaining)
    else:
        remaining.sort()

    val_count, test_count = _choose_counts(len(remaining), int(args.val_count), int(args.test_count))
    val_scenes = sorted(remaining[:val_count])
    test_scenes = sorted(remaining[val_count : val_count + test_count])
    unused_scenes = sorted(remaining[val_count + test_count :])

    train_path = out_dir / args.train_out
    val_path = out_dir / args.val_out
    test_path = out_dir / args.test_out
    unused_path = out_dir / args.unused_out
    summary_path = out_dir / args.summary_out

    _write_list(train_path, train_in_all)
    _write_list(val_path, val_scenes)
    _write_list(test_path, test_scenes)
    _write_list(unused_path, unused_scenes)

    summary = {
        "all_scenes_source": all_source,
        "train_scenes_file": str(train_file),
        "strategy": args.strategy,
        "seed": int(args.seed),
        "counts": {
            "all": len(all_scenes),
            "train_requested": len(train_scenes),
            "train_present_in_all": len(train_in_all),
            "remaining_after_train": len(remaining),
            "val": len(val_scenes),
            "test": len(test_scenes),
            "unused": len(unused_scenes),
        },
        "outputs": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "unused": str(unused_path),
        },
        "missing_train_scenes": missing_train,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "[done] "
        f"all={len(all_scenes)} train={len(train_in_all)} val={len(val_scenes)} "
        f"test={len(test_scenes)} unused={len(unused_scenes)}"
    )
    print(f"[out] train: {train_path}")
    print(f"[out] val:   {val_path}")
    print(f"[out] test:  {test_path}")
    print(f"[out] summary: {summary_path}")
    if missing_train:
        print(f"[warn] {len(missing_train)} train scenes were not found in the available scene universe.")


if __name__ == "__main__":
    main()
