#!/usr/bin/env python3
"""
convert_depth_npz.py
--------------------
Small helper that turns each depth `.npz` (with keys `depth` and `confidence`)
into an 8-bit PNG for quick inspection.  Usage:

    python scripts/convert_depth_npz.py \
        --input-dir path/to/depth_maps \
        --output-dir ./depth_pngs

The script linearly scales depth to [0, 255] based on the min/max values found
inside each file (or optionally a global range) and writes uint8 PNGs via
imageio.  Confidence is ignored for now, but can be re-enabled easily if
desired.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VGGT depth npz files to 8-bit PNGs.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing *.npz depth files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to drop the PNGs (defaults to <input-dir>/png).",
    )
    parser.add_argument(
        "--global-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional global min/max for scaling. If omitted, each file is auto-scaled.",
    )
    return parser.parse_args()


def scale_to_uint8(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    depth_clipped = np.clip(depth, depth_min, depth_max)
    scaled = (depth_clipped - depth_min) / (depth_max - depth_min + 1e-8)
    return (scaled * 255.0).astype(np.uint8)


def main():
    args = parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"{input_dir} is not a directory")

    output_dir = args.output_dir or (input_dir / "png")
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files found in {input_dir}")

    if args.global_range is not None:
        gmin, gmax = args.global_range
    else:
        gmin = gmax = None

    for npz_path in npz_files:
        data = np.load(npz_path)
        if "depth" not in data:
            print(f"[warn] {npz_path.name} has no 'depth' key, skipping.")
            continue
        depth = data["depth"]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        dmin = gmin if gmin is not None else float(depth.min())
        dmax = gmax if gmax is not None else float(depth.max())
        png = scale_to_uint8(depth, dmin, dmax)
        out_path = output_dir / f"{npz_path.stem}.png"
        imageio.imwrite(out_path, png)
        print(f"[convert] {npz_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
