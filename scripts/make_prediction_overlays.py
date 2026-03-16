#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = Path("/Users/giovannichiementin/Desktop/2D seg examples")
MASK_PATTERN = re.compile(r"^sample_\d+_[^_]+_(.+)_(pred|gt)\.png$")


def _load_pillow():
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Pillow is required for this script. Install it with:\n"
            "  python3 -m pip install Pillow"
        ) from exc
    return Image


def _rgb_name_from_mask(mask_path: Path) -> str:
    match = MASK_PATTERN.match(mask_path.name)
    if not match:
        raise ValueError(f"Unsupported mask filename format: {mask_path.name}")
    return match.group(1)


def _overlay_mask(
    image_mod,
    rgb_path: Path,
    mask_path: Path,
    output_path: Path,
    alpha: float,
) -> None:
    with image_mod.open(rgb_path).convert("RGB") as rgb_img, image_mod.open(mask_path).convert("RGB") as mask_img:
        if mask_img.size != rgb_img.size:
            mask_img = mask_img.resize(rgb_img.size, image_mod.NEAREST)
        overlay = image_mod.blend(rgb_img, mask_img, alpha=alpha)
        overlay.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create RGB/prediction overlay images using the same PIL blend used by eval_film_m2f.py."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing the RGB JPGs and *_pred.png masks. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where *_overlay.png files are written. Defaults to the input directory.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Blend alpha used for the prediction overlay. Default: 0.45",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = (args.output_dir or input_dir).expanduser().resolve()
    alpha = max(0.0, min(1.0, float(args.alpha)))

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    image_mod = _load_pillow()
    pred_paths = sorted(input_dir.glob("*_pred.png"))
    if not pred_paths:
        raise SystemExit(f"No *_pred.png files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for pred_path in pred_paths:
        rgb_name = _rgb_name_from_mask(pred_path)
        rgb_path = input_dir / rgb_name
        if not rgb_path.exists():
            print(f"Skipping {pred_path.name}: missing RGB image {rgb_name}", file=sys.stderr)
            continue

        output_name = pred_path.name.replace("_pred.png", "_overlay.png")
        output_path = output_dir / output_name
        _overlay_mask(image_mod, rgb_path, pred_path, output_path, alpha)
        print(f"Wrote {output_path}")
        generated += 1

    if generated == 0:
        raise SystemExit("No overlays were generated.")

    print(f"Generated {generated} overlay(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
