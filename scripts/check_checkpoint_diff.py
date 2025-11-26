#!/usr/bin/env python3
"""
check_checkpoint_diff.py
========================

Utility to compare two FusionMask2Former checkpoints. It reports parameter
norms and L2 diffs for FiLM fusion MLPs and the Mask2Former head.

Usage:
    python scripts/check_checkpoint_diff.py \
        --checkpoint-a checkpoints/film_m2f_base.pt \
        --checkpoint-b checkpoints/film_m2f_20251126_run_01.pt \
        --num-classes 2878

If you omit --checkpoint-a, it uses a freshly initialized model as the baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from scripts.train_film_m2f import FusionMask2Former


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diff two FiLM+Mask2Former checkpoints.")
    p.add_argument("--checkpoint-a", help="Baseline checkpoint (optional). If omitted, uses freshly initialized model.")
    p.add_argument("--checkpoint-b", required=True, help="Target checkpoint to compare.")
    p.add_argument("--num-classes", type=int, required=True, help="Number of semantic classes (must match training).")
    p.add_argument("--config-path", help="Mask2Former config path if different from default.")
    p.add_argument("--weights-path", help="Mask2Former weights path if different from default.")
    return p.parse_args()


def load_model_state(path: str, num_classes: int, config_path: str | None, weights_path: str | None) -> Dict[str, torch.Tensor]:
    device = torch.device("cpu")
    model = FusionMask2Former(device=device, num_classes=num_classes, config_path=config_path, weights_path=weights_path)
    if path:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    return dict(model.named_parameters())


def main() -> None:
    args = parse_args()
    base_params = load_model_state(args.checkpoint_a, args.num_classes, args.config_path, args.weights_path) if args.checkpoint_a else load_model_state("", args.num_classes, args.config_path, args.weights_path)
    target_params = load_model_state(args.checkpoint_b, args.num_classes, args.config_path, args.weights_path)

    def param_stats(prefix: str) -> Dict[str, float]:
        stats = {}
        for name, p in target_params.items():
            if name.startswith(prefix):
                stats[name] = p.norm().item()
        return stats

    def diff_stats(prefix: str) -> Dict[str, float]:
        stats = {}
        for name, p in target_params.items():
            if name.startswith(prefix) and name in base_params:
                stats[name] = (p - base_params[name]).norm().item()
        return stats

    film_norms = param_stats("fusion")
    film_diffs = diff_stats("fusion")
    head_norms = param_stats("sem_head")
    head_diffs = diff_stats("sem_head")

    def summarize(title: str, norms: Dict[str, float], diffs: Dict[str, float]):
        if not norms:
            print(f"[warn] No params found for prefix in {title}")
            return
        print(f"\n[{title}] {len(norms)} params")
        print(f"  norm min/mean/max: {min(norms.values()):.4f} / {sum(norms.values())/len(norms):.4f} / {max(norms.values()):.4f}")
        if diffs:
            print(f"  diff min/mean/max: {min(diffs.values()):.4f} / {sum(diffs.values())/len(diffs):.4f} / {max(diffs.values()):.4f}")
        else:
            print("  diff: (baseline not provided)")

    summarize("FiLM fusion", film_norms, film_diffs)
    summarize("Mask2Former head", head_norms, head_diffs)


if __name__ == "__main__":
    main()
