#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _append_opt(cmd: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    text = str(value)
    if text == "":
        return
    cmd.extend([flag, text])


def _append_bool(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def _load_ranked_results(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"No ranked results found in {path}")
    return payload


def _build_shared_eval_args(args: argparse.Namespace) -> List[str]:
    out: List[str] = [
        "--dataset-root",
        str(Path(args.dataset_root).expanduser()),
        "--checkpoint-dir",
        str(Path(args.checkpoint_dir).expanduser()),
        "--checkpoint-select",
        str(args.checkpoint_select),
        "--epoch-min",
        str(int(args.epoch_min)),
        "--epoch-max",
        str(int(args.epoch_max)),
        "--batch-size",
        str(int(args.batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--prefetch-factor",
        str(int(args.prefetch_factor)),
        "--chunk-cache-size",
        str(int(args.chunk_cache_size)),
        "--label-cache-size",
        str(int(args.label_cache_size)),
        "--label-chunk-cache-size",
        str(int(args.label_chunk_cache_size)),
        "--chunk-format",
        str(args.chunk_format),
        "--num-classes",
        str(int(args.num_classes)),
        "--ignore-index",
        str(int(args.ignore_index)),
        "--focal-alpha",
        str(float(args.focal_alpha)),
        "--focal-gamma",
        str(float(args.focal_gamma)),
        "--overlay-alpha",
        str(float(args.overlay_alpha)),
        "--sort-by",
        "miou",
        "--sort-order",
        "desc",
    ]
    _append_opt(out, "--labels-root", args.labels_root)
    _append_opt(out, "--label-chunks-root", args.label_chunks_root)
    _append_opt(out, "--label-chunk-subdir", args.label_chunk_subdir)
    _append_opt(out, "--label-chunk-ext", args.label_chunk_ext)
    _append_opt(out, "--index-cache", args.index_cache)
    _append_opt(out, "--ignore-classes", args.ignore_classes)
    _append_opt(out, "--ignore-classes-file", args.ignore_classes_file)
    _append_opt(out, "--remap-classes-file", args.remap_classes_file)
    _append_opt(out, "--config-path", args.config_path)
    _append_opt(out, "--weights-path", args.weights_path)
    _append_opt(out, "--class-names-file", args.class_names_file)
    _append_bool(out, "--no-label-chunks", not args.use_label_chunks)
    return out


def _run(cmd: List[str], *, cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validation-then-test semantic evaluation protocol.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--labels-root")
    ap.add_argument("--label-chunks-root")
    ap.add_argument("--label-chunk-subdir", default="label_chunks")
    ap.add_argument("--label-chunk-ext", default=".pt")
    ap.add_argument("--chunk-format", choices=["auto", "pt", "safetensors"], default="auto")
    ap.add_argument("--val-scenes-file", required=True)
    ap.add_argument("--test-scenes-file", required=True)
    ap.add_argument("--index-cache", help="Optional shared index cache path passed through to eval_film_m2f.py.")
    ap.add_argument("--num-classes", type=int, default=200)
    ap.add_argument("--ignore-index", type=int, default=65535)
    ap.add_argument("--ignore-classes")
    ap.add_argument("--ignore-classes-file")
    ap.add_argument("--remap-classes-file")
    ap.add_argument("--class-names-file")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--chunk-cache-size", type=int, default=4)
    ap.add_argument("--label-cache-size", type=int, default=0)
    ap.add_argument("--label-chunk-cache-size", type=int, default=2)
    ap.add_argument("--no-label-chunks", dest="use_label_chunks", action="store_false")
    ap.set_defaults(use_label_chunks=True)
    ap.add_argument("--config-path", default="mask2former/configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml")
    ap.add_argument("--weights-path", default="models/m2f_ade20k.pkl")
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint-select", choices=["all", "epoch-end"], default="epoch-end")
    ap.add_argument("--epoch-min", type=int, default=15)
    ap.add_argument("--epoch-max", type=int, default=26)
    ap.add_argument("--overlay-alpha", type=float, default=0.45)
    ap.add_argument("--test-export-masks", action="store_true")
    ap.add_argument("--test-export-raw", action="store_true")
    ap.add_argument("--test-export-samples", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parent
    eval_script = here / "eval_film_m2f.py"

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    shared = _build_shared_eval_args(args)

    val_json = out_dir / "val_ranked.json"
    val_csv = out_dir / "val_ranked.csv"
    val_manifest = out_dir / "val_samples_manifest.json"
    val_cmd = [
        sys.executable,
        str(eval_script),
        *shared,
        "--split-name",
        "val",
        "--scenes-file",
        str(Path(args.val_scenes_file).expanduser()),
        "--out-json",
        str(val_json),
        "--out-csv",
        str(val_csv),
        "--sample-manifest-out",
        str(val_manifest),
    ]
    _run(val_cmd, cwd=repo_root)

    val_payload = _load_ranked_results(val_json)
    best = val_payload["results"][0]
    best_epoch = int(best["epoch"])
    best_checkpoint = str(best["checkpoint"])

    test_json = out_dir / "test_selected.json"
    test_csv = out_dir / "test_selected.csv"
    test_manifest = out_dir / "test_samples_manifest.json"
    test_per_class_csv = out_dir / "test_per_class_iou.csv"
    test_confusion_csv = out_dir / "test_confusion_matrix.csv"
    test_cmd = [
        sys.executable,
        str(eval_script),
        *shared,
        "--split-name",
        "test",
        "--scenes-file",
        str(Path(args.test_scenes_file).expanduser()),
        "--epoch-min",
        str(best_epoch),
        "--epoch-max",
        str(best_epoch),
        "--max-checkpoints",
        "1",
        "--out-json",
        str(test_json),
        "--out-csv",
        str(test_csv),
        "--sample-manifest-out",
        str(test_manifest),
        "--out-per-class-csv",
        str(test_per_class_csv),
        "--out-confusion-csv",
        str(test_confusion_csv),
    ]
    if args.test_export_masks:
        test_export_dir = out_dir / "test_qualitative"
        test_cmd.extend(
            [
                "--export-masks",
                "--export-top-k",
                "1",
                "--export-samples",
                str(int(args.test_export_samples)),
                "--export-dir",
                str(test_export_dir),
            ]
        )
    if args.test_export_raw:
        test_cmd.append("--export-raw")
    _run(test_cmd, cwd=repo_root)

    summary = {
        "protocol": "validation-rank-then-held-out-test",
        "val_scenes_file": str(Path(args.val_scenes_file).expanduser()),
        "test_scenes_file": str(Path(args.test_scenes_file).expanduser()),
        "epoch_range": [int(args.epoch_min), int(args.epoch_max)],
        "selected_checkpoint": {
            "path": best_checkpoint,
            "epoch": best_epoch,
            "step": int(best["step"]),
            "val_loss": float(best["val_loss"]),
            "miou": float(best["miou"]),
            "pixel_acc": float(best["pixel_acc"]),
        },
        "outputs": {
            "val_ranked_json": str(val_json),
            "val_ranked_csv": str(val_csv),
            "val_samples_manifest": str(val_manifest),
            "test_json": str(test_json),
            "test_csv": str(test_csv),
            "test_samples_manifest": str(test_manifest),
            "test_per_class_iou_csv": str(test_per_class_csv),
            "test_confusion_csv": str(test_confusion_csv),
        },
    }
    if args.test_export_masks:
        summary["outputs"]["test_qualitative_dir"] = str(out_dir / "test_qualitative")
    summary_path = out_dir / "protocol_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[info] wrote {summary_path}")
    print(
        "[done] "
        f"selected epoch={best_epoch} "
        f"miou={float(best['miou']):.4f} "
        f"checkpoint={Path(best_checkpoint).name}"
    )


if __name__ == "__main__":
    main()
