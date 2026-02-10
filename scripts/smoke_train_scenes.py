#!/usr/bin/env python3
import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


LOG_PATTERN = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)\]\s+step\s+(?P<step>\d+)(?:/\d+)?\s+"
    r"loss\s+(?P<loss>[0-9.eE+-]+)\s+"
    r"grad_norm\s+fusion=(?P<fusion>[0-9.eE+-]+)\s+head=(?P<head>[0-9.eE+-]+)"
)


def _load_scenes(path: Path) -> List[str]:
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def _parse_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _iter_metrics(log_text: str) -> Iterable[Tuple[int, int, float, float, float]]:
    for match in LOG_PATTERN.finditer(log_text):
        epoch = int(match.group("epoch"))
        step = int(match.group("step"))
        loss = float(match.group("loss"))
        fusion = float(match.group("fusion"))
        head = float(match.group("head"))
        yield epoch, step, loss, fusion, head


def _append_metrics_csv(
    csv_path: Path,
    rows: Iterable[Tuple[str, int, int, float, float, float, str, str]],
) -> None:
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "scene",
                    "epoch",
                    "step",
                    "loss",
                    "grad_fusion",
                    "grad_head",
                    "run_dir",
                    "log_path",
                ]
            )
        for row in rows:
            writer.writerow(row)


def _append_runs_csv(
    csv_path: Path,
    scene: str,
    return_code: int,
    metrics_count: int,
    run_dir: Path,
    log_path: Path,
) -> None:
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                ["scene", "return_code", "metrics_count", "run_dir", "log_path"]
            )
        writer.writerow([scene, return_code, metrics_count, str(run_dir), str(log_path)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 1-epoch smoke tests per scene and log loss/grad metrics."
    )
    parser.add_argument(
        "--scenes-file",
        type=Path,
        default=Path(
            "/home/s2984792/scannetpp_raster_dec2024/label_stats/scenes_with_labels.txt"
        ),
        help="File with scene ids (one per line).",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Comma-separated scene ids to override scenes-file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/deepstore/datasets/itc/eos/scannetpp/dataset_ready"),
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("/home/s2984792/scannetpp_raster_dec2024/semantics_2d/semantics"),
    )
    parser.add_argument(
        "--remap-dir",
        type=Path,
        default=Path("/home/s2984792/scannetpp_raster_dec2024/label_stats"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/home/s2984792/src/VGGT-SLAM/runs/smoke_scenes"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("/home/s2984792/src/VGGT-SLAM/runs/smoke_scenes/metrics.csv"),
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("/home/s2984792/src/VGGT-SLAM/runs/smoke_scenes/runs.csv"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss-input", type=str, default="logprobs")
    parser.add_argument("--loss-eps", type=float, default=1e-6)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--loss-reduction", type=str, default="batch")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(
            "/home/s2984792/src/VGGT-SLAM/mask2former/configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml"
        ),
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("/home/s2984792/src/VGGT-SLAM/models/m2f_ade20k.pkl"),
    )
    parser.add_argument("--chunk-format", type=str, default="safetensors")
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("scripts/train_film_m2f_optimized_png.py"),
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for training runs.",
    )
    args = parser.parse_args()

    if args.scenes:
        scenes = _parse_list(args.scenes)
    else:
        scenes = _load_scenes(args.scenes_file)

    if not scenes:
        raise SystemExit("No scenes to process.")

    for scene in scenes:
        remap_file = args.remap_dir / f"remap_{scene}.txt"
        if not remap_file.is_file():
            print(f"[skip] missing remap file: {remap_file}")
            _append_runs_csv(
                args.runs_csv,
                scene,
                return_code=2,
                metrics_count=0,
                run_dir=args.runs_root / scene,
                log_path=args.runs_root / scene / "train.log",
            )
            continue

        run_dir = args.runs_root / scene
        ckpt_dir = run_dir / "checkpoints"
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"

        cmd = [
            args.python,
            "-u",
            str(args.train_script),
            "--dataset-root",
            str(args.dataset_root),
            "--labels-root",
            str(args.labels_root),
            "--run-dir",
            str(run_dir),
            "--ckpt-dir",
            str(ckpt_dir),
            "--scenes",
            scene,
            "--remap-classes-file",
            str(remap_file),
            "--ignore-index",
            "65535",
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--loss-input",
            args.loss_input,
            "--loss-eps",
            str(args.loss_eps),
            "--focal-alpha",
            str(args.focal_alpha),
            "--focal-gamma",
            str(args.focal_gamma),
            "--loss-reduction",
            args.loss_reduction,
            "--log-every",
            str(args.log_every),
            "--config-path",
            str(args.config_path),
            "--weights-path",
            str(args.weights_path),
            "--chunk-format",
            args.chunk_format,
        ]

        print(f"[run] {scene}")
        with log_path.open("w") as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)

        log_text = log_path.read_text()
        metrics = list(_iter_metrics(log_text))
        metrics_rows = [
            (scene, epoch, step, loss, fusion, head, str(run_dir), str(log_path))
            for epoch, step, loss, fusion, head in metrics
        ]
        _append_metrics_csv(args.metrics_csv, metrics_rows)
        _append_runs_csv(
            args.runs_csv,
            scene,
            return_code=proc.returncode,
            metrics_count=len(metrics),
            run_dir=run_dir,
            log_path=log_path,
        )

        if proc.returncode != 0:
            print(f"[fail] {scene} (exit {proc.returncode})")
        elif not metrics:
            print(f"[fail] {scene} (no loss lines)")
        else:
            min_loss = min(m[2] for m in metrics)
            max_grad = max(m[3] for m in metrics)
            print(
                f"[done] {scene} min_loss={min_loss:.4f} "
                f"max_fusion_grad={max_grad:.2e}"
            )


if __name__ == "__main__":
    main()
