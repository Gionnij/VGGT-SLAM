#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _read_metrics(path: Path) -> Dict[str, List[Tuple[int, int, float, float, float]]]:
    data: Dict[str, List[Tuple[int, int, float, float, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scene = row["scene"]
            epoch = int(row["epoch"])
            step = int(row["step"])
            loss = float(row["loss"])
            grad_fusion = float(row["grad_fusion"])
            grad_head = float(row["grad_head"])
            data[scene].append((epoch, step, loss, grad_fusion, grad_head))
    for scene in data:
        data[scene].sort(key=lambda x: (x[0], x[1]))
    return data


def _write_summary(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "metrics_count",
                "first_loss",
                "min_loss",
                "drop_ratio",
                "max_fusion_grad",
                "max_head_grad",
                "ok",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze smoke-test metrics and write valid scenes."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("/home/s2984792/src/VGGT-SLAM/runs/smoke_scenes/metrics.csv"),
    )
    parser.add_argument(
        "--out-valid",
        type=Path,
        default=Path(
            "/home/s2984792/scannetpp_raster_dec2024/label_stats/scenes_with_labels_valid.txt"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/home/s2984792/src/VGGT-SLAM/runs/smoke_scenes/summary.csv"),
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.05,
        help="Minimum relative drop in loss to mark OK.",
    )
    parser.add_argument(
        "--min-grad",
        type=float,
        default=1e-6,
        help="Minimum max fusion grad to mark OK.",
    )
    args = parser.parse_args()

    data = _read_metrics(args.metrics_csv)
    summary_rows: List[Dict[str, str]] = []
    valid: List[str] = []

    for scene, records in data.items():
        if not records:
            summary_rows.append(
                {
                    "scene": scene,
                    "metrics_count": "0",
                    "first_loss": "nan",
                    "min_loss": "nan",
                    "drop_ratio": "nan",
                    "max_fusion_grad": "nan",
                    "max_head_grad": "nan",
                    "ok": "False",
                }
            )
            continue

        first_loss = records[0][2]
        min_loss = min(r[2] for r in records)
        max_fusion_grad = max(r[3] for r in records)
        max_head_grad = max(r[4] for r in records)
        drop_ratio = (first_loss - min_loss) / max(first_loss, 1e-12)

        ok = (
            math.isfinite(first_loss)
            and math.isfinite(min_loss)
            and math.isfinite(max_fusion_grad)
            and drop_ratio >= args.drop_threshold
            and max_fusion_grad >= args.min_grad
        )

        summary_rows.append(
            {
                "scene": scene,
                "metrics_count": str(len(records)),
                "first_loss": f"{first_loss:.6f}",
                "min_loss": f"{min_loss:.6f}",
                "drop_ratio": f"{drop_ratio:.6f}",
                "max_fusion_grad": f"{max_fusion_grad:.6e}",
                "max_head_grad": f"{max_head_grad:.6e}",
                "ok": str(ok),
            }
        )
        if ok:
            valid.append(scene)

    args.out_valid.parent.mkdir(parents=True, exist_ok=True)
    args.out_valid.write_text("\n".join(valid))
    _write_summary(args.summary_csv, summary_rows)
    print(f"Wrote {len(valid)} valid scenes to: {args.out_valid}")
    print(f"Wrote summary to: {args.summary_csv}")


if __name__ == "__main__":
    main()
