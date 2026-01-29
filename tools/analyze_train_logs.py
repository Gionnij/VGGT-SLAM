#!/usr/bin/env python3
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_LOGS = [
    "/home/s2984792/src/VGGT-SLAM/runs/film_m2f_20260127_113413/logs/train_434494.log",
    "/home/s2984792/src/VGGT-SLAM/runs/film_m2f_20260127_113413/logs/train_435467.log",
]

LOSS_RE = re.compile(r"\bloss\s+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
EPOCH_STEP_RE = re.compile(
    r"\[epoch\s+(\d+)\].*?step\s+(\d+)(?:/(\d+))?",
    re.IGNORECASE,
)


def parse_log(path: Path, file_index: int) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            m = LOSS_RE.search(line)
            if not m:
                continue
            loss = float(m.group(1))
            epoch = step = step_total = None
            m2 = EPOCH_STEP_RE.search(line)
            if m2:
                epoch = int(m2.group(1))
                step = int(m2.group(2))
                step_total = int(m2.group(3)) if m2.group(3) else None
            records.append(
                {
                    "file": str(path),
                    "file_index": file_index,
                    "line": line_no,
                    "epoch": epoch,
                    "step": step,
                    "step_total": step_total,
                    "loss": loss,
                }
            )
    return records


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    out = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= window:
            s -= values[i - window]
        if i >= window - 1:
            out.append(s / window)
    return out


def slope_last(values: List[float], n: int) -> Optional[float]:
    if len(values) < 2:
        return None
    k = min(n, len(values))
    y = values[-k:]
    x_mean = (k - 1) / 2.0
    y_mean = sum(y) / k
    num = sum((i - x_mean) * (y[i] - y_mean) for i in range(k))
    den = sum((i - x_mean) ** 2 for i in range(k))
    return num / den if den > 0 else None


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else float("nan"),
        "median": statistics.median(values) if values else float("nan"),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
        "stdev": statistics.pstdev(values) if len(values) > 1 else float("nan"),
    }


def _sort_key(rec: Dict) -> Tuple:
    # Prefer (epoch, step) when present; otherwise fall back to file order + line number.
    if rec["epoch"] is not None and rec["step"] is not None:
        return (0, rec["epoch"], rec["step"], rec["file_index"], rec["line"])
    return (1, rec["file_index"], rec["line"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse loss from training logs and summarize trends."
    )
    ap.add_argument(
        "--logs",
        nargs="*",
        help="Optional log file paths; defaults to the two hardcoded log files.",
    )
    ap.add_argument("--window", type=int, default=200, help="Moving average window")
    ap.add_argument(
        "--slope-n", type=int, default=1000, help="Slope computed over last N points"
    )
    ap.add_argument("--out-csv", type=Path, help="Optional CSV output")
    ap.add_argument("--out-json", type=Path, help="Optional JSON output")
    ap.add_argument("--plot", type=Path, help="Optional PNG plot (requires matplotlib)")
    args = ap.parse_args()

    logs = [Path(p) for p in (args.logs or DEFAULT_LOGS)]
    all_records: List[Dict] = []
    for idx, p in enumerate(logs):
        if not p.exists():
            print(f"[warn] missing log file: {p}")
            continue
        all_records.extend(parse_log(p, idx))

    if not all_records:
        print("No loss values found.")
        return

    all_records.sort(key=_sort_key)
    losses = [r["loss"] for r in all_records]

    overall = summarize(losses)
    ma = moving_average(losses, args.window)
    slope = slope_last(losses, args.slope_n)
    trend = {
        "moving_avg_window": args.window,
        "first_ma": ma[0] if ma else None,
        "last_ma": ma[-1] if ma else None,
        "ma_change_pct": ((ma[-1] - ma[0]) / ma[0] * 100.0)
        if len(ma) >= 2 and ma[0] != 0
        else None,
        "slope_last_n": slope,
    }

    by_file = defaultdict(list)
    for r in all_records:
        by_file[r["file"]].append(r["loss"])
    per_file = {k: summarize(v) for k, v in by_file.items()}

    by_epoch = defaultdict(list)
    for r in all_records:
        if r["epoch"] is not None:
            by_epoch[r["epoch"]].append(r["loss"])
    per_epoch = {k: summarize(v) for k, v in sorted(by_epoch.items())}

    print("Overall:", overall)
    print("Trend:", trend)
    print("Per-file:", {Path(k).name: v for k, v in per_file.items()})
    if per_epoch:
        print("Per-epoch:", per_epoch)

    if args.out_csv:
        import csv

        with args.out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["file", "line", "epoch", "step", "step_total", "loss"]
            )
            for r in all_records:
                w.writerow(
                    [
                        r["file"],
                        r["line"],
                        r["epoch"],
                        r["step"],
                        r["step_total"],
                        r["loss"],
                    ]
                )

    if args.out_json:
        payload = {
            "overall": overall,
            "trend": trend,
            "per_file": per_file,
            "per_epoch": per_epoch,
            "records": all_records,
        }
        args.out_json.write_text(json.dumps(payload, indent=2))

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Plot skipped: {e}")
        else:
            plt.figure(figsize=(10, 4))
            plt.plot(losses, alpha=0.4, label="loss")
            if ma:
                offset = args.window - 1
                plt.plot(
                    range(offset, offset + len(ma)),
                    ma,
                    label=f"MA({args.window})",
                )
            plt.title("Training loss")
            plt.xlabel("log index")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot)
            print(f"Wrote plot to {args.plot}")


if __name__ == "__main__":
    main()
