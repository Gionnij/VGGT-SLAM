#!/usr/bin/env python3
"""
Summarize live runtime metrics emitted by main_live_stream.py / solver.py.

Inputs:
  - runtime JSONL (VGGT_RUNTIME_METRICS_PATH)
  - optional bridge log (unitree_tcp_ros2_bridge stats lines)

Outputs:
  - console summary
  - optional JSON/CSV files
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


BRIDGE_RE = re.compile(
    r"stats:\s*rx_fps=(?P<rx>[0-9]+(?:\.[0-9]+)?),\s*pub_fps=(?P<pub>[0-9]+(?:\.[0-9]+)?)"
)


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(mean(vals))


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Runtime JSONL not found: {path}")
    out: List[Dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            obj["_line"] = line_no
            out.append(obj)
    return out


def _parse_bridge_log(path: Path) -> Dict[str, Optional[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Bridge log not found: {path}")
    rx_vals: List[float] = []
    pub_vals: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = BRIDGE_RE.search(line)
        if not m:
            continue
        rx_vals.append(float(m.group("rx")))
        pub_vals.append(float(m.group("pub")))
    return {
        "rx_fps_mean": _safe_mean(rx_vals),
        "pub_fps_mean": _safe_mean(pub_vals),
        "num_samples": len(rx_vals),
    }


def _rate_from_timed_events(events: List[Dict], *, count_key: Optional[str] = None) -> Optional[float]:
    if not events:
        return None
    ordered = sorted(events, key=lambda x: float(x.get("t", 0.0)))
    t0 = float(ordered[0].get("t", 0.0))
    t1 = float(ordered[-1].get("t", 0.0))
    dt = t1 - t0
    if dt <= 0:
        return None
    if count_key is None:
        count = len(ordered)
    else:
        count = sum(max(0.0, float(e.get(count_key, 0.0))) for e in ordered)
    return float(count / dt)


def _summarize_runtime(events: List[Dict]) -> Dict[str, Optional[float]]:
    input_events = [e for e in events if e.get("event") == "input_frame"]
    window_events = [e for e in events if e.get("event") == "window_processed"]
    sem_events = [e for e in events if e.get("event") == "semantic_inference"]
    failed_events = [e for e in events if e.get("event") == "window_failed"]

    input_fps = _rate_from_timed_events(input_events)
    processed_fps = _rate_from_timed_events(window_events, count_key="processed_frames")

    sem_from_windows = [float(e["semantic_inference_s"]) for e in window_events if e.get("semantic_inference_s") is not None]
    sem_from_solver = [float(e["elapsed_s"]) for e in sem_events if e.get("elapsed_s") is not None]
    sem_values = sem_from_windows if sem_from_windows else sem_from_solver

    geometry_values = [float(e["geometry_window_s"]) for e in window_events if e.get("geometry_window_s") is not None]
    graph_values = [float(e["graph_optimization_s"]) for e in window_events if e.get("graph_optimization_s") is not None]

    processed_frames_total = int(sum(max(0.0, float(e.get("processed_frames", 0.0))) for e in window_events))

    return {
        "input_fps_runtime": input_fps,
        "processed_fps_runtime": processed_fps,
        "avg_semantic_inference_s": _safe_mean(sem_values),
        "avg_geometry_window_s": _safe_mean(geometry_values),
        "avg_graph_optimization_s": _safe_mean(graph_values),
        "num_input_events": float(len(input_events)),
        "num_window_events": float(len(window_events)),
        "num_window_failures": float(len(failed_events)),
        "processed_frames_total": float(processed_frames_total),
    }


def _print_summary(summary: Dict[str, Optional[float]]) -> None:
    def _fmt(key: str, digits: int = 4) -> str:
        val = summary.get(key)
        if val is None:
            return "n/a"
        return f"{float(val):.{digits}f}"

    print("Runtime Summary")
    print(f"- input FPS: {_fmt('input_fps')}")
    print(f"- processed FPS: {_fmt('processed_fps')}")
    print(f"- avg semantic inference (s): {_fmt('avg_semantic_inference_s')}")
    print(f"- avg geometry window (s): {_fmt('avg_geometry_window_s')}")
    print(f"- avg graph optimization (s): {_fmt('avg_graph_optimization_s')}")
    print(f"- windows processed: {_fmt('num_window_events', digits=0)}")
    print(f"- windows failed: {_fmt('num_window_failures', digits=0)}")
    print(f"- frames processed total: {_fmt('processed_frames_total', digits=0)}")


def _write_csv(path: Path, summary: Dict[str, Optional[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "input_fps",
        "processed_fps",
        "avg_semantic_inference_s",
        "avg_geometry_window_s",
        "avg_graph_optimization_s",
        "num_window_events",
        "num_window_failures",
        "processed_frames_total",
        "bridge_rx_fps_mean",
        "bridge_pub_fps_mean",
        "bridge_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: summary.get(k) for k in fields})


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize live runtime metrics for evaluation tables.")
    p.add_argument(
        "--runtime-jsonl",
        default="tap_logs/runtime_metrics.jsonl",
        help="Path to runtime JSONL emitted when VGGT_RUNTIME_METRICS=1.",
    )
    p.add_argument(
        "--bridge-log",
        help="Optional unitree_tcp_ros2_bridge log file to include rx_fps/pub_fps stats.",
    )
    p.add_argument("--out-json", type=Path)
    p.add_argument("--out-csv", type=Path)
    args = p.parse_args()

    events = _read_jsonl(Path(args.runtime_jsonl).expanduser())
    runtime_summary = _summarize_runtime(events)

    bridge_summary = {
        "rx_fps_mean": None,
        "pub_fps_mean": None,
        "num_samples": 0,
    }
    if args.bridge_log:
        bridge_summary = _parse_bridge_log(Path(args.bridge_log).expanduser())

    summary = {
        # Prefer bridge rx_fps as "input FPS" when available.
        "input_fps": (
            bridge_summary["rx_fps_mean"]
            if bridge_summary.get("rx_fps_mean") is not None
            else runtime_summary["input_fps_runtime"]
        ),
        "processed_fps": runtime_summary["processed_fps_runtime"],
        "avg_semantic_inference_s": runtime_summary["avg_semantic_inference_s"],
        "avg_geometry_window_s": runtime_summary["avg_geometry_window_s"],
        "avg_graph_optimization_s": runtime_summary["avg_graph_optimization_s"],
        "num_window_events": runtime_summary["num_window_events"],
        "num_window_failures": runtime_summary["num_window_failures"],
        "processed_frames_total": runtime_summary["processed_frames_total"],
        "bridge_rx_fps_mean": bridge_summary.get("rx_fps_mean"),
        "bridge_pub_fps_mean": bridge_summary.get("pub_fps_mean"),
        "bridge_samples": bridge_summary.get("num_samples"),
    }

    _print_summary(summary)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[out] {args.out_json}")
    if args.out_csv:
        _write_csv(args.out_csv, summary)
        print(f"[out] {args.out_csv}")


if __name__ == "__main__":
    main()
