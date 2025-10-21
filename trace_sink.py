"""
trace_sink.py

Thread-safe JSONL logger for tracing frame selection metadata during VGGT
forward passes. Created per the instrumentation plan so downstream modules
can emit ground-truth frame indices.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterable, Optional


class TraceSink:
    """Append-only JSONL sink with a simple locking protocol."""

    def __init__(self, path: str = "tap_logs/trace.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        step: int,
        kind: str,
        indices: Iterable[int],
        note: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Record a trace event.

        Args:
            step: Monotonic step identifier provided by the caller.
            kind: Short string describing what is being traced
                  (e.g. ``"dino_indices"`` or ``"dpt_indices"``).
            indices: Iterable of frame indices selected for this branch.
            note: Optional contextual payload, typically the frame ids.
        """
        record = {
            "t": time.time(),
            "step": int(step),
            "kind": str(kind),
            "indices": [int(i) for i in indices],
            "note": list(note) if note is not None else None,
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)

