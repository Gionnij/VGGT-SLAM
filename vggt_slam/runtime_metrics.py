from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_LOGGER: Optional["RuntimeMetricsLogger"] = None


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


class RuntimeMetricsLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, **fields: Any) -> None:
        record: Dict[str, Any] = {
            "t": float(time.time()),
            "event": str(event),
        }
        record.update(fields)
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)


def get_runtime_logger() -> Optional[RuntimeMetricsLogger]:
    global _LOGGER
    if not _truthy(os.getenv("VGGT_RUNTIME_METRICS", "0")):
        return None
    if _LOGGER is None:
        path = Path(os.getenv("VGGT_RUNTIME_METRICS_PATH", "tap_logs/runtime_metrics.jsonl")).expanduser()
        _LOGGER = RuntimeMetricsLogger(path)
    return _LOGGER
