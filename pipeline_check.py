"""
pipeline_check.py

Lightweight helper to record pipeline sanity checks as the system runs.
Enabled via environment variable `VGGT_PIPELINE_CHECK=1`. Optionally set
`VGGT_PIPELINE_CHECK_PATH` (default: ./pipeline_check.txt).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

_LOGGER: Optional["PipelineCheck"] = None


class PipelineCheck:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_line(f"\n=== Pipeline check started {timestamp} ===")

    def _write_line(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line.rstrip() + "\n")

    def log(
        self,
        stage: str,
        message: str,
        *,
        expected: Optional[str] = None,
        observed: Optional[str] = None,
        status: str = "info",
    ) -> None:
        parts = [
            f"[{stage}]",
            f"status={status}",
            f"msg={message}",
        ]
        if expected is not None:
            parts.append(f"expected={expected}")
        if observed is not None:
            parts.append(f"observed={observed}")
        self._write_line(" | ".join(parts))


def get_pipeline_logger() -> Optional[PipelineCheck]:
    """Return a singleton logger when VGGT_PIPELINE_CHECK is enabled."""
    global _LOGGER
    if os.getenv("VGGT_PIPELINE_CHECK", "0") != "1":
        return None
    if _LOGGER is None:
        path = os.getenv("VGGT_PIPELINE_CHECK_PATH", "pipeline_check.txt")
        _LOGGER = PipelineCheck(Path(path))
    return _LOGGER
