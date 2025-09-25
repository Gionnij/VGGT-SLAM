#!/usr/bin/env python3
"""move_results.py – archive VGGT‑SLAM outputs into ./results/<N>_<DDMMYY>_<HHMM>

USAGE
-----
CLI:
    $ python move_results.py                  # archives from CWD

Import:
    from move_results import archive_results
    archive_results()                         # archives from CWD
    archive_results("/path/to/run/folder")    # or pass a folder explicitly

The function looks for these artefacts in the *base directory*:
    * poses.txt
    * poses_logs/ (directory)
    * poses_points.pcd (optional)

A sibling directory `results/` must already exist inside *base directory*.
It then creates a sub-folder named:

        <ordinal>_<DDMMYY>_<HHMM>

where <ordinal> = 0,1,2,… one higher than any existing numeric prefix in
`./results`.  It then moves the artefacts inside that sub-folder.
"""

from __future__ import annotations
import datetime as dt
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

__all__ = ["archive_results", "main"]

# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------

def log(msg: str, *a):
    print("[move] " + msg.format(*a), flush=True)


# -----------------------------------------------------------------------------
# Core API
# -----------------------------------------------------------------------------

def _next_results_dir(results_dir: Path) -> Path:
    """Return the next run directory path inside results_dir.

    The directory name is `<ordinal>_<DDMMYY>_<HHMM>` where ordinal is one
    higher than any existing numeric prefix in results_dir.
    """
    def extract_ord(p: Path) -> int:
        m = re.match(r"^(\d+)_", p.name)
        return int(m.group(1)) if m else -1

    existing = [extract_ord(p) for p in results_dir.iterdir() if p.is_dir()]
    next_ord = (max(existing) + 1) if existing else 0
    stamp = dt.datetime.now().strftime("%d%m%y_%H%M")
    dirname = f"{next_ord:02d}_{stamp}"
    return results_dir / dirname


def archive_results(base_dir: Optional[Path | str] = None) -> Path:
    """Archive VGGT‑SLAM outputs from *base_dir* into a new results subfolder.

    Parameters
    ----------
    base_dir : Path | str | None
        The directory where `poses.txt`, `poses_logs/`, and optionally
        `poses_points.pcd` are located. Defaults to the current working dir.

    Returns
    -------
    Path
        The path to the created archive directory.

    Raises
    ------
    SystemExit
        If `results/` does not exist under *base_dir*.
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()

    results_dir = base / "results"
    if not results_dir.is_dir():
        sys.exit("results/ directory not found in base dir. Create it first.")

    dest_dir = _next_results_dir(results_dir)
    dest_dir.mkdir(parents=True, exist_ok=False)
    log("Created {}", dest_dir)

    # artefacts
    artefacts = [
        (base / "poses.txt", dest_dir / "poses.txt"),
        (base / "poses_logs", dest_dir / "poses_logs"),
        (base / "poses_points.pcd", dest_dir / "poses_points.pcd"),
    ]

    for src, dst in artefacts:
        if not src.exists():
            # silently skip optional artefacts (like poses_points.pcd)
            if src.name == "poses_points.pcd":
                continue
            else:
                # For required ones, warn and continue (or you could raise)
                log("WARNING: {} not found, skipping.", src.name)
                continue

        if src.is_dir():
            shutil.move(str(src), str(dest_dir))
            log("{} → {}/", src.name, dest_dir)
        else:
            shutil.move(str(src), str(dst))
            log("{} → {}", src.name, dest_dir)

    return dest_dir


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

def main():
    """CLI entrypoint: archive from current working directory."""
    archive_results(Path.cwd())


if __name__ == "__main__":
    main()