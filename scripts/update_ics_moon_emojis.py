#!/usr/bin/env python3
"""
Add moon phase emojis to SUMMARY lines in an existing ICS file.

Usage:
    python scripts/update_ics_moon_emojis.py path/to/calendar.ics

The script rewrites the file in-place, ensuring
  - "New Moon" summaries end with "🌚"
  - "Full Moon" summaries end with "🌝"
Existing emojis are preserved to avoid duplicates.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_emoji(summary: str) -> str:
    """Append the appropriate emoji to a SUMMARY payload if it's missing."""
    if summary.startswith("New Moon") and "🌚" not in summary:
        return "New Moon 🌚" + summary[len("New Moon") :]
    if summary.startswith("Full Moon") and "🌝" not in summary:
        return "Full Moon 🌝" + summary[len("Full Moon") :]
    return summary


def process_file(path: Path) -> None:
    original = path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    changes = 0

    for line in original:
        if line.startswith("SUMMARY"):
            prefix, _, payload = line.partition(":")
            if payload:
                new_payload = add_emoji(payload)
                if new_payload != payload:
                    changes += 1
                line = f"{prefix}:{new_payload}"
        updated_lines.append(line)

    if changes:
        path.write_text("\r\n".join(updated_lines) + "\r\n", encoding="utf-8")
        print(f"Updated {changes} SUMMARY lines in {path}")
    else:
        print(f"No SUMMARY lines needed changes in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ics_path", type=Path, help="Path to the ICS file to modify in-place")
    args = parser.parse_args()

    if not args.ics_path.exists():
        raise SystemExit(f"File not found: {args.ics_path}")

    process_file(args.ics_path)


if __name__ == "__main__":
    main()
