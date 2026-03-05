#!/usr/bin/env python3
"""Stream image-folder frames over Unitree TCP framing.

Protocol per frame:
  header: struct '!QdI' -> (seq, timestamp_sec, jpeg_len)
  payload: jpeg bytes
"""

from __future__ import annotations

import argparse
import signal
import socket
import struct
import time
from pathlib import Path
from typing import List

import cv2

HEADER = struct.Struct("!QdI")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scene image TCP streamer (Unitree framing)")
    p.add_argument("--folder", required=True, help="Image folder path")
    p.add_argument("--bind", default="127.0.0.1", help="Bind address")
    p.add_argument("--port", type=int, default=15001, help="TCP port")
    p.add_argument("--fps", type=float, default=2.0, help="Streaming FPS")
    p.add_argument("--loop", action="store_true", help="Loop over image set")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality [1..100]")
    p.add_argument("--max-width", type=int, default=0, help="Optional resize cap width (0 disables)")
    p.add_argument("--max-height", type=int, default=0, help="Optional resize cap height (0 disables)")
    p.add_argument("--log-every", type=int, default=20, help="Print every N sent frames")
    return p.parse_args()


def list_images(folder: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    out: List[Path] = []
    for pat in exts:
        out.extend(folder.glob(pat))
        out.extend(folder.glob(pat.upper()))
    out = sorted({p.resolve() for p in out})
    return out


def resize_if_needed(img, max_w: int, max_h: int):
    h, w = img.shape[:2]
    sx = 1.0
    sy = 1.0
    if max_w > 0 and w > max_w:
        sx = max_w / float(w)
    if max_h > 0 and h > max_h:
        sy = max_h / float(h)
    s = min(sx, sy)
    if s >= 1.0:
        return img
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.quality < 1 or args.quality > 100:
        raise ValueError("--quality must be in [1, 100]")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    images = list_images(folder)
    if not images:
        raise RuntimeError(f"No images found in: {folder}")

    stop = False

    def _sig_handler(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(1)
    server.settimeout(1.0)

    print(
        f"[scene-streamer] Serving {len(images)} images from {folder} "
        f"on {args.bind}:{args.port} @ {args.fps:.2f} FPS (loop={args.loop})"
    )

    seq = 0
    period = 1.0 / args.fps

    try:
        while not stop:
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue

            print(f"[scene-streamer] Client connected: {addr}")
            conn.settimeout(2.0)
            next_t = time.time()
            sent = 0
            idx = 0

            try:
                while not stop:
                    if idx >= len(images):
                        if args.loop:
                            idx = 0
                        else:
                            print("[scene-streamer] End of dataset, closing client.")
                            break

                    img_path = images[idx]
                    idx += 1

                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"[scene-streamer] Skipping unreadable file: {img_path}")
                        continue
                    img = resize_if_needed(img, int(args.max_width), int(args.max_height))
                    ok, enc = cv2.imencode(
                        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)]
                    )
                    if not ok:
                        print(f"[scene-streamer] JPEG encode failed: {img_path}")
                        continue

                    payload = enc.tobytes()
                    now = time.time()
                    frame = HEADER.pack(seq, now, len(payload)) + payload
                    conn.sendall(frame)
                    seq += 1
                    sent += 1

                    if args.log_every > 0 and (sent % int(args.log_every) == 0):
                        print(f"[scene-streamer] Sent {sent} frames to {addr}")

                    next_t += period
                    sleep_t = next_t - time.time()
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                    else:
                        next_t = time.time()

            except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                print("[scene-streamer] Client disconnected.")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        try:
            server.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
