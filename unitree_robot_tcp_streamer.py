#!/usr/bin/env python3
"""Pull Unitree camera frames on the robot and stream JPEG over TCP.

Protocol per frame:
  header: struct '!QdI' -> (seq, robot_ts_sec, jpeg_len)
  payload: jpeg bytes
"""

from __future__ import annotations

import argparse
import signal
import socket
import struct
import time
from typing import Any

# Some unitree-sdk2 builds import optional b2 unconditionally.
import sys
import types

sys.modules.setdefault("unitree_sdk2py.b2", types.ModuleType("unitree_sdk2py.b2"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

HEADER = struct.Struct("!QdI")


def to_bytes_payload(data: Any) -> bytes:
    """Normalize Unitree SDK image payload to bytes."""
    if data is None:
        return b""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    # unitree_sdk2py can return list[int] on some builds.
    if isinstance(data, list):
        return bytes(data)
    # fallback for array-like objects
    try:
        return bytes(data)
    except Exception:
        return b""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unitree VideoClient TCP streamer")
    p.add_argument("--interface", default="eth0", help="Robot NIC for Unitree SDK DDS")
    p.add_argument("--bind", default="0.0.0.0", help="Bind address for TCP server")
    p.add_argument("--port", type=int, default=5001, help="TCP port")
    p.add_argument("--fps", type=float, default=2.0, help="Stream rate")
    p.add_argument("--timeout", type=float, default=3.0, help="VideoClient timeout seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    stop = False

    def _sig_handler(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    ChannelFactoryInitialize(0, args.interface)
    client = VideoClient()
    client.SetTimeout(float(args.timeout))
    init_ret = client.Init()
    print(f"[robot-streamer] VideoClient.Init()={init_ret}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(1)
    server.settimeout(1.0)

    print(f"[robot-streamer] Listening on {args.bind}:{args.port} @ {args.fps:.2f} FPS")

    seq = 0
    period = 1.0 / args.fps

    try:
        while not stop:
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue

            print(f"[robot-streamer] Client connected: {addr}")
            conn.settimeout(2.0)
            next_t = time.time()

            try:
                while not stop:
                    code, data = client.GetImageSample()
                    payload = to_bytes_payload(data)
                    if code != 0 or not payload:
                        print(f"[robot-streamer] GetImageSample failed: code={code}")
                        time.sleep(0.2)
                        continue

                    now = time.time()
                    frame = HEADER.pack(seq, now, len(payload)) + payload
                    conn.sendall(frame)
                    seq += 1

                    next_t += period
                    sleep_t = next_t - time.time()
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                    else:
                        next_t = time.time()
            except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                print("[robot-streamer] Client disconnected.")
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
