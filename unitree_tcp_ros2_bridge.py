#!/usr/bin/env python3
"""Receive Unitree JPEG frames over TCP and publish ROS2 Image topics.

Expected robot-side protocol per frame:
  header: struct '!QdI' -> (seq, robot_ts_sec, jpeg_len)
  payload: jpeg bytes
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

HEADER = struct.Struct("!QdI")


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def decode_jpeg_resilient(payload: bytes) -> tuple[Optional[np.ndarray], bytes]:
    """Decode JPEG payload and recover from occasional framing garbage."""
    np_buf = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if frame is not None:
        return frame, payload

    # Some streams occasionally carry bytes before/after JPEG SOI/EOI markers.
    soi = payload.find(b"\xff\xd8")
    eoi = payload.rfind(b"\xff\xd9")
    if soi >= 0 and eoi > soi:
        cleaned = payload[soi : eoi + 2]
        np_buf = np.frombuffer(cleaned, dtype=np.uint8)
        frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if frame is not None:
            return frame, cleaned

    return None, payload


class UnitreeTcpRos2Bridge(Node):
    def __init__(
        self,
        robot_ip: str,
        port: int,
        fps: float,
        image_topic: str,
        info_topic: str,
        frame_id: str,
        publish_compressed: bool,
        compressed_topic: str,
        max_width: int,
        max_height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float,
        k2: float,
        p1: float,
        p2: float,
        k3: float,
        undistort: bool,
        undistort_alpha: float,
        stats_interval: float,
        save_every: int,
        save_dir: str,
        reconnect_sec: float,
    ):
        super().__init__("unitree_tcp_ros2_bridge")

        if fps <= 0.0:
            raise ValueError("fps must be > 0")

        self._robot_ip = robot_ip
        self._port = int(port)
        self._bridge = CvBridge()
        self._frame_id = frame_id
        self._reconnect_sec = max(0.2, reconnect_sec)
        self._max_width = max(0, int(max_width))
        self._max_height = max(0, int(max_height))
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)
        self._k1 = float(k1)
        self._k2 = float(k2)
        self._p1 = float(p1)
        self._p2 = float(p2)
        self._k3 = float(k3)
        self._dist = np.array([self._k1, self._k2, self._p1, self._p2, self._k3], dtype=np.float64)
        self._undistort = bool(undistort)
        self._undistort_alpha = float(undistort_alpha)
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._map_hw: Optional[tuple[int, int]] = None
        self._rect_k: Optional[np.ndarray] = None
        self._stats_interval = max(0.5, float(stats_interval))
        self._save_every = max(0, int(save_every))
        self._save_dir = save_dir.strip()
        if self._save_every > 0 and self._save_dir:
            os.makedirs(self._save_dir, exist_ok=True)

        if self._undistort and self._fx <= 0:
            self.get_logger().warn(
                "Undistortion enabled without calibrated fx/fy/cx/cy. "
                "Set --fx/--fy/--cx/--cy for meaningful rectification."
            )

        self._image_pub = self.create_publisher(Image, image_topic, 10)
        self._info_pub = self.create_publisher(CameraInfo, info_topic, 10)
        self._compressed_pub = (
            self.create_publisher(CompressedImage, compressed_topic, 10)
            if publish_compressed
            else None
        )

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_seq: Optional[int] = None
        self._latest_jpeg: Optional[bytes] = None

        self._rx_count = 0
        self._pub_count = 0
        self._last_pub_seq: Optional[int] = None
        self._last_stats_wall = time.time()
        self._last_stats_rx = 0
        self._last_stats_pub = 0
        self._last_hw: Optional[tuple[int, int]] = None

        self._rx_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._rx_thread.start()

        self._timer = self.create_timer(1.0 / fps, self._publish_tick)
        self._stats_timer = self.create_timer(self._stats_interval, self._log_stats)
        self.get_logger().info(
            f"Receiving TCP stream from {self._robot_ip}:{self._port}, publishing "
            f"{image_topic} + {info_topic} at {fps:.2f} FPS"
        )

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        sx = 1.0
        sy = 1.0
        if self._max_width > 0 and w > self._max_width:
            sx = self._max_width / float(w)
        if self._max_height > 0 and h > self._max_height:
            sy = self._max_height / float(h)
        s = min(sx, sy)
        if s >= 1.0:
            return frame
        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def _intrinsics_for_size(self, w: int, h: int) -> np.ndarray:
        fx = self._fx if self._fx > 0 else float(max(w, h))
        fy = self._fy if self._fy > 0 else float(max(w, h))
        cx = self._cx if self._cx > 0 else float(w) / 2.0
        cy = self._cy if self._cy > 0 else float(h) / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _undistort_if_enabled(self, frame: np.ndarray, k_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._undistort:
            return frame, k_in

        h, w = frame.shape[:2]
        if self._map1 is None or self._map_hw != (w, h):
            rect_k, _ = cv2.getOptimalNewCameraMatrix(
                k_in,
                self._dist,
                (w, h),
                alpha=self._undistort_alpha,
                newImgSize=(w, h),
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                k_in,
                self._dist,
                None,
                rect_k,
                (w, h),
                cv2.CV_16SC2,
            )
            self._map1 = map1
            self._map2 = map2
            self._rect_k = rect_k
            self._map_hw = (w, h)

        frame_ud = cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
        return frame_ud, self._rect_k if self._rect_k is not None else k_in

    def _receiver_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.get_logger().info(f"Connecting to {self._robot_ip}:{self._port} ...")
                sock = socket.create_connection((self._robot_ip, self._port), timeout=5.0)
                sock.settimeout(5.0)
                self.get_logger().info("TCP stream connected.")

                while not self._stop.is_set():
                    hdr = recv_exact(sock, HEADER.size)
                    seq, _robot_ts, jpeg_len = HEADER.unpack(hdr)
                    payload = recv_exact(sock, int(jpeg_len))
                    with self._lock:
                        self._latest_seq = int(seq)
                        self._latest_jpeg = payload
                    self._rx_count += 1
            except Exception as exc:
                self.get_logger().warn(f"TCP stream error: {exc}. Reconnecting in {self._reconnect_sec:.1f}s")
                time.sleep(self._reconnect_sec)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    def _publish_tick(self) -> None:
        with self._lock:
            seq = self._latest_seq
            jpeg = self._latest_jpeg

        if seq is None or jpeg is None:
            return
        if self._last_pub_seq is not None and seq == self._last_pub_seq:
            return

        frame, jpeg_for_compressed = decode_jpeg_resilient(jpeg)
        if frame is None:
            self.get_logger().warn("JPEG decode failed for received frame")
            return
        frame = self._resize_if_needed(frame)
        h, w = frame.shape[:2]
        k_in = self._intrinsics_for_size(w, h)
        frame, k_pub = self._undistort_if_enabled(frame, k_in)

        stamp = self.get_clock().now().to_msg()

        img_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self._frame_id
        self._image_pub.publish(img_msg)

        h, w = frame.shape[:2]
        self._last_hw = (w, h)
        fx = float(k_pub[0, 0])
        fy = float(k_pub[1, 1])
        cx = float(k_pub[0, 2])
        cy = float(k_pub[1, 2])
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._frame_id
        info.width = int(w)
        info.height = int(h)
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0] if self._undistort else self._dist.tolist()
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._info_pub.publish(info)

        if self._compressed_pub is not None:
            cmsg = CompressedImage()
            cmsg.header.stamp = stamp
            cmsg.header.frame_id = self._frame_id
            cmsg.format = "jpeg"
            if self._undistort:
                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not ok:
                    self.get_logger().warn("Failed to JPEG-encode undistorted frame")
                    return
                cmsg.data = enc.tobytes()
            else:
                cmsg.data = jpeg_for_compressed
            self._compressed_pub.publish(cmsg)

        self._last_pub_seq = seq
        self._pub_count += 1
        if self._save_every > 0 and self._save_dir and (self._pub_count % self._save_every == 0):
            out = os.path.join(self._save_dir, f"frame_{self._pub_count:06d}_{w}x{h}.jpg")
            try:
                cv2.imwrite(out, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            except Exception as exc:
                self.get_logger().warn(f"Failed to save sample frame: {exc}")

    def _log_stats(self) -> None:
        now = time.time()
        dt = now - self._last_stats_wall
        if dt <= 0:
            return

        rx_delta = self._rx_count - self._last_stats_rx
        pub_delta = self._pub_count - self._last_stats_pub
        rx_fps = rx_delta / dt
        pub_fps = pub_delta / dt
        hw = self._last_hw
        hw_txt = f"{hw[0]}x{hw[1]}" if hw else "n/a"

        self.get_logger().info(
            f"stats: rx_fps={rx_fps:.2f}, pub_fps={pub_fps:.2f}, "
            f"totals(rx={self._rx_count}, pub={self._pub_count}), "
            f"backlog={self._rx_count - self._pub_count}, hw={hw_txt}"
        )

        self._last_stats_wall = now
        self._last_stats_rx = self._rx_count
        self._last_stats_pub = self._pub_count

    def destroy_node(self) -> bool:
        self._stop.set()
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=2.0)
        return super().destroy_node()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="TCP-to-ROS2 bridge for Unitree camera")
    p.add_argument("--robot-ip", required=True, help="Robot IP reachable from this host")
    p.add_argument("--port", type=int, default=5001, help="TCP port")
    p.add_argument("--fps", type=float, default=2.0, help="Publish rate")
    p.add_argument("--image-topic", default="/unitree/front_camera/image_raw", help="ROS2 image topic")
    p.add_argument("--info-topic", default="/unitree/front_camera/camera_info", help="ROS2 CameraInfo topic")
    p.add_argument("--frame-id", default="unitree_front_camera", help="ROS2 frame_id")
    p.add_argument("--publish-compressed", action="store_true", help="Also publish CompressedImage")
    p.add_argument(
        "--compressed-topic",
        default="/unitree/front_camera/image/compressed",
        help="Compressed topic",
    )
    p.add_argument("--max-width", type=int, default=0, help="Optional max width for decoded image")
    p.add_argument("--max-height", type=int, default=0, help="Optional max height for decoded image")
    p.add_argument("--fx", type=float, default=-1.0, help="Camera fx (<=0 uses fallback)")
    p.add_argument("--fy", type=float, default=-1.0, help="Camera fy (<=0 uses fallback)")
    p.add_argument("--cx", type=float, default=-1.0, help="Camera cx (<=0 uses image center)")
    p.add_argument("--cy", type=float, default=-1.0, help="Camera cy (<=0 uses image center)")
    p.add_argument("--k1", type=float, default=0.0, help="Radial distortion k1")
    p.add_argument("--k2", type=float, default=0.0, help="Radial distortion k2")
    p.add_argument("--p1", type=float, default=0.0, help="Tangential distortion p1")
    p.add_argument("--p2", type=float, default=0.0, help="Tangential distortion p2")
    p.add_argument("--k3", type=float, default=0.0, help="Radial distortion k3")
    p.add_argument("--undistort", action="store_true", help="Apply undistortion to published images")
    p.add_argument("--undistort-alpha", type=float, default=0.0, help="Undistort crop alpha in [0,1]")
    p.add_argument("--stats-interval", type=float, default=5.0, help="Seconds between fps stats logs")
    p.add_argument("--save-every", type=int, default=0, help="Save one decoded frame every N published frames")
    p.add_argument("--save-dir", default="", help="Directory for saved sample frames")
    p.add_argument("--reconnect-sec", type=float, default=1.0, help="Reconnect delay")
    args, ros_args = p.parse_known_args()
    return args, ros_args


def main() -> None:
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)

    node = UnitreeTcpRos2Bridge(
        robot_ip=args.robot_ip,
        port=args.port,
        fps=args.fps,
        image_topic=args.image_topic,
        info_topic=args.info_topic,
        frame_id=args.frame_id,
        publish_compressed=args.publish_compressed,
        compressed_topic=args.compressed_topic,
        max_width=args.max_width,
        max_height=args.max_height,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        k1=args.k1,
        k2=args.k2,
        p1=args.p1,
        p2=args.p2,
        k3=args.k3,
        undistort=args.undistort,
        undistort_alpha=args.undistort_alpha,
        stats_interval=args.stats_interval,
        save_every=args.save_every,
        save_dir=args.save_dir,
        reconnect_sec=args.reconnect_sec,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
