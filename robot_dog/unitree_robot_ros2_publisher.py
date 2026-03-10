#!/usr/bin/env python3
"""Robot-side ROS2 publisher for Unitree front camera.

Runs on the robot, pulls JPEG frames from Unitree SDK2 VideoClient,
decodes them, and publishes sensor_msgs/Image at a fixed rate.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

# Some unitree-sdk2 builds import optional b2 unconditionally.
sys.modules.setdefault("unitree_sdk2py.b2", types.ModuleType("unitree_sdk2py.b2"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


def payload_to_bytes(data: Any) -> bytes:
    """Normalize Unitree payload variants (bytes/list/memoryview) to bytes."""
    if data is None:
        return b""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, list):
        return bytes(data)
    try:
        return bytes(data)
    except Exception:
        return b""


class UnitreeRobotRos2Publisher(Node):
    def __init__(
        self,
        interface_name: str,
        fps: float,
        timeout_s: float,
        image_topic: str,
        info_topic: str,
        frame_id: str,
        publish_raw: bool,
        publish_compressed: bool,
        compressed_topic: str,
        reliability: str,
        max_width: int,
        max_height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        stats_every: int,
    ):
        super().__init__("unitree_robot_ros2_publisher")

        if fps <= 0:
            raise ValueError("--fps must be > 0")

        self._bridge = CvBridge()
        self._frame_id = frame_id
        self._publish_raw = publish_raw
        self._publish_compressed = publish_compressed
        self._max_width = max(0, int(max_width))
        self._max_height = max(0, int(max_height))
        self._stats_every = max(1, int(stats_every))
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)

        ChannelFactoryInitialize(0, interface_name)
        self._client = VideoClient()
        self._client.SetTimeout(float(timeout_s))
        init_ret = self._client.Init()
        self.get_logger().info(f"VideoClient.Init() returned: {init_ret}")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=(
                ReliabilityPolicy.BEST_EFFORT
                if reliability == "best_effort"
                else ReliabilityPolicy.RELIABLE
            ),
            durability=DurabilityPolicy.VOLATILE,
        )

        self._image_pub = self.create_publisher(Image, image_topic, qos) if self._publish_raw else None
        self._info_pub = self.create_publisher(CameraInfo, info_topic, qos) if self._publish_raw else None
        self._compressed_pub = (
            self.create_publisher(CompressedImage, compressed_topic, qos)
            if publish_compressed
            else None
        )

        self._ok_count = 0
        self._fail_count = 0
        self._start_wall = time.time()
        self._timer = self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(
            f"Publishing raw={self._publish_raw} compressed={self._publish_compressed} "
            f"reliability={reliability} at {fps:.2f} FPS"
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

    def _tick(self) -> None:
        t0 = time.time()
        code, raw = self._client.GetImageSample()
        payload = payload_to_bytes(raw)
        if code != 0 or not payload:
            self._fail_count += 1
            if self._fail_count % 10 == 0:
                self.get_logger().warn(f"GetImageSample failed: code={code}, failures={self._fail_count}")
            return

        stamp = self.get_clock().now().to_msg()

        if self._publish_compressed and self._compressed_pub is not None:
            cmsg = CompressedImage()
            cmsg.header.stamp = stamp
            cmsg.header.frame_id = self._frame_id
            cmsg.format = "jpeg"
            cmsg.data = payload
            self._compressed_pub.publish(cmsg)

        if self._publish_raw and self._image_pub is not None:
            np_buf = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            if frame is None:
                self._fail_count += 1
                if self._fail_count % 10 == 0:
                    self.get_logger().warn("JPEG decode failed")
                return
            frame = self._resize_if_needed(frame)

            msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = self._frame_id
            self._image_pub.publish(msg)

            if self._info_pub is not None:
                h, w = frame.shape[:2]
                fx = self._fx if self._fx > 0 else float(max(w, h))
                fy = self._fy if self._fy > 0 else float(max(w, h))
                cx = self._cx if self._cx > 0 else float(w) / 2.0
                cy = self._cy if self._cy > 0 else float(h) / 2.0

                info = CameraInfo()
                info.header.stamp = stamp
                info.header.frame_id = self._frame_id
                info.width = int(w)
                info.height = int(h)
                info.distortion_model = "plumb_bob"
                info.d = []
                info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
                info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
                self._info_pub.publish(info)

        self._ok_count += 1
        if self._ok_count % self._stats_every == 0:
            dt = time.time() - self._start_wall
            eff_fps = self._ok_count / max(1e-6, dt)
            tick_ms = (time.time() - t0) * 1000.0
            self.get_logger().info(
                f"Published {self._ok_count} frames, effective_fps={eff_fps:.2f}, "
                f"last_tick_ms={tick_ms:.1f}, failures={self._fail_count}"
            )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Robot-side ROS2 publisher for Unitree front camera")
    p.add_argument("--interface", default="eth0", help="Robot NIC for Unitree SDK DDS")
    p.add_argument("--fps", type=float, default=2.0, help="ROS2 publish rate")
    p.add_argument("--timeout", type=float, default=0.8, help="VideoClient timeout seconds")
    p.add_argument("--image-topic", default="/unitree/front_camera/image_raw", help="Image topic")
    p.add_argument("--info-topic", default="/unitree/front_camera/camera_info", help="CameraInfo topic")
    p.add_argument("--frame-id", default="unitree_front_camera", help="Message frame_id")
    p.add_argument("--no-raw", action="store_true", help="Disable raw Image publishing")
    p.add_argument("--publish-compressed", action="store_true", help="Also publish CompressedImage")
    p.add_argument(
        "--compressed-topic",
        default="/unitree/front_camera/image/compressed",
        help="Compressed image topic",
    )
    p.add_argument(
        "--reliability",
        choices=["best_effort", "reliable"],
        default="best_effort",
        help="QoS reliability for publishers",
    )
    p.add_argument("--max-width", type=int, default=0, help="Optional max width for raw image (0 disables resize)")
    p.add_argument("--max-height", type=int, default=0, help="Optional max height for raw image (0 disables resize)")
    p.add_argument("--fx", type=float, default=-1.0, help="Camera fx (<=0 uses fallback)")
    p.add_argument("--fy", type=float, default=-1.0, help="Camera fy (<=0 uses fallback)")
    p.add_argument("--cx", type=float, default=-1.0, help="Camera cx (<=0 uses image center)")
    p.add_argument("--cy", type=float, default=-1.0, help="Camera cy (<=0 uses image center)")
    p.add_argument("--stats-every", type=int, default=20, help="Log statistics every N published frames")
    args, ros_args = p.parse_known_args()
    return args, ros_args


def main() -> None:
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)

    node = UnitreeRobotRos2Publisher(
        interface_name=args.interface,
        fps=args.fps,
        timeout_s=args.timeout,
        image_topic=args.image_topic,
        info_topic=args.info_topic,
        frame_id=args.frame_id,
        publish_raw=(not args.no_raw),
        publish_compressed=args.publish_compressed,
        compressed_topic=args.compressed_topic,
        reliability=args.reliability,
        max_width=args.max_width,
        max_height=args.max_height,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        stats_every=args.stats_every,
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
