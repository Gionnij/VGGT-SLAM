#!/usr/bin/env python3
"""ROS2 bridge using Unitree SDK2 VideoClient (no multicast required).

Pulls JPEG frames from Unitree VideoClient and republishes to ROS2 at a fixed rate.
"""

from __future__ import annotations

import argparse
import sys
import types
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
# Compatibility shim for unitree-sdk2 wheels that import optional `b2`
# unconditionally in unitree_sdk2py/__init__.py on some platforms.
sys.modules.setdefault("unitree_sdk2py.b2", types.ModuleType("unitree_sdk2py.b2"))

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.go2.video.video_client import VideoClient


class UnitreeSdk2Ros2Bridge(Node):
    def __init__(
        self,
        client: VideoClient,
        fps: float,
        image_topic: str,
        frame_id: str,
        publish_compressed: bool,
        compressed_topic: str,
        jpeg_quality: int,
    ):
        super().__init__("unitree_sdk2_ros2_bridge")

        if fps <= 0.0:
            raise ValueError("fps must be > 0")

        self._frame_id = frame_id
        self._bridge = CvBridge()
        self._publish_compressed = publish_compressed
        self._jpeg_quality = max(10, min(100, jpeg_quality))

        self._image_pub = self.create_publisher(Image, image_topic, 10)
        self._compressed_pub: Optional[rclpy.publisher.Publisher] = (
            self.create_publisher(CompressedImage, compressed_topic, 10)
            if publish_compressed
            else None
        )

        self._client = client

        self._ok_count = 0
        self._err_count = 0

        self._timer = self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(f"Publishing to {image_topic} at {fps:.2f} FPS")

    def _tick(self) -> None:
        code, data = self._client.GetImageSample()
        if code != 0 or data is None or len(data) == 0:
            self._err_count += 1
            if self._err_count % 10 == 0:
                self.get_logger().warn(f"GetImageSample failed: code={code}, failures={self._err_count}")
            return

        np_buf = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if frame is None:
            self._err_count += 1
            if self._err_count % 10 == 0:
                self.get_logger().warn("JPEG decode failed from VideoClient payload")
            return

        stamp = self.get_clock().now().to_msg()

        msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        self._image_pub.publish(msg)

        if self._publish_compressed and self._compressed_pub is not None:
            ok, enc = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
            if ok:
                cmsg = CompressedImage()
                cmsg.header.stamp = stamp
                cmsg.header.frame_id = self._frame_id
                cmsg.format = "jpeg"
                cmsg.data = enc.tobytes()
                self._compressed_pub.publish(cmsg)

        self._ok_count += 1
        if self._ok_count % 20 == 0:
            self.get_logger().info(f"Published {self._ok_count} frames")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Bridge Unitree SDK2 VideoClient frames to ROS2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interface", required=True, help="Local NIC name to reach robot (Mac often en0)")
    parser.add_argument(
        "--robot-ip",
        default="",
        help="Optional robot IP for CycloneDDS unicast peer discovery (e.g., 192.168.225.122)",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Publish rate")
    parser.add_argument("--timeout", type=float, default=3.0, help="VideoClient timeout in seconds")
    parser.add_argument("--image-topic", default="/unitree/front_camera/image_raw", help="ROS2 image topic")
    parser.add_argument("--frame-id", default="unitree_front_camera", help="Header frame_id")
    parser.add_argument(
        "--publish-compressed",
        action="store_true",
        help="Also publish CompressedImage topic",
    )
    parser.add_argument(
        "--compressed-topic",
        default="/unitree/front_camera/image/compressed",
        help="ROS2 compressed topic",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for compressed topic")

    args, ros_args = parser.parse_known_args()
    return args, ros_args


def init_video_client(interface_name: str, timeout_s: float) -> VideoClient:
    """Initialize Unitree VideoClient with interface fallback."""
    init_err = None

    # 1) Try explicit interface first
    if interface_name:
        try:
            channel_mod.ChannelFactoryInitialize(0, interface_name)
            client = VideoClient()
            client.SetTimeout(float(timeout_s))
            init_ret = client.Init()
            print(f"[SDK2] ChannelFactory interface='{interface_name}' ok, VideoClient.Init()={init_ret}")
            return client
        except Exception as exc:
            init_err = exc
            print(f"[SDK2] Interface init failed on '{interface_name}': {exc}")

    # 2) Fallback: autodetermine interface
    try:
        channel_mod.ChannelFactoryInitialize(0)
        client = VideoClient()
        client.SetTimeout(float(timeout_s))
        init_ret = client.Init()
        print(f"[SDK2] ChannelFactory auto-interface ok, VideoClient.Init()={init_ret}")
        return client
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Unitree SDK2 ChannelFactory. "
            f"Explicit interface error: {init_err}. Auto-detect error: {exc}"
        ) from exc


def set_unicast_peer_config(interface_name: str, robot_ip: str) -> None:
    """Override Unitree ChannelFactory config to use unicast DDS peer discovery."""
    iface = interface_name if interface_name else "autodetermine"
    iface_xml = (
        f'<NetworkInterface name="{iface}" priority="default" multicast="default" />'
        if interface_name
        else '<NetworkInterface autodetermine="true" priority="default" multicast="default" />'
    )
    config = f"""<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <AllowMulticast>false</AllowMulticast>
      <Interfaces>
        {iface_xml}
      </Interfaces>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="{robot_ip}" />
      </Peers>
    </Discovery>
    <Tracing>
      <Verbosity>config</Verbosity>
      <OutputFile>/tmp/cdds.LOG</OutputFile>
    </Tracing>
  </Domain>
</CycloneDDS>"""
    channel_mod.ChannelConfigHasInterface = config
    channel_mod.ChannelConfigAutoDetermine = config
    print(f"[SDK2] Using explicit CycloneDDS unicast peer: {robot_ip}")


def main() -> None:
    args, ros_args = parse_args()
    if args.robot_ip:
        set_unicast_peer_config(args.interface, args.robot_ip)
    client = init_video_client(args.interface, args.timeout)
    rclpy.init(args=ros_args)

    node = UnitreeSdk2Ros2Bridge(
        client=client,
        fps=args.fps,
        image_topic=args.image_topic,
        frame_id=args.frame_id,
        publish_compressed=args.publish_compressed,
        compressed_topic=args.compressed_topic,
        jpeg_quality=args.jpeg_quality,
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
