#!/usr/bin/env python3
"""ROS2 bridge for Unitree multimedia UDP video stream.

This node subscribes to Unitree's H264 RTP multicast stream through a
GStreamer pipeline and republishes frames as ROS 2 image topics.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


def build_gstreamer_pipeline(
    interface_name: str,
    multicast_address: str,
    port: int,
    width: int,
    height: int,
) -> str:
    """Create a GStreamer pipeline string for Unitree multimedia stream."""
    raw_caps = "video/x-raw,format=BGR"
    if width > 0 and height > 0:
        raw_caps += f",width={width},height={height}"

    return (
        f"udpsrc address={multicast_address} port={port} "
        f"multicast-iface={interface_name} auto-multicast=true ! "
        "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        f"{raw_caps} ! appsink drop=1 max-buffers=1 sync=false"
    )


class UnitreeMultimediaRos2Bridge(Node):
    def __init__(
        self,
        interface_name: str,
        multicast_address: str,
        port: int,
        width: int,
        height: int,
        fps: float,
        image_topic: str,
        publish_compressed: bool,
        compressed_topic: str,
        frame_id: str,
        reconnect_sec: float,
    ):
        super().__init__("unitree_multimedia_ros2_bridge")

        if fps <= 0.0:
            raise ValueError("fps must be > 0")

        self._fps = fps
        self._frame_id = frame_id
        self._reconnect_sec = max(0.1, reconnect_sec)
        self._bridge = CvBridge()
        self._publish_compressed = publish_compressed

        self._image_pub = self.create_publisher(Image, image_topic, 10)
        self._compressed_pub = (
            self.create_publisher(CompressedImage, compressed_topic, 10)
            if publish_compressed
            else None
        )

        self._pipeline = build_gstreamer_pipeline(
            interface_name=interface_name,
            multicast_address=multicast_address,
            port=port,
            width=width,
            height=height,
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._rx_count = 0
        self._pub_count = 0

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self._publish_timer = self.create_timer(1.0 / self._fps, self._publish_latest)

        self.get_logger().info(f"Publishing to: {image_topic} at {self._fps:.2f} FPS")
        self.get_logger().info(f"GStreamer pipeline: {self._pipeline}")
        if self._publish_compressed:
            self.get_logger().info(f"Also publishing compressed topic: {compressed_topic}")

    def _open_capture(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            return False
        self._cap = cap
        return True

    def _capture_loop(self) -> None:
        last_open_log = 0.0
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                opened = self._open_capture()
                if not opened:
                    now = time.time()
                    if now - last_open_log > 2.0:
                        self.get_logger().error(
                            "Failed to open multimedia stream. "
                            "Check interface name, network route, and GStreamer plugins."
                        )
                        last_open_log = now
                    time.sleep(self._reconnect_sec)
                    continue

                self.get_logger().info("Connected to multimedia stream.")

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.get_logger().warn("Stream read failed. Reconnecting...")
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                time.sleep(self._reconnect_sec)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            self._rx_count += 1

    def _publish_latest(self) -> None:
        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()

        stamp = self.get_clock().now().to_msg()

        msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        self._image_pub.publish(msg)

        if self._publish_compressed and self._compressed_pub is not None:
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                cmsg = CompressedImage()
                cmsg.header.stamp = stamp
                cmsg.header.frame_id = self._frame_id
                cmsg.format = "jpeg"
                cmsg.data = encoded.tobytes()
                self._compressed_pub.publish(cmsg)

        self._pub_count += 1
        if self._pub_count % 20 == 0:
            self.get_logger().info(
                f"Published {self._pub_count} frames (received {self._rx_count} raw frames)"
            )

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        return super().destroy_node()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Bridge Unitree multimedia stream to ROS2 image topics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interface", required=True, help="Network interface connected to robot (e.g., enp3s0)")
    parser.add_argument("--multicast-address", default="230.1.1.1", help="Multicast address")
    parser.add_argument("--port", type=int, default=1720, help="UDP port (front camera is typically 1720)")
    parser.add_argument("--width", type=int, default=1280, help="Expected frame width (set <=0 to disable)")
    parser.add_argument("--height", type=int, default=720, help="Expected frame height (set <=0 to disable)")
    parser.add_argument("--fps", type=float, default=2.0, help="ROS2 publish rate")
    parser.add_argument("--image-topic", default="/unitree/front_camera/image_raw", help="ROS2 image topic")
    parser.add_argument(
        "--publish-compressed",
        action="store_true",
        help="Also publish sensor_msgs/CompressedImage",
    )
    parser.add_argument(
        "--compressed-topic",
        default="/unitree/front_camera/image/compressed",
        help="ROS2 compressed image topic",
    )
    parser.add_argument("--frame-id", default="unitree_front_camera", help="frame_id for published messages")
    parser.add_argument("--reconnect-sec", type=float, default=1.0, help="Reconnect delay when stream drops")

    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main() -> None:
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)

    node = UnitreeMultimediaRos2Bridge(
        interface_name=args.interface,
        multicast_address=args.multicast_address,
        port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        image_topic=args.image_topic,
        publish_compressed=args.publish_compressed,
        compressed_topic=args.compressed_topic,
        frame_id=args.frame_id,
        reconnect_sec=args.reconnect_sec,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
