#!/usr/bin/env python3
import os
import time
import glob
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys, threading, signal, select, tty, termios
from std_msgs.msg import Empty

from sensor_msgs.msg import CameraInfo

def _f(x):  # ensure native Python float (not numpy.float32/64)
    return float(x)

def make_camera_info(w, h, fx=None, fy=None, cx=None, cy=None):
    # Simple pinhole intrinsics (good enough for testing)
    if fx is None or fy is None:
        fx = fy = max(w, h)  # naive focal for test
    if cx is None or cy is None:
        cx = w / 2.0
        cy = h / 2.0

    K = [_f(fx), _f(0),   _f(cx),
         _f(0),  _f(fy),  _f(cy),
         _f(0),  _f(0),   _f(1)]
    P = [_f(fx), _f(0),   _f(cx), _f(0),
         _f(0),  _f(fy),  _f(cy), _f(0),
         _f(0),  _f(0),   _f(1),  _f(0)]

    msg = CameraInfo()
    msg.width = int(w)
    msg.height = int(h)
    msg.k = K              # length 9, all Python floats ✅
    msg.p = P              # length 12, all Python floats ✅
    msg.d = []
    msg.distortion_model = "plumb_bob"
    return msg

class FolderCamera(Node):
    def __init__(self, folder, topic_image, topic_info, hz, loop, long_side_cap):
        super().__init__('folder_camera')
        self.bridge = CvBridge()
        self.pub_img = self.create_publisher(Image, topic_image, 10)
        self.pub_info = self.create_publisher(CameraInfo, topic_info, 10)
        self.pub_stop = self.create_publisher(Empty, '/stream/stop', 10)

        self.paths = sorted([p for p in glob.glob(os.path.join(folder, "*"))
                             if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")])
        if not self.paths:
            raise RuntimeError(f"No images found in {folder}")
        self.index = 0
        self.loop = loop
        self.dt = 1.0 / hz
        self.long_side_cap = long_side_cap

        # prime camera info
        img0 = cv2.imread(self.paths[0]); img0 = self._resize(img0)
        h, w = img0.shape[:2]
        self.cam_info = make_camera_info(w, h)

        self._stop = threading.Event()
        self._setup_signals()
        self._start_key_listener()   # non-blocking key listener

        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(f"Publishing {len(self.paths)} images at {hz} Hz from {folder}")

    def _setup_signals(self):
        def _sig_handler(sig, frame):
            self.get_logger().info(f"Received signal {sig}; stopping publisher…")
            self._stop.set()
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

    def _start_key_listener(self):
        def _listen():
            # Non-blocking stdin: press 'q' then Enter to quit
            try:
                while not self._stop.is_set():
                    r, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if r:
                        line = sys.stdin.readline().strip().lower()
                        if line == 'q':
                            self.get_logger().info("Keypress 'q' received; stopping publisher…")
                            self._stop.set()
                            break
            except Exception:
                pass
        t = threading.Thread(target=_listen, daemon=True)
        t.start()

    def _resize(self, img):
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side <= self.long_side_cap:
            return img
        scale = self.long_side_cap / float(long_side)
        return cv2.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_AREA)

    def _tick(self):
        if self._stop.is_set():
            # notify consumers, then shutdown
            try:
                self.pub_stop.publish(Empty())
            except Exception:
                pass
            self.get_logger().info("Publisher shutting down.")
            rclpy.shutdown()
            return

        path = self.paths[self.index]
        img = cv2.imread(path)
        if img is None:
            self.get_logger().warn(f"Could not read {path}")
            self.index = (self.index + 1) % len(self.paths)
            return

        img = self._resize(img)
        h, w = img.shape[:2]

        if self.cam_info.width != w or self.cam_info.height != h:
            self.cam_info = make_camera_info(w, h)

        stamp = self.get_clock().now().to_msg()
        self.cam_info.header.stamp = stamp
        self.cam_info.header.frame_id = "camera"
        self.pub_info.publish(self.cam_info)

        msg_img = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg_img.header.stamp = stamp
        msg_img.header.frame_id = "camera"
        self.pub_img.publish(msg_img)

        self.index += 1
        if self.index >= len(self.paths):
            if self.loop:
                self.index = 0
            else:
                self.get_logger().info("Finished publishing all images.")
                self._stop.set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with images")
    ap.add_argument("--hz", type=float, default=2.0, help="Publish rate (Hz)")
    ap.add_argument("--loop", action="store_true", help="Loop images forever")
    ap.add_argument("--image_topic", default="/camera/image_color")
    ap.add_argument("--info_topic", default="/camera/camera_info")
    ap.add_argument("--long_side_cap", type=int, default=1920, help="Resize long side (px)")
    args = ap.parse_args()

    rclpy.init()
    node = FolderCamera(args.folder, args.image_topic, args.info_topic, args.hz, args.loop, args.long_side_cap)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()