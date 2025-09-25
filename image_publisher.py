#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import threading
import select
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Empty
from cv_bridge import CvBridge


# ---------- helpers ----------
def _f(x):  # ensure native Python float
    return float(x)

def make_camera_info(w, h, fx=None, fy=None, cx=None, cy=None):
    if fx is None or fy is None:
        fx = fy = max(w, h)
    if cx is None or cy is None:
        cx = w / 2.0
        cy = h / 2.0

    K = [_f(fx), _f(0),  _f(cx),
         _f(0),  _f(fy), _f(cy),
         _f(0),  _f(0),  _f(1)]
    P = [_f(fx), _f(0),  _f(cx), _f(0),
         _f(0),  _f(fy), _f(cy), _f(0),
         _f(0),  _f(0),  _f(1),  _f(0)]

    msg = CameraInfo()
    msg.width = int(w)
    msg.height = int(h)
    msg.k = K
    msg.p = P
    msg.d = []
    msg.distortion_model = "plumb_bob"
    return msg


# ---------- node ----------
class FolderCamera(Node):
    def __init__(self, folder, topic_image, topic_info, hz, loop, long_side_cap):
        super().__init__('folder_camera')
        self.bridge = CvBridge()
        self.pub_img  = self.create_publisher(Image,      topic_image, 10)
        self.pub_info = self.create_publisher(CameraInfo, topic_info,  10)
        self.pub_stop = self.create_publisher(Empty, '/stream/stop', 10)

        self.paths = sorted([p for p in glob.glob(os.path.join(folder, "*"))
                             if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")])
        if not self.paths:
            raise RuntimeError(f"No images found in {folder}")

        self.total = len(self.paths)
        self.index = 0
        self.loop = loop
        self.dt = 1.0 / hz
        self.long_side_cap = int(long_side_cap)
        self.published_count = 0
        self._stopping = False
        self._stop_sent = False

        # prime camera info
        img0 = cv2.imread(self.paths[0])
        img0 = self._resize(img0)
        h, w = img0.shape[:2]
        self.cam_info = make_camera_info(w, h)

        # non-blocking key listener: press 'q' + Enter to stop gracefully
        self._start_key_listener()

        # main publish timer
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"Publishing {self.total} images at {hz} Hz from {folder}. "
            f"Press 'q' + Enter to stop gracefully."
        )

    def _start_key_listener(self):
        def _listen():
            try:
                while not self._stopping and rclpy.ok():
                    r, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if r:
                        line = sys.stdin.readline().strip().lower()
                        if line == 'q':
                            self.get_logger().info("Keypress 'q' received; initiating graceful stop…")
                            self._stopping = True
                            break
            except Exception:
                # stdin may not be a TTY in some schedulers; ignore
                pass
        t = threading.Thread(target=_listen, daemon=True)
        t.start()

    def _resize(self, img):
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side <= self.long_side_cap:
            return img
        scale = self.long_side_cap / float(long_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _tick(self):
        if self._stopping:
            # send /stream/stop once, then shutdown
            if not self._stop_sent:
                try:
                    self.pub_stop.publish(Empty())
                except Exception:
                    pass
                self._stop_sent = True
                self.get_logger().info("Sent /stream/stop. Shutting down publisher…")
            rclpy.shutdown()
            return

        path = self.paths[self.index]
        img = cv2.imread(path)
        if img is None:
            self.get_logger().warn(f"Could not read {path}")
            self._advance_index()
            return

        img = self._resize(img)
        h, w = img.shape[:2]

        # update CameraInfo if size changes
        if self.cam_info.width != w or self.cam_info.height != h:
            self.cam_info = make_camera_info(w, h)

        stamp = self.get_clock().now().to_msg()

        # publish CameraInfo
        self.cam_info.header.stamp = stamp
        self.cam_info.header.frame_id = "camera"
        self.pub_info.publish(self.cam_info)

        # publish Image
        msg_img = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg_img.header.stamp = stamp
        msg_img.header.frame_id = "camera"
        self.pub_img.publish(msg_img)

        self.published_count += 1
        if (self.published_count % 10) == 0 or self.published_count == self.total:
            self.get_logger().info(f"published {self.published_count}/{self.total} frames")

        self._advance_index()

    def _advance_index(self):
        self.index += 1
        if self.index >= self.total:
            if self.loop:
                self.index = 0
            else:
                self.get_logger().info("Finished publishing all images; stopping gracefully…")
                self._stopping = True


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with images")
    ap.add_argument("--hz", type=float, default=2.0, help="Publish rate (Hz)")
    ap.add_argument("--loop", action="store_true", help="Loop images forever")
    ap.add_argument("--image_topic", default="/camera/image_color")
    ap.add_argument("--info_topic",  default="/camera/camera_info")
    ap.add_argument("--long_side_cap", type=int, default=1920, help="Resize long side (px)")
    args = ap.parse_args()

    try:
        rclpy.init()
        node = FolderCamera(args.folder, args.image_topic, args.info_topic,
                            args.hz, args.loop, args.long_side_cap)
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Allow Ctrl-C to hard-exit immediately
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()