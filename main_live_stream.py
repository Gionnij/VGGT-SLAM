#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import glob
import argparse
import time
import signal
import threading
import tempfile
import shutil
import json
import traceback
import importlib
from collections import deque
from queue import Queue, Empty as QueueEmpty
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def _gtsam_ok(mod) -> bool:
    if mod is None:
        return False
    if hasattr(mod, "NonlinearFactorGraph"):
        return True
    core = getattr(mod, "gtsam", None)
    if core is not None and hasattr(core, "NonlinearFactorGraph"):
        return True
    try:
        c = importlib.import_module("gtsam.gtsam")
        return hasattr(c, "NonlinearFactorGraph")
    except Exception:
        return False


def _bootstrap_gtsam_from_sitepkg() -> None:
    # Optional fallback: load gtsam from a secondary site-packages path
    # without permanently contaminating sys.path for other deps.
    gtsam_site = os.getenv("VGGT_GTSAM_SITEPKG", "").strip()
    if not gtsam_site or not os.path.isdir(gtsam_site):
        return
    try:
        import gtsam as _gtsam  # type: ignore
        if _gtsam_ok(_gtsam):
            return
    except Exception:
        pass

    old_path = list(sys.path)
    try:
        if gtsam_site not in sys.path:
            sys.path.insert(0, gtsam_site)
        sys.modules.pop("gtsam", None)
        sys.modules.pop("gtsam.gtsam", None)
        import gtsam as _gtsam  # type: ignore
        if _gtsam_ok(_gtsam):
            print(f"[gtsam-path] loaded from {gtsam_site}")
        else:
            print(f"[gtsam-path][WARN] gtsam loaded from {gtsam_site} but symbols still missing")
    except Exception as exc:
        print(f"[gtsam-path][WARN] failed fallback import from {gtsam_site}: {exc}")
    finally:
        sys.path[:] = old_path


_bootstrap_gtsam_from_sitepkg()

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from hiding_folder.vggt import VGGT
from taps_runtime import attach_vggt_taps
from trace_sink import TraceSink
from trace_hooks import install_trace_probes

# ---------- Optional ROS 2 imports (guarded) ----------
ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    from std_msgs.msg import Empty as StopMsg  # for /stream/stop
    ROS_AVAILABLE = True
except Exception:
    pass


# ----------------- Data structures --------------------
@dataclass
class Frame:
    img: np.ndarray           # HxWx3 uint8 (BGR from cv_bridge)
    ts: float                 # seconds (float)
    seq: int                  # monotonic increasing
    K: np.ndarray             # 3x3 intrinsics


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(np.sum(ex, axis=axis, keepdims=True), 1e-8, None)


def _palette_bgr(n: int = 2048) -> np.ndarray:
    idx = np.arange(n, dtype=np.uint32)
    b = (idx * 37 + 17) % 255
    g = (idx * 73 + 29) % 255
    r = (idx * 109 + 53) % 255
    return np.stack([b, g, r], axis=1).astype(np.uint8)


def _depth_to_vis(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    out = np.zeros((*d.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    vals = d[valid]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        hi = lo + 1e-3
    dn = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    dn_u8 = (dn * 255.0).astype(np.uint8)
    out = cv2.applyColorMap(dn_u8, cv2.COLORMAP_TURBO)
    out[~valid] = 0
    return out


def _sem_logits_to_mask_per_frame(
    sem_mask_logits: np.ndarray,
    sem_cls_logits: np.ndarray,
    target_hw: tuple[int, int],
    max_frames: int,
) -> Optional[List[np.ndarray]]:
    if sem_mask_logits is None or sem_cls_logits is None:
        return None
    sm = np.asarray(sem_mask_logits)
    sc = np.asarray(sem_cls_logits)
    # Accept both [S,Q,H,W]/[S,Q,C] and batched [B,S,Q,H,W]/[B,S,Q,C].
    if sm.ndim == 5 and sc.ndim == 4:
        sm = sm.reshape(-1, *sm.shape[-3:])
        sc = sc.reshape(-1, *sc.shape[-2:])
    if sm.ndim == 3:
        sm = sm[None, ...]
    if sc.ndim == 2:
        sc = sc[None, ...]
    if sm.ndim != 4 or sc.ndim != 3:
        return None

    S = min(max_frames, sm.shape[0], sc.shape[0])
    if S <= 0:
        return None
    sm = sm[:S]
    sc = sc[:S]
    if sc.shape[-1] > 1:
        sc = sc[..., :-1]  # drop no-object class
    cls = _softmax_np(sc, axis=-1)
    mask_prob = 1.0 / (1.0 + np.exp(-sm))
    dense = np.einsum("sqc,sqhw->schw", cls, mask_prob)
    pred = np.argmax(dense, axis=1).astype(np.uint16)  # (S,H,W)

    th, tw = target_hw
    out = []
    for i in range(S):
        m = pred[i]
        if m.shape != (th, tw):
            m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        out.append(m)
    return out


class DemoExporter:
    def __init__(self, root: str):
        base = Path(root).expanduser()
        self.run_dir = _mkdir(base / f"demo_{_now_tag()}")
        self.rgb_dir = _mkdir(self.run_dir / "rgb")
        self.depth_npz_dir = _mkdir(self.run_dir / "depth_maps")
        self.depth_vis_dir = _mkdir(self.run_dir / "depth_maps_vis")
        self.mask_dir = _mkdir(self.run_dir / "segmentation_masks")
        self.overlay_dir = _mkdir(self.run_dir / "segmentation_overlays")
        self.saved_seq: set[int] = set()
        self.saved_sem_seq: set[int] = set()
        self.palette = _palette_bgr()
        self.num_saved = 0
        os.environ["VGGT_ACTIVE_DEMO_RUN_DIR"] = str(self.run_dir)
        print(f"[DEMO] Saving run artifacts to: {self.run_dir}")

    def export_batch(self, frames: List[Frame], frame_ids_window: List[str], predictions: dict) -> None:
        if not frames:
            return

        n = len(frames)
        depth = predictions.get("depth")
        depth_conf = predictions.get("depth_conf")
        sem_masks = predictions.get("sem_mask_logits")
        sem_cls = predictions.get("sem_cls_logits")
        sem_frame_indices = predictions.get("sem_frame_indices")

        h0, w0 = frames[0].img.shape[:2]
        sem_per_frame = _sem_logits_to_mask_per_frame(sem_masks, sem_cls, target_hw=(h0, w0), max_frames=n)
        sem_by_frame_idx: dict[int, np.ndarray] = {}
        if sem_per_frame is not None:
            if isinstance(sem_frame_indices, (list, tuple)) and len(sem_frame_indices) == len(sem_per_frame):
                for src_i, mask in zip(sem_frame_indices, sem_per_frame):
                    try:
                        fi = int(src_i)
                    except Exception:
                        continue
                    if 0 <= fi < n:
                        sem_by_frame_idx[fi] = mask
            else:
                for fi, mask in enumerate(sem_per_frame):
                    if fi >= n:
                        break
                    sem_by_frame_idx[fi] = mask

        depth_arr = np.asarray(depth) if depth is not None else None
        depth_conf_arr = np.asarray(depth_conf) if depth_conf is not None else None
        if depth_arr is not None:
            if depth_arr.ndim == 4 and depth_arr.shape[-1] == 1:
                depth_arr = depth_arr[..., 0]
            elif depth_arr.ndim != 3:
                depth_arr = None

        for i, fr in enumerate(frames):
            stem = Path(frame_ids_window[i]).stem if i < len(frame_ids_window) else f"seq_{fr.seq:08d}"
            if fr.seq not in self.saved_seq:
                self.saved_seq.add(fr.seq)
                cv2.imwrite(str(self.rgb_dir / f"{stem}.jpg"), fr.img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                if depth_arr is not None and i < depth_arr.shape[0]:
                    d = depth_arr[i].astype(np.float32)
                    conf = None
                    if depth_conf_arr is not None and depth_conf_arr.ndim >= 3 and i < depth_conf_arr.shape[0]:
                        conf = depth_conf_arr[i].astype(np.float32)
                    np.savez_compressed(self.depth_npz_dir / f"{stem}.npz", depth=d, confidence=conf)
                    cv2.imwrite(str(self.depth_vis_dir / f"{stem}.png"), _depth_to_vis(d))
                self.num_saved += 1

            # Save semantic artifacts independently from RGB/depth dedup so overlap
            # frames can still receive masks if they were produced in a later window.
            if i in sem_by_frame_idx and fr.seq not in self.saved_sem_seq:
                m = sem_by_frame_idx[i]
                cv2.imwrite(str(self.mask_dir / f"{stem}.png"), m.astype(np.uint16))
                color = self.palette[(m.astype(np.int64) % len(self.palette))]
                overlay = cv2.addWeighted(fr.img, 0.5, color, 0.5, 0.0)
                cv2.imwrite(str(self.overlay_dir / f"{stem}.png"), overlay)
                self.saved_sem_seq.add(fr.seq)

    def finalize(self, solver: Solver) -> None:
        if solver.map.get_num_submaps() == 0:
            (self.run_dir / "run_meta.json").write_text(
                json.dumps({"saved_frames": self.num_saved, "run_dir": str(self.run_dir), "note": "No submaps created"}, indent=2)
            )
            print(f"[DEMO] No submaps created; only frame-level artifacts were saved in {self.run_dir}")
            return

        pcd_path = self.run_dir / "fused_pointcloud.pcd"
        framewise_dir = self.run_dir / "framewise_pointclouds"
        solver.map.write_points_to_file(str(pcd_path))
        solver.map.save_framewise_pointclouds(str(framewise_dir))

        meta = {
            "saved_frames": self.num_saved,
            "run_dir": str(self.run_dir),
            "fused_pointcloud": str(pcd_path),
            "framewise_pointclouds_dir": str(framewise_dir),
        }
        (self.run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[DEMO] Saved fused point cloud: {pcd_path}")
        print(f"[DEMO] Saved framewise point clouds/depth: {framewise_dir}")


# ----------------- Utilities -------------------------
def resize_to_approx_2mp(img: np.ndarray, long_side_cap: int = 1920) -> np.ndarray:
    """Resize keeping aspect ratio so that the long side <= long_side_cap (~2MP at 1920x1080)."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= long_side_cap:
        return img
    scale = long_side_cap / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# Global tapper for taps_runtime
tapper = None

# ----------------- ROS 2 Ingest -----------------------
class Ros2Ingest:
    """
    ROS 2 subscriber that feeds a bounded Queue[Frame].
    Also listens to /stream/stop (std_msgs/Empty) and sets a stop_event.
    """
    def __init__(self,
                 topic_image: str,
                 topic_info: str,
                 topic_stop: str,
                 out_queue: Queue,
                 frame_id_start: int = 0,
                 bgr_to_rgb: bool = False,
                 stop_event: Optional[threading.Event] = None):
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS 2 not available. Please install rclpy, sensor_msgs, cv_bridge.")
        self._queue = out_queue
        self._topic_image = topic_image
        self._topic_info = topic_info
        self._topic_stop = topic_stop
        self._seq = frame_id_start
        self._bridge = CvBridge()
        self._bgr_to_rgb = bgr_to_rgb
        self._stop_event = stop_event

        rclpy.init(args=None)
        self.node = Node('vggt_slam_live_ingest')
        self._K: Optional[np.ndarray] = None

        self._sub_info = self.node.create_subscription(CameraInfo, self._topic_info, self._on_info, 10)
        self._sub_img = self.node.create_subscription(Image, self._topic_image, self._on_image, 10)
        self._sub_stop = self.node.create_subscription(StopMsg, self._topic_stop, self._on_stop, 10)
        self._executor_thread = threading.Thread(target=self._spin, daemon=True)

    def start(self):
        self._executor_thread.start()

    def shutdown(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _spin(self):
        rclpy.spin(self.node)

    def _on_stop(self, _: StopMsg):
        if self._stop_event is not None:
            self.node.get_logger().info(f"Received {self._topic_stop}; stopping ingest…")
            self._stop_event.set()

    def _on_info(self, msg: CameraInfo):
        try:
            K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self._K = K
        except Exception:
            pass

    def _on_image(self, msg: Image):
        if self._K is None:
            return  # wait for CameraInfo
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        if self._bgr_to_rgb:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # Resize to ~2 MP
        cv_img = resize_to_approx_2mp(cv_img, long_side_cap=1920)

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        seq = self._seq
        self._seq += 1

        frame = Frame(img=cv_img, ts=ts, seq=seq, K=self._K.copy())
        # Best-effort non-blocking put; short blocking fallback
        try:
            self._queue.put(frame, block=False)
        except Exception:
            try:
                self._queue.put(frame, block=True, timeout=0.01)
            except Exception:
                # Drop as last resort; consumer also decimates on pressure
                pass


# ----------------- Args -------------------------------
parser = argparse.ArgumentParser(description="VGGT-SLAM live/offline")
# Original args
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--vis_stride", type=int, default=1, help="Stride interval in the 3D point cloud image for visualization. Try increasing (such as 4) to reduce lag in visualizing large maps.")
parser.add_argument("--vis_point_size", type=float, default=0.003, help="Visualization point size")
# Live-stream args
parser.add_argument("--live", action="store_true", help="Enable live streaming mode via ROS 2")
parser.add_argument("--ros_image_topic", type=str, default="/camera/image_color", help="ROS 2 image topic")
parser.add_argument("--ros_info_topic", type=str, default="/camera/camera_info", help="ROS 2 CameraInfo topic")
parser.add_argument("--ros_stop_topic", type=str, default="/stream/stop", help="ROS 2 stop topic (std_msgs/Empty)")
parser.add_argument("--ros2_queue_size", type=int, default=60, help="Capacity of the ingest queue (frames)")
parser.add_argument("--target_fps", type=float, default=2.0, help="Target processing frequency (Hz)")
parser.add_argument("--window_size", type=int, default=15, help="Sliding window size for live processing")
parser.add_argument("--decimate_floor", type=int, default=1, help="Minimum decimation stride (>=1)")
parser.add_argument("--max_latency_s", type=float, default=1.0, help="Max acceptable staleness of a frame (s)")
parser.add_argument("--temp_dir", type=str, default="", help="Optional directory to buffer live frames as images (falls back to tmp or /dev/shm)")
parser.add_argument("--local_model", type=str, default=os.path.expanduser("~/models/VGGT-1B/model.pt"),
                    help="Path to local VGGT weights to avoid internet download")
parser.add_argument("--finetune-checkpoint", type=str, default="",
                    help="Optional fine-tuned checkpoint loaded on top of base VGGT weights")
parser.add_argument("--demo-root", type=str, default=os.getenv("VGGT_DEMO_ROOT", ""),
                    help="If set, save demo artifacts to this root folder")
parser.add_argument("--max-live-steps", type=int, default=0,
                    help="Auto-stop live run after N processed submaps (0 disables)")


# ----------------- Temp writer shim -------------------
def write_window_to_temp(paths_dir: str, frames: List[Frame], to_rgb: bool = False) -> List[str]:
    """
    Write frames to disk and return list of file paths in the same order.
    Controlled via env:
      - VGGT_LIVE_TEMP_IMAGE_FORMAT: jpg|jpeg|png (default: jpg)
      - VGGT_LIVE_TEMP_JPEG_QUALITY: 1..100 (default: 90)
      - VGGT_LIVE_TEMP_PNG_COMPRESSION: 0..9 (default: 1)
    """
    os.makedirs(paths_dir, exist_ok=True)
    img_fmt = os.getenv("VGGT_LIVE_TEMP_IMAGE_FORMAT", "jpg").strip().lower()
    if img_fmt not in ("jpg", "jpeg", "png"):
        img_fmt = "jpg"
    jpg_q = int(os.getenv("VGGT_LIVE_TEMP_JPEG_QUALITY", "90"))
    jpg_q = max(1, min(100, jpg_q))
    png_c = int(os.getenv("VGGT_LIVE_TEMP_PNG_COMPRESSION", "1"))
    png_c = max(0, min(9, png_c))

    file_paths = []
    for f in frames:
        img = f.img
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ext = "png" if img_fmt == "png" else "jpg"
        fname = f"seq_{f.seq:08d}_ts_{f.ts:.6f}.{ext}"
        path = os.path.join(paths_dir, fname)
        if img_fmt == "png":
            cv2.imwrite(path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_c])
        else:
            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_q])
        file_paths.append(path)
    return file_paths


# ----------------- Live loop --------------------------
def live_loop(args, solver: Solver, model: VGGT, device: str, trace_sink: Optional[TraceSink] = None):
    if not ROS_AVAILABLE:
        raise RuntimeError("--live was set but ROS 2 is not available. Install rclpy, sensor_msgs, cv_bridge.")

    stop_event = threading.Event()

    # Handle Ctrl-C / SIGTERM too
    def _sig_handler(sig, _):
        print(f"[LIVE] Received signal {sig}; stopping…")
        stop_event.set()
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    frame_queue: Queue = Queue(maxsize=args.ros2_queue_size)
    ros = Ros2Ingest(
        args.ros_image_topic,
        args.ros_info_topic,
        args.ros_stop_topic,
        frame_queue,
        stop_event=stop_event,
    )
    ros.start()
    demo = DemoExporter(args.demo_root) if args.demo_root else None

    base_tmp = args.temp_dir if args.temp_dir else ("/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir())
    tmp_dir = tempfile.mkdtemp(prefix="vggt_live_", dir=base_tmp)

    window = deque(maxlen=args.window_size + args.overlapping_window_size)
    last_proc_wall = time.time() - 10.0
    decimate = max(1, args.decimate_floor)

    print(f"[LIVE] Using temp dir: {tmp_dir}")
    print(f"[LIVE] Window={args.window_size} Overlap={args.overlapping_window_size} TargetFPS={args.target_fps}")
    tmp_fmt = os.getenv("VGGT_LIVE_TEMP_IMAGE_FORMAT", "jpg").strip().lower()
    if tmp_fmt not in ("jpg", "jpeg", "png"):
        tmp_fmt = "jpg"
    if tmp_fmt == "png":
        print(
            "[LIVE] Temp frame encoding: png "
            f"(compression={max(0, min(9, int(os.getenv('VGGT_LIVE_TEMP_PNG_COMPRESSION', '1'))))})"
        )
    else:
        print(
            "[LIVE] Temp frame encoding: jpg "
            f"(quality={max(1, min(100, int(os.getenv('VGGT_LIVE_TEMP_JPEG_QUALITY', '90'))))})"
        )

    try:
        while True:
            if stop_event.is_set():
                break

            try:
                frame: Frame = frame_queue.get(timeout=0.05)
            except QueueEmpty:
                continue

            # Optional latency guard using wall time (assumes clocks are roughly synced)
            if (time.time() - frame.ts) > args.max_latency_s:
                continue

            # Backpressure → adjust decimation based on queue fill level
            qlen = frame_queue.qsize()
            hi = int(0.8 * args.ros2_queue_size)
            lo = int(0.3 * args.ros2_queue_size)
            if qlen > hi:
                decimate = max(decimate + 1, args.decimate_floor)
            elif qlen < lo and decimate > args.decimate_floor:
                decimate -= 1

            # Respect decimation stride by sequence id
            if (frame.seq % decimate) != 0:
                continue

            window.append(frame)

            # Pace by target FPS
            now = time.time()
            if (now - last_proc_wall) < (1.0 / max(1e-6, args.target_fps)):
                continue

            need = args.window_size + args.overlapping_window_size
            if len(window) < need:
                continue

            # Convert window frames to image paths via temp files
            subdir = os.path.join(tmp_dir, f"batch_{window[0].seq:08d}")
            img_paths = write_window_to_temp(subdir, list(window))
            frame_ids_window = [os.path.basename(p) for p in img_paths]
            step_id = int(window[0].seq)

            if tapper is not None:
                try:
                    # Frame IDs from filenames (deterministic and human-readable)
                    tapper.set_batch_meta(
                        step=step_id,            # any monotonically increasing int is fine
                        t=time.time(),
                        window_len=len(img_paths),
                        frame_ids_window=frame_ids_window,  # what we asked the model to process
                        # input_hw is optional; tapper already sniffs it from the patch_embed pre-hook
                    )
                except Exception:
                    pass

            # Run solver with current window; fail-soft on bad geometry windows.
            try:
                predictions = solver.run_predictions(
                    img_paths,
                    model,
                    args.max_loops,
                    trace_sink=trace_sink,
                    step_id=step_id,
                    frame_ids_window=frame_ids_window,
                )
                if demo is not None:
                    demo.export_batch(list(window), frame_ids_window, predictions)

                solver.add_points(predictions)
                solver.graph.optimize()
                solver.map.update_submap_homographies(solver.graph)
            except Exception as exc:
                print(f"[LIVE][WARN] Window processing failed at step={step_id}: {exc}")
                if str(os.getenv("VGGT_LIVE_TRACEBACK", "0")).strip().lower() in ("1", "true", "yes", "on"):
                    traceback.print_exc()
                # Best-effort cleanup so failed windows do not leave large tensors
                # pinned in hook caches across retries.
                if tapper is not None:
                    try:
                        if hasattr(tapper, "_cache") and isinstance(tapper._cache, dict):
                            tapper._cache.clear()
                        if hasattr(tapper, "_meta") and isinstance(tapper._meta, dict):
                            tapper._meta.clear()
                    except Exception:
                        pass
                msg = str(exc).lower()
                if "cuda" in msg and "out of memory" in msg:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            try:
                                torch.cuda.ipc_collect()
                            except Exception:
                                pass
                            print("[LIVE][WARN] CUDA OOM recovery: cleared CUDA cache.")
                    except Exception:
                        pass
                # Advance window to avoid retrying the exact same failing batch forever.
                while len(window) > args.overlapping_window_size:
                    window.popleft()
                last_proc_wall = now
                if stop_event.is_set():
                    break
                continue

            loop_closure_detected = len(predictions.get("detected_loops", [])) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            # Keep overlap frames only
            while len(window) > args.overlapping_window_size:
                window.popleft()

            last_proc_wall = now
            if args.max_live_steps > 0 and solver.map.get_num_submaps() >= args.max_live_steps:
                print(f"[LIVE] Reached max_live_steps={args.max_live_steps}; stopping.")
                stop_event.set()

            # Log compact JSON record once per live step
            if tapper is not None:
                try:
                    tapper.maybe_log()
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("[LIVE] Interrupted by user. Shutting down…")
    finally:
        ros.shutdown()
        if not args.vis_map:
            solver.update_all_submap_vis()

        if args.log_results:
            solver.map.write_poses_to_file(args.log_path)
            # Log the full point cloud as one file, used for visualization.
            solver.map.write_points_to_file(args.log_path.replace(".txt", "_points.pcd"))
            print("Saved points to file", args.log_path.replace(".txt", "_points.pcd"))
            if not args.skip_dense_log:
                solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))
            
            from move_results import archive_results
            try:
                archive_results()
            except BaseException as exc:
                print(f"[WARN] archive_results failed: {exc}")

        if args.plot_focal_lengths:
            colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
            plt.figure(figsize=(8, 6))
            for i, values in enumerate(data):
                y = values
                x = [i] * len(values)
                plt.scatter(x, y, color=colors[i], label=f'List {i+1}')
            plt.xlabel("poses"); plt.ylabel("Focal lengths"); plt.grid(); plt.show()
        if demo is not None:
            demo.finalize(solver)
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass



# ----------------- Offline loop (unchanged) -----------
def offline_loop(args, solver: Solver, model: VGGT, device: str, trace_sink: Optional[TraceSink] = None):
    use_optical_flow_downsample = True

    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*"))
                   if "depth" not in os.path.basename(f).lower()
                   and "txt" not in os.path.basename(f).lower()
                   and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    data = []
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            print(image_names_subset)
            frame_ids_window = [os.path.basename(p) for p in image_names_subset]
            step_id = len(data)

            if tapper is not None:
                try:
                    tapper.set_batch_meta(
                        step=step_id,                   # or any step counter you prefer
                        t=time.time(),
                        window_len=len(image_names_subset),
                        frame_ids_window=frame_ids_window,
                    )
                except Exception:
                    pass

            predictions = solver.run_predictions(
                image_names_subset,
                model,
                args.max_loops,
                trace_sink=trace_sink,
                step_id=step_id,
                frame_ids_window=frame_ids_window,
            )

            if tapper is not None:
                try:
                    tapper.maybe_log()
                except Exception:
                    pass

            data.append(predictions["intrinsic"][:, 0, 0])

            solver.add_points(predictions)
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if not args.vis_map:
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path)
        # Log the full point cloud as one file, used for visualization.
        solver.map.write_points_to_file(args.log_path.replace(".txt", "_points.pcd"))
        print("Saved points to file", args.log_path.replace(".txt", "_points.pcd"))
        if not args.skip_dense_log:
            solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))
        from move_results import archive_results
        try:
            archive_results()
        except BaseException as exc:
            print(f"[WARN] archive_results failed: {exc}")

    if args.plot_focal_lengths:
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        plt.figure(figsize=(8, 6))
        for i, values in enumerate(data):
            y = values
            x = [i] * len(values)
            plt.scatter(x, y, color=colors[i], label=f'List {i+1}')
        plt.xlabel("poses"); plt.ylabel("Focal lengths"); plt.grid(); plt.show()


# ----------------- Main -------------------------------
def main():
    args = parser.parse_args()

    # Rotate tap log (keep last 3 runs) before starting a new session
    log_path = Path("tap_logs") / "taps.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _rotate_log(path: Path, keep: int = 3) -> None:
        if not path.exists():
            return
        for idx in range(keep, 0, -1):
            src = path.with_name(f"{path.name}.{idx}")
            dst = path.with_name(f"{path.name}.{idx + 1}")
            if src.exists():
                if idx == keep:
                    src.unlink()
                else:
                    src.replace(dst)
        path.replace(path.with_name(f"{path.name}.1"))

    _rotate_log(log_path, keep=3)
    log_path.touch()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False,
        vis_stride=args.vis_stride,
        vis_point_size=args.vis_point_size,
    )

    print("Initializing and loading VGGT model...")
    model = VGGT()
    try:
        import inspect
        model_file = inspect.getsourcefile(model.__class__) or "<unknown>"
    except Exception:
        model_file = "<unknown>"
    print(f"[VGGT] implementation: {model.__class__.__module__}.{model.__class__.__name__} ({model_file})")

    # Prefer local weights if available (GPU nodes without internet)
    state = None
    if args.local_model and os.path.exists(args.local_model):
        print(f"Loading VGGT weights from local path: {args.local_model}")
        state = torch.load(args.local_model, map_location="cpu")
    else:
        print("Local model not found; attempting to download (requires internet)…")
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state = torch.hub.load_state_dict_from_url(_URL)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[VGGT] Loaded with non-strict. Missing keys (new params): {len(missing)}")
        for key in missing[:10]:
            print("  missing:", key)
        if len(missing) > 10:
            print("  …")

    if unexpected:
        print(f"[VGGT] Unexpected keys in checkpoint (ignored): {len(unexpected)}")
        for key in unexpected[:10]:
            print("  unexpected:", key)
        if len(unexpected) > 10:
            print("  …")

    if args.finetune_checkpoint:
        if not os.path.exists(args.finetune_checkpoint):
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {args.finetune_checkpoint}")
        print(f"[FT] Loading fine-tuned checkpoint: {args.finetune_checkpoint}")
        ft = torch.load(args.finetune_checkpoint, map_location="cpu")
        if isinstance(ft, dict) and "state_dict" in ft and isinstance(ft["state_dict"], dict):
            ft_state = ft["state_dict"]
        elif isinstance(ft, dict) and "model_state" in ft and isinstance(ft["model_state"], dict):
            ft_state = ft["model_state"]
        elif isinstance(ft, dict) and "model" in ft and isinstance(ft["model"], dict):
            ft_state = ft["model"]
        elif isinstance(ft, dict):
            ft_state = ft
        else:
            raise RuntimeError("Unsupported fine-tuned checkpoint format.")

        remap = {}
        for k, v in ft_state.items():
            nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
            remap[nk] = v

        ft_missing, ft_unexpected = model.load_state_dict(remap, strict=False)
        loaded_count = max(0, len(remap) - len(ft_unexpected))
        print(
            f"[FT] load strict=False: loaded={loaded_count} missing={len(ft_missing)} unexpected={len(ft_unexpected)}"
        )
        if loaded_count == 0:
            sample = list(remap.keys())[:10]
            raise RuntimeError(
                "Fine-tuned checkpoint appears incompatible with VGGT model (0 parameters loaded). "
                f"Sample keys: {sample}. "
                "This usually means you passed a Fusion+Mask2Former checkpoint "
                "(e.g., from train_film_m2f_optimized_png.py) to main_live_stream.py."
            )

    # Keep depth-head side pyramids aligned with the full live window so semantic
    # frame selection (e.g., VGGT_SEM_FRAME_MODE=all) sees all frames.
    try:
        dh = getattr(model, "depth_head", None)
        if dh is not None and not getattr(dh, "_no_chunk_patch", False):
            _orig_depth_forward = dh.forward

            def _depth_forward_no_chunk(self, aggregated_tokens_list, images, patch_start_idx, *args, **kwargs):
                kwargs["frames_chunk_size"] = None
                return _orig_depth_forward(aggregated_tokens_list, images, patch_start_idx, *args, **kwargs)

            dh.forward = _depth_forward_no_chunk.__get__(dh, type(dh))
            setattr(dh, "_no_chunk_patch", True)
            print("[VGGT] depth_head chunking: disabled (frames_chunk_size=None)")
    except Exception as exc:
        print(f"[VGGT][WARN] failed to disable depth_head chunking: {exc}")

    try:
        dh = getattr(model, "depth_head", None)
        if dh is not None and getattr(dh, "film_enabled", False):
            gates = (
                torch.sigmoid(dh.film_gates).detach().cpu().tolist()
                if hasattr(dh, "film_gates")
                else None
            )
            print("[VGGT] FiLM enabled?", True, "gates:", gates)
        else:
            print("[VGGT] FiLM enabled?", False)
    except Exception as exc:
        print(f"[VGGT] FiLM inspection failed: {exc}")

    model.eval()

    model = model.to(device)
    
    # Optional taps/tracing (disabled for plain VGGT runs).
    enable_taps = str(os.getenv("VGGT_ENABLE_TAPS", "1")).strip().lower() not in ("0", "false", "no", "off")
    global tapper
    trace = None
    if enable_taps:
        tapper = attach_vggt_taps(model, logdir="tap_logs", capture_every=1)
        setattr(model, "_feature_tapper", tapper)
        install_trace_probes(model)
        trace = TraceSink("tap_logs/trace.jsonl")
    else:
        tapper = None
        print("[VGGT] Taps/tracing disabled (VGGT_ENABLE_TAPS=0)")

    if args.live:
        if args.overlapping_window_size != 1:
            print("[WARN] overlapping_window_size other than 1 is not supported; forcing to 1 for live mode.")
            args.overlapping_window_size = 1
        live_loop(args, solver, model, device, trace_sink=trace)
    else:
        offline_loop(args, solver, model, device, trace_sink=trace)


if __name__ == "__main__":
    main()
