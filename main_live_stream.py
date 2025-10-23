#!/usr/bin/env python3
from __future__ import annotations
import os
import glob
import argparse
import time
import signal
import threading
import tempfile
import shutil
from collections import deque
from queue import Queue, Empty as QueueEmpty
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT
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
                 out_queue: Queue,
                 frame_id_start: int = 0,
                 bgr_to_rgb: bool = False,
                 stop_event: Optional[threading.Event] = None):
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS 2 not available. Please install rclpy, sensor_msgs, cv_bridge.")
        self._queue = out_queue
        self._topic_image = topic_image
        self._topic_info = topic_info
        self._seq = frame_id_start
        self._bridge = CvBridge()
        self._bgr_to_rgb = bgr_to_rgb
        self._stop_event = stop_event

        rclpy.init(args=None)
        self.node = Node('vggt_slam_live_ingest')
        self._K: Optional[np.ndarray] = None

        self._sub_info = self.node.create_subscription(CameraInfo, self._topic_info, self._on_info, 10)
        self._sub_img = self.node.create_subscription(Image, self._topic_image, self._on_image, 10)
        self._sub_stop = self.node.create_subscription(StopMsg, '/stream/stop', self._on_stop, 10)
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
            self.node.get_logger().info("Received /stream/stop; stopping ingest…")
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
parser.add_argument("--ros2_queue_size", type=int, default=60, help="Capacity of the ingest queue (frames)")
parser.add_argument("--target_fps", type=float, default=2.0, help="Target processing frequency (Hz)")
parser.add_argument("--window_size", type=int, default=15, help="Sliding window size for live processing")
parser.add_argument("--decimate_floor", type=int, default=1, help="Minimum decimation stride (>=1)")
parser.add_argument("--max_latency_s", type=float, default=1.0, help="Max acceptable staleness of a frame (s)")
parser.add_argument("--temp_dir", type=str, default="", help="Optional directory to buffer live frames as images (falls back to tmp or /dev/shm)")
parser.add_argument("--local_model", type=str, default=os.path.expanduser("~/models/VGGT-1B/model.pt"),
                    help="Path to local VGGT weights to avoid internet download")


# ----------------- Temp writer shim -------------------
def write_window_to_temp(paths_dir: str, frames: List[Frame], to_rgb: bool = False) -> List[str]:
    """Write frames to disk and return list of file paths in the same order (JPEG q=90)."""
    os.makedirs(paths_dir, exist_ok=True)
    file_paths = []
    for f in frames:
        img = f.img
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fname = f"seq_{f.seq:08d}_ts_{f.ts:.6f}.jpg"
        path = os.path.join(paths_dir, fname)
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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
    ros = Ros2Ingest(args.ros_image_topic, args.ros_info_topic, frame_queue, stop_event=stop_event)
    ros.start()

    base_tmp = args.temp_dir if args.temp_dir else ("/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir())
    tmp_dir = tempfile.mkdtemp(prefix="vggt_live_", dir=base_tmp)

    window = deque(maxlen=args.window_size + args.overlapping_window_size)
    last_proc_wall = time.time() - 10.0
    decimate = max(1, args.decimate_floor)

    print(f"[LIVE] Using temp dir: {tmp_dir}")
    print(f"[LIVE] Window={args.window_size} Overlap={args.overlapping_window_size} TargetFPS={args.target_fps}")

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

            # Run solver with current window
            predictions = solver.run_predictions(
                img_paths,
                model,
                args.max_loops,
                trace_sink=trace_sink,
                step_id=step_id,
                frame_ids_window=frame_ids_window,
            )

            solver.add_points(predictions)
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

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
            archive_results()

        if args.plot_focal_lengths:
            colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
            plt.figure(figsize=(8, 6))
            for i, values in enumerate(data):
                y = values
                x = [i] * len(values)
                plt.scatter(x, y, color=colors[i], label=f'List {i+1}')
            plt.xlabel("poses"); plt.ylabel("Focal lengths"); plt.grid(); plt.show()
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
        archive_results()

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
    
    # Attach taps right after model instantiation/eval
    global tapper
    tapper = attach_vggt_taps(model, logdir="tap_logs", capture_every=1)
    install_trace_probes(model)

    trace = TraceSink("tap_logs/trace.jsonl")

    if args.live:
        if args.overlapping_window_size != 1:
            print("[WARN] overlapping_window_size other than 1 is not supported; forcing to 1 for live mode.")
            args.overlapping_window_size = 1
        live_loop(args, solver, model, device, trace_sink=trace)
    else:
        offline_loop(args, solver, model, device, trace_sink=trace)


if __name__ == "__main__":
    main()
