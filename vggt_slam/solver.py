import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import viser
import viser.transforms as viser_tf
import os
import traceback
from termcolor import colored
from typing import Optional, Sequence

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer
from pipeline_check import get_pipeline_logger

_SEM_RUNNER_LOGGED = False
_SEM_ERROR_COUNT = 0


def _gtsam_sym(name):
    if hasattr(gtsam, name):
        return getattr(gtsam, name)
    core = getattr(gtsam, "gtsam", None)
    if core is not None and hasattr(core, name):
        return getattr(core, name)
    try:
        import gtsam.gtsam as gtsam_core  # type: ignore
        if hasattr(gtsam_core, name):
            return getattr(gtsam_core, name)
    except Exception:
        pass
    raise ImportError(f"gtsam symbol not found: {name}")

def color_point_cloud_by_confidence(pcd, confidence, cmap='viridis'):
    """
    Color a point cloud based on per-point confidence values.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud.
        confidence (np.ndarray): Confidence values, shape (N,).
        cmap (str): Matplotlib colormap name.
    """
    assert len(confidence) == len(pcd.points), "Confidence length must match number of points"

    # Normalize confidence to [0, 1]
    confidence_normalized = (confidence - np.min(confidence)) / (np.ptp(confidence) + 1e-8)
    
    # Map to colors using matplotlib colormap
    colormap = plt.get_cmap(cmap)
    colors = colormap(confidence_normalized)[:, :3]  # Drop alpha channel

    # Assign to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # Global toggle for all frames and frustums
        self.gui_show_frames = self.server.gui.add_checkbox(
            "Show Cameras",
            initial_value=True,
        )
        self.gui_show_frames.on_update(self._on_update_show_frames)

        # Store frames and frustums by submap
        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

        num_rand_colors = 250
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int, image_scale: float=0.5) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """

        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            self.submap_frames[submap_id] = []
            self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum
            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Downsample for visualization with `image_scale`
            img_resized = cv2.resize(
                img,
                (int(img.shape[1] * image_scale), int(img.shape[0] * image_scale)),
                interpolation=cv2.INTER_AREA
            )

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img_resized,
                line_width=3.0,
                color=self.random_colors[submap_id]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps."""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible



class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        use_point_map: bool = False,
        visualize_global_map: bool = False,
        use_sim3: bool = False,
        gradio_mode: bool = False,
        vis_stride: int = 1,         # represents how much the visualized point clouds are sparsified
        vis_point_size: float = 0.001):
        
        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.gradio_mode = gradio_mode

        if self.gradio_mode:
            self.viewer = TrimeshViewer()
        else:
            self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.use_sim3 = use_sim3
        if self.use_sim3:
            from vggt_slam.graph_se3 import PoseGraph
        else:
            from vggt_slam.graph import PoseGraph
        self.graph = PoseGraph()

        self.image_retrieval = None
        self._loop_retrieval_disabled = str(os.getenv("VGGT_DISABLE_LOOP_RETRIEVAL", "0")).strip().lower() in ("1", "true", "yes", "on")
        self.current_working_submap = None

        self.first_edge = True

        self.T_w_kf_minus = None

        self.prior_pcd = None
        self.prior_conf = None

        self.vis_stride = vis_stride
        self.vis_point_size = vis_point_size

        print("Starting viser server...")

    def _get_image_retrieval(self):
        if self._loop_retrieval_disabled:
            return None
        if self.image_retrieval is None:
            print("[LoopRetrieval] Initializing SALAD retrieval model...")
            self.image_retrieval = ImageRetrieval()
            print("[LoopRetrieval] Ready.")
        return self.image_retrieval

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.gradio_mode:
            self.viewer.add_point_cloud(points_in_world_frame, points_colors)
        else:
            self.viewer.server.scene.add_point_cloud(
                name="pcd_"+name,
                points=points_in_world_frame,
                colors=points_colors,
                point_size=point_size,
                point_shape="circle",
            )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        # NOTE(hlim): `stride` is used only to reduce the visualization cost in viser,
        # and does not affect the underlying point cloud data.
        points_in_world_frame = submap.get_points_in_world_frame(stride = self.vis_stride)
        points_colors = submap.get_points_colors(stride = self.vis_stride)
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, self.vis_point_size)

    def set_submap_poses(self, submap):
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world()
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_all_frames()
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def export_3d_scene(self, output_path="output.glb"):
        return self.viewer.export(output_path)

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        images = pred_dict["images"]  # (S, 3, H, W)

        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
        # print(intrinsics_cam)

        detected_loops = pred_dict["detected_loops"]

        if self.use_point_map:
            world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]  # (S, H, W)
            world_points = world_points_map
        else:
            depth_map = pred_dict["depth"]  # (S, H, W, 1)
            conf = pred_dict["depth_conf"]  # (S, H, W)
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        # Convert images from (S, 3, H, W) to (S, H, W, 3)
        # Then flatten everything for the point cloud
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)

        # Flatten
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)

        # estimate focal length from points
        points_in_first_cam = world_points[0,...]
        h, w = points_in_first_cam.shape[0:2]

        new_pcd_num = self.current_working_submap.get_id()
        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[-1,...].reshape(-1, 3)
            self.prior_conf = conf[-1,...].reshape(-1)

            # Add node to graph.
            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
        else:
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)

            current_pts = world_points[0,...].reshape(-1, 3)
        
            # TODO conf should be using the threshold in its own submap
            good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[0,...,:].reshape(-1) > prior_submap.get_conf_threshold())
            
            if self.use_sim3:
                # Note we still use H and not T in variable names so we can share code with the Sim3 case, 
                # and SIM3 and SE3 are also subsets of the SL4 group
                R_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,0:3]
                t_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,3]
                T_temp = np.eye(4)
                T_temp[0:3,0:3] = R_temp
                T_temp[0:3,3] = t_temp
                T_temp = np.linalg.inv(T_temp)
                scale_factor = np.mean(np.linalg.norm((T_temp[0:3,0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3,3], axis=1) / np.linalg.norm(current_pts[good_mask], axis=1))
                print(colored("scale factor", 'green'), scale_factor)
                H_relative = np.eye(4)
                H_relative[0:3,0:3] = R_temp
                H_relative[0:3,3] = t_temp

                # apply scale factor to points and poses
                world_points *= scale_factor
                cam_to_world[:, 0:3, 3] *= scale_factor
            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])
            
            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            # Visualize the point clouds
            # pcd1 = o3d.geometry.PointCloud()
            # pcd1.points = o3d.utility.Vector3dVector(self.prior_pcd)
            # pcd1 = color_point_cloud_by_confidence(pcd1, self.prior_conf)
            # pcd2 = o3d.geometry.PointCloud()
            # current_pts = world_points[0,...].reshape(-1, 3)
            # points = apply_homography(H_relative, current_pts)
            # pcd2.points = o3d.utility.Vector3dVector(points)
            # # pcd2 = color_point_cloud_by_confidence(pcd2, conf_flat, cmap='jet')
            # o3d.visualization.draw_geometries([pcd1, pcd2])

            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame,...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame,...].reshape(-1)

            # Add node to graph.
            self.graph.add_homography(new_pcd_num, H_w_submap)

            # Add between factor.
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)
            # print("added between factor", prior_pcd_num, new_pcd_num, H_relative)
            print(f"[VGGT-SLAM] Added odometry factor {prior_pcd_num}→{new_pcd_num}")

        # Create and add submap.
        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        self.current_working_submap.set_conf_masks(conf) # TODO should make this work for point cloud conf as well
        depth_maps = pred_dict.get("depth")
        depth_confidence = pred_dict.get("depth_conf")
        if depth_maps is not None and depth_confidence is not None:
            self.current_working_submap.add_all_depths(depth_maps, depth_confidence)

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1

            if self.use_sim3:
                pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                Pose3 = _gtsam_sym("Pose3")
                pose_world_detected = Pose3(pose_world_detected)
                pose_world_query = Pose3(pose_world_query)
                H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
            else:
                points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                H_relative_lc = ransac_projective(points_world_query, points_world_detected)


            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure() # Just for debugging and analysis, keep track of total number of loop closures
            # print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id, H_relative_lc)
            # print("homography between nodes estimated to be", np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap)
            print(f"[VGGT-SLAM] Added loop closure {loop.detected_submap_id}↔{loop.query_submap_id}")

            # print("relative_pose factor added", relative_pose)

            # Visualize query and detected frames
            # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            # axes[0].imshow(self.map.get_submap(loop.detected_submap_id).get_frame_at_index(loop.detected_submap_frame).cpu().numpy().transpose(1,2,0))
            # axes[0].set_title("Detect")
            # axes[0].axis("off")  # Hide axis
            # axes[1].imshow(self.current_working_submap.get_frame_at_index(loop.query_submap_frame).cpu().numpy().transpose(1,2,0))
            # axes[1].set_title("Query")
            # axes[1].axis("off")
            # plt.show()

            # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            # axes[0].imshow(self.map.get_submap(loop.detected_submap_id).get_frame_at_index(0).cpu().numpy().transpose(1,2,0))
            # axes[0].set_title("Detect")
            # axes[0].axis("off")  # Hide axis
            # axes[1].imshow(self.current_working_submap.get_frame_at_index(0).cpu().numpy().transpose(1,2,0))
            # axes[1].set_title("Query")
            # axes[1].axis("off")
            # plt.show()


        self.map.add_submap(self.current_working_submap)


    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(
        self,
        image_names,
        model,
        max_loops,
        trace_sink=None,
        step_id: Optional[int] = None,
        frame_ids_window: Optional[Sequence[str]] = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = load_and_preprocess_images(image_names).to(device)
        # print(f"Preprocessed images shape: {images.shape}")
        print(f"[VGGT-SLAM] Batch ready: tensor shape {tuple(images.shape)}")
        pipeline_logger = get_pipeline_logger()
        if pipeline_logger:
            pipeline_logger.log(
                "VGGT",
                "Window tensor prepared",
                expected="[B,S,3,H,W]",
                observed=str(tuple(images.shape)),
                status="ok",
            )

        # print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        window_tensor = images

        # Check for loop closures
        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        # new_submap.add_all_frames(images)
        new_submap.add_all_frames(images)
        if frame_ids_window is not None:
            new_submap.set_frame_ids(list(frame_ids_window))
        else:
            new_submap.set_frame_ids(image_names)
        detected_loops = []
        retrieved_frames = []
        if max_loops > 0:
            retriever = self._get_image_retrieval()
            if retriever is not None:
                new_submap.set_all_retrieval_vectors(retriever.get_all_submap_embeddings(new_submap))
                detected_loops = retriever.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops)
                if len(detected_loops) > 0:
                    print(colored("detected_loops", "yellow"), detected_loops)
                retrieved_frames = self.map.get_frames_from_loops(detected_loops)

        num_loop_frames = len(retrieved_frames)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        if num_loop_frames > 0:
            image_tensor = torch.stack(retrieved_frames)  # Shape (n, 3, w, h)
            images = torch.cat([images, image_tensor], dim=0) # Shape (s+n, 3, w, h)

            # TODO we don't really need to store the loop closure frame again, but this makes lookup easier for the visualizer.
            # We added the frame to the submap once before to get the retrieval vectors,
            new_submap.add_all_frames(images)

        self.current_working_submap = new_submap

        prev_sink = getattr(model, "_trace_sink", None)
        prev_step = getattr(model, "_trace_step", None)
        prev_frame_ids = getattr(model, "_trace_frame_ids", None)
        prev_window_tensor = getattr(model, "_trace_window_tensor", None)
        prev_window_length = getattr(model, "_trace_window_length", None)
        prev_trace_state = getattr(model, "_trace_state", None)

        if trace_sink is not None:
            if not getattr(model, "_trace_wrapped", False):
                try:
                    from trace_hooks import install_trace_probes
                    install_trace_probes(model)
                    model._trace_wrapped = True
                except Exception:
                    pass
            model._trace_sink = trace_sink
            model._trace_step = step_id
            model._trace_frame_ids = frame_ids_window
            model._trace_window_tensor = window_tensor
            model._trace_window_length = len(frame_ids_window) if frame_ids_window is not None else images.shape[0]
            model._trace_state = {}

        try:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    predictions = model(images)
        finally:
            if trace_sink is not None:
                model._trace_sink = prev_sink
                model._trace_step = prev_step
                model._trace_frame_ids = prev_frame_ids
                model._trace_window_tensor = prev_window_tensor
                model._trace_window_length = prev_window_length
                model._trace_state = prev_trace_state

        # Optional semantic head sidecar (no effect if module/env absent)
        global _SEM_RUNNER_LOGGED, _SEM_ERROR_COUNT
        try:
            runner_source = "hiding_folder.semantic_runner"
            try:
                from hiding_folder.semantic_runner import run_semantic_if_enabled
            except Exception:
                runner_source = "vggt.heads.semantic_runner"
                from vggt.heads.semantic_runner import run_semantic_if_enabled
            if not _SEM_RUNNER_LOGGED:
                print(
                    f"[SEM runner] source={runner_source} "
                    f"backend={os.getenv('VGGT_SEM_BACKEND', '<unset>')} "
                    f"dpt_source={os.getenv('VGGT_SEM_DPT_SOURCE', 'raw')}"
                )
                _SEM_RUNNER_LOGGED = True
            device = next(model.parameters()).device
            predictions["_sem_step_id"] = int(step_id)
            predictions["_sem_frame_ids"] = list(frame_ids_window) if frame_ids_window is not None else []
            run_semantic_if_enabled(model, predictions, device)
            sem_masks = predictions.get("sem_mask_logits")
            sem_cls = predictions.get("sem_cls_logits")
            if sem_masks is not None and sem_cls is not None:
                print("[SEM] masks:", tuple(sem_masks.shape), "cls:", tuple(sem_cls.shape))
                # Quick numeric probe so we know tensors are non-trivial.
                try:
                    mask_mean = float(sem_masks.mean().item())
                    mask_std = float(sem_masks.std().item())
                    cls_scores = sem_cls.softmax(dim=-1)
                    top_scores = cls_scores.max(dim=-1).values
                    print(
                        "[SEM stats] mask mean/std:",
                        f"{mask_mean:.4f}",
                        f"{mask_std:.4f}",
                        "cls max avg:",
                        f"{float(top_scores.mean().item()):.4f}",
                        "cls max min/max:",
                        f"{float(top_scores.min().item()):.4f}",
                        f"{float(top_scores.max().item()):.4f}",
                    )
                    if pipeline_logger:
                        pipeline_logger.log(
                            "SEMHEAD",
                            "Semantic logits emitted",
                            expected="mask (B,S,Q,H,W), cls (B,S,Q,C)",
                            observed=(
                                f"mask {tuple(sem_masks.shape)}, cls {tuple(sem_cls.shape)}, "
                                f"mask mean {mask_mean:.4f} std {mask_std:.4f}"
                            ),
                            status="ok",
                        )
                except Exception:
                    print("[SEM stats] probe failed")
                    if pipeline_logger:
                        pipeline_logger.log("SEMHEAD", "Stat probe failed", status="warn")
            else:
                if pipeline_logger:
                    pipeline_logger.log(
                        "SEMHEAD",
                        "Semantic logits missing",
                        expected="Outputs when VGGT_SEMANTIC_HEAD=1",
                        observed="None",
                        status="warn",
                    )
        except Exception as exc:
            _SEM_ERROR_COUNT += 1
            if _SEM_ERROR_COUNT <= 5 or (_SEM_ERROR_COUNT % 50) == 0:
                print(f"[SEM runner][WARN] failed ({_SEM_ERROR_COUNT}): {exc}")
                if str(os.getenv("VGGT_SEM_TRACEBACK", "0")).strip().lower() in ("1", "true", "yes", "on"):
                    traceback.print_exc()

        # TODO: remove FiLM delta probe after validation.
        dh = getattr(model, "depth_head", None)
        raw_pyr = getattr(dh, "raw_pyramid", None) if dh is not None else None
        film_pyr = getattr(dh, "film_side_pyramid", None) if dh is not None else None
        if pipeline_logger and dh is not None:
            film_enabled = getattr(dh, "film_enabled", False)
            gates = getattr(dh, "film_gates", None)
            observed_gates = None
            if gates is not None:
                observed_gates = [float(x) for x in torch.sigmoid(gates.detach()).cpu().tolist()]  # type: ignore[arg-type]
            pipeline_logger.log(
                "FiLM",
                "FiLM configuration",
                expected="Passive FiLM with gates",
                observed=str({"enabled": film_enabled, "gates": observed_gates}),
                status="ok" if film_enabled else "warn",
            )
        if raw_pyr and film_pyr:
            try:
                deltas = []
                for raw_lvl, film_lvl in zip(raw_pyr, film_pyr):
                    if raw_lvl is None or film_lvl is None:
                        deltas.append(("none", "none"))
                        continue
                    rr = raw_lvl.reshape(-1, *raw_lvl.shape[-3:]).detach()
                    ff = film_lvl.reshape(-1, *film_lvl.shape[-3:]).detach()
                    mad = (ff - rr).abs().mean().item()
                    rr_flat = rr.flatten(1)
                    ff_flat = ff.flatten(1)
                    num = (ff_flat * rr_flat).sum(1)
                    den = ff_flat.norm(dim=1) * rr_flat.norm(dim=1) + 1e-6
                    cos = (num / den).mean().item()
                    deltas.append((mad, cos))
                print("[FiLM Δ] per-level MAD/COS:", deltas)
                if pipeline_logger:
                    pipeline_logger.log(
                        "FiLM",
                        "FiLM side pyramid available",
                        expected="Non-zero deltas",
                        observed=str(deltas),
                        status="ok",
                    )
            except Exception as exc:
                print("[FiLM Δ] probe failed:", exc)
                if pipeline_logger:
                    pipeline_logger.log("FiLM", "Probe failed", observed=str(exc), status="warn")
        else:
            print("[FiLM Δ] pyramid unavailable (raw or FiLM missing)")
            if pipeline_logger:
                pipeline_logger.log(
                    "FiLM",
                    "FiLM side pyramid unavailable",
                    expected="film_side_pyramid populated",
                    observed="missing",
                    status="warn",
                )

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        predictions["detected_loops"] = detected_loops

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

        return predictions
