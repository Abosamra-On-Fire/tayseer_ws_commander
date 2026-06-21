import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import sensor_msgs_py.point_cloud2 as pc2


# ═══════════════════════════════════════════════════════════════════════════════
# GRASP CANDIDATE DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraspCandidate:
    """A single grasp pose candidate with quality metrics."""
    score: float              # 0.0 - 1.0, higher is better
    position: np.ndarray      # [x, y, z] in camera frame
    orientation: np.ndarray   # [qx, qy, qz, qw] in camera frame
    approach: np.ndarray      # unit vector: direction of gripper approach
    gripper_y: np.ndarray     # unit vector: jaw opening direction
    gripper_width: float      # required opening width (metres)
    grasp_type: str           # "top_down", "side", "angled"
    pre_grasp_offset: float   # metres to back off for pre-grasp


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC GRASP DETECTOR (Option 2: Segmented Point Cloud + Geometric Planning)
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricGraspDetector:
    def __init__(
        self,
        gripper_max_width: float = 0.065,
        gripper_finger_depth: float = 0.04,
        gripper_finger_length: float = 0.035,
        voxel_size: float = 0.003,
        min_points: int = 80,
        plane_dist_thresh: float = 0.015,
        cluster_eps: float = 0.015,
        cluster_min_points: int = 50,
    ):
        self._table_normal: Optional[np.ndarray] = None
        self.gripper_max_width = gripper_max_width
        self.gripper_finger_depth = gripper_finger_depth
        self.gripper_finger_length = gripper_finger_length
        self.voxel_size = voxel_size
        self.min_points = min_points
        self.plane_dist_thresh = plane_dist_thresh
        self.cluster_eps = cluster_eps
        self.cluster_min_points = cluster_min_points

    def detect(
        self,
        depth_img: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        fx: float, fy: float, cx: float, cy: float,
        max_depth: float,
    ) -> Optional[GraspCandidate]:
        self._table_normal = None   # ← reset stale state from previous frame
        pcd = self._bbox_to_pointcloud(depth_img, x1, y1, x2, y2,
                                       fx, fy, cx, cy, max_depth)
        if pcd is None or len(pcd.points) < self.min_points:
            return None

        pcd = pcd.voxel_down_sample(self.voxel_size)
        if len(pcd.points) < self.min_points:
            return None

        pcd_object = self._remove_table_plane(pcd)
        if pcd_object is None or len(pcd_object.points) < self.min_points:
            return None

        clusters = self._extract_clusters(pcd_object)
        if not clusters:
            return None

        object_pcd = max(clusters, key=lambda c: len(c.points))
        if len(object_pcd.points) < self.min_points:
            return None

        object_pcd, _ = object_pcd.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=2.0
        )
        if len(object_pcd.points) < self.min_points:
            return None

        bbox = object_pcd.get_oriented_bounding_box()
        center = bbox.center
        extent = bbox.extent
        bbox_R = bbox.R

        object_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02, max_nn=30
            )
        )
        normals = np.asarray(object_pcd.normals)
        avg_normal = np.mean(normals, axis=0)
        avg_normal /= (np.linalg.norm(avg_normal) + 1e-8)
        if avg_normal[2] < 0:
            avg_normal = -avg_normal

        candidates = self._generate_grasp_candidates(
            object_pcd, center, extent, bbox_R, avg_normal
        )
        if not candidates:
            return None

        best = max(candidates, key=lambda c: c.score)
        return best

    def _bbox_to_pointcloud(
        self,
        depth_img: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        fx: float, fy: float, cx: float, cy: float,
        max_depth: float,
    ) -> Optional[o3d.geometry.PointCloud]:
        h, w = depth_img.shape
        u1 = max(0, int(x1))
        v1 = max(0, int(y1))
        u2 = min(w, int(x2))
        v2 = min(h, int(y2))
        if u2 <= u1 or v2 <= v1:
            return None

        roi = depth_img[v1:v2, u1:u2]
        u_grid, v_grid = np.meshgrid(np.arange(u1, u2), np.arange(v1, v2))
        valid = (roi > 0.01) & (roi <= max_depth)
        z = roi[valid]
        if len(z) == 0:
            return None

        u = u_grid[valid]
        v = v_grid[valid]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts = np.stack([x, y, z], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def _remove_table_plane(self, pcd):
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.plane_dist_thresh,
            ransac_n=3,
            num_iterations=1000,
        )
        n = np.array(plane_model[:3])
        n /= (np.linalg.norm(n) + 1e-8)
        object_pcd = pcd.select_by_index(inliers, invert=True)
        if len(object_pcd.points) > 0:
            obj_center = np.asarray(object_pcd.points).mean(axis=0)
            plane_pt = np.array(plane_model[:3]) * (-plane_model[3])
            if np.dot(n, obj_center - plane_pt) < 0:
                n = -n
        self._table_normal = n
        return object_pcd

    def _extract_clusters(self, pcd: o3d.geometry.PointCloud) -> List[o3d.geometry.PointCloud]:
        labels = np.array(
            pcd.cluster_dbscan(
                eps=self.cluster_eps,
                min_points=self.cluster_min_points,
                print_progress=False,
            )
        )
        if len(labels) == 0:
            return []

        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_idx = np.where(labels == label)[0]
            cluster_pcd = pcd.select_by_index(cluster_idx)
            clusters.append(cluster_pcd)
        return clusters

    def _generate_grasp_candidates(self, pcd, center, extent, bbox_R, avg_normal):
        candidates = []
        pts = np.asarray(pcd.points)
        if self._table_normal is None:
            return candidates

        up = self._table_normal.copy()
        down = -up

        proj = pts @ up
        z_top = proj.max()
        z_bottom = proj.min()
        height = z_top - z_bottom

        lateral_mask = proj > (z_bottom + 0.02)
        if lateral_mask.sum() < 10:
            lateral_mask = np.ones(len(pts), dtype=bool)
        lateral_center = pts[lateral_mask].mean(axis=0)

        # --- Strategy 1: Top-Down ---
        grasp_pt_td = lateral_center - up * (lateral_center @ up) + up * z_top
        approach_td = down.copy()
        pos_td = grasp_pt_td - approach_td * self.gripper_finger_depth

        major_horiz = bbox_R[:, np.argsort(extent)[2]].copy()
        major_horiz -= up * np.dot(major_horiz, up)
        norm = np.linalg.norm(major_horiz)
        if norm < 0.1:
            major_horiz = np.cross(up, np.array([1,0,0]))
            norm = np.linalg.norm(major_horiz)
            if norm < 0.1:
                major_horiz = np.cross(up, np.array([0,1,0]))
            norm = np.linalg.norm(major_horiz)
        major_horiz /= norm

        gripper_y_td = major_horiz
        gripper_x_td = np.cross(gripper_y_td, approach_td)
        gripper_x_td /= (np.linalg.norm(gripper_x_td) + 1e-8)
        gripper_y_td = np.cross(approach_td, gripper_x_td)
        gripper_y_td /= (np.linalg.norm(gripper_y_td) + 1e-8)
        rot_td = np.column_stack([gripper_x_td, gripper_y_td, approach_td])
        quat_td = R.from_matrix(rot_td).as_quat()

        score_td = self._score_surface_grasp(
            pts, pos_td, approach_td, gripper_y_td, height,
            strategy="top_down", table_dist=z_bottom
        )
        candidates.append(GraspCandidate(
            score=score_td,
            position=pos_td,
            orientation=quat_td,
            approach=approach_td,
            gripper_y=gripper_y_td,
            gripper_width=min(extent[np.argsort(extent)[0]] * 1.2, self.gripper_max_width),
            grasp_type="top_down",
            pre_grasp_offset=0.08,
        ))

        # --- Strategy 2: Side ---
        if height > 0.06:
            side_dir = np.cross(up, major_horiz)
            side_dir /= (np.linalg.norm(side_dir) + 1e-8)
            approach_side = side_dir

            grasp_pt_side = lateral_center - up * (lateral_center @ up) + up * (z_bottom + height * 0.5)
            pos_side = grasp_pt_side - approach_side * self.gripper_finger_depth

            gripper_y_side = major_horiz.copy()
            gripper_x_side = np.cross(gripper_y_side, approach_side)
            gripper_x_side /= (np.linalg.norm(gripper_x_side) + 1e-8)
            gripper_y_side = np.cross(approach_side, gripper_x_side)
            gripper_y_side /= (np.linalg.norm(gripper_y_side) + 1e-8)
            rot_side = np.column_stack([gripper_x_side, gripper_y_side, approach_side])
            quat_side = R.from_matrix(rot_side).as_quat()

            score_side = self._score_surface_grasp(
                pts, pos_side, approach_side, gripper_y_side, height,
                strategy="side", table_dist=z_bottom
            )
            candidates.append(GraspCandidate(
                score=score_side,
                position=pos_side,
                orientation=quat_side,
                approach=approach_side,
                gripper_y=gripper_y_side,
                gripper_width=min(extent[np.argsort(extent)[0]] * 1.2, self.gripper_max_width),
                grasp_type="side",
                pre_grasp_offset=0.10,
            ))

        # --- Filter: table collision ---
        filtered = []
        for c in candidates:
            finger_tip = c.position + c.approach * self.gripper_finger_depth
            finger_tip_proj = finger_tip @ up
            if finger_tip_proj < z_bottom + 0.005:
                continue
            gripper_body_proj = c.position @ up
            if gripper_body_proj < z_bottom + 0.015:
                continue
            filtered.append(c)

        return filtered

    def _score_surface_grasp(self, pts, position, approach, gripper_y, height, strategy, table_dist):
        score = 0.0
        n_points = len(pts)
        score += min(1.0, n_points / 300.0) * 0.15

        if strategy == "top_down":
            score += min(1.0, height / 0.10) * 0.15
        elif strategy == "side":
            score += min(1.0, height / 0.15) * 0.25

        centroid = pts.mean(axis=0)
        lateral_err = np.linalg.norm(position - centroid)
        score += max(0.0, 1.0 - lateral_err / 0.05) * 0.20

        score += 0.25
        score += 0.15

        return min(1.0, max(0.0, score))


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def project_pose_to_pixel(
    pose_cam: PoseStamped,
    fx: float, fy: float, cx: float, cy: float
) -> Optional[Tuple[int, int]]:
    """Project a 3D point in camera frame to image (u, v)."""
    x = pose_cam.pose.position.x
    y = pose_cam.pose.position.y
    z = pose_cam.pose.position.z
    if z <= 0.01:
        return None
    u = int((x / z) * fx + cx)
    v = int((y / z) * fy + cy)
    return (u, v)


def draw_grasp_on_image(
    img: np.ndarray,
    grasp_cam: PoseStamped,
    fx: float, fy: float, cx: float, cy: float,
    color: Tuple[int, int, int] = (0, 0, 255),
    label: str = "GRASP",
) -> np.ndarray:
    """Draw grasp point + approach arrow on the image."""
    px = project_pose_to_pixel(grasp_cam, fx, fy, cx, cy)
    if px is None:
        return img

    u, v = px
    cv2.circle(img, (u, v), 6, color, -1)
    cv2.circle(img, (u, v), 8, (255, 255, 255), 2)

    # Approach arrow
    quat = [
        grasp_cam.pose.orientation.x,
        grasp_cam.pose.orientation.y,
        grasp_cam.pose.orientation.z,
        grasp_cam.pose.orientation.w,
    ]
    rot = R.from_quat(quat)
    approach = rot.as_matrix()[:, 2]  # Z-axis = approach

    tip = np.array([
        grasp_cam.pose.position.x,
        grasp_cam.pose.position.y,
        grasp_cam.pose.position.z,
    ]) + approach * 0.04

    if tip[2] > 0.01:
        tip_u = int((tip[0] / tip[2]) * fx + cx)
        tip_v = int((tip[1] / tip[2]) * fy + cy)
        cv2.arrowedLine(img, (u, v), (tip_u, tip_v), color, 2, tipLength=8)

    cv2.putText(img, label, (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def draw_pregrasp_on_image(
    img: np.ndarray,
    pregrasp_cam: PoseStamped,
    grasp_cam: PoseStamped,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Draw pre-grasp to grasp line."""
    px_pre = project_pose_to_pixel(pregrasp_cam, fx, fy, cx, cy)
    px_grasp = project_pose_to_pixel(grasp_cam, fx, fy, cx, cy)
    if px_pre and px_grasp:
        cv2.line(img, px_pre, px_grasp, (255, 128, 0), 2)
        cv2.circle(img, px_pre, 4, (255, 128, 0), -1)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECT TRACKER (unchanged, works well)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    class_name: str
    position: np.ndarray
    last_seen: float
    hit_count: int = 1

    def update(self, new_pos: np.ndarray, alpha: float) -> None:
        self.position = alpha * new_pos + (1.0 - alpha) * self.position
        self.last_seen = time.monotonic()
        self.hit_count += 1


class ObjectTracker:
    def __init__(
        self,
        max_distance: float = 1.0,
        max_age: float = 5.0,
        alpha: float = 0.3,
    ) -> None:
        self.max_distance = max_distance
        self.max_age = max_age
        self.alpha = alpha
        self._tracks: List[TrackedObject] = []

    def update(self, class_name: str, raw_position: np.ndarray) -> np.ndarray:
        self._prune_stale()
        best_track: Optional[TrackedObject] = None
        best_dist = float("inf")

        for track in self._tracks:
            if track.class_name != class_name:
                continue
            dist = float(np.linalg.norm(track.position - raw_position))
            if dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is not None and best_dist < self.max_distance:
            best_track.update(raw_position, self.alpha)
            return best_track.position.copy()

        new_track = TrackedObject(
            class_name=class_name,
            position=raw_position.copy(),
            last_seen=time.monotonic(),
        )
        self._tracks.append(new_track)
        return raw_position.copy()

    def _prune_stale(self) -> None:
        now = time.monotonic()
        self._tracks = [
            t for t in self._tracks if now - t.last_seen < self.max_age
        ]

    @property
    def track_count(self) -> int:
        return len(self._tracks)


# ═══════════════════════════════════════════════════════════════════════════════
# DEPTH SAMPLING HELPER (unchanged, works well)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_surface_depth(
    depth_img: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    max_depth: float,
    depth_percentile: float = 20.0,
) -> Optional[float]:
    h, w = depth_img.shape
    u1 = max(0, int(x1))
    v1 = max(0, int(y1))
    u2 = min(w, int(x2))
    v2 = min(h, int(y2))

    roi = depth_img[v1:v2, u1:u2]
    valid = roi[(roi > 0.0) & (roi <= max_depth)]

    if valid.size == 0:
        return None
    return float(np.percentile(valid, depth_percentile))


# ═══════════════════════════════════════════════════════════════════════════════
# ROS 2 NODE
# ═══════════════════════════════════════════════════════════════════════════════

class YoloDepthDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_depth_detector_node")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── ROS parameters ─────────────────────────────────────────────────
        self.declare_parameter("model_path", "yolov8m.pt")
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("output_image_topic", "/yolo/annotated_image")
        self.declare_parameter("output_detections_topic", "/yolo/detections")
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("device", "auto")
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("sync_slop", 0.05)

        self.declare_parameter("fx", 0.0)
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("cx", 0.0)
        self.declare_parameter("cy", 0.0)

        self.declare_parameter("tracker_max_distance", 1.0)
        self.declare_parameter("tracker_max_age", 5.0)
        self.declare_parameter("tracker_alpha", 0.3)
        self.declare_parameter("depth_percentile", 20.0)

        # NEW: workspace / reachability
        self.declare_parameter("arm_base_frame", "g_base")
        self.declare_parameter("max_reach", 0.28)

        # ── Read parameters ─────────────────────────────────────────────────
        model_path = self.get_parameter("model_path").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        cam_info_topic = self.get_parameter("camera_info_topic").value
        out_img_topic = self.get_parameter("output_image_topic").value
        out_det_topic = self.get_parameter("output_detections_topic").value
        self.conf_threshold = self.get_parameter("conf_threshold").value
        device_param = self.get_parameter("device").value
        self.max_depth = self.get_parameter("max_depth").value
        sync_slop = self.get_parameter("sync_slop").value
        self.depth_percentile = self.get_parameter("depth_percentile").value

        tracker_max_dist = self.get_parameter("tracker_max_distance").value
        tracker_max_age = self.get_parameter("tracker_max_age").value
        tracker_alpha = self.get_parameter("tracker_alpha").value

        self.arm_base_frame = self.get_parameter("arm_base_frame").value
        self.max_reach = self.get_parameter("max_reach").value

        # ── Object tracker ──────────────────────────────────────────────────
        self.tracker = ObjectTracker(
            max_distance=tracker_max_dist,
            max_age=tracker_max_age,
            alpha=tracker_alpha,
        )
        self.get_logger().info(
            f"ObjectTracker: max_dist={tracker_max_dist}m, "
            f"max_age={tracker_max_age}s, alpha={tracker_alpha}"
        )

        # ── Geometric grasp detector ──────────────────────────────────────
        self.grasp_detector = GeometricGraspDetector(
            gripper_max_width=0.065,
            gripper_finger_depth=0.04,
            gripper_finger_length=0.035,
            voxel_size=0.003,
            min_points=80,
            plane_dist_thresh=0.015,
            cluster_eps=0.015,
            cluster_min_points=50,
        )
        self.get_logger().info(
            "GeometricGraspDetector initialized with full point cloud pipeline"
        )

        # ── Device / model ──────────────────────────────────────────────────
        if device_param == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_param

        self.get_logger().info(f"Loading YOLO model on {self.device.upper()}…")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.get_logger().info("Model loaded successfully.")

        self.bridge = CvBridge()

        # ── Publishers ──────────────────────────────────────────────────────
        self.image_pub = self.create_publisher(Image, out_img_topic, 10)
        self.det_pub = self.create_publisher(Detection2DArray, out_det_topic, 10)
        self.coord_pub = self.create_publisher(Detection2DArray, "relative_coordinates", 10)
        self.world_pub = self.create_publisher(PoseStamped, "/detected_object", 10)

        self.grasp_candidates_pub = self.create_publisher(PoseArray, "/grasp_candidates", 10)
        self.best_grasp_pub = self.create_publisher(PoseStamped, "/best_grasp", 10)
        self.pre_grasp_pub = self.create_publisher(PoseStamped, "/pre_grasp", 10)

        # ── QoS ─────────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Camera intrinsics ────────────────────────────────────────────────
        self.fx = self.get_parameter("fx").value
        self.fy = self.get_parameter("fy").value
        self.cx = self.get_parameter("cx").value
        self.cy = self.get_parameter("cy").value

        if self.fx > 0.0 and self.fy > 0.0 and self.cx > 0.0 and self.cy > 0.0:
            self.intrinsics_ready = True
            self.cam_info_sub = None
            self.get_logger().info(
                f"Using manual intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
        else:
            self.intrinsics_ready = False
            self.cam_info_sub = self.create_subscription(
                CameraInfo, cam_info_topic, self.camera_info_callback, 10
            )
            self.get_logger().info(f"Waiting for camera info on {cam_info_topic}…")

        # ── Synchronized subscribers ─────────────────────────────────────────
        rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=qos)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=sync_slop
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info(
            f"Synchronized subscriber ready: {self.rgb_topic} + {depth_topic}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _is_reachable(self, pose_cam: PoseStamped) -> bool:
        """Crude spherical workspace check around arm base."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.arm_base_frame,
                pose_cam.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            p = do_transform_pose_stamped(pose_cam, t).pose.position
            dist = (p.x**2 + p.y**2 + p.z**2) ** 0.5
            return dist <= self.max_reach
        except Exception as e:
            self.get_logger().debug(f"Reachability check failed: {e}")
            return True  # fail-open if TF is missing
        
    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.intrinsics_ready = True
        self.get_logger().info(
            f"Camera intrinsics received: fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f}"
        )
        if self.cam_info_sub is not None:
            self.destroy_subscription(self.cam_info_sub)
            self.cam_info_sub = None

    # ──────────────────────────────────────────────────────────────────────────

    def sync_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        if not self.intrinsics_ready:
            self.get_logger().warn("Camera intrinsics not available yet, skipping frame")
            return

        # ── Convert RGB ──────────────────────────────────────────────────────
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
            return

        # ── YOLO inference ───────────────────────────────────────────────────
        results = self.model(cv_image, verbose=False, device=self.device)[0]
        det_array = Detection2DArray()
        det_array.header = rgb_msg.header
        annotated = cv_image.copy()

        # ── Parse depth ──────────────────────────────────────────────────────
        depth_img = self._parse_depth(depth_msg)
        if depth_img is None:
            self.get_logger().warn("Depth parsing failed, skipping frame")
            return

        h, w = depth_img.shape

        if results.boxes is None or len(results.boxes) == 0:
            self._publish_annotated(annotated, det_array)
            return

        # Collect all grasp candidates for visualization
        best_grasp_cam = None
        best_pregrasp_cam = None
        best_score = -1.0
        all_candidates_cam = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id].lower().replace(" ", "_")
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            # ── Depth estimation for object position (tracker) ────────────
            z_cam = estimate_surface_depth(
                depth_img, x1, y1, x2, y2,
                self.max_depth, self.depth_percentile
            )
            if z_cam is None:
                continue

            center_u = int(np.clip((x1 + x2) / 2.0, 0, w - 1))
            center_v = int(np.clip((y1 + y2) / 2.0, 0, h - 1))

            x_cam = (center_u - self.cx) * z_cam / self.fx
            y_cam = (center_v - self.cy) * z_cam / self.fy

            # ── TF: camera → map (for object tracking) ─────────────────────
            pose_camera = PoseStamped()
            pose_camera.header.frame_id = "camera_link"
            pose_camera.pose.position.x = x_cam
            pose_camera.pose.position.y = y_cam
            pose_camera.pose.position.z = z_cam
            pose_camera.pose.orientation.w = 1.0

            try:
                transform = self.tf_buffer.lookup_transform(
                    "map", "camera_link",
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
                pose_map = do_transform_pose_stamped(pose_camera, transform)
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                continue

            raw_pos = np.array([
                pose_map.pose.position.x,
                pose_map.pose.position.y,
                pose_map.pose.position.z,
            ])
            filtered_pos = self.tracker.update(cls_name, raw_pos)
            x, y, z = filtered_pos

            # ── Publish stable world pose ──────────────────────────────────
            world_msg = PoseStamped()
            world_msg.header.stamp = pose_map.header.stamp
            world_msg.header.frame_id = cls_name
            world_msg.pose = pose_map.pose
            world_msg.pose.position.x = float(x)
            world_msg.pose.position.y = float(y)
            world_msg.pose.position.z = float(z)
            self.world_pub.publish(world_msg)

            # ── vision_msgs Detection2D ────────────────────────────────────
            detection = Detection2D()
            detection.header.frame_id = "map"
            detection.header.stamp = rgb_msg.header.stamp

            box_cx = (x1 + x2) / 2.0
            box_cy = (y1 + y2) / 2.0
            try:
                detection.bbox.center.x = box_cx
                detection.bbox.center.y = box_cy
                detection.bbox.center.theta = 0.0
            except AttributeError:
                try:
                    detection.bbox.center.position.x = box_cx
                    detection.bbox.center.position.y = box_cy
                except AttributeError:
                    detection.bbox.center.x = box_cx
                    detection.bbox.center.y = box_cy

            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = cls_name
            hypothesis.hypothesis.score = conf
            hypothesis.pose.pose.position.x = float(x)
            hypothesis.pose.pose.position.y = float(y)
            hypothesis.pose.pose.position.z = float(z)
            hypothesis.pose.pose.orientation.w = 1.0
            detection.results.append(hypothesis)
            det_array.detections.append(detection)

            # ═══════════════════════════════════════════════════════════════
            # GEOMETRIC GRASP DETECTION
            # ═══════════════════════════════════════════════════════════════
            grasp = self.grasp_detector.detect(
                depth_img, x1, y1, x2, y2,
                self.fx, self.fy, self.cx, self.cy,
                self.max_depth,
            )

            if grasp is not None:
                # Build PoseStamped in camera_link frame
                grasp_cam = PoseStamped()
                grasp_cam.header = rgb_msg.header
                grasp_cam.header.frame_id = "camera_link"
                grasp_cam.pose.position.x = float(grasp.position[0])
                grasp_cam.pose.position.y = float(grasp.position[1])
                grasp_cam.pose.position.z = float(grasp.position[2])
                grasp_cam.pose.orientation.x = float(grasp.orientation[0])
                grasp_cam.pose.orientation.y = float(grasp.orientation[1])
                grasp_cam.pose.orientation.z = float(grasp.orientation[2])
                grasp_cam.pose.orientation.w = float(grasp.orientation[3])

                # CORRECTED: pre-grasp = back off along approach
                pregrasp_cam = PoseStamped()
                pregrasp_cam.header = grasp_cam.header
                pregrasp_cam.pose.position.x = float(
                    grasp.position[0] - grasp.approach[0] * grasp.pre_grasp_offset
                )
                pregrasp_cam.pose.position.y = float(
                    grasp.position[1] - grasp.approach[1] * grasp.pre_grasp_offset
                )
                pregrasp_cam.pose.position.z = float(
                    grasp.position[2] - grasp.approach[2] * grasp.pre_grasp_offset
                )
                pregrasp_cam.pose.orientation = grasp_cam.pose.orientation

                # Store for visualization
                all_candidates_cam.append(grasp_cam)

                # Track best grasp across ALL objects this frame
                if grasp.score > best_score and self._is_reachable(grasp_cam):
                    best_score = grasp.score
                    best_grasp_cam = grasp_cam
                    best_pregrasp_cam = pregrasp_cam

                # Draw on image
                color = {"top_down": (0, 0, 255), "side": (0, 255, 0)}.get(
                    grasp.grasp_type, (0, 255, 255)
                )
                annotated = draw_grasp_on_image(
                    annotated, grasp_cam, self.fx, self.fy, self.cx, self.cy,
                    color=color, label=f"{grasp.grasp_type.upper()} ({grasp.score:.2f})",
                )
                annotated = draw_pregrasp_on_image(
                    annotated, pregrasp_cam, grasp_cam,
                    self.fx, self.fy, self.cx, self.cy,
                )

                self.get_logger().debug(
                    f"Grasp candidate [{cls_name}]: type={grasp.grasp_type}, "
                    f"score={grasp.score:.3f}, width={grasp.gripper_width:.3f}m"
                )

            # ── Annotate detection bbox ─────────────────────────────────────
            ix1 = max(0, int(x1))
            iy1 = max(0, int(y1))
            ix2 = min(w, int(x2))
            iy2 = min(h, int(y2))
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

            label = f"{cls_name} x:{x:.2f} y:{y:.2f} z:{z:.2f}"
            text_y = iy1 - 10 if iy1 - 10 >= 15 else iy1 + 15
            cv2.putText(annotated, label, (ix1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ── Publish all grasp candidates as PoseArray ──────────────────────
        if all_candidates_cam:
            pose_array = PoseArray()
            pose_array.header = rgb_msg.header
            pose_array.header.frame_id = "camera_link"
            for pc in all_candidates_cam:
                pose_array.poses.append(pc.pose)
            self.grasp_candidates_pub.publish(pose_array)

        # ── Publish the single best grasp (after all objects processed) ──────
        if best_grasp_cam is not None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "map", "camera_link",
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
                best_grasp_map = do_transform_pose_stamped(best_grasp_cam, transform)
                best_pregrasp_map = do_transform_pose_stamped(best_pregrasp_cam, transform)
                self.best_grasp_pub.publish(best_grasp_map)
                self.pre_grasp_pub.publish(best_pregrasp_map)
            except Exception as e:
                self.get_logger().warn(f"Best grasp TF failed: {e}")

        # ── Publish annotated image and detections ─────────────────────────
        self._publish_annotated(annotated, det_array)

        if det_array.detections:
            self.det_pub.publish(det_array)
            self.coord_pub.publish(det_array)

    # ──────────────────────────────────────────────────────────────────────────

    def _publish_annotated(self, annotated: np.ndarray, header_msg: Detection2DArray) -> None:
        try:
            msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            msg.header = header_msg.header
            self.image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

    # ──────────────────────────────────────────────────────────────────────────

    def _parse_depth(self, depth_msg: Image) -> Optional[np.ndarray]:
        """Convert depth image to float32 metres."""
        try:
            depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth CV Bridge error: {e}")
            return None

        if depth_msg.encoding == "16UC1":
            return depth_np.astype(np.float32) / 1000.0
        elif depth_msg.encoding == "32FC1":
            return depth_np.astype(np.float32)
        else:
            self.get_logger().warn(f"Unsupported depth encoding: {depth_msg.encoding}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
