import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import message_filters
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


# ─────────────────────────────────────────────────────────────────────────────
# Tracker helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackedObject:
    """Single object instance maintained by the spatial tracker."""
    class_name: str
    position: np.ndarray        # [x, y, z] in map frame (filtered)
    last_seen: float            # wall-clock seconds (time.monotonic)
    hit_count: int = 1

    def update(self, new_pos: np.ndarray, alpha: float) -> None:
        """Exponential Moving Average position update."""
        self.position = alpha * new_pos + (1.0 - alpha) * self.position
        self.last_seen = time.monotonic()
        self.hit_count += 1


class ObjectTracker:
    """
    Lightweight spatial tracker that associates detections to existing tracks
    by class name + Euclidean distance, then applies EMA smoothing.
    """

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


# ─────────────────────────────────────────────────────────────────────────────
# Depth-sampling helper
# ─────────────────────────────────────────────────────────────────────────────

def estimate_surface_depth(
    depth_img: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
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


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 node (Foxy compatible)
# ─────────────────────────────────────────────────────────────────────────────

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

        # ── Read parameters ─────────────────────────────────────────────────
        model_path       = self.get_parameter("model_path").value
        self.rgb_topic   = self.get_parameter("rgb_topic").value
        depth_topic      = self.get_parameter("depth_topic").value
        cam_info_topic   = self.get_parameter("camera_info_topic").value
        out_img_topic    = self.get_parameter("output_image_topic").value
        out_det_topic    = self.get_parameter("output_detections_topic").value
        self.conf_threshold = self.get_parameter("conf_threshold").value
        device_param     = self.get_parameter("device").value
        self.max_depth   = self.get_parameter("max_depth").value
        sync_slop        = self.get_parameter("sync_slop").value
        self.depth_percentile = self.get_parameter("depth_percentile").value

        tracker_max_dist = self.get_parameter("tracker_max_distance").value
        tracker_max_age  = self.get_parameter("tracker_max_age").value
        tracker_alpha    = self.get_parameter("tracker_alpha").value

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

        # ── Device / model ──────────────────────────────────────────────────
        if device_param == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_param

        self.get_logger().info(f"Loading YOLO model on {self.device.upper()}...")
        self.model = YOLO("yolo26x.pt")
        self.model.to(self.device)
        self.get_logger().info("Model loaded successfully.")

        self.bridge = CvBridge()

        # ── Publishers ──────────────────────────────────────────────────────
        self.image_pub = self.create_publisher(Image, out_img_topic, 10)
        self.det_pub   = self.create_publisher(Detection2DArray, out_det_topic, 10)
        self.coord_pub = self.create_publisher(Detection2DArray, "relative_coordinates", 10)
        self.world_pub = self.create_publisher(PoseStamped, "/detected_object", 10)

        # ── QoS (Foxy style) ──────────────────────────────────────────────
        # Foxy uses QoSReliabilityPolicy / QoSHistoryPolicy (not ReliabilityPolicy / HistoryPolicy)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
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
            self.get_logger().info(f"Waiting for camera info on {cam_info_topic}...")

        # ── Synchronized subscribers ─────────────────────────────────────────
        # Foxy: message_filters.Subscriber uses `qos=` parameter (not `qos_profile=`)
        rgb_sub   = message_filters.Subscriber(self, Image, self.rgb_topic,  qos_profile=qos)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic,     qos_profile=qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=sync_slop
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info(
            f"Synchronized subscriber ready: {self.rgb_topic} + {depth_topic}"
        )

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

    def sync_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        if not self.intrinsics_ready:
            self.get_logger().warn("Camera intrinsics not available yet, skipping frame")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
            return

        results  = self.model(cv_image, verbose=False, device=self.device)[0]
        det_array          = Detection2DArray()
        det_array.header   = rgb_msg.header
        annotated          = cv_image.copy()

        depth_img = self._parse_depth(depth_msg)
        if depth_img is None:
            self.get_logger().warn("Depth parsing failed, skipping frame")
            return

        h, w = depth_img.shape

        if results.boxes is None or len(results.boxes) == 0:
            self._publish_annotated(annotated, det_array)
            return

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_id   = int(box.cls[0])
            cls_name = results.names[cls_id].lower().replace(" ", "_")
            x1, y1, x2, y2 = map(float, box.xyxy[0])

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

            pose_camera = PoseStamped()
            pose_camera.header.frame_id       = "camera_link"
            pose_camera.pose.position.x       = x_cam
            pose_camera.pose.position.y       = y_cam
            pose_camera.pose.position.z       = z_cam
            pose_camera.pose.orientation.w    = 1.0

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

            world_msg                    = PoseStamped()
            world_msg.header.stamp       = pose_map.header.stamp
            world_msg.header.frame_id    = cls_name
            world_msg.pose               = pose_map.pose
            world_msg.pose.position.x    = float(x)
            world_msg.pose.position.y    = float(y)
            world_msg.pose.position.z    = float(z)
            self.world_pub.publish(world_msg)

            # ── vision_msgs Detection2D (Foxy compatible) ──────────────────
            detection           = Detection2D()
            detection.header.frame_id = "map"
            detection.header.stamp    = rgb_msg.header.stamp

            box_cx = (x1 + x2) / 2.0
            box_cy = (y1 + y2) / 2.0

            # Foxy: BoundingBox2D.center is geometry_msgs/Pose2D (x, y, theta)
            detection.bbox.center.x     = box_cx
            detection.bbox.center.y     = box_cy
            detection.bbox.center.theta = 0.0

            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1

            # Foxy: ObjectHypothesisWithPose.hypothesis has `id` (int64), not `class_id` (string)
            # In Humble/Rolling it was renamed to `class_id`. For Foxy we use `id`.
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.id = cls_id      # Foxy: int64 `id` field
            hypothesis.hypothesis.score = conf
            hypothesis.pose.pose.position.x = float(x)
            hypothesis.pose.pose.position.y = float(y)
            hypothesis.pose.pose.position.z = float(z)
            hypothesis.pose.pose.orientation.w = 1.0
            detection.results.append(hypothesis)
            det_array.detections.append(detection)

            # ── Annotate image ────────────────────────────────────────────────
            ix1 = max(0, int(x1));  iy1 = max(0, int(y1))
            ix2 = min(w, int(x2));  iy2 = min(h, int(y2))
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

            label  = f"{cls_name} x:{x:.2f} y:{y:.2f} z:{z:.2f}"
            text_y = iy1 - 10 if iy1 - 10 >= 15 else iy1 + 15
            cv2.putText(annotated, label, (ix1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self._publish_annotated(annotated, det_array)

        if det_array.detections:
            self.det_pub.publish(det_array)
            self.coord_pub.publish(det_array)

    def _publish_annotated(self, annotated: np.ndarray, header_msg: Detection2DArray) -> None:
        try:
            msg        = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            msg.header = header_msg.header
            self.image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

    def _parse_depth(self, depth_msg: Image) -> Optional[np.ndarray]:
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
