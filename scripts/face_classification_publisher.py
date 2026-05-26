#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import String
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

import math
import time
import message_filters
import tf2_ros
from tf2_ros import TransformException


def _transform_to_matrix(tf_stamped) -> np.ndarray:
    """Convert geometry_msgs/TransformStamped to a 4x4 numpy matrix."""
    q = tf_stamped.transform.rotation
    t = tf_stamped.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


class FaceClassificationPublisher(Node):
    def __init__(self):
        super().__init__("face_classification_publisher")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("image_topic", "/oakd/rgb/preview/image_raw"),
                ("pointcloud_topic", "/oakd/rgb/preview/depth/points"),
                ("publish_topic", "/face_class"),
                ("detector_model", "yolov8n.pt"),
                ("classifier_model", "/home/gamma/colcon_ws/face_classification/runs/classify/train/weights/best.pt"),
                ("device", ""),
                ("map_frame", "map"),
                ("standoff_distance", 1.0),
                ("max_detection_distance", 2.0),
                ("normal_window_size", 20),
                ("min_valid_depth_points", 10),
                ("marker_z_offset", 0.0),
                ("confirmed_face_min_dist", 0.50),
            ],
        )

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        self.publish_topic = self.get_parameter("publish_topic").get_parameter_value().string_value
        self.detector_model_path = self.get_parameter("detector_model").get_parameter_value().string_value
        self.classifier_model_path = self.get_parameter("classifier_model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.standoff_distance = float(self.get_parameter("standoff_distance").value)
        self.max_detection_distance = float(self.get_parameter("max_detection_distance").value)
        self.normal_window_size = int(self.get_parameter("normal_window_size").value)
        self.min_valid_depth_points = int(self.get_parameter("min_valid_depth_points").value)
        self.marker_z_offset = float(self.get_parameter("marker_z_offset").value)
        self.confirmed_face_min_dist = float(self.get_parameter("confirmed_face_min_dist").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.detector = YOLO(self.detector_model_path)
        self.classifier = YOLO(self.classifier_model_path)

        self.processing = False

        # Accumulation buffer for stable face goal confirmation
        self.detection_buffer = []   # list of (p_center, p_goal) tuples in map frame
        self.confirmed_face_count = 0
        self.confirmed_goals = []    # avg_goal positions of all committed faces

        # Image + pointcloud time-sync. Per-frame markers are deliberately NOT published —
        # we only emit /face_classifier/confirmed_markers once 15 detections are buffered.
        self.image_sub_mf = message_filters.Subscriber(
            self, Image, self.image_topic, qos_profile=qos_profile_sensor_data
        )
        self.pc_sub_mf = message_filters.Subscriber(
            self, PointCloud2, self.pointcloud_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub_mf, self.pc_sub_mf], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.synced_callback)

        self.class_pub = self.create_publisher(String, self.publish_topic, 10)
        self.confirmed_marker_pub = self.create_publisher(
            MarkerArray, "/face_classifier/confirmed_markers", 10
        )
        self.face_goal_pub = self.create_publisher(PoseStamped, "/face_goal", 10)

    def _commit_face(self) -> None:
        centers = np.array([c for c, _, _, _ in self.detection_buffer])
        goals   = np.array([g for _, g, _, _ in self.detection_buffer])
        classes = [cls for _, _, cls, _ in self.detection_buffer]
        self.detection_buffer = []
        avg_center = np.median(centers, axis=0)
        avg_goal   = np.median(goals, axis=0)
        committed_class = max(set(classes), key=classes.count)

        self.confirmed_face_count += 1
        face_id = self.confirmed_face_count
        self.confirmed_goals.append(avg_goal)
        print(f"Confirmed face #{face_id} at goal ({avg_goal[0]:.2f}, {avg_goal[1]:.2f}, {avg_goal[2]:.2f})")

        # Yaw so the robot faces from avg_goal toward avg_center
        dx = avg_center[0] - avg_goal[0]
        dy = avg_center[1] - avg_goal[1]
        yaw = math.atan2(dy, dx)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        now_stamp = self.get_clock().now().to_msg()

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.map_frame
        goal_msg.header.stamp = now_stamp
        goal_msg.pose.position.x = float(avg_goal[0])
        goal_msg.pose.position.y = float(avg_goal[1])
        goal_msg.pose.position.z = float(avg_goal[2]) + self.marker_z_offset
        goal_msg.pose.orientation.z = qz
        goal_msg.pose.orientation.w = qw
        self.face_goal_pub.publish(goal_msg)

        # Confirmed publish: face center (red), goal/standoff point (yellow),
        # arrow center→goal (cyan), and a label. Robot_commander relays into /confirmed_objects.
        confirmed_array = MarkerArray()

        center_sphere = Marker()
        center_sphere.header.frame_id = self.map_frame
        center_sphere.header.stamp = now_stamp
        center_sphere.ns = "face_center"
        center_sphere.id = face_id * 4
        center_sphere.type = Marker.SPHERE
        center_sphere.action = Marker.ADD
        center_sphere.pose.position.x = float(avg_center[0])
        center_sphere.pose.position.y = float(avg_center[1])
        center_sphere.pose.position.z = float(avg_center[2]) + self.marker_z_offset
        center_sphere.pose.orientation.w = 1.0
        center_sphere.scale.x = center_sphere.scale.y = center_sphere.scale.z = 0.2
        center_sphere.color.r = 1.0
        center_sphere.color.a = 1.0
        confirmed_array.markers.append(center_sphere)

        goal_sphere = Marker()
        goal_sphere.header.frame_id = self.map_frame
        goal_sphere.header.stamp = now_stamp
        goal_sphere.ns = "face_goal"
        goal_sphere.id = face_id * 4 + 1
        goal_sphere.type = Marker.SPHERE
        goal_sphere.action = Marker.ADD
        goal_sphere.pose.position.x = float(avg_goal[0])
        goal_sphere.pose.position.y = float(avg_goal[1])
        goal_sphere.pose.position.z = float(avg_goal[2]) + self.marker_z_offset
        goal_sphere.pose.orientation.w = 1.0
        goal_sphere.scale.x = goal_sphere.scale.y = goal_sphere.scale.z = 0.3
        goal_sphere.color.r = 1.0
        goal_sphere.color.g = 1.0
        goal_sphere.color.a = 1.0
        confirmed_array.markers.append(goal_sphere)

        arrow = Marker()
        arrow.header.frame_id = self.map_frame
        arrow.header.stamp = now_stamp
        arrow.ns = "face_normal"
        arrow.id = face_id * 4 + 2
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        tail = Point(); tail.x, tail.y, tail.z = float(avg_center[0]), float(avg_center[1]), float(avg_center[2]) + self.marker_z_offset
        head = Point(); head.x, head.y, head.z = float(avg_goal[0]),   float(avg_goal[1]),   float(avg_goal[2])   + self.marker_z_offset
        arrow.points = [tail, head]
        arrow.scale.x = 0.05
        arrow.scale.y = 0.10
        arrow.scale.z = 0.10
        arrow.color.g = 0.8
        arrow.color.b = 1.0
        arrow.color.a = 0.9
        confirmed_array.markers.append(arrow)

        label = Marker()
        label.header.frame_id = self.map_frame
        label.header.stamp = now_stamp
        label.ns = "face_label"
        label.id = face_id * 4 + 3
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(avg_center[0])
        label.pose.position.y = float(avg_center[1])
        label.pose.position.z = float(avg_center[2]) + self.marker_z_offset + 0.25
        label.pose.orientation.w = 1.0
        label.scale.z = 0.15
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f"Face #{face_id}"
        confirmed_array.markers.append(label)

        # Carry the YOLO classifier label downstream; robot_commander parses
        # the text on arrival to decide gender etc.
        cls_marker = Marker()
        cls_marker.header.frame_id = self.map_frame
        cls_marker.header.stamp = now_stamp
        cls_marker.ns = "face_class"
        cls_marker.id = face_id
        cls_marker.type = Marker.TEXT_VIEW_FACING
        cls_marker.action = Marker.ADD
        cls_marker.pose.position.x = float(avg_center[0])
        cls_marker.pose.position.y = float(avg_center[1])
        cls_marker.pose.position.z = float(avg_center[2]) + self.marker_z_offset + 0.45
        cls_marker.pose.orientation.w = 1.0
        cls_marker.scale.z = 0.15
        cls_marker.color.r = cls_marker.color.g = cls_marker.color.b = cls_marker.color.a = 1.0
        cls_marker.text = committed_class
        confirmed_array.markers.append(cls_marker)

        self.confirmed_marker_pub.publish(confirmed_array)

    def synced_callback(self, img_msg: Image, pc_msg: PointCloud2) -> None:
        if self.processing:
            return
        self.processing = True
        try:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            except Exception:
                return

            pc_array = None
            try:
                points_np = pc2.read_points_numpy(pc_msg, field_names=("x", "y", "z"))
                if (
                    pc_msg.height > 1
                    and pc_msg.width > 1
                    and points_np.shape[0] == pc_msg.height * pc_msg.width
                ):
                    pc_array = points_np.reshape((pc_msg.height, pc_msg.width, 3))
            except Exception as ex:
                self.get_logger().warn(f"Failed to parse point cloud: {ex}")

            img_h, img_w = cv_image.shape[:2]

            det_results = self.detector.predict(
                source=cv_image, classes=[0], verbose=False, device=self.device
            )

            for box in det_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = cv_image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_results = self.classifier.predict(
                    source=crop, verbose=False, device=self.device
                )
                probs = cls_results[0].probs
                if probs is None:
                    continue

                top_class_id = probs.top1
                top_class_name = cls_results[0].names[top_class_id]

                out_msg = String()
                out_msg.data = top_class_name
                self.class_pub.publish(out_msg)

                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(cv_image, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(
                    cv_image,
                    top_class_name,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                if pc_array is None:
                    continue

                pc_h, pc_w = pc_array.shape[:2]
                u_pc = int(cx * pc_w / img_w)
                v_pc = int(cy * pc_h / img_h)
                u_pc = max(0, min(pc_w - 1, u_pc))
                v_pc = max(0, min(pc_h - 1, v_pc))

                half_w = self.normal_window_size
                u0 = max(0, u_pc - half_w)
                u1 = min(pc_w, u_pc + half_w + 1)
                v0 = max(0, v_pc - half_w)
                v1 = min(pc_h, v_pc + half_w + 1)

                neighborhood = pc_array[v0:v1, u0:u1, :].reshape(-1, 3)
                valid_mask = np.isfinite(neighborhood).all(axis=1)
                neighborhood = neighborhood[valid_mask]

                if neighborhood.shape[0] < self.min_valid_depth_points:
                    self.get_logger().warn(
                        f"Skip marker: only {neighborhood.shape[0]} valid depth points"
                    )
                    continue

                center_pt = pc_array[v_pc, u_pc, :]
                if not np.isfinite(center_pt).all():
                    center_pt = neighborhood.mean(axis=0)
                center_pt = np.asarray(center_pt, dtype=np.float64)

                if float(np.linalg.norm(center_pt)) > self.max_detection_distance:
                    continue

                centered = neighborhood - neighborhood.mean(axis=0)
                cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
                eig_vals, eig_vecs = np.linalg.eigh(cov)
                normal = eig_vecs[:, 0]
                n_len = float(np.linalg.norm(normal))
                if n_len < 1e-9:
                    continue
                normal = normal / n_len

                # Flip normal so it points from wall toward the camera/room.
                # In the camera optical frame the wall point has positive depth (z>0);
                # the outward normal should have a negative component along P_center.
                if float(np.dot(normal, center_pt)) > 0:
                    normal = -normal

                try:
                    tf = self.tf_buffer.lookup_transform(
                        self.map_frame,
                        pc_msg.header.frame_id,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.2),
                    )
                except TransformException as ex:
                    self.get_logger().warn(
                        f"TF lookup failed ({pc_msg.header.frame_id} -> {self.map_frame}): {ex}"
                    )
                    continue

                T_map_cam = _transform_to_matrix(tf)
                R_map_cam = T_map_cam[:3, :3]

                p_cam_h = np.array([center_pt[0], center_pt[1], center_pt[2], 1.0])
                p_map = T_map_cam @ p_cam_h
                n_map = R_map_cam @ normal
                n_norm = float(np.linalg.norm(n_map))
                if n_norm < 1e-9:
                    continue
                n_map = n_map / n_norm

                p_center = (float(p_map[0]), float(p_map[1]), float(p_map[2]))
                p_goal = (
                    p_center[0] + self.standoff_distance * float(n_map[0]),
                    p_center[1] + self.standoff_distance * float(n_map[1]),
                    p_center[2] + self.standoff_distance * float(n_map[2]),
                )

                self.get_logger().debug(
                    f"face={top_class_name} "
                    f"P_center=({p_center[0]:.2f},{p_center[1]:.2f},{p_center[2]:.2f}) "
                    f"P_goal=({p_goal[0]:.2f},{p_goal[1]:.2f},{p_goal[2]:.2f})"
                )

                # Skip if this detection is near an already-confirmed face goal
                p_goal_arr = np.array(p_goal[:2])  # XY only for proximity check
                too_close = any(
                    float(np.linalg.norm(p_goal_arr - np.array(cg[:2]))) < self.confirmed_face_min_dist
                    for cg in self.confirmed_goals
                )
                if too_close:
                    continue

                # Accumulate toward confirmed detection (median over 25 frames)
                now = time.time()
                self.detection_buffer = [e for e in self.detection_buffer if now - e[3] <= 10.0]
                self.detection_buffer.append((p_center, p_goal, top_class_name, now))
                n_buf = len(self.detection_buffer)
                cv2.putText(
                    cv_image,
                    f"Buffering: {n_buf}/25",
                    (x1, min(y2 + 20, cv_image.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 255),
                    2,
                )
                if n_buf >= 25:
                    self._commit_face()

            cv2.imshow("face_classification", cv_image)
            cv2.waitKey(1)
        finally:
            self.processing = False


def main() -> None:
    rclpy.init(args=None)
    node = FaceClassificationPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
