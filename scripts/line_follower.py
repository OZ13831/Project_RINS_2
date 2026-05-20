#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import Marker, MarkerArray
from irobot_create_msgs.msg import HazardDetectionVector

import tf2_ros
from tf2_ros import Buffer, TransformListener

from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time

# State machine states
FOLLOWING   = 'following'    # PID line follow
TURNING     = 'turning'      # rotate until yaw == target_heading (closed-loop on TF)
DRIVING_OUT = 'driving_out'  # drive straight to clear a junction before re-classifying
TURNING_180 = 'turning_180'  # open-loop 180° for cliff/obstacle/cutoff/dead-end/dead crossroad


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class LineFollower(Node):
    """
    DFS-based maze explorer.

    At every crossroad, the robot stops, samples the line mask in three robot-relative
    sectors (Forward / Left / Right — 'Back' is implicit, that's where we came from),
    and stores the unexplored absolute headings on a stack indexed by map-frame pose.
    It then turns to the highest-priority unexplored branch (Left > Forward > Right).

    On any dead-end / cliff / obstacle / line-cutoff event, it just spins 180° and
    follows the line back — the line itself is the breadcrumb. When it re-arrives at
    a known crossroad (pose within `crossroad_match_radius` of a stack entry), it pops
    the next unexplored branch and turns to it. When the entry has no branches left,
    it backtracks one level further.

    No homing-to-(x,y) controller is involved, so AMCL drift only has to be small
    enough to distinguish crossroads from one another — not to land on an absolute
    point.
    """

    def __init__(self) -> None:
        super().__init__("line_follower")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("image_topic",           "/top_camera/rgb/preview/image_raw"),
                ("cmd_vel_topic",         "/cmd_vel"),
                ("scan_topic",            "/scan"),
                ("cmd_frame_id",          ""),
                ("linear_speed",          0.1),
                ("angular_gain",          1.5),
                ("max_angular_speed",     1.0),
                ("search_angular_speed",  0.3),
                ("min_line_area",         500),
                ("roi_y_start_ratio",     0.5),
                ("roi_y_end_ratio",       1.0),
                ("hsv_lower",             [90, 80, 50]),
                ("hsv_upper",             [130, 255, 255]),
                ("obstacle_dist",         0.2),
                ("crossway_corner_threshold", 80),
                ("dead_end_frames",       20),
                ("hazard_topic",          "/hazard_detection"),
                # DFS branch detection / matching
                ("branch_pixels_min",       150),   # min line pixels in an F/L/R sector
                ("branch_waypoint_offset",  0.5),   # rviz marker offset from crossroad
                ("crossroad_match_radius",  0.5),   # "same crossroad" tolerance in m
                ("crossroad_cooldown_sec",  6.0),   # ignore crossway re-detect after handle
                ("turn_tolerance",          0.10),  # yaw error to leave TURNING (rad)
                ("crossway_exit_drive_sec", 1.5),   # straight-drive after a turn
                # Look-ahead cutoff
                ("lookahead_y_start_ratio", 0.5),
                ("lookahead_y_end_ratio",   0.85),
                ("lookahead_min_pixels",    100),
                ("line_cutoff_frames",      3),
            ],
        )

        self.image_topic  = self.get_parameter("image_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.scan_topic   = self.get_parameter("scan_topic").value
        self.hazard_topic = self.get_parameter("hazard_topic").value

        self.bridge     = CvBridge()
        self.processing = False

        # State machine
        self.state          = FOLLOWING
        self.state_start_t  = time.time()
        self.obstacle_ahead = False
        self.cliff_detected = False
        self.no_line_frames = 0
        self._line_cutoff_count = 0

        # DFS crossroad memory.
        # Each entry: {'pose': (x, y), 'branches': [absolute_heading_rad, ...]}.
        # Empty 'branches' = fully explored; entry stays in the stack so a re-visit
        # is recognised and triggers a further backtrack instead of re-pushing.
        self.crossroad_stack: list = []
        self.target_heading = None        # absolute heading set when entering TURNING
        self._crossroad_cooldown_until = 0.0  # wall-time suppression of crossway re-trigger

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # IO
        self.cmd_pub   = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.wp_pub    = self.create_publisher(MarkerArray, '/line_follower/waypoints', 10)
        self.img_sub   = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        self.scan_sub  = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data)
        self.hazard_sub = self.create_subscription(
            HazardDetectionVector, self.hazard_topic, self._hazard_cb, 10)

        self._update_params()
        self.get_logger().info(
            f"LineFollower: camera={self.image_topic}  "
            f"cmd={self.cmd_vel_topic}  scan={self.scan_topic}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_params(self):
        gp = self.get_parameter
        self.linear_speed         = float(gp("linear_speed").value)
        self.angular_gain         = float(gp("angular_gain").value)
        self.max_angular_speed    = float(gp("max_angular_speed").value)
        self.search_angular_speed = float(gp("search_angular_speed").value)
        self.min_line_area        = int(gp("min_line_area").value)
        self.roi_y_start_ratio    = max(0.0, min(1.0, float(gp("roi_y_start_ratio").value)))
        self.roi_y_end_ratio      = max(0.0, min(1.0, float(gp("roi_y_end_ratio").value)))
        self.cmd_frame_id         = str(gp("cmd_frame_id").value)
        self.obstacle_dist        = float(gp("obstacle_dist").value)
        self.crossway_corner_threshold = int(gp("crossway_corner_threshold").value)
        self.dead_end_frames      = int(gp("dead_end_frames").value)
        self.branch_pixels_min       = int(gp("branch_pixels_min").value)
        self.branch_waypoint_offset  = float(gp("branch_waypoint_offset").value)
        self.crossroad_match_radius  = float(gp("crossroad_match_radius").value)
        self.crossroad_cooldown_sec  = float(gp("crossroad_cooldown_sec").value)
        self.turn_tolerance          = float(gp("turn_tolerance").value)
        self.crossway_exit_drive_sec = float(gp("crossway_exit_drive_sec").value)
        self.lookahead_y_start_ratio = max(0.0, min(1.0, float(gp("lookahead_y_start_ratio").value)))
        self.lookahead_y_end_ratio   = max(0.0, min(1.0, float(gp("lookahead_y_end_ratio").value)))
        self.lookahead_min_pixels    = int(gp("lookahead_min_pixels").value)
        self.line_cutoff_frames      = int(gp("line_cutoff_frames").value)

        lower = gp("hsv_lower").value
        upper = gp("hsv_upper").value
        if len(lower) != 3 or len(upper) != 3:
            lower, upper = [90, 80, 50], [130, 255, 255]
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)

    def _twist(self, linear: float, angular: float) -> TwistStamped:
        t = TwistStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        if self.cmd_frame_id:
            t.header.frame_id = self.cmd_frame_id
        t.twist.linear.x  = float(linear)
        t.twist.angular.z = float(angular)
        return t

    def _transition(self, new_state: str):
        self.state         = new_state
        self.state_start_t = time.time()
        self.get_logger().info(f"[state] -> {new_state}")

    # ------------------------------------------------------------------
    # LaserScan
    # ------------------------------------------------------------------

    def scan_callback(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        step      = (msg.angle_max - msg.angle_min) / max(n - 1, 1)
        half_cone = int(math.radians(30) / step)
        front = list(msg.ranges[:half_cone]) + list(msg.ranges[max(0, n - half_cone):])
        valid = [r for r in front if msg.range_min < r < msg.range_max]
        self.obstacle_ahead = bool(valid and min(valid) < self.obstacle_dist)

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    def _get_mask(self, frame):
        h, w = frame.shape[:2]
        y0 = int(h * self.roi_y_start_ratio)
        y1 = int(h * self.roi_y_end_ratio)
        if y1 <= y0:
            y0, y1 = int(h * 0.5), h
        roi  = frame[y0:y1, :]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask,  None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask, w

    def _count_lookahead_pixels(self, frame) -> int:
        h = frame.shape[0]
        y0 = int(h * self.lookahead_y_start_ratio)
        y1 = int(h * self.lookahead_y_end_ratio)
        if y1 <= y0:
            return 0
        hsv = cv2.cvtColor(frame[y0:y1, :], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return int(np.sum(mask > 0))

    def _classify_junction(self, mask, width: int):
        """Returns 'crossway' or None — strong Harris response area heuristic."""
        if np.sum(mask > 0) < self.min_line_area:
            return None
        resp = cv2.cornerHarris(np.float32(mask), 5, 3, 0.04)
        if int(np.sum(resp > 0.01 * resp.max())) > self.crossway_corner_threshold:
            return 'crossway'
        return None

    def _detect_branches_local(self, frame):
        """Return robot-relative branch directions present at the current crossroad,
        a subset of {'F', 'L', 'R'}. 'B' (where we came from) is omitted since DFS
        backtracks via TURNING_180 + line-follow, not via the stack."""
        h, w = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        full = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        # Sectors are picked so they correspond to F/L/R as a top-down camera sees them.
        regions = {
            'F': full[int(h * 0.20):int(h * 0.50), int(w * 0.35):int(w * 0.65)],
            'L': full[int(h * 0.50):int(h * 0.80), int(w * 0.00):int(w * 0.30)],
            'R': full[int(h * 0.50):int(h * 0.80), int(w * 0.70):int(w * 1.00)],
        }
        return [d for d, m in regions.items() if int(np.sum(m > 0)) > self.branch_pixels_min]

    # ------------------------------------------------------------------
    # Hazard + TF
    # ------------------------------------------------------------------

    def _hazard_cb(self, msg: HazardDetectionVector):
        self.cliff_detected = any(d.type == 2 for d in msg.detections)

    def _robot_pose_map(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, yaw)

    # ------------------------------------------------------------------
    # DFS crossroad logic
    # ------------------------------------------------------------------

    def _select_branch(self, frame):
        """At a detected crossway, decide what to do.

        Returns an absolute heading (rad) to turn the robot to, or None to backtrack
        (caller should transition to TURNING_180). Side effects: pushes a new stack
        entry on first visit, pops a branch on re-visit, arms the re-detect cooldown."""
        pose = self._robot_pose_map()
        if pose is None:
            self.get_logger().warn('TF unavailable at crossroad — backtracking')
            return None
        x, y, yaw = pose

        # Re-visit? (any stack entry within match radius)
        for entry in self.crossroad_stack:
            ex, ey = entry['pose']
            if math.hypot(ex - x, ey - y) < self.crossroad_match_radius:
                if entry['branches']:
                    target = entry['branches'].pop()
                    self.get_logger().info(
                        f'Re-entering crossroad ({x:.2f},{y:.2f}); '
                        f'{len(entry["branches"])} branches left; '
                        f'turning to {math.degrees(target):.0f}°'
                    )
                    self._publish_waypoint_markers(entry['pose'], entry['branches'] + [target])
                    self._arm_cooldown()
                    return target
                # Empty — already explored fully, signal further backtrack
                self.get_logger().info(
                    f'Crossroad ({x:.2f},{y:.2f}) already fully explored — backtracking'
                )
                return None

        # First visit: detect branches, push, pick the highest-priority one
        local = self._detect_branches_local(frame)
        if not local:
            self.get_logger().info(
                f'Crossroad at ({x:.2f},{y:.2f}) — no branches visible — backtracking'
            )
            return None

        # Priority: Left > Forward > Right. We pop from the tail, so build the list
        # so that L sits last (popped first).
        priority_push_order = ['R', 'F', 'L']
        headings = {
            'F': _wrap_angle(yaw),
            'L': _wrap_angle(yaw + math.pi / 2.0),
            'R': _wrap_angle(yaw - math.pi / 2.0),
        }
        ordered = [headings[d] for d in priority_push_order if d in local]
        target = ordered.pop()
        # Add an entry even when ordered is empty — preserves the "already visited"
        # marker so a future re-arrival can backtrack instead of re-pushing.
        self.crossroad_stack.append({'pose': (x, y), 'branches': ordered})

        self.get_logger().info(
            f'New crossroad ({x:.2f},{y:.2f}) branches={local} '
            f'turning to {math.degrees(target):.0f}° stack={len(self.crossroad_stack)}'
        )
        self._publish_waypoint_markers((x, y), ordered + [target])
        self._arm_cooldown()
        return target

    def _arm_cooldown(self):
        self._crossroad_cooldown_until = time.time() + self.crossroad_cooldown_sec

    def _publish_waypoint_markers(self, crossroad_xy, branch_headings):
        cx, cy = crossroad_xy
        off = self.branch_waypoint_offset
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        center = Marker()
        center.header.frame_id = 'map'
        center.header.stamp = stamp
        center.ns = 'crossroad'
        center.id = 0
        center.type = Marker.SPHERE
        center.action = Marker.ADD
        center.pose.position.x, center.pose.position.y = float(cx), float(cy)
        center.pose.orientation.w = 1.0
        center.scale.x = center.scale.y = center.scale.z = 0.10
        center.color.r, center.color.a = 1.0, 1.0
        arr.markers.append(center)

        for i, h in enumerate(branch_headings):
            ex = cx + off * math.cos(h)
            ey = cy + off * math.sin(h)
            sphere = Marker()
            sphere.header = center.header
            sphere.ns = 'branch'
            sphere.id = i + 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y = float(ex), float(ey)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.12
            sphere.color.g, sphere.color.a = 1.0, 1.0
            arr.markers.append(sphere)
        self.wp_pub.publish(arr)

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def image_callback(self, msg: Image):
        if self.processing:
            return
        self.processing = True
        self._update_params()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            self.processing = False
            return

        mask, width = self._get_mask(frame)
        m           = cv2.moments(mask)
        line_found  = m["m00"] > self.min_line_area
        far_pixels  = self._count_lookahead_pixels(frame)
        elapsed     = time.time() - self.state_start_t

        cv2.imshow("Line Mask", mask)
        cv2.waitKey(1)

        # ---- FOLLOWING ------------------------------------------------
        if self.state == FOLLOWING:
            cooldown_active = time.time() < self._crossroad_cooldown_until
            classification = self._classify_junction(mask, width) if line_found else None

            if self.obstacle_ahead or self.cliff_detected:
                reason = 'cliff' if self.cliff_detected else 'obstacle'
                self.cliff_detected = False
                self.get_logger().info(f'TURNING_180 reason={reason}')
                self._transition(TURNING_180)
                self.cmd_pub.publish(self._twist(0.0, 0.0))

            elif not line_found:
                self.no_line_frames += 1
                if self.no_line_frames >= self.dead_end_frames:
                    self.no_line_frames = 0
                    self.get_logger().info('TURNING_180 reason=dead_end')
                    self._transition(TURNING_180)
                else:
                    self.cmd_pub.publish(self._twist(0.0, self.search_angular_speed))

            elif classification == 'crossway' and not cooldown_active:
                self.no_line_frames = 0
                self._line_cutoff_count = 0
                target = self._select_branch(frame)
                if target is None:
                    # Either dead crossroad (no branches) or fully-explored — backtrack
                    self._transition(TURNING_180)
                else:
                    self.target_heading = target
                    self._transition(TURNING)
                self.cmd_pub.publish(self._twist(0.0, 0.0))

            elif far_pixels < self.lookahead_min_pixels:
                self._line_cutoff_count += 1
                if self._line_cutoff_count >= self.line_cutoff_frames:
                    self._line_cutoff_count = 0
                    self.get_logger().info(f'TURNING_180 reason=line_cutoff far={far_pixels}')
                    self._transition(TURNING_180)
                self.cmd_pub.publish(self._twist(0.0, 0.0))

            else:
                # Plain PID line follow
                self.no_line_frames = 0
                self._line_cutoff_count = 0
                cx = int(m["m10"] / m["m00"])
                error_norm = (cx - width / 2.0) / (width / 2.0)
                angular = max(-self.max_angular_speed,
                              min(self.max_angular_speed, -self.angular_gain * error_norm))
                linear = self.linear_speed * (1.0 - error_norm ** 2)
                self.cmd_pub.publish(self._twist(linear, angular))

        # ---- TURNING (TF-closed-loop yaw to target_heading) ----------
        elif self.state == TURNING:
            pose = self._robot_pose_map()
            if pose is None or self.target_heading is None:
                # TF dropped — bail to following; PID will recover heading via the line
                self.target_heading = None
                self._transition(FOLLOWING)
            else:
                _, _, yaw = pose
                err = _wrap_angle(self.target_heading - yaw)
                if abs(err) < self.turn_tolerance:
                    # Explicit zero-velocity so inertia doesn't carry past the target
                    self.cmd_pub.publish(self._twist(0.0, 0.0))
                    self.target_heading = None
                    self._transition(DRIVING_OUT)
                else:
                    # P-control with a floor speed so we don't stall on small errors.
                    # Gain 2.5: saturates max_angular_speed at err ≈ 0.4 rad (~23°)
                    # and brakes smoothly below that, eliminating overshoot.
                    ang = max(-self.max_angular_speed,
                              min(self.max_angular_speed, 2.5 * err))
                    if abs(ang) < 0.2:
                        ang = math.copysign(0.2, ang)
                    self.cmd_pub.publish(self._twist(0.0, ang))

        # ---- DRIVING_OUT (PID-follow through the junction grace window) ----
        elif self.state == DRIVING_OUT:
            if elapsed >= self.crossway_exit_drive_sec:
                self._transition(FOLLOWING)
            elif line_found:
                # Line PID while transiting the junction — this is what keeps the
                # robot on the new branch when the turn was slightly off-target.
                cx = int(m["m10"] / m["m00"])
                error_norm = (cx - width / 2.0) / (width / 2.0)
                angular = max(-self.max_angular_speed,
                              min(self.max_angular_speed, -self.angular_gain * error_norm))
                linear = self.linear_speed * (1.0 - error_norm ** 2)
                self.cmd_pub.publish(self._twist(linear, angular))
            else:
                # Line briefly out of view post-turn — creep forward, PID will
                # re-acquire on the next frame.
                self.cmd_pub.publish(self._twist(self.linear_speed, 0.0))

        # ---- TURNING_180 (emergency / backtrack) ----------------------
        elif self.state == TURNING_180:
            # Open-loop full π. On exit, FOLLOWING resumes and the line takes us
            # back — the line is the breadcrumb, no homing controller needed.
            full_turn = math.pi / self.max_angular_speed
            if elapsed >= full_turn:
                self._transition(FOLLOWING)
            else:
                self.cmd_pub.publish(self._twist(0.0, self.max_angular_speed))

        self.processing = False


def main():
    rclpy.init(args=None)
    node = LineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
