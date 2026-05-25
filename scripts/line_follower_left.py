#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from irobot_create_msgs.msg import HazardDetectionVector

import tf2_ros
from tf2_ros import Buffer, TransformListener

from cv_bridge import CvBridge
import cv2
import numpy as np

FOLLOW    = 'follow'
TURN      = 'turn'
TURN_180  = 'turn_180'
DRIVE_OUT = 'drive_out'


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _image_angle_to_robot(theta_img: float) -> float:
    """Convert an angle measured in OpenCV image coords (atan2(dy, dx), y-down,
    so image-up = -π/2) to robot frame (forward = 0, left = +π/2, back = ±π,
    right = -π/2). The top-camera is rigidly mounted with image-up == robot-forward."""
    return _wrap_angle(-math.pi / 2.0 - theta_img)


class LineFollowerLeft(Node):
    """Reactive blue-line follower with a left-hand rule that handles
    arbitrary-shape crossroads (Y, T, +, asymmetric, n-way).

    Branch detection: angular histogram of line pixels in a ring around a
    fixed robot anchor in image space. Consecutive hot angular bins form
    a branch; the branch closest to image-down is the "back" we came from
    and is filtered out. Among the remaining branches, the one with the
    largest robot-frame angle (most-left) is chosen. If that branch is
    near forward (|angle| < forward_snap_rad) we stay in FOLLOW.

    Anti-spin defense: hysteresis (N consecutive junction frames) + time
    cooldown + map-frame spatial cooldown + forced DRIVE_OUT after every
    turn.

    On a dead end (no line in PID ROI) the robot spins ~180° and re-acquires
    the line backwards.
    """

    def __init__(self) -> None:
        super().__init__("line_follower_left")

        self.declare_parameters(
            namespace="",
            parameters=[
                # ── I/O topics ────────────────────────────────────────────────
                ("image_topic",                 "/top_camera/rgb/preview/image_raw"),  # downward-facing camera feed used to find the line
                ("cmd_vel_topic",               "/cmd_vel"),                            # where to publish drive commands (TwistStamped)
                ("hazard_topic",                "/hazard_detection"),                   # iRobot Create bump/hazard events; triggers TURN_180
                ("cmd_frame_id",                ""),                                    # frame_id stamped on cmd_vel (empty = leave blank)

                # ── Drive speeds ──────────────────────────────────────────────
                ("linear_speed",                0.115),  # forward cruise speed (m/s) when centred on the line
                ("angular_gain",                1.5),    # P-gain mapping centroid error → angular velocity
                ("max_angular_speed",           1.0),    # hard cap on |angular.z| (rad/s) for PID, TURN and TURN_180
                ("search_angular_speed",        0.3),    # in-place spin rate (rad/s) used while line is lost (FOLLOW dead-end probe)

                # ── Line color (HSV) ──────────────────────────────────────────
                ("hsv_lower",                   [90, 80, 50]),     # lower HSV bound for the blue line — must be a 3-element list
                ("hsv_upper",                   [130, 255, 255]),  # upper HSV bound; defaults span typical sim blue

                # ── PID ROI (vertical fraction of the frame) ──────────────────
                ("roi_y_start_ratio",           0.05),   # top edge of the PID centroid ROI as a fraction of image height
                ("roi_y_end_ratio",             0.9),    # bottom edge of the PID centroid ROI; tighten to ignore far-field clutter
                ("min_line_area",               500),    # min mask area (pixels) inside the ROI to count the line as "found"

                # ── Dead-end / junction hysteresis ────────────────────────────
                ("dead_end_frames",             15),     # consecutive line-lost frames in FOLLOW before triggering TURN_180
                ("crossroad_hysteresis_frames", 3),      # consecutive junction frames required before committing to a TURN

                # ── Branch-detection ring (full-image angular histogram) ──────
                ("robot_anchor_x_ratio",        0.5),    # anchor X (fraction of width) — where the "robot is" in image space
                ("robot_anchor_y_ratio",        0.95),   # anchor Y — pushed near the bottom so back = image-down
                ("ring_inner_ratio",            0.3),    # inner radius of the sampling ring as a fraction of min(w, h)
                ("ring_outer_ratio",            0.45),   # outer radius of the sampling ring as a fraction of min(w, h)
                ("angle_bins",                  36),     # number of angular histogram bins around the anchor (here 10° each)
                ("min_pixels_per_bin",          5),      # min line pixels in a bin for it to count as "hot" (part of a branch)
                ("back_tolerance_rad",          0.25),    # tolerance (rad ≈ 29°) within which a branch is treated as the back-pointing one and filtered out
                ("forward_snap_rad",            0.1),   # if the chosen branch is within this rad (≈ 14°) of straight ahead, stay in FOLLOW instead of TURN

                # ── Anti-spin guards around junctions ─────────────────────────
                ("crossroad_match_radius",      0.4),    # if the bot is within this many map-frame metres of the last junction, suppress re-firing
                ("crossroad_cooldown_sec",      4.0),    # time cooldown (s) after a junction before another can be committed
                ("drive_out_sec",               7.0),    # forced DRIVE_OUT duration (s) after every TURN, to push past the junction before re-checking
                ("turn_tolerance",              0.10),   # |yaw error| (rad) under which TURN is considered complete
                ("turn_180_settle_sec",         1.0),    # extra stationary settle time (s) at the end of TURN_180 before resuming forward motion

                # ── Debug ─────────────────────────────────────────────────────
                ("debug_show_mask",             True),   # show the OpenCV debug window with ring / branches / state overlay
            ],
        )

        self.bridge = CvBridge()
        self.processing = False

        self.state = FOLLOW
        self.state_start_t = time.time()
        self.target_yaw = None
        self.no_line_frames = 0
        self.crossroad_streak = 0
        self.last_crossroad_xy = None
        self.crossroad_cooldown_until = 0.0
        self.collision_detected = False

        # Set by _step_follow each frame for the debug renderer
        self._debug_chosen_idx = None
        self._debug_non_back_mask = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._update_params()
        self.cmd_pub = self.create_publisher(
            TwistStamped, self.cmd_vel_topic, 10)
        self.img_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        self.hazard_sub = self.create_subscription(
            HazardDetectionVector, self.hazard_topic, self._hazard_cb, 10)

        self.get_logger().info(
            f"LineFollowerLeft: camera={self.image_topic}  cmd={self.cmd_vel_topic}  "
            f"hazard={self.hazard_topic}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_params(self):
        gp = self.get_parameter
        self.image_topic              = gp("image_topic").value
        self.cmd_vel_topic            = gp("cmd_vel_topic").value
        self.hazard_topic             = gp("hazard_topic").value
        self.cmd_frame_id             = str(gp("cmd_frame_id").value)
        self.linear_speed             = float(gp("linear_speed").value)
        self.angular_gain             = float(gp("angular_gain").value)
        self.max_angular_speed        = float(gp("max_angular_speed").value)
        self.search_angular_speed     = float(gp("search_angular_speed").value)
        self.roi_y_start_ratio        = max(0.0, min(1.0, float(gp("roi_y_start_ratio").value)))
        self.roi_y_end_ratio          = max(0.0, min(1.0, float(gp("roi_y_end_ratio").value)))
        self.min_line_area            = int(gp("min_line_area").value)
        self.dead_end_frames          = int(gp("dead_end_frames").value)
        self.crossroad_hysteresis_frames = int(gp("crossroad_hysteresis_frames").value)
        self.robot_anchor_x_ratio     = float(gp("robot_anchor_x_ratio").value)
        self.robot_anchor_y_ratio     = float(gp("robot_anchor_y_ratio").value)
        self.ring_inner_ratio         = float(gp("ring_inner_ratio").value)
        self.ring_outer_ratio         = float(gp("ring_outer_ratio").value)
        self.angle_bins               = max(8, int(gp("angle_bins").value))
        self.min_pixels_per_bin       = int(gp("min_pixels_per_bin").value)
        self.back_tolerance_rad       = float(gp("back_tolerance_rad").value)
        self.forward_snap_rad         = float(gp("forward_snap_rad").value)
        self.crossroad_match_radius   = float(gp("crossroad_match_radius").value)
        self.crossroad_cooldown_sec   = float(gp("crossroad_cooldown_sec").value)
        self.drive_out_sec            = float(gp("drive_out_sec").value)
        self.turn_tolerance           = float(gp("turn_tolerance").value)
        self.turn_180_settle_sec      = max(0.0, float(gp("turn_180_settle_sec").value))
        self.debug_show_mask          = bool(gp("debug_show_mask").value)

        lo = gp("hsv_lower").value
        hi = gp("hsv_upper").value
        if len(lo) != 3 or len(hi) != 3:
            lo, hi = [90, 80, 50], [130, 255, 255]
        self.hsv_lower = np.array(lo, dtype=np.uint8)
        self.hsv_upper = np.array(hi, dtype=np.uint8)

    def _twist(self, linear: float, angular: float) -> TwistStamped:
        t = TwistStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        if self.cmd_frame_id:
            t.header.frame_id = self.cmd_frame_id
        t.twist.linear.x  = float(linear)
        t.twist.angular.z = float(angular)
        return t

    def _transition(self, new_state: str):
        self.state = new_state
        self.state_start_t = time.time()
        self.get_logger().info(f"[state] -> {new_state}")

    def _hazard_cb(self, msg: HazardDetectionVector):
        # type 1 == BUMP in irobot_create_msgs/HazardDetection. Latches True until
        # the image_callback consumes it and starts a TURN_180.
        if any(d.type == 1 for d in msg.detections):
            self.collision_detected = True

    def _robot_pose_map(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, yaw)

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    def _get_pid_mask(self, frame):
        """Lower-ROI HSV mask used for PID centroid tracking & line-present check."""
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

    def _full_mask_and_anchor(self, frame):
        """Full-frame HSV mask used for branch detection and debug viz."""
        h, w = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.erode(mask,  None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        cx = int(w * self.robot_anchor_x_ratio)
        cy = int(h * self.robot_anchor_y_ratio)
        rmin = self.ring_inner_ratio * min(w, h)
        rmax = self.ring_outer_ratio * min(w, h)
        return mask, (cx, cy), (rmin, rmax)

    def _detect_branches(self, full_mask, anchor, radii):
        """Return list of (robot_angle_rad, pixel_count, mean_image_angle).

        Builds an angular histogram of mask pixels within the [rmin, rmax]
        ring centred on `anchor`, groups consecutive bins exceeding
        `min_pixels_per_bin` into runs (handling wrap-around), and reports
        one branch per run with the circular-mean angle. The line itself,
        viewed end-on, lights up exactly one run; every additional run is
        an extra branch."""
        cx, cy = anchor
        rmin, rmax = radii
        ys, xs = np.where(full_mask > 0)
        if xs.size == 0:
            return []
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        r2 = dx * dx + dy * dy
        sel = (r2 >= rmin * rmin) & (r2 <= rmax * rmax)
        if not np.any(sel):
            return []
        theta = np.arctan2(dy[sel], dx[sel])  # image frame, [-π, π]

        n = self.angle_bins
        bin_width = 2.0 * math.pi / n
        idx = ((theta + math.pi) / bin_width).astype(np.int32)
        idx = np.clip(idx, 0, n - 1)
        counts = np.zeros(n, dtype=np.int32)
        np.add.at(counts, idx, 1)

        hot = counts >= self.min_pixels_per_bin
        if not np.any(hot):
            return []
        if np.all(hot):
            # Ring entirely lit — degenerate, treat as one omnidirectional branch
            return [(0.0, int(counts.sum()), 0.0)]

        # Find a cold bin so runs don't get split across the array boundary
        start = int(np.where(~hot)[0][0])
        runs = []
        cur = None
        for i in range(n):
            b = (start + i) % n
            if hot[b]:
                if cur is None:
                    cur = {'sum_cos': 0.0, 'sum_sin': 0.0, 'sum_n': 0}
                ang = -math.pi + (b + 0.5) * bin_width
                w = int(counts[b])
                cur['sum_cos'] += math.cos(ang) * w
                cur['sum_sin'] += math.sin(ang) * w
                cur['sum_n']   += w
            else:
                if cur is not None:
                    runs.append(cur)
                    cur = None
        if cur is not None:
            runs.append(cur)

        branches = []
        for run in runs:
            mean_img_ang = math.atan2(run['sum_sin'], run['sum_cos'])
            robot_ang = _image_angle_to_robot(mean_img_ang)
            branches.append((robot_ang, run['sum_n'], mean_img_ang))
        return branches

    # ------------------------------------------------------------------
    # Crossroad gating
    # ------------------------------------------------------------------

    def _cooldown_blocks_crossroad(self) -> bool:
        if time.time() < self.crossroad_cooldown_until:
            return True
        if self.last_crossroad_xy is not None:
            pose = self._robot_pose_map()
            if pose is not None:
                x, y, _ = pose
                lx, ly = self.last_crossroad_xy
                if math.hypot(x - lx, y - ly) < self.crossroad_match_radius:
                    return True
        return False

    def _arm_cooldowns(self):
        self.crossroad_cooldown_until = time.time() + self.crossroad_cooldown_sec
        pose = self._robot_pose_map()
        if pose is not None:
            self.last_crossroad_xy = (pose[0], pose[1])
        else:
            self.last_crossroad_xy = None
            self.get_logger().warn("TF unavailable at crossroad — spatial cooldown skipped")

    # ------------------------------------------------------------------
    # PID
    # ------------------------------------------------------------------

    def _pid_cmd(self, mask, width):
        m = cv2.moments(mask)
        if m["m00"] <= 0:
            return 0.0, self.search_angular_speed
        cx = int(m["m10"] / m["m00"])
        error_norm = (cx - width / 2.0) / (width / 2.0)
        angular = max(-self.max_angular_speed,
                      min(self.max_angular_speed,
                          -self.angular_gain * error_norm))
        linear = self.linear_speed * (1.0 - error_norm ** 2)
        return linear, angular

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

        pid_mask, width = self._get_pid_mask(frame)
        m = cv2.moments(pid_mask)
        line_found = m["m00"] > self.min_line_area
        elapsed = time.time() - self.state_start_t

        # Collision takes precedence over anything except an already-running
        # TURN_180 — we don't want to restart the recovery turn on every bump frame.
        if self.collision_detected and self.state != TURN_180:
            self.collision_detected = False
            self.get_logger().info("collision — TURN_180")
            self.cmd_pub.publish(self._twist(0.0, 0.0))
            self._transition(TURN_180)

        full_mask, anchor, radii = self._full_mask_and_anchor(frame)
        branches = self._detect_branches(full_mask, anchor, radii)
        # Mark which branches are non-back for the debug overlay
        non_back_flags = [
            abs(_wrap_angle(b[0] - math.pi)) > self.back_tolerance_rad
            for b in branches
        ]
        self._debug_non_back_mask = non_back_flags
        self._debug_chosen_idx = None

        if self.state == FOLLOW:
            self._step_follow(pid_mask, width, line_found, branches, non_back_flags)
        elif self.state == TURN:
            self._step_turn()
        elif self.state == TURN_180:
            self._step_turn_180(elapsed)
        elif self.state == DRIVE_OUT:
            self._step_drive_out(pid_mask, width, line_found, elapsed)

        if self.debug_show_mask:
            self._show_debug(full_mask, anchor, radii, branches)

        self.processing = False

    def _step_follow(self, pid_mask, width, line_found, branches, non_back_flags):
        # Dead-end via line absence
        if not line_found:
            self.no_line_frames += 1
            self.crossroad_streak = 0
            if self.no_line_frames >= self.dead_end_frames:
                self.no_line_frames = 0
                self.get_logger().info("dead end — TURN_180")
                self._transition(TURN_180)
            else:
                self.cmd_pub.publish(self._twist(0.0, self.search_angular_speed))
            return
        self.no_line_frames = 0

        if not self._cooldown_blocks_crossroad():
            non_back = [(i, b) for i, (b, ok) in enumerate(zip(branches, non_back_flags)) if ok]

            # Junction = at least 2 outgoing (non-back) branches
            if len(non_back) >= 2:
                self.crossroad_streak += 1
            else:
                self.crossroad_streak = 0

            if self.crossroad_streak >= self.crossroad_hysteresis_frames:
                self.crossroad_streak = 0
                # Most-left = largest robot-frame angle
                chosen_idx, (chosen_angle, chosen_pix, _) = max(
                    non_back, key=lambda ib: ib[1][0])
                self._debug_chosen_idx = chosen_idx

                pose = self._robot_pose_map()
                if pose is None:
                    self.get_logger().warn("TF unavailable at junction — deferring")
                    # Don't arm cooldowns; retry next frame
                else:
                    yaw = pose[2]
                    self.cmd_pub.publish(self._twist(0.0, 0.0))
                    self._arm_cooldowns()
                    deg = math.degrees(chosen_angle)
                    if abs(chosen_angle) < self.forward_snap_rad:
                        self.get_logger().info(
                            f"junction F (angle={deg:.0f}° px={chosen_pix} "
                            f"branches={len(non_back)}) -> stay FOLLOW")
                        # Fall through to PID
                    else:
                        self.target_yaw = _wrap_angle(yaw + chosen_angle)
                        side = "L" if chosen_angle > 0 else "R"
                        self.get_logger().info(
                            f"junction {side} (angle={deg:.0f}° px={chosen_pix} "
                            f"branches={len(non_back)}) -> TURN")
                        self._transition(TURN)
                        return

        linear, angular = self._pid_cmd(pid_mask, width)
        self.cmd_pub.publish(self._twist(linear, angular))

    def _step_turn(self):
        pose = self._robot_pose_map()
        if pose is None or self.target_yaw is None:
            self.target_yaw = None
            self.get_logger().warn("TF lost during TURN — falling back to FOLLOW")
            self._transition(FOLLOW)
            return
        _, _, yaw = pose
        err = _wrap_angle(self.target_yaw - yaw)
        if abs(err) < self.turn_tolerance:
            self.cmd_pub.publish(self._twist(0.0, 0.0))
            self.target_yaw = None
            self._transition(DRIVE_OUT)
            return
        ang = max(-self.max_angular_speed,
                  min(self.max_angular_speed, 2.5 * err))
        if abs(ang) < 0.2:
            ang = math.copysign(0.2, ang)
        self.cmd_pub.publish(self._twist(0.0, ang))

    def _step_turn_180(self, elapsed):
        # In-place rotate until ~π is covered, then hold stationary for a brief
        # settle so inertia stops and the camera stabilises before DRIVE_OUT
        # commands any forward motion.
        full_turn = math.pi / self.max_angular_speed
        settle_end = full_turn + self.turn_180_settle_sec
        if elapsed < full_turn:
            self.cmd_pub.publish(self._twist(0.0, self.max_angular_speed))
        elif elapsed < settle_end:
            self.cmd_pub.publish(self._twist(0.0, 0.0))
        else:
            self.cmd_pub.publish(self._twist(0.0, 0.0))
            self._transition(DRIVE_OUT)

    def _step_drive_out(self, pid_mask, width, line_found, elapsed):
        if elapsed >= self.drive_out_sec:
            self._transition(FOLLOW)
            return
        if line_found:
            linear, angular = self._pid_cmd(pid_mask, width)
            self.cmd_pub.publish(self._twist(linear, angular))
        else:
            self.cmd_pub.publish(self._twist(self.linear_speed, 0.0))

    # ------------------------------------------------------------------
    # Debug rendering
    # ------------------------------------------------------------------

    def _show_debug(self, full_mask, anchor, radii, branches):
        cx, cy = anchor
        rmin, rmax = radii
        bgr = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(bgr, (cx, cy), int(rmin), (0, 0, 255), 1)
        cv2.circle(bgr, (cx, cy), int(rmax), (0, 0, 255), 1)
        cv2.circle(bgr, (cx, cy), 4, (255, 255, 0), -1)

        # Draw each branch as a ray from anchor; chosen = yellow, back = grey, other = cyan
        nb_flags = self._debug_non_back_mask or [True] * len(branches)
        chosen_idx = self._debug_chosen_idx
        ray_r = int(rmax * 1.15)
        for i, (robot_ang, pix, img_ang) in enumerate(branches):
            ex = int(cx + ray_r * math.cos(img_ang))
            ey = int(cy + ray_r * math.sin(img_ang))
            if i == chosen_idx:
                color = (0, 255, 255)   # yellow = chosen
                thickness = 3
            elif not nb_flags[i]:
                color = (128, 128, 128) # grey = back (filtered)
                thickness = 1
            else:
                color = (255, 200, 0)   # cyan-ish = candidate
                thickness = 2
            cv2.line(bgr, (cx, cy), (ex, ey), color, thickness)
            cv2.putText(bgr, f"{math.degrees(robot_ang):+.0f}",
                        (ex + 5, ey + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cooldown_left = max(0.0, self.crossroad_cooldown_until - time.time())
        header = (f"state={self.state}  branches={len(branches)}  "
                  f"streak={self.crossroad_streak}  cd={cooldown_left:.1f}s")
        cv2.putText(bgr, header, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("LineFollowerLeft", bgr)
        cv2.waitKey(1)


def main():
    rclpy.init(args=None)
    node = LineFollowerLeft()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
