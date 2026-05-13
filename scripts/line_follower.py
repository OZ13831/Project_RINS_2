#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped

from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time

# State machine states
FOLLOWING   = 'following'   # Normal PID line following
CROSSWAY    = 'crossway'    # Turning left 90° at a detected junction
TURNING_180 = 'turning_180' # Spinning 180° after obstacle or dead end


class LineFollower(Node):
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
                ("roi_y_start_ratio",     0.85),
                ("roi_y_end_ratio",       1.0),
                ("hsv_lower",             [90, 80, 50]),
                ("hsv_upper",             [130, 255, 255]),
                # Stop and spin 180° when obstacle closer than this (meters)
                ("obstacle_dist",         0.2),
                # Crossway: minimum number of strong Harris corner pixels to be a junction
                ("crossway_corner_threshold", 70),
                # Consecutive frames without any line before declaring a dead end
                ("dead_end_frames",       20),
            ],
        )

        self.image_topic   = self.get_parameter("image_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.scan_topic    = self.get_parameter("scan_topic").value

        self.bridge     = CvBridge()
        self.processing = False

        # State machine
        self.state          = FOLLOWING
        self.state_start_t  = time.time()
        self.obstacle_ahead = False
        self.no_line_frames = 0  # consecutive frames with no visible line

        self.cmd_pub  = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.img_sub  = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data)

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
    # LaserScan: check for obstacles in the frontal 60° cone
    # ------------------------------------------------------------------

    def scan_callback(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        step      = (msg.angle_max - msg.angle_min) / max(n - 1, 1)
        half_cone = int(math.radians(30) / step)  # 30° on each side of front

        # In most scan conventions index 0 is the front; last indices wrap around
        front = list(msg.ranges[:half_cone]) + list(msg.ranges[max(0, n - half_cone):])
        valid = [r for r in front if msg.range_min < r < msg.range_max]
        self.obstacle_ahead = bool(valid and min(valid) < self.obstacle_dist)

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------

    def _get_mask(self, frame):
        """Crop the configured ROI and return a binary line mask and frame width."""
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

    def _is_crossway(self, mask, width: int) -> bool:
        """
        Detect a junction using Harris corner detection on the line mask.
        A T-junction or crossway produces strong corner responses where lines meet.
        A plain straight line produces almost no interior corners.
        Tune crossway_corner_threshold: raise it if straight curves false-trigger,
        lower it if real junctions are missed.
        """
        if np.sum(mask > 0) < self.min_line_area:
            return False
        # cornerHarris requires float32
        response = cv2.cornerHarris(np.float32(mask), blockSize=5, ksize=3, k=0.04)
        # Count pixels with a strong corner response (>1% of peak)
        strong_corner_pixels = int(np.sum(response > 0.01 * response.max()))
        return strong_corner_pixels > self.crossway_corner_threshold

    # ------------------------------------------------------------------
    # Main image callback — runs the state machine
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
        m          = cv2.moments(mask)
        line_found = m["m00"] > self.min_line_area
        elapsed    = time.time() - self.state_start_t

        cv2.imshow("Line Mask", mask)
        cv2.waitKey(1)

        # ---- FOLLOWING ------------------------------------------------
        if self.state == FOLLOWING:

            if self.obstacle_ahead:
                # Something is blocking the path — stop and turn around
                self._transition(TURNING_180)
                self.cmd_pub.publish(self._twist(0.0, 0.0))

            elif not line_found:
                # Spin slowly; after dead_end_frames with no line → dead end
                self.no_line_frames += 1
                if self.no_line_frames >= self.dead_end_frames:
                    self.no_line_frames = 0
                    self._transition(TURNING_180)
                else:
                    self.cmd_pub.publish(self._twist(0.0, self.search_angular_speed))

            elif self._is_crossway(mask, width):
                # Junction detected: stop and then turn left to explore left branch first
                self.no_line_frames = 0
                self._transition(CROSSWAY)
                self.cmd_pub.publish(self._twist(0.0, 0.0))

            else:
                # Normal line following: steer toward line centroid
                self.no_line_frames = 0
                cx = int(m["m10"] / m["m00"])
                # Normalize error to [-1, 1] so the gain is resolution-independent
                error_norm = (cx - width / 2.0) / (width / 2.0)
                angular = max(-self.max_angular_speed,
                              min(self.max_angular_speed, -self.angular_gain * error_norm))
                # Scale forward speed down quadratically with error:
                # centered → full speed, at image edge → stopped, only turning
                linear = self.linear_speed * (1.0 - error_norm ** 2)
                self.cmd_pub.publish(self._twist(linear, angular))

        # ---- CROSSWAY -------------------------------------------------
        elif self.state == CROSSWAY:
            # Turn left (positive angular_z = CCW = left in ROS convention) by 90°
            # Duration = (π/2) / angular_speed.  Tune max_angular_speed if the turn
            # overshoots or undershoots (real robots may not match commanded speed).
            if elapsed >= (math.pi / 4.0) / self.max_angular_speed:
                self._transition(FOLLOWING)
            else:
                self.cmd_pub.publish(self._twist(0.0, self.max_angular_speed))

        # ---- TURNING_180 ----------------------------------------------
        elif self.state == TURNING_180:
            full_turn  = math.pi / self.max_angular_speed
            # Don't exit early until past 90° — prevents immediately re-seeing
            # the dead-end line we just turned away from
            past_90    = elapsed >= (math.pi / 2.0) / self.max_angular_speed
            if elapsed >= full_turn or (past_90 and line_found):
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
