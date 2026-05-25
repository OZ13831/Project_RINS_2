#!/usr/bin/env python3
"""Simple blue-line follower with a left-hand-rule bias.

Looks at the bottom strip of the top camera, computes the centroid of blue
pixels, then shifts the PID target rightward (equivalently, draws a virtual
"offsetted centroid" to the LEFT of the true centroid) so the robot hugs the
left edge of the line. At a Y/T junction, the centroid of the union of branches
drags toward the leftmost branch, so left-hand rule emerges naturally — no
explicit branch detection needed.

On a dead-end (line lost for N consecutive frames) or a bump collision
(HazardDetectionVector type=1), the robot spins ~180° open-loop, waits one
second, then resumes following.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from irobot_create_msgs.msg import HazardDetectionVector

from cv_bridge import CvBridge
import cv2
import numpy as np


FOLLOW   = 'follow'
TURN_180 = 'turn_180'
WAIT     = 'wait'


class LineFollowerNew(Node):
    def __init__(self) -> None:
        super().__init__("line_follower_new")

        self.declare_parameters(
            namespace="",
            parameters=[
                # I/O topics
                ("image_topic",            "/top_camera/rgb/preview/image_raw"),
                ("cmd_vel_topic",          "/cmd_vel"),
                ("hazard_topic",           "/hazard_detection"),
                ("cmd_frame_id",           ""),

                # Bottom-strip ROI — fraction of image height taken from the bottom.
                # Raise toward 0.4 to look further ahead; lower toward 0.1 for tight tracking.
                ("bottom_strip_ratio",     0.25),

                # Left-hand-rule offset: how many pixels the virtual centroid is
                # drawn to the left of the true centroid. Larger = stronger left bias.
                ("left_offset_px",         40),

                # Blue line HSV (matches the rest of the package's followers).
                ("hsv_lower",              [90, 80, 50]),
                ("hsv_upper",              [130, 255, 255]),

                # Drive
                ("linear_speed",           0.115),
                ("angular_gain",           1.5),
                ("max_angular_speed",      1.0),

                # Dead-end / recovery
                ("min_line_area",          200),
                ("dead_end_frames",        15),
                ("turn_180_angular_speed", 1.0),
                ("turn_180_duration_sec",  3.3),
                ("wait_after_turn_sec",    1.0),

                # Debug
                ("debug_show_mask",        True),
            ],
        )

        gp = self.get_parameter
        self.image_topic            = gp("image_topic").value
        self.cmd_vel_topic          = gp("cmd_vel_topic").value
        self.hazard_topic           = gp("hazard_topic").value
        self.cmd_frame_id           = gp("cmd_frame_id").value
        self.bottom_strip_ratio     = float(gp("bottom_strip_ratio").value)
        self.left_offset_px         = int(gp("left_offset_px").value)
        self.linear_speed           = float(gp("linear_speed").value)
        self.angular_gain           = float(gp("angular_gain").value)
        self.max_angular_speed      = float(gp("max_angular_speed").value)
        self.min_line_area          = int(gp("min_line_area").value)
        self.dead_end_frames        = int(gp("dead_end_frames").value)
        self.turn_180_angular_speed = float(gp("turn_180_angular_speed").value)
        self.turn_180_duration_sec  = float(gp("turn_180_duration_sec").value)
        self.wait_after_turn_sec    = float(gp("wait_after_turn_sec").value)
        self.debug_show_mask        = bool(gp("debug_show_mask").value)

        lo = gp("hsv_lower").value
        hi = gp("hsv_upper").value
        if len(lo) != 3 or len(hi) != 3:
            lo, hi = [90, 80, 50], [130, 255, 255]
        self.hsv_lower = np.array(lo, dtype=np.uint8)
        self.hsv_upper = np.array(hi, dtype=np.uint8)

        self.bridge = CvBridge()
        self.processing = False

        self.state = FOLLOW
        self.state_start_t = time.time()
        self.no_line_frames = 0
        self.collision_detected = False

        self.cmd_pub = self.create_publisher(
            TwistStamped, self.cmd_vel_topic, 10)
        self.img_sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, qos_profile_sensor_data)
        self.hazard_sub = self.create_subscription(
            HazardDetectionVector, self.hazard_topic, self._hazard_cb, 10)

        self.get_logger().info(
            f"line_follower_new up. image={self.image_topic} cmd={self.cmd_vel_topic} "
            f"hazard={self.hazard_topic} bottom_strip={self.bottom_strip_ratio} "
            f"left_offset_px={self.left_offset_px}")

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
        if any(d.type == 1 for d in msg.detections):
            self.collision_detected = True

    def _image_cb(self, msg: Image):
        if self.processing:
            return
        self.processing = True
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._step(bgr)
        except Exception as e:
            self.get_logger().error(f"image_cb: {e}")
        finally:
            self.processing = False

    def _step(self, bgr: np.ndarray):
        H, W = bgr.shape[:2]
        now = time.time()

        if self.state == FOLLOW:
            if self.collision_detected:
                self._transition(TURN_180)
                self.collision_detected = False
                self.no_line_frames = 0
                self.cmd_pub.publish(self._twist(0.0, self.turn_180_angular_speed))
                return

            strip_y0 = int(H * (1.0 - self.bottom_strip_ratio))
            strip = bgr[strip_y0:H, :]
            hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)

            m = cv2.moments(mask)
            cx = cy = None
            display_cx = None
            error_norm = 0.0

            if m["m00"] >= self.min_line_area:
                self.no_line_frames = 0
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                display_cx = cx - self.left_offset_px
                error_norm = (display_cx - W / 2.0) / (W / 2.0)
                angular = -self.angular_gain * error_norm
                angular = max(-self.max_angular_speed,
                              min(self.max_angular_speed, angular))
                linear = self.linear_speed * max(0.0, 1.0 - error_norm * error_norm)
                self.cmd_pub.publish(self._twist(linear, angular))
            else:
                self.no_line_frames += 1
                self.cmd_pub.publish(self._twist(0.0, 0.0))
                if self.no_line_frames >= self.dead_end_frames:
                    self._transition(TURN_180)
                    self.no_line_frames = 0

            if self.debug_show_mask:
                self._show_debug(mask, W, cx, cy, display_cx, error_norm)

        elif self.state == TURN_180:
            if now - self.state_start_t < self.turn_180_duration_sec:
                self.cmd_pub.publish(self._twist(0.0, self.turn_180_angular_speed))
            else:
                self.cmd_pub.publish(self._twist(0.0, 0.0))
                self._transition(WAIT)

        elif self.state == WAIT:
            self.cmd_pub.publish(self._twist(0.0, 0.0))
            if now - self.state_start_t >= self.wait_after_turn_sec:
                self._transition(FOLLOW)

    def _show_debug(self, mask, W, cx, cy, display_cx, error_norm):
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        h = vis.shape[0]

        cv2.line(vis, (W // 2, 0), (W // 2, h), (0, 255, 0), 1)

        if cx is not None:
            cv2.circle(vis, (int(cx), int(cy)), 4, (180, 180, 180), -1)

            dx = int(display_cx)
            dy = int(cy)
            cv2.circle(vis, (dx, dy), 8, (0, 0, 255), -1)
            cv2.line(vis, (dx - 12, dy), (dx + 12, dy), (0, 0, 255), 1)
            cv2.line(vis, (dx, dy - 12), (dx, dy + 12), (0, 0, 255), 1)

        cv2.putText(vis, f"state={self.state}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(vis, f"no_line={self.no_line_frames}/{self.dead_end_frames}",
                    (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if cx is not None:
            cv2.putText(vis,
                        f"cx={cx:.0f} off={self.left_offset_px} err={error_norm:+.2f}",
                        (5, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.putText(vis, "line lost",
                        (5, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.imshow("LineFollowerNew", vis)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNew()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
