#!/usr/bin/env python3
"""
detect_spill2.py — vision-based spill detection using the arm's top_camera.

Strategy:
  1. Send the arm to 'look_for_spill' so the top_camera looks at the floor ahead.
  2. On every depth frame from /top_camera/rgb/preview/depth, run
     cv2.HoughCircles to find circular elevated features (the spill pool edge).
  3. Require CONFIRM_FRAMES consecutive hits before publishing True on
     /spill_detected, to suppress transient false positives during arm motion.

The arm command is sent once, 2 s after startup, to give the arm controller
time to come online.  detect_spill2 must run alongside arm_mover_actions.py.
"""

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
import cv2
import numpy as np


class SpillDetector2(Node):

    # ── HoughCircles knobs ────────────────────────────────────────────────────
    # Camera FOV ≈ 1.25 rad over 320 px → ~256 px/rad.
    # Spill radius ~0.2 m at camera distance ~0.5 m → ~26° → ~66 px.
    # Tune MIN/MAX_RADIUS if the spill appears bigger or smaller in the image.
    HOUGH_DP        = 1      # inverse accumulator resolution ratio
    HOUGH_MIN_DIST  = 40     # minimum px between detected circle centres
    HOUGH_PARAM1    = 60     # Canny high threshold
    HOUGH_PARAM2    = 22     # accumulator threshold (lower = more circles)
    MIN_RADIUS      = 15     # px — smallest circle to accept
    MAX_RADIUS      = 110    # px — largest circle to accept

    # ── Temporal stability ────────────────────────────────────────────────────
    CONFIRM_FRAMES = 5       # consecutive frames needed before publishing True

    # ── Arm startup delay ─────────────────────────────────────────────────────
    ARM_SEND_DELAY = 2.0     # seconds — wait for arm controller before commanding

    def __init__(self):
        super().__init__('spill_detector2')

        self._consec  = 0
        self._arm_sent = False

        # Publisher: spill result
        self.spill_pub = self.create_publisher(Bool, '/spill_detected', 10)

        # Publisher: arm command (arm_mover_actions.py must be running)
        self.arm_pub = self.create_publisher(String, '/arm_command', 1)

        # Subscriber: top_camera depth image
        self.create_subscription(
            Image,
            '/top_camera/rgb/preview/depth',
            self.depth_callback,
            qos_profile_sensor_data,
        )

        # One-shot timer to send the arm command after the controller is up
        self._arm_timer = self.create_timer(self.ARM_SEND_DELAY, self._send_arm_command)

        self.get_logger().info('SpillDetector2 started — waiting to command arm...')

    # ── Arm positioning ───────────────────────────────────────────────────────

    def _send_arm_command(self):
        msg = String()
        msg.data = 'look_for_spill'
        self.arm_pub.publish(msg)
        self._arm_sent = True
        self.get_logger().info('Arm commanded to look_for_spill')
        self._arm_timer.cancel()

    # ── Depth image processing ────────────────────────────────────────────────

    def depth_callback(self, msg: Image):
        try:
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        except Exception as e:
            self.get_logger().warn(f'image decode: {e}', throttle_duration_sec=2.0)
            return

        # Clean up invalid values
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        valid = depth[depth > 0]
        if valid.size == 0:
            return

        # Normalise to uint8.  Use 95th-percentile as the far end to avoid
        # stretching caused by a few outlier distant readings.
        d_far = float(np.percentile(valid, 95))
        if d_far < 1e-3:
            return

        norm = np.clip(depth / d_far, 0.0, 1.0)
        # Invert: closer (elevated spill) → brighter pixel
        img8 = ((1.0 - norm) * 255).astype(np.uint8)
        img8[depth == 0] = 0   # mask invalid pixels as black

        blurred = cv2.GaussianBlur(img8, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.HOUGH_DP,
            minDist=self.HOUGH_MIN_DIST,
            param1=self.HOUGH_PARAM1,
            param2=self.HOUGH_PARAM2,
            minRadius=self.MIN_RADIUS,
            maxRadius=self.MAX_RADIUS,
        )

        found = circles is not None

        if found:
            self._consec += 1
        else:
            self._consec = 0

        result = self._consec >= self.CONFIRM_FRAMES
        if result:
            self.get_logger().info('Spill detected', throttle_duration_sec=1.0)
        self.spill_pub.publish(Bool(data=result))


def main():
    rclpy.init()
    node = SpillDetector2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
