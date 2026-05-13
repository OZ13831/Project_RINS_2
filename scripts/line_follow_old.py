#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped

from cv_bridge import CvBridge
import cv2
import numpy as np


class LineFollower(Node):
    def __init__(self) -> None:
        super().__init__("line_follower")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("image_topic", "/top_camera/rgb/preview/image_raw"),
                ("cmd_vel_topic", "/cmd_vel"),
                ("cmd_frame_id", ""),
                ("linear_speed", 0.2),
                ("angular_gain", 0.004),
                ("max_angular_speed", 1.2),
                ("search_angular_speed", 0.3),
                ("min_line_area", 500),
                ("roi_y_start_ratio", 0.5),
                ("roi_y_end_ratio", 1.0),
                ("hsv_lower", [90, 80, 50]),
                ("hsv_upper", [130, 255, 255]),
            ],
        )

        self.image_topic = self.get_parameter("image_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.bridge = CvBridge()
        self.processing = False

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )

        self._update_params()

        self.get_logger().info(
            f"Line follower listening on {self.image_topic}, publishing to {self.cmd_vel_topic}."
        )

    def _clamp_ratio(self, value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    def _update_params(self) -> None:
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.search_angular_speed = float(self.get_parameter("search_angular_speed").value)
        self.min_line_area = int(self.get_parameter("min_line_area").value)
        self.roi_y_start_ratio = self._clamp_ratio(
            self.get_parameter("roi_y_start_ratio").value
        )
        self.roi_y_end_ratio = self._clamp_ratio(
            self.get_parameter("roi_y_end_ratio").value
        )
        self.cmd_frame_id = str(self.get_parameter("cmd_frame_id").value)

        lower = self.get_parameter("hsv_lower").value
        upper = self.get_parameter("hsv_upper").value
        if len(lower) != 3 or len(upper) != 3:
            lower = [90, 80, 50]
            upper = [130, 255, 255]
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)

    def _build_twist(self, linear_x: float, angular_z: float) -> TwistStamped:
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        if self.cmd_frame_id:
            twist.header.frame_id = self.cmd_frame_id
        twist.twist.linear.x = float(linear_x)
        twist.twist.angular.z = float(angular_z)
        return twist

    def image_callback(self, msg: Image) -> None:
        if self.processing:
            return

        self.processing = True
        self._update_params()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            self.processing = False
            return

        height, width = cv_image.shape[:2]
        y_start = int(height * self.roi_y_start_ratio)
        y_end = int(height * self.roi_y_end_ratio)
        if y_end <= y_start:
            y_start = int(height * 0.5)
            y_end = height

        roi = cv_image[y_start:y_end, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        moments = cv2.moments(mask)
        if moments["m00"] > self.min_line_area:
            cx = int(moments["m10"] / moments["m00"])
            error = cx - (width / 2.0)
            angular = -self.angular_gain * error
            angular = max(-self.max_angular_speed, min(self.max_angular_speed, angular))
            cmd = self._build_twist(self.linear_speed, angular)
        else:
            cmd = self._build_twist(0.0, self.search_angular_speed)

        self.cmd_pub.publish(cmd)
        self.processing = False


def main() -> None:
    rclpy.init(args=None)
    node = LineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
