#!/usr/bin/env python3

import argparse
import os
import re
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class OakdImageCapture(Node):
    def __init__(self, name_prefix: str, image_dir: str) -> None:
        super().__init__("oakd_image_capture")

        self.name_prefix = name_prefix
        self.image_dir = image_dir
        self.bridge = CvBridge()
        self.saved = False

        self.image_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self.image_callback, qos_profile_sensor_data
        )

    def _next_filename(self) -> str:
        os.makedirs(self.image_dir, exist_ok=True)
        pattern = re.compile(rf"^{re.escape(self.name_prefix)}(\d+)$")
        max_index = 0
        for entry in os.listdir(self.image_dir):
            stem, _ext = os.path.splitext(entry)
            match = pattern.match(stem)
            if match:
                try:
                    max_index = max(max_index, int(match.group(1)))
                except ValueError:
                    continue
        next_index = max_index + 1
        return os.path.join(self.image_dir, f"{self.name_prefix}{next_index}.png")

    def image_callback(self, msg: Image) -> None:
        if self.saved:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")
            return

        file_path = self._next_filename()
        if cv2.imwrite(file_path, cv_image):
            self.get_logger().info(f"Saved image to {file_path}")
        else:
            self.get_logger().error(f"Failed to write image to {file_path}")
        self.saved = True
        self.destroy_subscription(self.image_sub)
        rclpy.shutdown()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a single OAK-D image.")
    parser.add_argument(
        "--name-prefix",
        required=True,
        help="Filename prefix for the saved image.",
    )
    parser.add_argument(
        "--image-dir",
        default="/home/gamma/colcon_ws/images",
        help="Directory to save images.",
    )
    filtered = rclpy.utilities.remove_ros_args(args=argv)
    return parser.parse_args(filtered[1:])


def main() -> None:
    rclpy.init(args=None)
    args = parse_args(sys.argv)
    node = OakdImageCapture(args.name_prefix, args.image_dir)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
