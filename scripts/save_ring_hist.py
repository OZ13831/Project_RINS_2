#!/usr/bin/env python3
"""
save_ring_hist.py

Listens to the top camera depth stream, detects ring masks via HoughCircles
(same method as detect_rings.py), and saves an HSV-H histogram of the masked
RGB region whenever a String is published to /save.

Saved file: ~/colcon_ws/task2_hist/<msg>_<N>.npy
where N is the next unused index for that prefix.
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
import numpy as np

HIST_DIR = os.path.expanduser('~/colcon_ws/task2_hist')


def _decode_bgr(msg: Image) -> np.ndarray:
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding.lower() == 'rgb8':
        return raw[:, :, ::-1]
    return raw  # bgr8 / bgra8 — first three channels are BGR


def _decode_depth(msg: Image) -> np.ndarray:
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.dilate(cv2.erode(mask, kernel, iterations=1), kernel, iterations=1)


def _next_index(prefix: str) -> int:
    """Return the next unused integer suffix for files named <prefix>_N.npy."""
    existing = [
        f for f in os.listdir(HIST_DIR)
        if f.startswith(prefix + '_') and f.endswith('.npy')
    ]
    indices = []
    for f in existing:
        stem = f[len(prefix) + 1:-4]  # strip "prefix_" and ".npy"
        if stem.isdigit():
            indices.append(int(stem))
    return max(indices, default=0) + 1


class RingHistSaver(Node):

    def __init__(self):
        super().__init__('ring_hist_saver')

        os.makedirs(HIST_DIR, exist_ok=True)

        self._rgb: np.ndarray | None = None
        self._mask: np.ndarray | None = None

        self.create_subscription(Image, '/top_camera/rgb/preview/image_raw',
                                 self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/top_camera/rgb/preview/depth',
                                 self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(String, '/save', self._save_cb, 10)

        self.get_logger().info(f'Ready — publish to /save with a label to capture a histogram')
        self.get_logger().info(f'Saving to {HIST_DIR}')

    # ── Image subscribers ─────────────────────────────────────────────────────

    def _rgb_cb(self, msg: Image):
        try:
            self._rgb = _decode_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f'RGB decode failed: {e}', throttle_duration_sec=2.0)

    def _depth_cb(self, msg: Image):
        try:
            depth = _decode_depth(msg)
        except Exception as e:
            self.get_logger().warn(f'Depth decode failed: {e}', throttle_duration_sec=2.0)
            return

        depth[depth == np.inf] = 0

        img8 = depth / 65536.0 * 255
        if img8.max() > 0:
            img8 = img8 / img8.max() * 255
        img8 = img8.astype(np.uint8)

        circles = cv2.HoughCircles(
            img8, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=70, param2=20, minRadius=1, maxRadius=30,
        )

        if circles is None:
            self._mask = None
            return

        mask = np.zeros(img8.shape, dtype=np.uint8)
        for x, y, r in np.uint16(np.around(circles[0])):
            cv2.circle(mask, (x, y), r + 5, 255, -1)
            cv2.circle(mask, (x, y), max(0, r - 2), 0, -1)

        self._mask = _cleanup_mask(mask)

    # ── Save trigger ──────────────────────────────────────────────────────────

    def _save_cb(self, msg: String):
        label = msg.data.strip()
        if not label:
            self.get_logger().warn('Received empty label on /save — ignoring')
            return

        if self._rgb is None:
            self.get_logger().warn('No RGB frame yet — ignoring save request')
            return

        if self._mask is None or np.sum(self._mask) == 0:
            self.get_logger().warn('No ring mask available — ignoring save request')
            return

        mask = self._mask
        rgb = self._rgb

        # Resize mask if shapes differ (shouldn't normally happen)
        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]

        hist = cv2.calcHist([h_channel], [0], mask, [180], [0, 180])
        hist = cv2.normalize(hist, hist).flatten()

        idx = _next_index(label)
        path = os.path.join(HIST_DIR, f'{label}_{idx}.npy')
        np.save(path, hist)
        self.get_logger().info(f'Saved histogram → {path}')


def main():
    rclpy.init()
    rclpy.spin(RingHistSaver())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
