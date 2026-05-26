#!/usr/bin/env python3
"""
belt_calibrate.py — Calibrate BELT_MEDIAN_MAX for detect_tile_anomaly.

Subscribes to /top_camera/rgb/preview/image_raw and displays the central
ROI (matching BELT_ROI_FRACTION in detect_tile_anomaly.py). Tap labelled
keys to capture the median grey value at each position you care about.

Procedure (do this twice — once per belt):
  1. Drive the robot to the anomaly-cell standoff for the RED cell (the
     pose robot_commander navigates to). Tap `1` 3-5 times — captures
     "far" (floor in view) for the red approach.
  2. Hand-drive the robot forward (rqt teleop, joystick, whatever) until
     the belt fills the ROI as much as you'd want before STATE_APPROACH
     stops. Tap `2` 3-5 times — captures "near" (at-belt) for red.
  3. Repeat for the GREEN cell with `3` (far) and `4` (near).
  4. Press `q`. The summary table is printed AND written to
     /tmp/belt_calibration.txt — paste either one back.

Keys: 1/2 = red far/near, 3/4 = green far/near, q = quit.

Run:
  python3 src/dis_tutorial7/scripts/belt_calibrate.py
"""
from collections import defaultdict

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

# Must match BELT_ROI_FRACTION in detect_tile_anomaly.py
ROI_FRACTION = 0.60

LABELS = {
    ord('1'): 'red_far',
    ord('2'): 'red_near',
    ord('3'): 'green_far',
    ord('4'): 'green_near',
}
ORDER = ('red_far', 'red_near', 'green_far', 'green_near')
OUTPUT_PATH = '/tmp/belt_calibration.txt'
TOPIC = '/top_camera/rgb/preview/image_raw'


def _ros_image_to_bgr(msg: Image) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    enc = msg.encoding.lower()
    if enc in ('bgr8', 'bgra8'):
        return data[:, :, :3]
    if enc in ('rgb8', 'rgba8'):
        return data[:, :, 2::-1]
    if enc == 'mono8':
        return cv2.cvtColor(data[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f'Unsupported encoding: {msg.encoding}')


class BeltCalibrate(Node):

    def __init__(self):
        super().__init__('belt_calibrate')
        self._samples = defaultdict(list)
        self.create_subscription(Image, TOPIC, self._cb, qos_profile_sensor_data)
        cv2.namedWindow('Belt Calibrate', cv2.WINDOW_NORMAL)
        self.get_logger().info(
            'Ready. Drive the robot and tap 1/2/3/4 to label samples. Press q to quit.'
        )

    def _cb(self, msg: Image) -> None:
        bgr = _ros_image_to_bgr(msg)
        H, W = bgr.shape[:2]
        f = ROI_FRACTION
        rx0 = int(W * (1 - f) / 2); rx1 = W - rx0
        ry0 = int(H * (1 - f) / 2); ry1 = H - ry0

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        roi = blurred[ry0:ry1, rx0:rx1]
        median = int(np.median(roi))
        p10 = int(np.percentile(roi, 10))
        p90 = int(np.percentile(roi, 90))

        disp = bgr.copy()
        cv2.rectangle(disp, (rx0, ry0), (rx1 - 1, ry1 - 1), (0, 255, 255), 2)

        # Header: live readout.
        info = f'median={median}   p10={p10}   p90={p90}'
        cv2.rectangle(disp, (0, 0), (W, 32), (30, 30, 30), cv2.FILLED)
        cv2.putText(disp, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 220, 220), 1, cv2.LINE_AA)

        # Footer: bucket counts so you know each one has enough samples.
        counts = '   '.join(f'{lbl}={len(self._samples[lbl])}' for lbl in ORDER)
        cv2.rectangle(disp, (0, H - 24), (W, H), (30, 30, 30), cv2.FILLED)
        cv2.putText(disp, counts, (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Belt Calibrate', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._print_and_save()
            rclpy.shutdown()
            return
        if key in LABELS:
            lbl = LABELS[key]
            self._samples[lbl].append((median, p10, p90))
            print(f'[{lbl}] median={median} p10={p10} p90={p90} '
                  f'(bucket size: {len(self._samples[lbl])})')

    def _print_and_save(self) -> None:
        lines = []
        lines.append('================== BELT CALIBRATION ==================')
        lines.append(f'(ROI_FRACTION={ROI_FRACTION}, topic={TOPIC})')
        lines.append('')
        lines.append('| label      | n |   median   |    p10     |    p90     |')
        lines.append('|------------|---|------------|------------|------------|')
        for lbl in ORDER:
            s = self._samples.get(lbl, [])
            if not s:
                lines.append(f'| {lbl:10s} | 0 |     --     |     --     |     --     |')
                continue
            meds = [x[0] for x in s]
            p10s = [x[1] for x in s]
            p90s = [x[2] for x in s]
            lines.append(
                f'| {lbl:10s} | {len(s):1d} | '
                f'{int(np.mean(meds)):3d} ({min(meds):3d}-{max(meds):3d}) | '
                f'{int(np.mean(p10s)):3d} ({min(p10s):3d}-{max(p10s):3d}) | '
                f'{int(np.mean(p90s)):3d} ({min(p90s):3d}-{max(p90s):3d}) |'
            )
        lines.append('======================================================')
        text = '\n'.join(lines)
        print('\n' + text + '\n')
        try:
            with open(OUTPUT_PATH, 'w') as f:
                f.write(text + '\n')
            print(f'Saved to {OUTPUT_PATH}')
        except OSError as e:
            print(f'Could not save: {e}')


def main():
    rclpy.init()
    node = BeltCalibrate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._print_and_save()
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
