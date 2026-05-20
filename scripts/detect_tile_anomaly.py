#!/opt/ultralytics/bin/python3
"""
detect_tile_anomaly.py — tile anomaly detection via SuperSimpleNet + top_camera.

Pipeline per frame:
  1. Subscribe to /top_camera/rgb/preview/image_raw.
  2. Threshold the dark background to isolate the bright tile region.
  3. Find the tile's square boundary via contour approximation, validated
     and refined with HoughLinesP on Canny edges.
  4. Perspective-warp the tile ROI to a square crop.
  5. Run SuperSimpleNet inference → sigmoid(score) > ANOMALY_THRESHOLD.
  6. Publish Bool on /tile_anomaly (True = anomaly present).
"""

import os
import sys
sys.path.insert(0, '/home/gamma/colcon_ws/SuperSimpleNet')

# Ensure cv2.imshow has a display to draw on
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

import cv2
import numpy as np
import torch

from model.supersimplenet import SuperSimpleNet

# ── Constants ─────────────────────────────────────────────────────────────────

WEIGHTS_PATH = '/home/gamma/colcon_ws/SuperSimpleNet/weights/weights.pt'

IMAGE_SIZE = (256, 256)   # (H, W) expected by the loaded SSN weights

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# SuperSimpleNet model config (must match the training run for weights.pt)
SSN_CONFIG = {
    'backbone':       'wide_resnet50_2',
    'layers':         ['layer2', 'layer3'],
    'patch_size':     3,
    'adapt_cls_feat': False,
    # training-only keys (not used at inference, but required by __init__)
    'noise':          False,
    'noise_std':      0.015,
    'stop_grad':      False,
    'perlin':         False,
    'perlin_thr':     0.5,
    'no_anomaly':     'none',
    'bad':            False,
    'overlap':        False,
    'gamma':          0.4,
    'epochs':         160,
    'adapt_lr':       1e-3,
    'seg_lr':         1e-3,
    'dec_lr':         1e-3,
}


class TileAnomalyDetector(Node):

    # ── Tunable detection parameters ──────────────────────────────────────────

    # Tile must cover at least this fraction of the frame to count
    MIN_TILE_FRACTION = 0.05

    # Poly approximation: epsilon = this * arc length
    POLY_EPSILON_RATIO = 0.02

    # HoughLinesP params for edge validation on the tile boundary
    HOUGH_RHO            = 1
    HOUGH_THETA          = np.pi / 180
    HOUGH_THRESHOLD      = 50
    HOUGH_MIN_LINE_LEN   = 30
    HOUGH_MAX_LINE_GAP   = 20

    # Anomaly decision threshold on sigmoid(score) output
    ANOMALY_THRESHOLD = 0.5

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('tile_anomaly_detector')

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self._device}')

        self._model = self._load_model()

        self._pub = self.create_publisher(Bool, '/tile_anomaly', 10)

        self.create_subscription(
            Image,
            '/top_camera/rgb/preview/image_raw',
            self._image_callback,
            qos_profile_sensor_data,
        )

        # Open the window immediately so it's visible before any frame arrives
        cv2.namedWindow('Tile Anomaly Detector', cv2.WINDOW_NORMAL)
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'Waiting for camera...',
                    (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.imshow('Tile Anomaly Detector', placeholder)
        cv2.waitKey(1)

        self.get_logger().info('TileAnomalyDetector ready.')

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> SuperSimpleNet:
        model = SuperSimpleNet(image_size=IMAGE_SIZE, config=SSN_CONFIG)
        model.load_model(WEIGHTS_PATH)
        model.to(self._device)
        model.eval()
        return model

    # ── Image callback ────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image) -> None:
        display = None
        try:
            bgr = _ros_image_to_bgr(msg)
            display = bgr.copy()

            result = self._find_tile(bgr)

            if result is None:
                self.get_logger().info('No tile found.', throttle_duration_sec=2.0)
                self._pub.publish(Bool(data=False))
                _put_label(display, 'No tile', score=None, anomaly=False)
            else:
                roi, corners = result
                score = self._infer(roi)
                anomaly = bool(score > self.ANOMALY_THRESHOLD)

                self.get_logger().info(
                    f'score={score:.3f}  result={"ANOMALY" if anomaly else "OK"}',
                    throttle_duration_sec=1.0,
                )
                self._pub.publish(Bool(data=anomaly))

                color = (0, 0, 220) if anomaly else (0, 200, 0)
                cv2.polylines(display, [corners.astype(np.int32)],
                              isClosed=True, color=color, thickness=3)
                _put_label(display, 'ANOMALY' if anomaly else 'No anomaly', score, anomaly)

        except Exception as exc:
            self.get_logger().error(f'Callback error: {exc}', throttle_duration_sec=2.0)

        finally:
            if display is not None:
                cv2.imshow('Tile Anomaly Detector', display)
            cv2.waitKey(1)

    # ── Tile extraction ───────────────────────────────────────────────────────

    def _find_tile(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Locate the square tile on the dark background.

        Returns (warped_roi, corners) where corners is a (4, 2) int32 array
        in the original frame's coordinate space, or None if no tile is found.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Step 1: blur then Otsu-binarize to separate tile from background
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean up small holes and specks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

        # Step 2: find the largest connected component (the tile)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        if num_labels < 2:
            return None

        # Label 0 is background; pick the non-background label with maximum area
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        tile_area = int(stats[largest_label, cv2.CC_STAT_AREA])
        frame_area = float(bgr.shape[0] * bgr.shape[1])

        if tile_area / frame_area < self.MIN_TILE_FRACTION:
            return None

        # Extract contour from the isolated component mask
        mask = (labels == largest_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = contours[0]

        # Step 3: approximate contour as a quadrilateral
        peri   = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, self.POLY_EPSILON_RATIO * peri, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            # Fallback: run HoughLinesP on Canny edges to find the 4 sides,
            # then derive corners from the bounding rect of the tile mask.
            edges = cv2.Canny(mask, 50, 150)
            lines = cv2.HoughLinesP(
                edges,
                self.HOUGH_RHO,
                self.HOUGH_THETA,
                threshold=self.HOUGH_THRESHOLD,
                minLineLength=self.HOUGH_MIN_LINE_LEN,
                maxLineGap=self.HOUGH_MAX_LINE_GAP,
            )
            x, y, w, h = cv2.boundingRect(largest)

            if lines is not None:
                pts = lines.reshape(-1, 4)
                xs = np.concatenate([pts[:, 0], pts[:, 2]])
                ys = np.concatenate([pts[:, 1], pts[:, 3]])
                x  = int(np.clip(xs.min(), 0, bgr.shape[1]))
                y  = int(np.clip(ys.min(), 0, bgr.shape[0]))
                x2 = int(np.clip(xs.max(), 0, bgr.shape[1]))
                y2 = int(np.clip(ys.max(), 0, bgr.shape[0]))
                w, h = x2 - x, y2 - y

            corners = np.float32([
                [x,     y    ],
                [x + w, y    ],
                [x + w, y + h],
                [x,     y + h],
            ])

        # Step 4: order corners and perspective-warp to a square crop
        corners = _order_corners(corners)
        side    = max(IMAGE_SIZE)
        dst     = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
        M       = cv2.getPerspectiveTransform(corners, dst)
        warped  = cv2.warpPerspective(bgr, M, (side, side))

        return warped, corners

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self, roi_bgr: np.ndarray) -> float:
        """Return sigmoid(anomaly_score) in [0, 1]."""
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_rgb = cv2.resize(roi_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

        tensor = _to_tensor(roi_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            _, score = self._model(tensor)

        return float(torch.sigmoid(score).item())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _put_label(img: np.ndarray, text: str, score: float | None, anomaly: bool) -> None:
    """Overlay a status label (and score) in the top-left corner of img."""
    color = (0, 0, 220) if anomaly else (0, 200, 0)
    line1 = text
    line2 = f'score: {score:.3f}' if score is not None else ''
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2

    # Dark backing rectangle so text is readable on any background
    (w1, h1), _ = cv2.getTextSize(line1, font, scale, thick)
    (w2, h2), _ = cv2.getTextSize(line2, font, scale * 0.75, thick - 1)
    pad = 8
    box_w = max(w1, w2) + pad * 2
    box_h = h1 + (h2 + 6 if line2 else 0) + pad * 2
    cv2.rectangle(img, (0, 0), (box_w, box_h), (30, 30, 30), cv2.FILLED)

    cv2.putText(img, line1, (pad, pad + h1), font, scale, color, thick, cv2.LINE_AA)
    if line2:
        cv2.putText(img, line2, (pad, pad + h1 + 6 + h2),
                    font, scale * 0.75, (200, 200, 200), thick - 1, cv2.LINE_AA)


def _ros_image_to_bgr(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image to a BGR uint8 numpy array without cv_bridge."""
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    enc = msg.encoding.lower()
    if enc in ('bgr8', 'bgra8'):
        return data[:, :, :3]
    if enc in ('rgb8', 'rgba8'):
        return data[:, :, 2::-1]  # RGB → BGR
    if enc in ('mono8',):
        return cv2.cvtColor(data[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f'Unsupported image encoding: {msg.encoding}')


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners ordered as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.float32([tl, tr, br, bl])


def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8 RGB numpy array to a normalised CxHxW float tensor."""
    t = torch.from_numpy(rgb).float() / 255.0   # [0, 1]
    t = t.permute(2, 0, 1)                       # HWC → CHW
    return (t - IMAGENET_MEAN) / IMAGENET_STD


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = TileAnomalyDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
