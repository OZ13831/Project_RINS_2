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
import json
import math
sys.path.insert(0, '/home/gamma/colcon_ws/SuperSimpleNet')

# Ensure cv2.imshow has a display to draw on
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from geometry_msgs.msg import TwistStamped

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

    # ── Driving / alignment control ──
    # The robot drives *along* the belt, using the tile's x-axis orientation only
    # to keep its heading straight. It does NOT recentre on a tile — the tiles are
    # stationary, so once aligned the robot just keeps moving forward to the next
    # tile, holding the previous tile's heading while none is in view.
    APPROACH_SPEED      = 0.05   # slow forward speed while driving toward the belt
    BELT_DARK_FRACTION  = 0.60   # Otsu dark-pixel fraction that means "belt fills frame"
    SPIN_ANGULAR        = -0.8   # angular.z for 90° right turn (negative = clockwise)
    LINEAR_SPEED = 0.08    # forward cruise speed while a tile is tracked (m/s)
    CREEP_SPEED  = 0.06    # forward speed when no tile in view (m/s)
    K_THETA      = 1.2     # gain on tile x-axis angle error (rad → rad/s)
    MAX_ANGULAR  = 1.0     # clamp on |angular.z| (rad/s)

    # ── Startup centring (runs once, before traversal begins) ──
    # Brings the first tile to the middle of the image. Rotation swings the tile
    # onto the robot's forward (image-vertical) axis; forward/back drives it to the
    # vertical centre. Flip FWD_MOVES_TILE_UP if the robot drives the wrong way.
    K_CENTER_YAW      = 1.2     # gain: horizontal offset → angular.z
    K_CENTER_LIN      = 0.20    # gain: vertical offset → linear.x
    CENTER_LINEAR_MAX = 0.05    # clamp on centring linear speed (m/s)
    CENTER_TOL        = 0.12    # radial offset (image fraction) deemed "centred"
    SEARCH_ANGULAR    = 0.3     # in-place rotation speed when searching for a tile
    FWD_MOVES_TILE_UP = True    # driving forward moves the tile toward the image top

    # ── Anomaly-inference gating ──
    # Only trust the anomaly verdict when the tile sits near the centre column and
    # is not jammed against the top/bottom edge of the frame.
    INFER_OFFSET_X_MAX = 0.35   # |horizontal offset| must be ≤ this to infer
    EDGE_BAND          = 0.10   # ignore tiles fully within the top/bottom 10%

    # ── Tile counting ──
    # Each time the tile's bounding box covers the image centre pixel and then
    # stops covering it (tile moves on or disappears), count += 1.
    # Counting only starts after the centering phase completes.
    TILE_LIMIT_RED   = 4   # tiles to inspect when commander says "red"
    TILE_LIMIT_GREEN = 5   # tiles to inspect when commander says "green"
    RESULTS_PATH  = '/home/gamma/colcon_ws/tile_anomaly_results.json'
    ANOMALIES_DIR = '/home/gamma/colcon_ws/anomalies'

    # ── Drive states ──
    STATE_APPROACH    = 'approach'      # drive forward until belt fills the frame
    STATE_SPIN        = 'spin'          # rotate 90° right to face along the belt
    STATE_COLOR_CHECK = 'color_check'   # floor color verification (arm still down)
    STATE_ARM_TO_BELT = 'arm_to_belt'   # waiting for arm to reach look_at_belt_left
    STATE_CENTERING   = 'centering'     # bring first tile to image centre
    STATE_TRAVERSING  = 'traversing'
    STATE_IDLE        = 'idle'         # waiting for /start_detection trigger    # cruise along the belt, inspecting tiles

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('tile_anomaly_detector')

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self._device}')

        self._model = self._load_model()

        self._pub           = self.create_publisher(Bool,         '/tile_anomaly', 10)
        self._cmd_pub       = self.create_publisher(TwistStamped, '/cmd_vel',      10)
        self._arm_pub       = self.create_publisher(String,       '/arm_command',  10)
        self._color_match_pub    = self.create_publisher(Bool,    '/color_match',   10)
        self._detection_done_pub = self.create_publisher(Bool,    '/detection_done', 10)

        self._state          = self.STATE_IDLE
        self._expected_color = 'green'              # hardcoded for demo
        self.TILE_LIMIT      = self.TILE_LIMIT_GREEN

        # Tile counting state
        self._tile_count     = 0     # total tiles whose bbox has covered then left centre
        self._results        = []    # list of {'index','anomaly','score'}
        self._center_covered = False # True while the tile bbox is over the image centre pixel
        self._best_err       = float('inf')
        self._best_anomaly   = None  # None = tile never hit centre while can_infer was True
        self._best_score     = None
        self._best_heatmap   = None  # numpy heatmap from the best inference
        self._best_mask      = None  # binary anomaly mask (0/255) from the best inference
        self._best_roi       = None  # original tile ROI (BGR) from the best inference
        self._last_angular   = 0.0   # heading correction held while no tile in view

        os.makedirs(self.ANOMALIES_DIR, exist_ok=True)

        self.create_subscription(String, '/color_command',    self._color_command_cb,    10)
        self.create_subscription(Bool,   '/start_detection', self._start_detection_cb,  10)
        self.create_subscription(
            Image,
            '/top_camera/rgb/preview/image_raw',
            self._image_callback,
            qos_profile_sensor_data,
        )

        self._arm_ready = False

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

    # ── Arm-ready gate ────────────────────────────────────────────────────────

    def _arm_ready_cb(self) -> None:
        self._arm_ready = True
        self._arm_ready_timer.cancel()
        self.get_logger().info('Arm settled — driving toward belt.')

    def _spin_done_cb(self) -> None:
        self._spin_timer.cancel()
        self._publish_twist(0.0, 0.0)
        self._state = self.STATE_COLOR_CHECK
        self.get_logger().info('90° turn complete — reading floor color.')

    def _belt_ready_cb(self) -> None:
        self._belt_ready_timer.cancel()
        self._state = self.STATE_CENTERING
        self.get_logger().info('Arm at belt position — starting tile detection.')

    # ── Color command callback ────────────────────────────────────────────────

    def _color_command_cb(self, msg: String) -> None:
        color = msg.data.strip().lower()
        if color not in ('red', 'green'):
            self.get_logger().warning(f'Unknown color command: {msg.data!r}')
            return
        if self._expected_color is None:
            self._expected_color = color
            self.TILE_LIMIT = self.TILE_LIMIT_RED if color == 'red' else self.TILE_LIMIT_GREEN
            self.get_logger().info(
                f'Color command received: {color} → TILE_LIMIT={self.TILE_LIMIT}')

    # ── Detection trigger ─────────────────────────────────────────────────────

    def _start_detection_cb(self, msg: Bool) -> None:
        if not msg.data or self._state != self.STATE_IDLE:
            return
        # Reset per-run tile state
        self._tile_count     = 0
        self._results        = []
        self._center_covered = False
        self._best_err       = float('inf')
        self._best_anomaly   = None
        self._best_score     = None
        self._best_heatmap   = None
        self._best_mask      = None
        self._best_roi       = None
        self._last_angular   = 0.0
        # Move arm to down and wait for it to settle before approaching
        self._arm_ready = False
        self._arm_pub.publish(String(data='down'))
        self.get_logger().info('Detection triggered — arm → down; waiting 3.5 s...')
        if hasattr(self, '_arm_ready_timer'):
            self._arm_ready_timer.cancel()
        self._arm_ready_timer = self.create_timer(5.0, self._arm_ready_cb)
        self._state = self.STATE_APPROACH

    # ── Belt proximity detection ──────────────────────────────────────────────

    def _belt_in_view(self, bgr: np.ndarray) -> bool:
        """True when the conveyor belt fills enough of the frame to start color check."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_frac = float(np.sum(blurred < thresh_val)) / blurred.size
        return dark_frac >= self.BELT_DARK_FRACTION

    # ── Floor color detection (Otsu on saturation) ────────────────────────────

    def _detect_floor_color(self, bgr: np.ndarray) -> str | None:
        """
        Returns 'red', 'green', or None if the dominant saturated hue is ambiguous.
        Uses Otsu thresholding on the HSV saturation channel to isolate the colored
        line, then votes by hue.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _, sat_mask = cv2.threshold(
            hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        colored = hsv[sat_mask > 0]
        if len(colored) < 200:
            return None
        hues = colored[:, 0]  # OpenCV hue: 0–180
        red_px   = int(np.sum((hues < 15) | (hues > 165)))
        green_px = int(np.sum((hues > 35) & (hues < 85)))
        if red_px == 0 and green_px == 0:
            return None
        if red_px > green_px * 1.5:
            return 'red'
        if green_px > red_px * 1.5:
            return 'green'
        return None

    # ── Image callback ────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image) -> None:
        display = None
        try:
            bgr = _ros_image_to_bgr(msg)
            display = bgr.copy()

            # ── Idle phase: waiting for /start_detection trigger ─────────────
            if self._state == self.STATE_IDLE:
                _put_label(display, 'Idle — waiting for trigger...', score=None, anomaly=False)
                cv2.imshow('Tile Anomaly Detector', display)
                cv2.waitKey(1)
                return

            # ── Approach phase: drive forward until belt fills the frame ──────
            if self._state == self.STATE_APPROACH:
                if not self._arm_ready:
                    _put_label(display, 'Waiting for arm...', score=None, anomaly=False)
                    cv2.imshow('Tile Anomaly Detector', display)
                    cv2.waitKey(1)
                    return

                if self._belt_in_view(bgr):
                    self._publish_twist(0.0, 0.0)
                    self.get_logger().info('Belt in view — spinning 90° right.')
                    self._state = self.STATE_SPIN
                    spin_duration = math.pi / 2.0 / abs(self.SPIN_ANGULAR)
                    self._spin_timer = self.create_timer(spin_duration, self._spin_done_cb)
                else:
                    self._publish_twist(self.APPROACH_SPEED, 0.0)
                    _put_label(display, 'Approaching belt...', score=None, anomaly=False)
                cv2.imshow('Tile Anomaly Detector', display)
                cv2.waitKey(1)
                return

            # ── Spin phase: rotate 90° right to face along the belt ─────────
            if self._state == self.STATE_SPIN:
                self._publish_twist(0.0, self.SPIN_ANGULAR)
                _put_label(display, 'Turning right...', score=None, anomaly=False)
                cv2.imshow('Tile Anomaly Detector', display)
                cv2.waitKey(1)
                return

            # ── Color-check phase ─────────────────────────────────────────────
            if self._state == self.STATE_COLOR_CHECK:
                detected = self._detect_floor_color(bgr)
                if detected is None:
                    _put_label(display, 'Checking floor color...', score=None, anomaly=False)
                    cv2.imshow('Tile Anomaly Detector', display)
                    cv2.waitKey(1)
                    return
                if detected != self._expected_color:
                    self.get_logger().warn(
                        f'Color MISMATCH: detected={detected}, expected={self._expected_color}. Stopping.')
                    self._color_match_pub.publish(Bool(data=False))
                    self._detection_done_pub.publish(Bool(data=True))
                    self._state = self.STATE_IDLE
                    return
                self.get_logger().info(f'Color matched ({detected}) — arm → look_at_belt_left.')
                self._arm_pub.publish(String(data='look_at_belt_left'))
                self._state = self.STATE_ARM_TO_BELT
                self._belt_ready_timer = self.create_timer(3.5, self._belt_ready_cb)
                return

            if self._state == self.STATE_ARM_TO_BELT:
                _put_label(display, 'Arm moving to belt...', score=None, anomaly=False)
                cv2.imshow('Tile Anomaly Detector', display)
                cv2.waitKey(1)
                return

            result = self._find_tile(bgr)

            if result is None:
                self.get_logger().info('No tile found.', throttle_duration_sec=2.0)
                self._pub.publish(Bool(data=False))

                # Tile left the screen while its bbox was over centre → count it.
                # (only after centering is complete)
                if self._center_covered and self._state == self.STATE_TRAVERSING:
                    self._commit_tile()
                    self._center_covered = False

                if self._state == self.STATE_CENTERING:
                    # No tile yet to centre on — rotate slowly to find one.
                    self._publish_twist(0.0, self.SEARCH_ANGULAR)
                    _put_label(display, 'Searching...', score=None, anomaly=False)
                else:
                    # Keep moving forward along the previous tile's axis.
                    self._publish_twist(self.CREEP_SPEED, self._last_angular)
                    _put_label(display, 'No tile', score=None, anomaly=False)
            else:
                roi, corners = result
                H, W = bgr.shape[:2]
                angle, off_x, off_y = self._tile_pose(corners, W, H)

                # ── Bounding-box centre check (traversal only) ───────────────
                cx, cy = W // 2, H // 2
                x0 = int(corners[:, 0].min()); x1 = int(corners[:, 0].max())
                y0 = int(corners[:, 1].min()); y1 = int(corners[:, 1].max())
                covers_center = x0 <= cx <= x1 and y0 <= cy <= y1

                if self._state == self.STATE_TRAVERSING:
                    if covers_center and not self._center_covered:
                        # Bbox just came over the centre pixel — start tracking.
                        self._center_covered = True
                        self._best_err = float('inf')
                        self._best_anomaly = None
                        self._best_score = None
                        self._best_heatmap = None
                        self._best_mask = None
                        self._best_roi = None

                    elif not covers_center and self._center_covered:
                        # Bbox just left the centre pixel — count this tile.
                        self._commit_tile()
                        self._center_covered = False

                # Anomaly inference is only trusted near the centre column and when
                # the tile is not fully jammed into the top/bottom 10% of the frame.
                ys = corners[:, 1]
                fully_in_edge = (ys.max() < self.EDGE_BAND * H or
                                 ys.min() > (1.0 - self.EDGE_BAND) * H)
                can_infer = (self._state == self.STATE_TRAVERSING
                             and abs(off_x) <= self.INFER_OFFSET_X_MAX
                             and not fully_in_edge)

                score = None
                anomaly = False
                heatmap = None
                if can_infer:
                    roi = _rectify_tile(roi)
                    score, heatmap, mask = self._infer(roi)
                    anomaly = bool(score > self.ANOMALY_THRESHOLD)
                    if self._center_covered:
                        self._update_best(anomaly, score, heatmap, mask, roi,
                                          abs(off_x) + abs(angle) / (math.pi / 4))
                    self._pub.publish(Bool(data=anomaly))
                else:
                    self._pub.publish(Bool(data=False))

                # Motion: centre the first tile, then traverse the belt.
                if self._state == self.STATE_CENTERING:
                    radial = math.hypot(off_x, off_y)
                    if radial < self.CENTER_TOL:
                        self.get_logger().info('Tile centred — starting traversal.')
                        self._state = self.STATE_TRAVERSING
                        self._publish_twist(self.LINEAR_SPEED, -self.K_THETA * angle)
                    else:
                        ang = -self.K_CENTER_YAW * off_x
                        lin = -self.K_CENTER_LIN * off_y
                        if not self.FWD_MOVES_TILE_UP:
                            lin = -lin
                        lin = max(-self.CENTER_LINEAR_MAX, min(self.CENTER_LINEAR_MAX, lin))
                        self._publish_twist(lin, ang)
                else:
                    # Orientation-only steering; cruise forward to the next tile.
                    angular = -self.K_THETA * angle
                    self._last_angular = max(-self.MAX_ANGULAR,
                                             min(self.MAX_ANGULAR, angular))
                    self._publish_twist(self.LINEAR_SPEED, angular)

                self.get_logger().info(
                    f'[{self._state}] tile#{self._tile_count} '
                    f'angle={math.degrees(angle):+.1f}deg off_x={off_x:+.2f} '
                    f'off_y={off_y:+.2f} infer={can_infer} '
                    f'score={score if score is not None else float("nan"):.3f}',
                    throttle_duration_sec=1.0,
                )

                color = (0, 0, 220) if anomaly else ((0, 200, 0) if can_infer else (160, 160, 160))
                cv2.polylines(display, [corners.astype(np.int32)],
                              isClosed=True, color=color, thickness=3)
                _draw_x_axis(display, corners)
                label = ('ANOMALY' if anomaly else 'No anomaly') if can_infer else 'Inspect skipped'
                _put_label(display, label, score, anomaly)
                _put_status(display, self._state, self._tile_count, angle, off_x, off_y)

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

    def _infer(self, roi_bgr: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Return (sigmoid(anomaly_score), heatmap_bgr, mask_u8).

        heatmap_bgr is a colourised [H, W, 3] uint8 image; mask_u8 is a
        binary [H, W] uint8 image (0 / 255) of pixels whose per-pixel
        sigmoid exceeds ANOMALY_THRESHOLD.
        """
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_rgb = cv2.resize(roi_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

        tensor = _to_tensor(roi_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            anomaly_map, score = self._model(tensor)

        score_val = float(torch.sigmoid(score).item())

        # Apply sigmoid to map (same transform as the scalar score) so values are
        # in [0, 1] with absolute meaning — no local contrast stretch.
        amap = torch.sigmoid(anomaly_map).squeeze().cpu().numpy()  # (H, W), [0, 1]
        amap_u8 = (amap * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
        mask_u8 = (amap > self.ANOMALY_THRESHOLD).astype(np.uint8) * 255

        return score_val, heatmap_bgr, mask_u8

    # ── Driving / alignment ─────────────────────────────────────────────────────

    def _tile_pose(self, corners: np.ndarray, img_w: int, img_h: int) -> tuple[float, float, float]:
        """
        Pose of the tile relative to the image frame.

        Returns (angle, offset_x, offset_y):
          angle    — orientation of the tile's x-axis (mean of top/bottom edge),
                     wrapped to (-pi/4, pi/4] since a square has 90° symmetry;
                     0 means the tile rows are aligned with the image x-axis.
          offset_x — horizontal offset of the tile centre from image centre,
                     normalised to [-1, 1] (negative = tile left of centre).
          offset_y — vertical offset of the tile centre from image centre,
                     normalised to [-1, 1] (negative = tile above centre).
        """
        top = corners[1] - corners[0]   # TL → TR
        bot = corners[2] - corners[3]   # BL → BR
        x_axis = (top + bot) / 2.0
        angle = math.atan2(float(x_axis[1]), float(x_axis[0]))
        angle = ((angle + math.pi / 4) % (math.pi / 2)) - math.pi / 4

        center = corners.mean(axis=0)
        offset_x = (float(center[0]) - img_w / 2.0) / (img_w / 2.0)
        offset_y = (float(center[1]) - img_h / 2.0) / (img_h / 2.0)
        return angle, offset_x, offset_y

    def _publish_twist(self, linear: float, angular: float) -> None:
        angular = max(-self.MAX_ANGULAR, min(self.MAX_ANGULAR, float(angular)))
        t = TwistStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.twist.linear.x  = float(linear)
        t.twist.angular.z = angular
        self._cmd_pub.publish(t)

    # ── Tile session / counting ──────────────────────────────────────────────────

    def _update_best(self, anomaly: bool, score: float, heatmap: np.ndarray,
                     mask: np.ndarray, roi: np.ndarray, err: float) -> None:
        """Keep the best (most-centred) verdict seen while in the centre zone."""
        if err < self._best_err:
            self._best_err = err
            self._best_anomaly = anomaly
            self._best_score = score
            self._best_heatmap = heatmap
            self._best_mask = mask
            self._best_roi = roi.copy() if roi is not None else None

    def _commit_tile(self) -> None:
        """Bbox left the image centre pixel — record verdict and increment count."""
        inspected = self._best_anomaly is not None
        entry = {
            'index':   self._tile_count,
            'anomaly': bool(self._best_anomaly) if inspected else None,
            'score':   round(float(self._best_score), 4) if inspected else None,
        }
        self._results.append(entry)
        self._save_heatmap(entry['index'], self._best_heatmap)
        self._save_normal(entry['index'], self._best_roi)
        self._save_mask(entry['index'], self._best_mask)
        self._tile_count += 1

        n_anom = sum(1 for r in self._results if r['anomaly'])
        verdict = (f'{"ANOMALY" if entry["anomaly"] else "OK"} (score={entry["score"]:.3f})'
                   if inspected else 'UNINSPECTED (no valid inference at centre)')
        self.get_logger().info(
            f'Tile #{entry["index"]}: {verdict}  '
            f'[{n_anom}/{self._tile_count} anomalous so far]'
        )
        self._save_results()

        if self._tile_count >= self.TILE_LIMIT:
            self._print_report()
            self._publish_twist(0.0, 0.0)
            self._color_match_pub.publish(Bool(data=True))
            self._detection_done_pub.publish(Bool(data=True))
            self.get_logger().info('Detection complete — color matched, /detection_done published. Returning to idle.')
            self._state = self.STATE_IDLE
            
    def _save_heatmap(self, tile_index: int, heatmap: np.ndarray | None) -> None:
        """Save the anomaly heatmap for a tile as a PNG in ANOMALIES_DIR."""
        if heatmap is None:
            return
        try:
            path = os.path.join(self.ANOMALIES_DIR, f'tile_{tile_index:03d}.png')
            cv2.imwrite(path, heatmap, )
            self.get_logger().info(f'Heatmap saved: {path}')
        except Exception as exc:
            self.get_logger().error(f'Could not save heatmap: {exc}',
                                    throttle_duration_sec=5.0)

    def _save_normal(self, tile_index: int, roi: np.ndarray | None) -> None:
        """Save the original tile ROI as 'tile_XXX_normal.png' next to the heatmap."""
        if roi is None:
            return
        try:
            path = os.path.join(self.ANOMALIES_DIR, f'tile_{tile_index:03d}_normal.png')
            cv2.imwrite(path, roi)
            self.get_logger().info(f'Normal tile saved: {path}')
        except Exception as exc:
            self.get_logger().error(f'Could not save normal tile: {exc}',
                                    throttle_duration_sec=5.0)

    def _save_mask(self, tile_index: int, mask: np.ndarray | None) -> None:
        """Save the binary anomaly mask as 'tile_XXX_mask.png' (0/255)."""
        if mask is None:
            return
        try:
            path = os.path.join(self.ANOMALIES_DIR, f'tile_{tile_index:03d}_mask.png')
            cv2.imwrite(path, mask)
            self.get_logger().info(f'Mask saved: {path}')
        except Exception as exc:
            self.get_logger().error(f'Could not save mask: {exc}',
                                    throttle_duration_sec=5.0)

    def _print_report(self) -> None:
        """Print a T/F anomaly summary for every inspected tile."""
        print('\n' + '=' * 36)
        print(f'  TILE INSPECTION REPORT ({self._tile_count} tiles)')
        print('=' * 36)
        for r in self._results:
            if r['anomaly'] is None:
                flag, detail = '?', 'uninspected'
            else:
                flag = 'T' if r['anomaly'] else 'F'
                detail = f'score={r["score"]:.3f}'
            print(f'  Tile {r["index"]}: {flag}  ({detail})')
        print('=' * 36 + '\n')

    def _save_results(self) -> None:
        try:
            with open(self.RESULTS_PATH, 'w') as f:
                json.dump({'tiles_inspected': self._tile_count,
                           'results': self._results}, f, indent=2)
        except Exception as exc:
            self.get_logger().error(f'Could not save results: {exc}',
                                    throttle_duration_sec=5.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rectify_tile(roi: np.ndarray, out: int = 256) -> np.ndarray:
    """Rectify the tile via a homography fitted to its four edge lines.

    Pipeline: foreground mask → silhouette edges → HoughLines → split into
    near-horizontal vs near-vertical → keep the extreme top/bottom and
    left/right lines → intersect pairwise to get four corners → homography
    to an `out × out` square. Falls back to a contour-based quad when Hough
    finds fewer than 2 lines in either direction, and returns the original
    ROI on any unexpected error so inference still runs.
    """
    if roi is None or roi.size == 0:
        return roi

    def _contour_fallback(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return roi
        c = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        pts = (approx.reshape(-1, 2).astype(np.float32) if len(approx) == 4
               else cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32))
        s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
        src = np.array([pts[s.argmin()], pts[d.argmin()],
                        pts[s.argmax()], pts[d.argmax()]], np.float32)
        dst = np.array([[0, 0], [out - 1, 0], [out - 1, out - 1], [0, out - 1]], np.float32)
        return cv2.warpPerspective(roi, cv2.getPerspectiveTransform(src, dst), (out, out))

    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        # Edges of the foreground silhouette only — keeps Hough focused on the
        # tile outline and not interior texture/cracks.
        edges = cv2.Canny(mask, 50, 150)

        h, w = roi.shape[:2]
        min_len = int(0.25 * min(h, w))
        lines = cv2.HoughLines(edges, 1, math.pi / 180, min_len)
        if lines is None:
            return _contour_fallback(mask)

        horiz, vert = [], []
        for rho, theta in lines[:, 0]:
            # |sin θ| ≈ 1 ⇒ horizontal image line; |cos θ| ≈ 1 ⇒ vertical.
            if abs(math.sin(theta)) > 0.7:
                horiz.append((rho, theta))
            elif abs(math.cos(theta)) > 0.7:
                vert.append((rho, theta))
        if len(horiz) < 2 or len(vert) < 2:
            return _contour_fallback(mask)

        # For mostly-axis-aligned lines, rho ≈ signed distance to origin (top-left).
        # Sorting by rho separates the two parallel edges cleanly.
        horiz.sort(key=lambda x: x[0])
        vert.sort(key=lambda x: x[0])
        top, bottom = horiz[0], horiz[-1]
        left, right = vert[0], vert[-1]

        def _intersect(l1, l2):
            r1, t1 = l1
            r2, t2 = l2
            A = np.array([[math.cos(t1), math.sin(t1)],
                          [math.cos(t2), math.sin(t2)]], dtype=np.float64)
            b = np.array([r1, r2], dtype=np.float64)
            return np.linalg.solve(A, b)

        tl = _intersect(top, left)
        tr = _intersect(top, right)
        br = _intersect(bottom, right)
        bl = _intersect(bottom, left)
        src = np.array([tl, tr, br, bl], dtype=np.float32)

        # Sanity: reject if any corner falls far outside the ROI (degenerate fit).
        margin = 0.5 * max(h, w)
        if (src[:, 0].min() < -margin or src[:, 0].max() > w + margin or
                src[:, 1].min() < -margin or src[:, 1].max() > h + margin):
            return _contour_fallback(mask)

        dst = np.array([[0, 0], [out - 1, 0], [out - 1, out - 1], [0, out - 1]], np.float32)
        H, _ = cv2.findHomography(src, dst)
        if H is None:
            return _contour_fallback(mask)
        return cv2.warpPerspective(roi, H, (out, out))
    except (cv2.error, np.linalg.LinAlgError):
        return roi


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


def _draw_x_axis(img: np.ndarray, corners: np.ndarray) -> None:
    """Draw the tile's x-axis (top/bottom edge direction) through its centre."""
    center = corners.mean(axis=0)
    x_axis = ((corners[1] - corners[0]) + (corners[2] - corners[3])) / 2.0
    norm = np.linalg.norm(x_axis)
    if norm < 1e-3:
        return
    half = (x_axis / norm) * (norm / 2.0)
    p1 = (center - half).astype(np.int32)
    p2 = (center + half).astype(np.int32)
    cv2.arrowedLine(img, tuple(p1), tuple(p2), (0, 220, 220), 2, cv2.LINE_AA, tipLength=0.15)


def _put_status(img: np.ndarray, state: str, tile_count: int,
                angle: float, off_x: float, off_y: float) -> None:
    """Overlay drive state, tile counter and pose info in the bottom-left corner."""
    text = (f'{state}  tile #{tile_count}  angle:{math.degrees(angle):+.1f}  '
            f'off_x:{off_x:+.2f}  off_y:{off_y:+.2f}')
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    y = img.shape[0] - 10
    cv2.rectangle(img, (0, y - h - 8), (w + 12, img.shape[0]), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, text, (6, y), font, scale, (220, 220, 220), thick, cv2.LINE_AA)


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
        # Stop the robot, commit any tile still in view, persist final results.
        try:
            node._publish_twist(0.0, 0.0)
            if node._center_covered:
                node._commit_tile()
            node._save_results()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
