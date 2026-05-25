#!/usr/bin/env python3
"""
Triggered spill detector for horizontally-fallen barrels.

Barrel localisation is delegated to cylinder_segmentation.cpp — we subscribe
to /cylinder_point_cloud_horizontal (camera-frame XYZRGB cloud of points that
RANSAC classified as horizontal cylinder inliers). The closest cluster in that
cloud is the barrel we're inspecting; its centroid gives us the 3D anchor,
and the average colour of its points gives the barrel colour.

Pipeline on /detect_spill service call:
  1. Take a snapshot of latest {RGB, scene-pointcloud, horizontal-cylinder-cloud}.
  2. From the horizontal cylinder cloud:
       a. Keep only points near the nearest one (clusters multi-barrel scenes).
       b. centroid = mean XYZ; barrel_colour = HSV-classified mean RGB.
  3. RANSAC fit a ground plane to the scene cloud (points away from the barrel).
  4. Build inner (0-0.6 m) and control (0.8-1.6 m) rings on the ground plane,
     measured as horizontal-projected distance from the barrel centroid.
  5. Score: inner-ring mean V vs control-ring mean V; require a contiguous
     dark blob (V < threshold) of minimum area inside the inner ring.

Service: std_srvs/Trigger.
  success = spilled
  message = JSON { spilled, color, confidence, v_inner, v_control, v_ratio,
                   spill_area_px, reason }
"""

import json
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_srvs.srv import Trigger

# ---- Topics ----------------------------------------------------------------

RGB_TOPIC          = "/oakd/rgb/preview/image_raw"
SCENE_CLOUD_TOPIC  = "/oakd/rgb/preview/depth/points"
BARREL_CLOUD_TOPIC = "/cylinder_point_cloud_horizontal"   # from cylinder_segmentation.cpp

# ---- Tunables --------------------------------------------------------------

# How close to the nearest barrel point we keep, when clustering multi-barrel
# scenes. Targets a single barrel that's typically <0.5 m across.
NEAR_CLUSTER_RADIUS_M = 0.6

# Maximum allowed timestamp difference between scene cloud and barrel cloud.
# Both stamps come from the same camera frame, so they should be identical or
# off by a few frames at most.
MAX_PAIR_DT_S = 1.0

# Ground plane fit
GROUND_PLANE_TOL  = 0.03      # m; points within this are "ground"
RANSAC_ITERS      = 200
RANSAC_INLIER_TOL = 0.02      # m

# Rings on the ground (horizontal distance from barrel centroid)
INNER_RING_M   = (0.0, 0.6)
CONTROL_RING_M = (0.8, 1.6)

# Spill scoring
V_DARK_THRESH     = 70        # V-channel cutoff for "dark"
MIN_SPILL_AREA_PX = 80        # min contiguous dark area in inner ring
V_RATIO_THRESH    = 0.65      # spilled if V_inner / V_control < this

DEBUG_WINDOW = "spill_detector"

# ---------------------------------------------------------------------------


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def classify_color(r_mean: float, g_mean: float, b_mean: float) -> str:
    """Map an average RGB (0-255) to one of {green, yellow, blue, red, black}."""
    bgr = np.array([[[b_mean, g_mean, r_mean]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 80 and s < 80:
        return "black"
    if h < 12 or h > 165:
        return "red"
    if 15 <= h <= 38:
        return "yellow"
    if 40 <= h <= 85:
        return "green"
    if 95 <= h <= 135:
        return "blue"
    return "unknown"


def ransac_plane(points_xyz: np.ndarray):
    """Fit n·x = d to (N,3). Returns (n, d) or (None, None)."""
    if points_xyz.shape[0] < 50:
        return None, None
    rng = np.random.default_rng(0)
    best_count = 0
    best_inliers = None
    for _ in range(RANSAC_ITERS):
        ids = rng.choice(points_xyz.shape[0], 3, replace=False)
        p1, p2, p3 = points_xyz[ids]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d = float(np.dot(normal, p1))
        inliers = np.abs(points_xyz @ normal - d) < RANSAC_INLIER_TOL
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None:
        return None, None
    inlier_pts = points_xyz[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    n = vh[-1] / (np.linalg.norm(vh[-1]) + 1e-9)
    d = float(np.dot(n, centroid))
    return n, d


def read_barrel_cloud(msg: PointCloud2):
    """Pull (xyz, rgb_uint32) from an XYZRGB cloud, ignoring NaNs."""
    pts = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
    if pts is None or len(pts) == 0:
        return None, None
    # pc2.read_points returns a structured numpy array; preserve field dtypes.
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=-1).astype(np.float32)
    rgb_f = np.ascontiguousarray(pts["rgb"]).astype(np.float32)
    rgb_u = rgb_f.view(np.uint32)
    return xyz, rgb_u


class SpillDetector(Node):
    def __init__(self):
        super().__init__("spill_detector")
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_rgb = None        # (stamp, ndarray HxWx3 BGR)
        self._latest_scene = None      # (stamp, ndarray (N,3) float32)
        self._latest_barrel = None     # (stamp, dict {xyz, rgb_u})

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, RGB_TOPIC, self._on_rgb, sensor_qos)
        self.create_subscription(PointCloud2, SCENE_CLOUD_TOPIC, self._on_scene, sensor_qos)
        self.create_subscription(PointCloud2, BARREL_CLOUD_TOPIC, self._on_barrel, 5)

        self.srv = self.create_service(Trigger, "detect_spill", self._on_service)
        self.get_logger().info(
            f"spill_detector ready. Service /detect_spill | "
            f"barrel topic {BARREL_CLOUD_TOPIC}"
        )

    # ---- subscriptions -----------------------------------------------------

    def _on_rgb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return
        with self._lock:
            self._latest_rgb = (msg.header.stamp, img)

    def _on_scene(self, msg: PointCloud2):
        try:
            arr = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        except Exception as e:
            self.get_logger().warn(f"scene cloud read: {e}")
            return
        with self._lock:
            self._latest_scene = (msg.header.stamp, arr.astype(np.float32))

    def _on_barrel(self, msg: PointCloud2):
        try:
            xyz, rgb_u = read_barrel_cloud(msg)
        except Exception as e:
            self.get_logger().warn(f"barrel cloud read: {e}")
            return
        if xyz is None or xyz.shape[0] == 0:
            return
        with self._lock:
            self._latest_barrel = (msg.header.stamp, {"xyz": xyz, "rgb_u": rgb_u})

    # ---- service handler ---------------------------------------------------

    def _on_service(self, _req, response):
        with self._lock:
            rgb_pkt = self._latest_rgb
            scene_pkt = self._latest_scene
            barrel_pkt = self._latest_barrel

        if rgb_pkt is None or scene_pkt is None:
            return self._fail(response, "RGB or scene pointcloud not received yet")
        if barrel_pkt is None:
            return self._fail(response, "no horizontal barrel detected by cylinder_segmentation")

        # Stamp pairing: scene cloud and barrel cloud should be from frames
        # close in time (cylinder_segmentation processes the same camera input).
        dt = abs(stamp_to_sec(scene_pkt[0]) - stamp_to_sec(barrel_pkt[0]))
        if dt > MAX_PAIR_DT_S:
            return self._fail(response,
                              f"scene/barrel cloud stamps differ by {dt:.2f}s (>{MAX_PAIR_DT_S})")

        rgb = rgb_pkt[1]
        flat_xyz = scene_pkt[1]
        H, W = rgb.shape[:2]
        if flat_xyz.shape[0] != H * W:
            return self._fail(response,
                              f"scene cloud N={flat_xyz.shape[0]} != H*W={H*W}")

        # --- 1. Pick nearest barrel cluster ---
        info = barrel_pkt[1]
        b_xyz_all = info["xyz"]
        b_rgb_all = info["rgb_u"]
        dist_to_cam = np.linalg.norm(b_xyz_all, axis=1)
        seed = b_xyz_all[int(np.argmin(dist_to_cam))]
        keep = np.linalg.norm(b_xyz_all - seed, axis=1) < NEAR_CLUSTER_RADIUS_M
        b_xyz = b_xyz_all[keep]
        b_rgb_u = b_rgb_all[keep]
        if b_xyz.shape[0] < 20:
            return self._fail(response,
                              f"barrel cluster too small ({b_xyz.shape[0]} pts after clustering)")

        barrel_centroid = b_xyz.mean(axis=0).astype(np.float64)

        # Decode packed RGB and classify colour
        r = ((b_rgb_u >> 16) & 0xFF).astype(np.float32)
        g = ((b_rgb_u >> 8)  & 0xFF).astype(np.float32)
        b = ( b_rgb_u        & 0xFF).astype(np.float32)
        color = classify_color(float(r.mean()), float(g.mean()), float(b.mean()))

        # --- 2. Ground plane fit (exclude barrel sphere) ---
        valid = np.isfinite(flat_xyz).all(axis=1)
        scene_pts = flat_xyz[valid]
        d_to_barrel = np.linalg.norm(scene_pts - barrel_centroid, axis=1)
        scene_for_plane = scene_pts[(d_to_barrel > 0.4) & (d_to_barrel < 4.0)]
        if scene_for_plane.shape[0] > 5000:
            idx = np.random.default_rng(1).choice(scene_for_plane.shape[0], 5000, replace=False)
            scene_for_plane = scene_for_plane[idx]
        n, d = ransac_plane(scene_for_plane)
        if n is None:
            return self._fail(response, "ground plane RANSAC failed", color=color)
        # Optical convention: +Y is down. Floor normal points "up" → -Y dominant.
        if n[1] > 0:
            n = -n
            d = -d

        # --- 3. Inner / control ring masks (horizontal in-plane distance) ---
        flat_all = flat_xyz.reshape(H, W, 3)
        signed = flat_all @ n - d
        on_ground = np.isfinite(signed) & (np.abs(signed) < GROUND_PLANE_TOL)

        barrel_signed = float(np.dot(barrel_centroid, n) - d)
        barrel_on_plane = barrel_centroid - barrel_signed * n

        delta = flat_all - barrel_on_plane                 # HxWx3
        in_plane = delta - (delta @ n)[..., None] * n      # remove normal component
        horiz_dist = np.linalg.norm(in_plane, axis=-1)
        horiz_dist[~np.isfinite(horiz_dist)] = 1e9

        inner_mask = (on_ground &
                      (horiz_dist >= INNER_RING_M[0]) &
                      (horiz_dist <= INNER_RING_M[1])).astype(np.uint8) * 255
        control_mask = (on_ground &
                        (horiz_dist >= CONTROL_RING_M[0]) &
                        (horiz_dist <= CONTROL_RING_M[1])).astype(np.uint8) * 255

        inner_count = int((inner_mask > 0).sum())
        control_count = int((control_mask > 0).sum())
        if inner_count < 50 or control_count < 50:
            self._show_debug(rgb, inner_mask, control_mask, None,
                             f"{color}: ROI too small in={inner_count} ctrl={control_count}")
            return self._fail(response,
                              f"ground ROI too small inner={inner_count} ctrl={control_count}",
                              color=color)

        # --- 4. Score ---
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        v_chan = hsv[:, :, 2]
        inner_mean = float(v_chan[inner_mask > 0].mean())
        control_mean = float(v_chan[control_mask > 0].mean())
        v_ratio = inner_mean / max(control_mean, 1.0)

        dark_inner = ((v_chan < V_DARK_THRESH) & (inner_mask > 0)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_inner = cv2.morphologyEx(dark_inner, cv2.MORPH_OPEN, k)
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(dark_inner, connectivity=8)
        spill_area = 0
        spill_blob = None
        if n_comp > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            spill_area = int(stats[idx, cv2.CC_STAT_AREA])
            spill_blob = (labels == idx).astype(np.uint8) * 255

        spilled = (v_ratio < V_RATIO_THRESH) and (spill_area >= MIN_SPILL_AREA_PX)
        ratio_conf = max(0.0, min(1.0, (V_RATIO_THRESH - v_ratio) / V_RATIO_THRESH))
        area_conf  = max(0.0, min(1.0, spill_area / 600.0))
        confidence = (0.5 * ratio_conf + 0.5 * area_conf) if spilled else 0.0

        label = (f"{color} | spilled={spilled} | V_in={inner_mean:.1f}"
                 f"/V_ctrl={control_mean:.1f} ratio={v_ratio:.2f} "
                 f"area={spill_area} barrel_pts={b_xyz.shape[0]}")
        self._show_debug(rgb, inner_mask, control_mask, spill_blob, label)

        response.success = bool(spilled)
        response.message = json.dumps({
            "spilled": bool(spilled),
            "color": color,
            "confidence": float(confidence),
            "v_inner": inner_mean,
            "v_control": control_mean,
            "v_ratio": v_ratio,
            "spill_area_px": int(spill_area),
            "barrel_pts": int(b_xyz.shape[0]),
            "barrel_centroid": [float(x) for x in barrel_centroid],
            "reason": label,
        })
        return response

    # ---- helpers -----------------------------------------------------------

    def _fail(self, response, reason, color="unknown"):
        response.success = False
        response.message = json.dumps({
            "spilled": False, "color": color, "confidence": 0.0, "reason": reason,
        })
        self.get_logger().info(f"detect_spill: {reason}")
        return response

    def _show_debug(self, rgb, inner_mask, control_mask, spill_blob, label):
        debug = rgb.copy()
        overlay = debug.copy()
        if control_mask is not None:
            overlay[control_mask > 0] = (0, 80, 0)
        if inner_mask is not None:
            overlay[inner_mask > 0] = (0, 0, 80)
        if spill_blob is not None:
            overlay[spill_blob > 0] = (255, 0, 255)
        debug = cv2.addWeighted(debug, 0.55, overlay, 0.45, 0)
        cv2.putText(debug, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)
        try:
            big = cv2.resize(debug, (debug.shape[1] * 2, debug.shape[0] * 2),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imshow(DEBUG_WINDOW, big)
            cv2.waitKey(1)
        except cv2.error:
            pass


def main():
    rclpy.init()
    node = SpillDetector()
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
