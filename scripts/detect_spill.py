#!/usr/bin/env python3
"""
Detects whether a spill 'hill' is currently visible in the OAK-D point cloud.
Exposes a std_srvs/Trigger service on /detect_spill: when called, the node
collects point-cloud frames for DETECT_WINDOW_SEC seconds and replies
success=True if a hill-shaped cluster was confirmed during that window.

Requires the camera_stf static transform from turtlebot4_spawn.launch.py so
that the OAK-D frame can be resolved to map or odom (z-up) for height filtering.

Key discriminators vs a horizontal barrel:
  - Flatness: a spill hill is wide relative to its height (footprint/z_range > MIN_FLATNESS).
    A barrel end-on has footprint ≈ z_range, so it fails this check.
  - Temporal stability: require CONFIRM_FRAMES consecutive detections within
    the request window before returning True, suppressing transient hits.
"""

import time

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_srvs.srv import Trigger

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree
import tf2_ros
from tf2_ros import TransformException

# >>> DEBUG DUMP imports — remove these along with the other DEBUG DUMP blocks.
import json
import os
from datetime import datetime
# <<< END DEBUG DUMP imports


class SpillDetector(Node):

    # Height band for a spill hill in map/odom frame (metres, z-up).
    # The world mesh shows barrel surfaces at ~0.13–0.27 m; spill hills
    # are much flatter, so we cap at 0.20 m and rely on the flatness check
    # (not height alone) to separate spills from barrels.
    HILL_MIN_Z = 0.02
    HILL_MAX_Z = 0.10

    # 2-D grid clustering
    GRID_RES   = 0.05   # voxel width (m)
    MIN_VOXELS = 30     # minimum occupied voxels — large enough to reject barrel-edge stripes

    # Hill shape constraints
    MIN_FOOTPRINT     = 0.15   # smallest meaningful spill (m)
    MAX_FOOTPRINT     = 2.00   # largest plausible spill (m)
    MAX_ASPECT        = 2.0    # reject elongated clusters; real spills are roundish
    MIN_MINOR_EXTENT  = 0.10   # both axes must span at least this — kills thin stripes
                               # left by the lower curvature of horizontal barrels.

    # Flatness: footprint / z_range must exceed this.
    # A spill hill is wide and low (ratio >> 1).
    # A horizontal barrel end-on has footprint ≈ z_range (ratio ≈ 1).
    MIN_FLATNESS = 3.0

    # Distance (in camera frame, metres) within which scene points are
    # discarded because they belong to / sit too close to a horizontal barrel
    # reported by cylinder_segmentation on /cylinder_point_cloud_horizontal.
    EXCLUDE_BARREL_RADIUS = 0.4

    # How long /detect_spill collects frames before answering.
    DETECT_WINDOW_SEC = 3.0

    def __init__(self):
        super().__init__('spill_detector')

        self.declare_parameter('pointcloud_topic', '/oakd/rgb/preview/depth/points')
        topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Reentrant group so the service callback can sleep on the detection
        # window while the point-cloud callback keeps firing on the same executor.
        self._cb_group = ReentrantCallbackGroup()

        self.create_subscription(PointCloud2, topic, self.pointcloud_callback,
                                 qos_profile_sensor_data,
                                 callback_group=self._cb_group)

        # Horizontal-barrel points in camera frame, supplied by cylinder_segmentation.
        # Used to mask out the barrel curvature so we don't classify it as a spill.
        self._barrel_pts_cam = None
        self.create_subscription(
            PointCloud2, '/cylinder_point_cloud_horizontal',
            self._barrel_cb, qos_profile_sensor_data,
            callback_group=self._cb_group,
        )

        self._detecting   = False  # gates pointcloud_callback work
        self._frames_total = 0     # frames processed during the current window
        self._frames_spill = 0     # of those, frames where a hill cluster was found

        self.create_service(Trigger, 'detect_spill', self._detect_spill_service,
                            callback_group=self._cb_group)

        # >>> DEBUG DUMP — call `ros2 service call /dump_spill_debug std_srvs/srv/Trigger`
        # >>> to save the latest frame to /home/gamma/colcon_ws/spill_debug/<ts>/.
        # >>> Remove this block + the matching block in pointcloud_callback +
        # >>> the _dump_debug method to disable.
        self._dump_last_msg = None
        self._dump_last_result = None
        self._dump_last_metrics = None
        self.create_service(Trigger, 'dump_spill_debug', self._dump_debug,
                            callback_group=self._cb_group)
        # <<< END DEBUG DUMP

        self.get_logger().info(
            f'SpillDetector ready — call /detect_spill (window={self.DETECT_WINDOW_SEC}s), '
            f'listening on {topic}'
        )

    # ── Service ───────────────────────────────────────────────────────────────

    def _detect_spill_service(self, _request, response):
        self.get_logger().info(
            f'/detect_spill called — sampling for {self.DETECT_WINDOW_SEC:.1f}s'
        )
        self._frames_total = 0
        self._frames_spill = 0
        self._detecting = True

        time.sleep(self.DETECT_WINDOW_SEC)

        self._detecting = False
        total = self._frames_total
        spill = self._frames_spill
        # Majority vote: > 50% of frames in the window must show a spill cluster.
        # Require at least one frame so an empty window doesn't trivially "pass".
        is_spill = total > 0 and spill * 2 > total

        response.success = is_spill
        response.message = (
            f"{'spill' if is_spill else 'not spill'} ({spill}/{total} frames)"
        )
        self.get_logger().info(f'/detect_spill -> {response.message}')
        return response

    # ── TF ────────────────────────────────────────────────────────────────────

    def _get_transform_matrix(self, source_frame):
        """Try map, fall back to odom. Time(0) = latest, avoids extrapolation errors."""
        for target in ('map', 'odom'):
            try:
                t = self.tf_buffer.lookup_transform(
                    target, source_frame, rclpy.time.Time()
                )
                tx = t.transform.translation
                q  = t.transform.rotation
                x, y, z, w = q.x, q.y, q.z, q.w
                return np.array([
                    [1-2*(y*y+z*z),  2*(x*y-z*w),    2*(x*z+y*w),   tx.x],
                    [2*(x*y+z*w),    1-2*(x*x+z*z),  2*(y*z-x*w),   tx.y],
                    [2*(x*z-y*w),    2*(y*z+x*w),    1-2*(x*x+y*y), tx.z],
                    [0.0, 0.0, 0.0, 1.0],
                ], dtype=np.float64)
            except TransformException:
                continue
        self.get_logger().warn(
            f'No transform from {source_frame} to map or odom',
            throttle_duration_sec=2.0
        )
        return None

    # ── Barrel exclusion ──────────────────────────────────────────────────────

    def _barrel_cb(self, msg):
        """Cache horizontal-barrel inlier points (camera frame) for masking."""
        pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        # skip_nans=True doesn't drop +/-inf, which the OAK-D emits for far returns.
        if len(pts) > 0:
            pts = pts[np.isfinite(pts).all(axis=1)]
        self._barrel_pts_cam = pts if len(pts) > 0 else None

    def _drop_near_barrel(self, pts_cam):
        """Remove scene points within EXCLUDE_BARREL_RADIUS of any barrel point."""
        if self._barrel_pts_cam is None or len(pts_cam) == 0:
            return pts_cam
        tree = cKDTree(self._barrel_pts_cam)
        dist, _ = tree.query(pts_cam, k=1)
        return pts_cam[dist > self.EXCLUDE_BARREL_RADIUS]

    # ── Clustering ────────────────────────────────────────────────────────────

    def _cluster_2d(self, pts_xy):
        origin = pts_xy.min(axis=0) - self.GRID_RES
        xi = ((pts_xy[:, 0] - origin[0]) / self.GRID_RES).astype(int)
        yi = ((pts_xy[:, 1] - origin[1]) / self.GRID_RES).astype(int)
        grid = np.zeros((yi.max() + 2, xi.max() + 2), dtype=np.int8)
        grid[yi, xi] = 1
        labeled_grid, n = cc_label(grid, structure=np.ones((3, 3), dtype=np.int8))
        return labeled_grid[yi, xi], n

    # ── Shape classifier ──────────────────────────────────────────────────────

    def _is_spill_hill(self, cluster):
        xy    = cluster[:, :2]
        ext_x = float(xy[:, 0].max() - xy[:, 0].min())
        ext_y = float(xy[:, 1].max() - xy[:, 1].min())
        footprint = max(ext_x, ext_y)
        z_range   = float(cluster[:, 2].max() - cluster[:, 2].min())

        if footprint < self.MIN_FOOTPRINT or footprint > self.MAX_FOOTPRINT:
            return False
        # Reject walls and elongated beams
        if footprint / (min(ext_x, ext_y) + 1e-6) > self.MAX_ASPECT:
            return False
        # Both axes must span MIN_MINOR_EXTENT — kills thin stripes (barrel curve).
        if min(ext_x, ext_y) < self.MIN_MINOR_EXTENT:
            return False
        # Key discriminator: spills are wide-and-flat; barrels are tall-and-round.
        # footprint / z_range < MIN_FLATNESS → too tower-like → barrel, not spill.
        if z_range < 1e-3 or footprint / z_range < self.MIN_FLATNESS:
            return False
        return True

    # ── Main callback ─────────────────────────────────────────────────────────

    def pointcloud_callback(self, msg):
        # Only do detection work while a /detect_spill request is in flight.
        # The debug-dump path stashes the latest frame either way (see end of fn).
        if not self._detecting:
            self._dump_last_msg = msg
            return

        pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        # skip_nans=True doesn't drop +/-inf, which the OAK-D emits for far returns.
        pts = pts[np.isfinite(pts).all(axis=1)]
        n_before_barrel_mask = len(pts)
        # Mask out points near any horizontal barrel reported by cylinder_segmentation,
        # so the lower curve of a fallen barrel doesn't fool the geometry classifier.
        pts = self._drop_near_barrel(pts)
        if len(pts) < self.MIN_VOXELS:
            return

        T = self._get_transform_matrix(msg.header.frame_id)
        if T is None:
            return

        pts_world = (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]

        z    = pts_world[:, 2]
        hill = pts_world[(z > self.HILL_MIN_Z) & (z <= self.HILL_MAX_Z)]

        found = False
        # >>> DEBUG DUMP — collect per-cluster stats for offline inspection.
        # >>> Without this block clusters are still classified the same way;
        # >>> we just don't retain shape stats. Remove block to disable.
        cluster_stats = []
        # <<< END DEBUG DUMP
        if len(hill) >= self.MIN_VOXELS:
            labels, n_clusters = self._cluster_2d(hill[:, :2])
            for cid in range(1, n_clusters + 1):
                cluster = hill[labels == cid]
                if len(cluster) < self.MIN_VOXELS:
                    continue
                accepted = self._is_spill_hill(cluster)
                # >>> DEBUG DUMP — record this cluster's shape stats.
                xy = cluster[:, :2]
                ext_x = float(xy[:, 0].max() - xy[:, 0].min())
                ext_y = float(xy[:, 1].max() - xy[:, 1].min())
                fp = max(ext_x, ext_y)
                zr = float(cluster[:, 2].max() - cluster[:, 2].min())
                cluster_stats.append({
                    'cid': int(cid),
                    'n_points': int(len(cluster)),
                    'ext_x': ext_x, 'ext_y': ext_y,
                    'footprint': fp, 'z_range': zr,
                    'aspect': fp / (min(ext_x, ext_y) + 1e-6),
                    'flatness': (fp / zr) if zr > 1e-3 else float('inf'),
                    'accepted': bool(accepted),
                })
                # <<< END DEBUG DUMP
                if accepted:
                    found = True
                    break

        self._frames_total += 1
        if found:
            self._frames_spill += 1

        # >>> DEBUG DUMP — stash latest frame state for /dump_spill_debug.
        # >>> Remove this block + the _dump_debug method + the imports + the
        # >>> __init__ block to disable.
        self._dump_last_msg = msg
        self._dump_last_result = bool(found)
        self._dump_last_metrics = {
            'stamp_sec': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'frame_id': msg.header.frame_id,
            'n_points_raw': int(n_before_barrel_mask),
            'n_points_after_barrel_mask': int(len(pts)),
            'n_barrel_pts_cached': int(0 if self._barrel_pts_cam is None else len(self._barrel_pts_cam)),
            'n_points_in_hill_band': int(len(hill)),
            'hill_band': [self.HILL_MIN_Z, self.HILL_MAX_Z],
            'frames_total': int(self._frames_total),
            'frames_spill': int(self._frames_spill),
            'found_this_frame': bool(found),
            'clusters': cluster_stats,
        }
        # <<< END DEBUG DUMP

    # >>> DEBUG DUMP — service callback. Remove this entire method along with
    # >>> the other DEBUG DUMP blocks to disable.
    def _dump_debug(self, _req, response):
        try:
            if self._dump_last_msg is None:
                response.success = False
                response.message = 'no pointcloud received yet'
                return response

            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            out_dir = os.path.join('/home/gamma/colcon_ws/spill_debug', ts)
            os.makedirs(out_dir, exist_ok=True)

            msg = self._dump_last_msg
            pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            np.savez_compressed(
                os.path.join(out_dir, 'pointcloud.npz'),
                xyz_camera_frame=pts,
                frame_id=np.array(msg.header.frame_id),
            )

            T = self._get_transform_matrix(msg.header.frame_id)
            if T is not None and len(pts) > 0:
                pts_world = (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
                np.savez_compressed(
                    os.path.join(out_dir, 'pointcloud_world.npz'),
                    xyz_world=pts_world,
                )

            with open(os.path.join(out_dir, 'result.json'), 'w') as f:
                json.dump({
                    'confirmed': self._dump_last_result,
                    'metrics': self._dump_last_metrics,
                    'thresholds': {
                        'HILL_MIN_Z': self.HILL_MIN_Z,
                        'HILL_MAX_Z': self.HILL_MAX_Z,
                        'GRID_RES': self.GRID_RES,
                        'MIN_VOXELS': self.MIN_VOXELS,
                        'MIN_FOOTPRINT': self.MIN_FOOTPRINT,
                        'MAX_FOOTPRINT': self.MAX_FOOTPRINT,
                        'MAX_ASPECT': self.MAX_ASPECT,
                        'MIN_MINOR_EXTENT': self.MIN_MINOR_EXTENT,
                        'MIN_FLATNESS': self.MIN_FLATNESS,
                        'EXCLUDE_BARREL_RADIUS': self.EXCLUDE_BARREL_RADIUS,
                        'CONFIRM_FRAMES': self.CONFIRM_FRAMES,
                    },
                }, f, indent=2)

            self.get_logger().info(f'debug dump: {out_dir}')
            response.success = True
            response.message = out_dir
        except Exception as e:
            self.get_logger().warn(f'debug dump failed: {e}')
            response.success = False
            response.message = f'dump failed: {e}'
        return response
    # <<< END DEBUG DUMP


def main():
    rclpy.init()
    node = SpillDetector()
    # MultiThreadedExecutor so the service callback's 3-second wait doesn't
    # block the point-cloud subscription it depends on.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
