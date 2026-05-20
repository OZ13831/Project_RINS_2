#!/usr/bin/env python3
"""
Detects whether a spill 'hill' is currently visible in the OAK-D point cloud.
Publishes std_msgs/Bool on /spill_detected (True = hill-shaped cluster found).

Requires the camera_stf static transform from turtlebot4_spawn.launch.py so
that the OAK-D frame can be resolved to map or odom (z-up) for height filtering.

Key discriminators vs a horizontal barrel:
  - Flatness: a spill hill is wide relative to its height (footprint/z_range > MIN_FLATNESS).
    A barrel end-on has footprint ≈ z_range, so it fails this check.
  - Temporal stability: require CONFIRM_FRAMES consecutive detections before
    publishing True, suppressing transient hits while the robot moves.
"""

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Bool

import numpy as np
from scipy.ndimage import label as cc_label
import tf2_ros
from tf2_ros import TransformException


class SpillDetector(Node):

    # Height band for a spill hill in map/odom frame (metres, z-up).
    # The world mesh shows barrel surfaces at ~0.13–0.27 m; spill hills
    # are much flatter, so we cap at 0.20 m and rely on the flatness check
    # (not height alone) to separate spills from barrels.
    HILL_MIN_Z = 0.02
    HILL_MAX_Z = 0.03

    # 2-D grid clustering
    GRID_RES   = 0.05   # voxel width (m)
    MIN_VOXELS = 15     # minimum occupied voxels — raised to reduce small noise hits

    # Hill shape constraints
    MIN_FOOTPRINT = 0.15   # smallest meaningful spill (m)
    MAX_FOOTPRINT = 2.00   # largest plausible spill (m)
    MAX_ASPECT    = 3.0    # reject elongated clusters (walls, beams)

    # Flatness: footprint / z_range must exceed this.
    # A spill hill is wide and low (ratio >> 1).
    # A horizontal barrel end-on has footprint ≈ z_range (ratio ≈ 1).
    MIN_FLATNESS = 3.0

    # Temporal stability: publish True only after this many consecutive frames
    # all independently detect a hill-shaped cluster.
    CONFIRM_FRAMES = 5

    def __init__(self):
        super().__init__('spill_detector')

        self.declare_parameter('pointcloud_topic', '/oakd/rgb/preview/depth/points')
        topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(PointCloud2, topic, self.pointcloud_callback,
                                 qos_profile_sensor_data)
        self.spill_pub = self.create_publisher(Bool, '/spill_detected', 10)

        self._consec = 0   # consecutive frames with a hill detection

        self.get_logger().info(f'SpillDetector started — listening on {topic}')

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
        # Key discriminator: spills are wide-and-flat; barrels are tall-and-round.
        # footprint / z_range < MIN_FLATNESS → too tower-like → barrel, not spill.
        if z_range < 1e-3 or footprint / z_range < self.MIN_FLATNESS:
            return False
        return True

    # ── Main callback ─────────────────────────────────────────────────────────

    def pointcloud_callback(self, msg):
        pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if len(pts) < self.MIN_VOXELS:
            return

        T = self._get_transform_matrix(msg.header.frame_id)
        if T is None:
            return

        pts_world = (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]

        z    = pts_world[:, 2]
        hill = pts_world[(z > self.HILL_MIN_Z) & (z <= self.HILL_MAX_Z)]

        found = False
        if len(hill) >= self.MIN_VOXELS:
            labels, n_clusters = self._cluster_2d(hill[:, :2])
            for cid in range(1, n_clusters + 1):
                cluster = hill[labels == cid]
                if len(cluster) >= self.MIN_VOXELS and self._is_spill_hill(cluster):
                    found = True
                    break

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
    node = SpillDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
