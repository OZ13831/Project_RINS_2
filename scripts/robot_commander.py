#!/usr/bin/env python3

from enum import Enum
import math
import statistics
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Quaternion
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose, Spin
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3


class RobotCommander(Node):

    def __init__(self, node_name='robot_commander'):
        super().__init__(node_name=node_name)

        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None

        amcl_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.initial_pose_received = False
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amcl_cb, amcl_qos)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')

        # ── Detection aggregation ────────────────────────────────────────────
        # Per-kind obs threshold before emitting a median marker. Rings/faces trust
        # the upstream node (each already averages internally before publishing).
        self.OBS_THRESHOLD = {'ring': 1, 'cyl_vert': 8, 'cyl_horiz': 8, 'face': 1}
        self.CLUSTER_DIST = 0.5
        self.DEDUP_DIST = 0.75
        # Entries: rings/cyls store (x, y); faces store (cx, cy, gx, gy) — center + goal
        self._pending = {'ring': [], 'cyl_vert': [], 'cyl_horiz': [], 'face': []}
        self._confirmed = {'ring': [], 'cyl_vert': [], 'cyl_horiz': [], 'face': []}
        # Diagnostics — first-msg flags and a counter to throttle per-obs logs
        self._rings_msg_count = 0
        self._cyl_msg_count = 0
        self._face_msg_count = 0
        self._obs_count = {'ring': 0, 'cyl_vert': 0, 'cyl_horiz': 0, 'face': 0}

        self.create_subscription(MarkerArray, '/rings', self._rings_cb, 10)
        self.create_subscription(Marker, '/cylinder_markers', self._cylinder_cb, 10)
        self.create_subscription(MarkerArray, '/face_classifier/confirmed_markers', self._face_cb, 10)
        self.objects_pub = self.create_publisher(MarkerArray, '/confirmed_objects', 10)

        self.get_logger().info('RobotCommander ready')

    # ── Navigation ────────────────────────────────────────────────────────────

    def go_to_pose(self, pose: PoseStamped, behavior_tree=''):
        self.get_logger().info(f'Navigating to ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})')
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for NavigateToPose server...')

        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.behavior_tree = behavior_tree

        future = self.nav_client.send_goal_async(goal, self._feedback_cb)
        rclpy.spin_until_future_complete(self, future)
        self.goal_handle = future.result()

        if not self.goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, angle: float, time_allowance=10):
        self.get_logger().info(f'Spinning {angle:.2f} rad')
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for Spin server...')

        goal = Spin.Goal()
        goal.target_yaw = angle
        goal.time_allowance = Duration(sec=time_allowance)

        future = self.spin_client.send_goal_async(goal, self._feedback_cb)
        rclpy.spin_until_future_complete(self, future)
        self.goal_handle = future.result()

        if not self.goal_handle.accepted:
            self.get_logger().error('Spin rejected')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def cancel_task(self):
        if self.result_future:
            rclpy.spin_until_future_complete(self, self.goal_handle.cancel_goal_async())

    def is_task_complete(self):
        if not self.result_future:
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.1)
        if self.result_future.result():
            self.status = self.result_future.result().status
            return True
        return False

    def get_result(self) -> TaskResult:
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        if self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        if self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        return TaskResult.UNKNOWN

    # ── Initialisation helpers ────────────────────────────────────────────────

    def wait_until_nav2_active(self, navigator='bt_navigator', localizer='amcl'):
        self._wait_for_node(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._wait_for_node(navigator)
        self.get_logger().info('Nav2 is ready')

    def set_initial_pose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        self.initial_pose_pub.publish(msg)

    def yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = quaternion_from_euler(0, 0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    # ── Internal ──────────────────────────────────────────────────────────────

    def _wait_for_node(self, node_name):
        client = self.create_client(GetState, f'{node_name}/get_state')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {node_name}...')
        req = GetState.Request()
        while True:
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() and future.result().current_state.label == 'active':
                return
            time.sleep(2)

    def _amcl_cb(self, msg):
        self.initial_pose_received = True

    def _feedback_cb(self, msg):
        self.feedback = msg.feedback

    # ── Detection callbacks ──────────────────────────────────────────────────

    def _rings_cb(self, msg):
        # Each /rings msg = one confirmed ring (sphere + label). Use the sphere only.
        self._rings_msg_count += 1
        if self._rings_msg_count == 1:
            self.get_logger().info(f'First /rings msg received ({len(msg.markers)} markers)')
        for m in msg.markers:
            if m.ns == 'rings':
                self._add_observation('ring', m.pose.position.x, m.pose.position.y)

    def _cylinder_cb(self, msg):
        # Vertical vs horizontal is encoded in the namespace by cylinder_segmentation.cpp
        self._cyl_msg_count += 1
        if self._cyl_msg_count == 1:
            self.get_logger().info(f'First /cylinder_markers msg (ns={msg.ns})')
        if msg.ns == 'cylinder':
            self._add_observation('cyl_vert', msg.pose.position.x, msg.pose.position.y)
        elif msg.ns == 'cylinder_horizontal':
            self._add_observation('cyl_horiz', msg.pose.position.x, msg.pose.position.y)

    def _face_cb(self, msg):
        # face_classification_publisher already medians 15 frames; one msg = one confirmed face.
        # Pull the center + goal positions out of the MarkerArray by namespace.
        self._face_msg_count += 1
        if self._face_msg_count == 1:
            self.get_logger().info(f'First /face_classifier/confirmed_markers msg received')
        center = goal = None
        for m in msg.markers:
            if m.ns == 'face_center':
                center = (m.pose.position.x, m.pose.position.y)
            elif m.ns == 'face_goal':
                goal = (m.pose.position.x, m.pose.position.y)
        if center is None or goal is None:
            return
        self._add_observation('face', center[0], center[1], goal[0], goal[1])

    def _add_observation(self, kind, *entry):
        # entry[0], entry[1] = primary (x, y); extra coords passed through (used by face goal)
        x, y = entry[0], entry[1]

        # Dedup: drop observations near an already-confirmed object of the same kind
        for known in self._confirmed[kind]:
            if math.hypot(known[0] - x, known[1] - y) < self.DEDUP_DIST:
                return

        # Cluster: pool with nearby pending observations
        self._pending[kind].append(entry)
        nearby = [e for e in self._pending[kind]
                  if math.hypot(e[0] - x, e[1] - y) < self.CLUSTER_DIST]

        # Throttled progress log: every 5th observation per kind
        self._obs_count[kind] += 1
        if self._obs_count[kind] % 5 == 0:
            self.get_logger().info(
                f'{kind}: obs={self._obs_count[kind]} pending={len(self._pending[kind])} '
                f'cluster_here={len(nearby)}/{self.OBS_THRESHOLD[kind]}'
            )

        # Confirm: median per coordinate once the cluster has OBS_THRESHOLD points
        if len(nearby) >= self.OBS_THRESHOLD[kind]:
            confirmed_entry = tuple(statistics.median(e[i] for e in nearby) for i in range(len(entry)))
            self._confirmed[kind].append(confirmed_entry)
            self._pending[kind] = [e for e in self._pending[kind]
                                   if math.hypot(e[0] - confirmed_entry[0], e[1] - confirmed_entry[1]) >= self.DEDUP_DIST]
            self.get_logger().info(f'Confirmed {kind} at ({confirmed_entry[0]:.2f}, {confirmed_entry[1]:.2f})')
            self._publish_objects()

    def _publish_objects(self):
        # Redraw every confirmed object in a single MarkerArray; color/ns differentiate kinds.
        # Faces additionally get a goal sphere + arrow from center → goal (the standoff point).
        palette = {
            'ring':      (1.0, 1.0, 0.0),  # yellow
            'cyl_vert':  (0.0, 1.0, 0.0),  # green
            'cyl_horiz': (0.0, 0.0, 1.0),  # blue
            'face':      (1.0, 0.0, 1.0),  # magenta
        }
        arr = MarkerArray()
        mid = 0
        stamp = self.get_clock().now().to_msg()
        for kind, entries in self._confirmed.items():
            r, g, b = palette[kind]
            for i, entry in enumerate(entries):
                x, y = entry[0], entry[1]

                sphere = Marker()
                sphere.header.frame_id = 'map'
                sphere.header.stamp = stamp
                sphere.ns = kind
                sphere.id = mid
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position.x = float(x)
                sphere.pose.position.y = float(y)
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
                sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, 1.0
                arr.markers.append(sphere)
                mid += 1

                label = Marker()
                label.header.frame_id = 'map'
                label.header.stamp = stamp
                label.ns = f'{kind}_label'
                label.id = mid
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = float(x)
                label.pose.position.y = float(y)
                label.pose.position.z = 0.3
                label.pose.orientation.w = 1.0
                label.scale.z = 0.15
                label.color.r = label.color.g = label.color.b = label.color.a = 1.0
                label.text = f'{kind} {i + 1}'
                arr.markers.append(label)
                mid += 1

                # Face-only: also draw the standoff goal point and an arrow from face → goal
                if kind == 'face' and len(entry) >= 4:
                    gx, gy = entry[2], entry[3]

                    goal = Marker()
                    goal.header.frame_id = 'map'
                    goal.header.stamp = stamp
                    goal.ns = 'face_goal'
                    goal.id = mid
                    goal.type = Marker.SPHERE
                    goal.action = Marker.ADD
                    goal.pose.position.x = float(gx)
                    goal.pose.position.y = float(gy)
                    goal.pose.orientation.w = 1.0
                    goal.scale.x = goal.scale.y = goal.scale.z = 0.2
                    goal.color.r, goal.color.g, goal.color.b, goal.color.a = 0.0, 1.0, 1.0, 1.0
                    arr.markers.append(goal)
                    mid += 1

                    arrow = Marker()
                    arrow.header.frame_id = 'map'
                    arrow.header.stamp = stamp
                    arrow.ns = 'face_arrow'
                    arrow.id = mid
                    arrow.type = Marker.ARROW
                    arrow.action = Marker.ADD
                    tail = Point(); tail.x, tail.y = float(x),  float(y)
                    head = Point(); head.x, head.y = float(gx), float(gy)
                    arrow.points = [tail, head]
                    arrow.scale.x = 0.04
                    arrow.scale.y = 0.08
                    arrow.scale.z = 0.08
                    arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.0, 1.0, 0.9
                    arr.markers.append(arrow)
                    mid += 1
        self.objects_pub.publish(arr)


def main():
    rclpy.init()
    rc = RobotCommander()
    rc.wait_until_nav2_active()

    # Waypoints copied from points.yaml (x, y, qz, qw)
    positions = [
        (-2.6569306611436088,   0.16201769689870127,  0.1107997044644789,   0.9938427569241445),
        (-4.3069673497168885,  -0.9610489716983373,   0.785944750453859,    0.6182967323494611),
        (-1.9658334873514014,  -3.248015988877124,    0.35379317043051733,  0.9353236833079354),
        (-1.2193167449510325,  -1.7361916636833115,   0.7089521216516188,   0.7052566123090718),
        (-0.21022026415384767,  -4.284759737487124,  -0.03349020160328896,  0.999439045863514),
        ( 0.7975906528003365,  -2.803675605193642,    0.024640921542908818, 0.9996963663960754),
        ( 0.024602486672440412, -0.6436527675802777,  0.7524657221444425,   0.6586314120945361),
    ]

    for i, (x, y, qz, qw) in enumerate(positions):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = rc.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        rc.get_logger().info(f'Waypoint {i + 1}/{len(positions)}')
        if not rc.go_to_pose(goal):
            rc.get_logger().error(f'Waypoint {i + 1} rejected, skipping')
            continue

        while not rc.is_task_complete():
            # spin instead of sleep so detection callbacks fire while we drive
            rclpy.spin_once(rc, timeout_sec=0.1)

        result = rc.get_result()
        rc.get_logger().info(f'Waypoint {i + 1} result: {result}')
        if result == TaskResult.FAILED:
            rc.get_logger().error('Navigation failed, aborting')
            break

    rc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
