#!/usr/bin/env python3

import ctypes
from ctypes.util import find_library
from collections import Counter
from datetime import datetime
from enum import Enum
import json
import math
import os
import re
import statistics
import subprocess
import time

import cv2
import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Quaternion, TwistStamped
from irobot_create_msgs.msg import HazardDetectionVector
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
import speech_recognition as sr
from speech_intent import get_intent
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

from inspection_report import generate_report, parse_face_class

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

SPILLS_DIR = '/home/gamma/colcon_ws/spills'
REPORT_DIR = '/home/gamma/colcon_ws'
SPEECH_DIR = '/home/gamma/colcon_ws/speech_sintesis'
AUDIO_DEVICE = 'plughw:1,0'  # HyperX Cloud II Wireless

# Silence ALSA's noisy error stream so it doesn't clobber ROS logs.
_ALSA_ERR_HANDLER = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
)(lambda *_: None)
try:
    _asound = ctypes.cdll.LoadLibrary(find_library('asound'))
    _asound.snd_lib_error_set_handler(_ALSA_ERR_HANDLER)
except OSError:
    pass

MIC_DEVICE_INDEX = 4
MIC_SAMPLE_RATE = 48000

TASK_INTENTS = {'BARRELS', 'RINGS', 'ANOMALY_RED', 'ANOMALY_GREEN', 'NOTHING'}


def play(*names):
    """Play one or more wav files from SPEECH_DIR sequentially via aplay.

    `names` may be bare basenames ('hi_man') or include the .wav extension.
    Missing files / aplay errors are logged and swallowed — speech is non-essential.
    """
    for n in names:
        if not n:
            continue
        fname = n if n.endswith('.wav') else f'{n}.wav'
        path = os.path.join(SPEECH_DIR, fname)
        if not os.path.exists(path):
            print(f'[AUDIO] missing: {fname}')
            continue
        print(f'[AUDIO] {fname}')
        try:
            subprocess.run(
                ['aplay', '-D', AUDIO_DEVICE, path],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print('[AUDIO] aplay not found')
            return


def _digits(n) -> list[str]:
    return list(str(int(n)))


def say_count(n: int, unit_wav: str) -> None:
    """Play 'there were <N> <unit>' from the prerecorded fragments."""
    play('there_were', *_digits(n), unit_wav)


def say_anomalies(n_anomalies: int, n_tiles: int) -> None:
    """Play 'there were <N> anomalies in <M> tiles'."""
    play('there_were', *_digits(n_anomalies),
         'anomalies_in', *_digits(n_tiles), 'tiles')


def _resolve_gender(face_class: str | None) -> str | None:
    """Map a face class label to 'man' / 'woman' / None by substring."""
    cls = (face_class or '').lower()
    if 'she_her' in cls:
        return 'woman'
    if 'he_him' in cls:
        return 'man'
    return None


def _respond_to_task(rc, task: str | None) -> None:
    """Play the task's starting sound; for BARRELS/RINGS also announce the count."""
    if task == 'BARRELS':
        play('inspecting_barrels')
        n = (len(rc._confirmed.get('cyl_vert', []))
             + len(rc._confirmed.get('cyl_horiz', [])))
        say_count(n, 'barrels')
    elif task == 'RINGS':
        play('inspecting_rings')
        n = len(rc._confirmed.get('ring', []))
        say_count(n, 'rings')
    elif task == 'ANOMALY_RED':
        play('red_anomaly')
    elif task == 'ANOMALY_GREEN':
        play('green_anomaly')
    elif task == 'NOTHING':
        play('nothing')


def _listen_for_task(recognizer, gender: str | None) -> str | None:
    """Block on the mic until a task is confirmed per gender rules; return it.

    Woman: same task intent must be heard twice IN A ROW (any different task
    intent resets the streak), or a YES after a task intent. NO clears state.
    Man / unknown: the first task intent wins.
    """
    if gender == 'woman':
        last_intent = None
        while True:
            text, intent = listen_for_command(recognizer)
            print(f'command: {text} | intent: {intent}')
            if intent is None:
                continue
            if intent == 'YES':
                if last_intent in TASK_INTENTS:
                    return last_intent
            elif intent == 'NO':
                last_intent = None
            elif intent in TASK_INTENTS:
                if last_intent == intent:
                    return intent
                last_intent = intent
                _task_wav = {
                    'RINGS':         'rings',
                    'BARRELS':       'barrels',
                    'ANOMALY_RED':   'red_anomaly',
                    'ANOMALY_GREEN': 'green_anomaly',
                    'NOTHING':       'nothing',
                }.get(intent, '')
                print(f'[SPEECH] Asking for confirmation: {intent}')
                play('confirm_task', _task_wav)
    while True:
        text, intent = listen_for_command(recognizer)
        print(f'command: {text} | intent: {intent}')
        if intent in TASK_INTENTS:
            return intent


def listen_for_command(recognizer, device_index=MIC_DEVICE_INDEX, sample_rate=MIC_SAMPLE_RATE):
    """Block until one utterance is recognized; return (text, intent), or (None, None) on failure."""
    try:
        with sr.Microphone(device_index=device_index, sample_rate=sample_rate) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            print('Listening for command...')
            audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio).lower()
        intent, _ = get_intent(text)
        return text, intent
    except (sr.UnknownValueError, sr.RequestError):
        return None, None


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
        self._latest_robot_pose = None
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amcl_cb, amcl_qos)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)

        self._detect_spill_client = self.create_client(Trigger, '/detect_spill')

        # Latest global costmap (transient_local + reliable per Nav2 convention).
        # Used to extend horizontal-barrel standoff goals to the furthest free
        # cell along the normal; see _furthest_free_along_normal.
        costmap_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._latest_costmap = None
        self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self._costmap_cb, costmap_qos,
        )

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')

        # ── Detection aggregation ────────────────────────────────────────────
        # Per-kind obs threshold before emitting a median marker. Rings/faces trust
        # the upstream node (each already averages internally before publishing).
        self.OBS_THRESHOLD = {'ring': 1, 'cyl_vert': 10, 'cyl_horiz': 10, 'face': 1, 'spill': 1}
        self.CLUSTER_DIST = 0.5
        self.DEDUP_DIST = 0.75
        # Seconds after which a pending observation is discarded (prevents stale
        # obs from clustering with fresh ones seen later at a different location).
        self.PENDING_TTL = {'ring': 10.0, 'cyl_vert': 60.0, 'cyl_horiz': 60.0, 'face': 60.0, 'spill': 60.0}
        # Entries: rings/cyls/spills store (x, y, r, g, b); faces store (cx, cy, gx, gy, class)
        self._pending = {'ring': [], 'cyl_vert': [], 'cyl_horiz': [], 'face': [], 'spill': []}
        self._pending_times = {'ring': [], 'cyl_vert': [], 'cyl_horiz': [], 'face': [], 'spill': []}
        self._confirmed = {'ring': [], 'cyl_vert': [], 'cyl_horiz': [], 'face': [], 'spill': []}
        # Diagnostics — first-msg flags and a counter to throttle per-obs logs
        self._rings_msg_count = 0
        self._cyl_msg_count = 0
        self._face_msg_count = 0
        self._obs_count = {'ring': 0, 'cyl_vert': 0, 'cyl_horiz': 0, 'face': 0, 'spill': 0}

        self.create_subscription(MarkerArray, '/rings', self._rings_cb, 10)
        self.create_subscription(Marker, '/cylinder_markers', self._cylinder_cb, 10)
        # cylinder_segmentation bundles horizontal barrels here as
        # (cylinder_horizontal / _goal / _normal) markers per detection so we
        # can approach from the side.
        self.create_subscription(MarkerArray, '/cylinder_markers_horizontal', self._cyl_horiz_cb, 10)
        self.create_subscription(MarkerArray, '/face_classifier/confirmed_markers', self._face_cb, 10)
        # Per-frame raw YOLO classifier label for the face currently in view.
        # face_classification_publisher pushes one of these for every face box
        # in every frame. We sample this stream during a face interaction
        # (see sample_face_class) to overwrite the stored class with a
        # close-up reading that's usually more accurate than the original
        # median taken from a few metres out.
        self._face_class_samples: list[str] = []
        self.create_subscription(String, '/face_class', self._face_class_cb, 10)
        self.create_subscription(MarkerArray, '/spills', self._spills_cb, 10)
        self.objects_pub = self.create_publisher(MarkerArray, '/confirmed_objects', 10)

        self.arm_pub = self.create_publisher(String, '/arm_command', 10)
        self._start_detection_pub = self.create_publisher(Bool, '/start_detection', 10)
        self._color_command_pub = self.create_publisher(String, '/color_command', 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # When False, the rings/cylinder/face/spill callbacks ignore incoming
        # observations. Anomaly callbacks (/color_match, /detection_done) are
        # always live. Set False during the anomaly phase, re-enabled before
        # blue-line following.
        self.detections_enabled = True

        # Inspection-report state.
        self.face_requests = []           # [{'identity': {...}, 'task': str}]
        self.anomaly_cells_visited = []   # 'red' / 'green'
        self.spill_snapshots = {}         # spill_index -> image path
        self._bridge = CvBridge()
        self._latest_bgr = None
        self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self._image_cb, qos_profile_sensor_data,
        )

        self._color_match    = None   # set by /color_match from detect_tile_anomaly
        self._detection_done = False  # set by /detection_done from detect_tile_anomaly
        self.create_subscription(Bool, '/color_match',    self._color_match_cb,    10)
        self.create_subscription(Bool, '/detection_done', self._detection_done_cb, 10)

        # ── Inline line follower ─────────────────────────────────────────────
        self._lf_active = False
        self._lf_processing = False
        self._lf_state = 'follow'
        self._lf_state_start_t = 0.0
        self._lf_no_line_frames = 0
        self._lf_collision = False
        self._lf_hsv_lower = np.array([90, 80, 50], dtype=np.uint8)
        self._lf_hsv_upper = np.array([130, 255, 255], dtype=np.uint8)
        self._lf_bottom_strip_ratio = 0.25
        self._lf_left_offset_px = 40
        self._lf_linear_speed = 0.115
        self._lf_angular_gain = 1.5
        self._lf_max_angular_speed = 1.0
        self._lf_min_line_area = 200
        self._lf_dead_end_frames = 15
        self._lf_turn_180_angular_speed = 1.0
        self._lf_turn_180_duration_sec = 3.3
        self._lf_wait_after_turn_sec = 1.0
        self.create_subscription(
            Image, '/top_camera/rgb/preview/image_raw', self._lf_image_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            HazardDetectionVector, '/hazard_detection', self._lf_hazard_cb, 10,
        )

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
        # (x, y, qz, qw) — orientation is kept so we can re-issue this pose as
        # a nav2 goal (e.g. "return to start" in the anomaly-only test).
        self._latest_robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

    def _costmap_cb(self, msg: OccupancyGrid):
        self._latest_costmap = msg

    def _costmap_value_at(self, x: float, y: float) -> int:
        """Return raw cell value at (x, y) in map frame, or -1 if no costmap / off-grid."""
        cm = self._latest_costmap
        if cm is None:
            return -1
        res = cm.info.resolution
        if res <= 0:
            return -1
        col = int((x - cm.info.origin.position.x) / res)
        row = int((y - cm.info.origin.position.y) / res)
        if col < 0 or col >= cm.info.width or row < 0 or row >= cm.info.height:
            return -1
        return int(cm.data[row * cm.info.width + col])

    def furthest_free_along_normal(
        self,
        bx: float, by: float,
        gx: float, gy: float,
        step: float = 0.05,
    ) -> tuple[float, float]:
        """Furthest free (cost == 0, i.e. white) cell on the segment (bx,by)→(gx,gy).

        Logic:
          * if (gx, gy) is already free, keep it;
          * otherwise walk back from (gx, gy) toward the barrel in `step`
            increments and return the first free cell — that is the furthest
            point along the same normal that still sits on a white pixel.
          * if no free cell exists on the whole segment, fall back to (gx, gy).
        """
        if self._latest_costmap is None:
            return gx, gy
        if self._costmap_value_at(gx, gy) == 0:
            return gx, gy

        dx = gx - bx
        dy = gy - by
        norm = math.hypot(dx, dy)
        if norm < 1e-3:
            return gx, gy
        ux, uy = dx / norm, dy / norm

        t = norm - step
        while t > 0:
            x = bx + ux * t
            y = by + uy * t
            if self._costmap_value_at(x, y) == 0:
                return x, y
            t -= step

        return gx, gy

    def call_detect_spill(self, timeout_sec: float = 10.0) -> bool:
        """Block on /detect_spill (Trigger). The service samples for ~3 s before responding."""
        if not self._detect_spill_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/detect_spill service unavailable')
            return False
        future = self._detect_spill_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if future.result() is None:
            self.get_logger().warn('/detect_spill call timed out')
            return False
        result = future.result()
        self.get_logger().info(f'/detect_spill -> success={result.success} ({result.message})')
        return bool(result.success)

    def _feedback_cb(self, msg):
        self.feedback = msg.feedback

    def _color_match_cb(self, msg: Bool):
        self._color_match = msg.data

    def _detection_done_cb(self, msg: Bool):
        if msg.data:
            self._detection_done = True

    # ── Detection callbacks ──────────────────────────────────────────────────

    def _rings_cb(self, msg):
        if not self.detections_enabled:
            return
        # Each /rings msg = one confirmed ring (sphere + label). Use the sphere only.
        self._rings_msg_count += 1
        if self._rings_msg_count == 1:
            self.get_logger().info(f'First /rings msg received ({len(msg.markers)} markers)')
        for m in msg.markers:
            if m.ns == 'rings':
                self._add_observation('ring', m.pose.position.x, m.pose.position.y,
                                      m.color.r, m.color.g, m.color.b)

    def _cylinder_cb(self, msg):
        if not self.detections_enabled:
            return
        # Vertical cylinders only here. Horizontal barrels arrive bundled with
        # their side-facing goal on /cylinder_markers_horizontal (see _cyl_horiz_cb).
        self._cyl_msg_count += 1
        if self._cyl_msg_count == 1:
            self.get_logger().info(f'First /cylinder_markers msg (ns={msg.ns})')
        if msg.ns == 'cylinder':
            self._add_observation('cyl_vert',  msg.pose.position.x, msg.pose.position.y,
                                  msg.color.r, msg.color.g, msg.color.b)

    def _cyl_horiz_cb(self, msg):
        if not self.detections_enabled:
            return
        # Each MarkerArray = one horizontal-barrel detection with matching ids:
        #   ns='cylinder_horizontal'        → centre pose + barrel colour
        #   ns='cylinder_horizontal_goal'   → side-facing standoff pose
        center = goal = None
        color = None
        for m in msg.markers:
            if m.ns == 'cylinder_horizontal':
                center = (m.pose.position.x, m.pose.position.y)
                color = (m.color.r, m.color.g, m.color.b)
            elif m.ns == 'cylinder_horizontal_goal':
                goal = (m.pose.position.x, m.pose.position.y)
        if center is None or goal is None or color is None:
            return
        self._add_observation('cyl_horiz', center[0], center[1], goal[0], goal[1], *color)

    def _image_cb(self, msg):
        try:
            self._latest_bgr = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as ex:
            self.get_logger().warn(f'image conversion failed: {ex}', throttle_duration_sec=5.0)

    def _save_spill_snapshot(self, spill_index):
        if self._latest_bgr is None:
            return
        try:
            os.makedirs(SPILLS_DIR, exist_ok=True)
            path = os.path.join(SPILLS_DIR, f'spill_{spill_index:03d}.png')
            cv2.imwrite(path, self._latest_bgr)
            self.spill_snapshots[spill_index] = path
            self.get_logger().info(f'Spill snapshot saved: {path}')
        except Exception as ex:
            self.get_logger().warn(f'Could not save spill snapshot: {ex}')

    def _spills_cb(self, msg):
        if not self.detections_enabled:
            return
        # spill_color_detector already dedupes/confirms. Mirror its list verbatim
        # and republish to /confirmed_objects so spills show up alongside rings/etc.
        new_list = []
        for m in msg.markers:
            if m.ns != 'spill':
                continue
            new_list.append((
                m.pose.position.x, m.pose.position.y,
                m.color.r, m.color.g, m.color.b,
            ))
        # Save a snapshot for each spill that wasn't in the previous confirmed list.
        new_indices = [
            i for i, s in enumerate(new_list)
            if all(math.hypot(s[0] - o[0], s[1] - o[1]) > 0.3
                   for o in self._confirmed.get('spill', []))
        ]
        if new_list != self._confirmed['spill']:
            self._confirmed['spill'] = new_list
            for i in new_indices:
                self._save_spill_snapshot(i)
            self.get_logger().info(f'Confirmed spills now: {len(new_list)}')
            self._publish_objects()

    def _face_cb(self, msg):
        if not self.detections_enabled:
            return
        # face_classification_publisher already medians 15 frames; one msg = one confirmed face.
        # Pull the center + goal positions out of the MarkerArray by namespace.
        self._face_msg_count += 1
        if self._face_msg_count == 1:
            self.get_logger().info(f'First /face_classifier/confirmed_markers msg received')
        center = goal = None
        face_class = ''
        for m in msg.markers:
            if m.ns == 'face_center':
                center = (m.pose.position.x, m.pose.position.y)
            elif m.ns == 'face_goal':
                goal = (m.pose.position.x, m.pose.position.y)
            elif m.ns == 'face_class':
                face_class = m.text
        if center is None or goal is None:
            return
        self._add_observation('face', center[0], center[1], goal[0], goal[1], face_class)

    def _face_class_cb(self, msg: String):
        """Append every per-frame /face_class label into the rolling buffer.

        Sampling is bounded by sample_face_class() which clears the buffer
        before each call, so unbounded growth between calls isn't a concern
        in practice (the list is reset on every refresh).
        """
        self._face_class_samples.append(msg.data)

    def sample_face_class(self, duration: float = 2.5, face_index: int = 0) -> str | None:
        """Take a close-up reading: clear the buffer, spin for `duration`
        seconds collecting /face_class messages, majority-vote, then overwrite
        the stored class on rc._confirmed['face'][face_index].

        Returns the new class on success, or None if no samples arrived
        (face_classification_publisher is down, or the face isn't in view).

        Caveat: /face_class fires once per detection box per frame, so if
        another face is also visible the buffer will mix labels. In our use
        case the robot is parked at the face's standoff and the close face
        dominates the view, which makes the mix safe in practice.
        """
        if face_index < 0 or face_index >= len(self._confirmed['face']):
            self.get_logger().warn(f'sample_face_class: bad face_index {face_index}')
            return None

        # Clear stale samples from before we arrived in front of the face.
        self._face_class_samples = []

        end = time.time() + duration
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self._face_class_samples:
            return None

        # Majority vote across all samples captured in the window.
        new_class = Counter(self._face_class_samples).most_common(1)[0][0]

        # Rewrite the stored 5-tuple — keep position + goal, swap the class.
        cx, cy, gx, gy, _old_class = self._confirmed['face'][face_index]
        self._confirmed['face'][face_index] = (cx, cy, gx, gy, new_class)
        return new_class

    def _add_observation(self, kind, *entry):
        # entry[0], entry[1] = primary (x, y); extra coords passed through (used by face goal)
        x, y = entry[0], entry[1]
        now = time.time()

        # Dedup: drop observations near an already-confirmed object of the same kind
        for known in self._confirmed[kind]:
            if math.hypot(known[0] - x, known[1] - y) < self.DEDUP_DIST:
                return

        # TTL: discard pending observations older than PENDING_TTL[kind].
        ttl = self.PENDING_TTL.get(kind)
        if ttl is not None and self._pending_times[kind]:
            keep = [i for i, t in enumerate(self._pending_times[kind]) if now - t < ttl]
            self._pending[kind]      = [self._pending[kind][i]       for i in keep]
            self._pending_times[kind] = [self._pending_times[kind][i] for i in keep]

        # Cluster: pool with nearby pending observations
        self._pending[kind].append(entry)
        self._pending_times[kind].append(now)
        nearby = [e for e in self._pending[kind]
                  if math.hypot(e[0] - x, e[1] - y) < self.CLUSTER_DIST]

        # Throttled progress log: every 5th observation per kind
        self._obs_count[kind] += 1
        if self._obs_count[kind] % 5 == 0:
            self.get_logger().info(
                f'{kind}: obs={self._obs_count[kind]} pending={len(self._pending[kind])} '
                f'cluster_here={len(nearby)}/{self.OBS_THRESHOLD[kind]}'
            )

        # Confirm: median per coordinate once the cluster has OBS_THRESHOLD points.
        # Non-numeric fields (e.g. face class label) pass through from the latest obs.
        if len(nearby) >= self.OBS_THRESHOLD[kind]:
            confirmed_entry = tuple(
                statistics.median(e[i] for e in nearby) if isinstance(entry[i], (int, float))
                else entry[i]
                for i in range(len(entry))
            )
            self._confirmed[kind].append(confirmed_entry)
            keep = [i for i, e in enumerate(self._pending[kind])
                    if math.hypot(e[0] - confirmed_entry[0], e[1] - confirmed_entry[1]) >= self.DEDUP_DIST]
            self._pending[kind]       = [self._pending[kind][i]       for i in keep]
            self._pending_times[kind] = [self._pending_times[kind][i] for i in keep]
            self.get_logger().info(f'Confirmed {kind} at ({confirmed_entry[0]:.2f}, {confirmed_entry[1]:.2f})')
            self._publish_objects()

    # ── Inline line follower ──────────────────────────────────────────────────

    def _lf_hazard_cb(self, msg: HazardDetectionVector):
        if any(d.type == 1 for d in msg.detections):
            self._lf_collision = True

    def _lf_image_cb(self, msg: Image):
        if not self._lf_active or self._lf_processing:
            return
        self._lf_processing = True
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._lf_step(bgr)
        except Exception as ex:
            self.get_logger().error(f'line_follower image_cb: {ex}')
        finally:
            self._lf_processing = False

    def _lf_step(self, bgr: np.ndarray):
        H, W = bgr.shape[:2]
        now = time.time()

        if self._lf_state == 'follow':
            if self._lf_collision:
                self._lf_transition('turn_180')
                self._lf_collision = False
                self._lf_no_line_frames = 0
                self._lf_pub_twist(0.0, self._lf_turn_180_angular_speed)
                return

            strip_y0 = int(H * (1.0 - self._lf_bottom_strip_ratio))
            strip = bgr[strip_y0:H, :]
            hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self._lf_hsv_lower, self._lf_hsv_upper)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)

            m = cv2.moments(mask)
            cx = cy = None
            display_cx = None
            error_norm = 0.0

            if m['m00'] >= self._lf_min_line_area:
                self._lf_no_line_frames = 0
                cx = m['m10'] / m['m00']
                cy = m['m01'] / m['m00']
                display_cx = cx - self._lf_left_offset_px
                error_norm = (display_cx - W / 2.0) / (W / 2.0)
                angular = -self._lf_angular_gain * error_norm
                angular = max(-self._lf_max_angular_speed, min(self._lf_max_angular_speed, angular))
                linear = self._lf_linear_speed * max(0.0, 1.0 - error_norm * error_norm)
                self._lf_pub_twist(linear, angular)
            else:
                self._lf_no_line_frames += 1
                self._lf_pub_twist(0.0, 0.0)
                if self._lf_no_line_frames >= self._lf_dead_end_frames:
                    self._lf_transition('turn_180')
                    self._lf_no_line_frames = 0

            self._lf_show_debug(mask, W, cx, cy, display_cx, error_norm)

        elif self._lf_state == 'turn_180':
            if now - self._lf_state_start_t < self._lf_turn_180_duration_sec:
                self._lf_pub_twist(0.0, self._lf_turn_180_angular_speed)
            else:
                self._lf_pub_twist(0.0, 0.0)
                self._lf_transition('wait')

        elif self._lf_state == 'wait':
            self._lf_pub_twist(0.0, 0.0)
            if now - self._lf_state_start_t >= self._lf_wait_after_turn_sec:
                self._lf_transition('follow')

    def _lf_transition(self, new_state: str):
        self._lf_state = new_state
        self._lf_state_start_t = time.time()
        self.get_logger().info(f'[line_follower] -> {new_state}')

    def _lf_pub_twist(self, linear: float, angular: float):
        tw = TwistStamped()
        tw.header.stamp = self.get_clock().now().to_msg()
        tw.twist.linear.x = float(linear)
        tw.twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(tw)

    def _lf_show_debug(self, mask, W, cx, cy, display_cx, error_norm):
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        h = vis.shape[0]
        cv2.line(vis, (W // 2, 0), (W // 2, h), (0, 255, 0), 1)
        if cx is not None:
            cv2.circle(vis, (int(cx), int(cy)), 4, (180, 180, 180), -1)
            dx = int(display_cx)
            dy = int(cy)
            cv2.circle(vis, (dx, dy), 8, (0, 0, 255), -1)
            cv2.line(vis, (dx - 12, dy), (dx + 12, dy), (0, 0, 255), 1)
            cv2.line(vis, (dx, dy - 12), (dx, dy + 12), (0, 0, 255), 1)
        cv2.putText(vis, f'state={self._lf_state}',
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(vis, f'no_line={self._lf_no_line_frames}/{self._lf_dead_end_frames}',
                    (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if cx is not None:
            cv2.putText(vis, f'cx={cx:.0f} off={self._lf_left_offset_px} err={error_norm:+.2f}',
                        (5, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.putText(vis, 'line lost',
                        (5, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imshow('LineFollower', vis)
        cv2.waitKey(1)

    def follow_line_until_face(self, baseline_faces: int, face_approach_dist: float = 1.0):
        """Run the blue-line follower in-process until a new face is within face_approach_dist metres.

        Returns the face entry tuple, or None if no face was found.
        """
        self._lf_state = 'follow'
        self._lf_state_start_t = time.time()
        self._lf_no_line_frames = 0
        self._lf_collision = False
        self._lf_processing = False
        self._lf_active = True
        self.get_logger().info(f'Line follower started (baseline faces={baseline_faces})')

        new_face = None
        try:
            while new_face is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self._latest_robot_pose is None:
                    continue
                rx, ry = self._latest_robot_pose[0], self._latest_robot_pose[1]
                for face_entry in self._confirmed['face'][baseline_faces:]:
                    cx, cy = face_entry[0], face_entry[1]
                    dist = math.hypot(rx - cx, ry - cy)
                    if dist <= face_approach_dist:
                        new_face = face_entry
                        self.get_logger().info(f'Face within {dist:.2f} m — stopping line follower.')
                        break
        finally:
            self._lf_active = False
            cv2.destroyWindow('LineFollower')
            stop = TwistStamped()
            stop.header.stamp = self.get_clock().now().to_msg()
            self.cmd_vel_pub.publish(stop)
            self.get_logger().info('Line follower stopped.')

        return new_face

    def _publish_objects(self):
        # Redraw every confirmed object in a single MarkerArray; color/ns differentiate kinds.
        # Faces additionally get a goal sphere + arrow from center → goal (the standoff point).
        palette = {
            'ring':      (1.0, 1.0, 0.0),  # fallback yellow if no color in entry
            'cyl_vert':  (0.0, 1.0, 0.0),  # green
            'cyl_horiz': (0.0, 0.0, 1.0),  # blue
            'face':      (1.0, 0.0, 1.0),  # magenta
            'spill':     (1.0, 0.5, 0.0),  # fallback orange (real spills carry barrel color)
        }
        arr = MarkerArray()
        mid = 0
        stamp = self.get_clock().now().to_msg()
        for kind, entries in self._confirmed.items():
            for i, entry in enumerate(entries):
                x, y = entry[0], entry[1]
                # Color positions: rings/cyl_vert/spill use (x, y, r, g, b);
                # cyl_horiz includes a goal at [2:4] so colour shifts to [4:7].
                if kind == 'cyl_horiz' and len(entry) >= 7:
                    r, g, b = entry[4], entry[5], entry[6]
                elif kind in ('ring', 'cyl_vert', 'spill') and len(entry) >= 5:
                    r, g, b = entry[2], entry[3], entry[4]
                else:
                    r, g, b = palette[kind]

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

                # Faces + horizontal barrels carry a side-facing goal at entry[2:4].
                # Render the standoff sphere and a center→goal arrow so the
                # approach point is visible in RViz.
                if (kind == 'face' or kind == 'cyl_horiz') and len(entry) >= 4:
                    gx, gy = entry[2], entry[3]
                    goal_ns = 'face_goal' if kind == 'face' else 'cyl_horiz_goal'
                    arrow_ns = 'face_arrow' if kind == 'face' else 'cyl_horiz_arrow'

                    goal = Marker()
                    goal.header.frame_id = 'map'
                    goal.header.stamp = stamp
                    goal.ns = goal_ns
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
                    arrow.ns = arrow_ns
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


def _visit_barrel_and_check_spill(rc, barrel_index, barrel):
    """Drive to the side-facing standoff for `barrel`, call /detect_spill, snapshot+record on hit.

    Entry layout (set by _cyl_horiz_cb): (cx, cy, gx, gy, r, g, b)
      cx, cy → barrel centre (map frame)
      gx, gy → side-facing standoff produced by cylinder_segmentation.cpp
               (perpendicular to the fitted axis, on the camera-facing side)
      r, g, b → barrel colour, copied through to the spill entry below.

    Detections are gated off during the detour so the passive /spills callback
    can't replace _confirmed['spill'] mid-flight and clobber our append.
    """
    bx, by = barrel[0], barrel[1]
    gx, gy = barrel[2], barrel[3]
    color = (barrel[4], barrel[5], barrel[6]) if len(barrel) >= 7 else (1.0, 0.5, 0.0)

    # If the C++ goal sits on a white (cost==0) cell, use it. Otherwise pull
    # it back along the same normal to the furthest free cell still on the
    # barrel→goal segment, so we stand as far back as the costmap allows.
    gx, gy = rc.furthest_free_along_normal(bx, by, gx, gy)

    yaw = math.atan2(by - gy, bx - gx)
    q = quaternion_from_euler(0, 0, yaw)
    label = f'Horizontal barrel {barrel_index + 1} spill check'
    rc.get_logger().info(
        f'{label}: barrel=({bx:.2f},{by:.2f}) free-extended goal=({gx:.2f},{gy:.2f}) '
        f'dist={math.hypot(gx - bx, gy - by):.2f}m'
    )

    was_enabled = rc.detections_enabled
    rc.detections_enabled = False
    try:
        if not _navigate(rc, gx, gy, float(q[2]), float(q[3]), label):
            rc.get_logger().warn(f'{label}: navigation failed, skipping spill check')
            return

        is_spill = rc.call_detect_spill()
        if not is_spill:
            rc.get_logger().info(f'{label}: no spill')
            return

        # Mirror the barrel's color so inspection_report can colour-name it and
        # _match_spills_to_horizontal can pair them up by position.
        rc._confirmed['spill'].append((bx, by, color[0], color[1], color[2]))
        spill_index = len(rc._confirmed['spill']) - 1
        rc._save_spill_snapshot(spill_index)
        rc._publish_objects()
        rc.get_logger().info(f'{label}: SPILL recorded (#{spill_index})')
    finally:
        rc.detections_enabled = was_enabled


def _drain_horizontal_barrels(rc, visited_indices):
    """Visit every newly-confirmed horizontal barrel that hasn't been inspected yet."""
    while True:
        barrels = list(rc._confirmed.get('cyl_horiz', []))
        next_unvisited = next(
            ((i, b) for i, b in enumerate(barrels) if i not in visited_indices),
            None,
        )
        if next_unvisited is None:
            return
        i, b = next_unvisited
        visited_indices.add(i)
        _visit_barrel_and_check_spill(rc, i, b)


def _navigate(rc, x, y, qz, qw, label):
    goal = PoseStamped()
    goal.header.frame_id = 'map'
    goal.header.stamp = rc.get_clock().now().to_msg()
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.orientation.z = qz
    goal.pose.orientation.w = qw

    rc.get_logger().info(f'→ {label}')
    if not rc.go_to_pose(goal):
        rc.get_logger().error(f'{label}: goal rejected')
        return False

    while not rc.is_task_complete():
        rclpy.spin_once(rc, timeout_sec=0.1)

    result = rc.get_result()
    rc.get_logger().info(f'{label} result: {result}')
    return result != TaskResult.FAILED


def _run_anomaly_trip(rc, cell_color, anomaly_positions, label_prefix, gx, gy, q):
    """Drive to the anomaly cell, run detection, escape, return to (gx, gy, q),
    then announce 'there were N anomalies in M tiles'."""
    rc.get_logger().info(f'{label_prefix} requested {cell_color}-cell anomaly check.')

    rc._color_command_pub.publish(String(data=cell_color))
    rc._detection_done = False
    rc._color_match = None
    _navigate(rc, *anomaly_positions[cell_color],
              f'Anomaly {cell_color} cell ({label_prefix})')
    rc._start_detection_pub.publish(Bool(data=True))
    rc.anomaly_cells_visited.append(cell_color)
    rc.get_logger().info(
        f'Detection triggered at {cell_color} cell. Waiting for /detection_done...'
    )
    while not rc._detection_done:
        rclpy.spin_once(rc, timeout_sec=0.1)
    rc.get_logger().info(f'Detection done at {cell_color} cell.')

    n_anomalies, n_tiles = 0, 0
    try:
        with open('/home/gamma/colcon_ws/tile_anomaly_results.json') as f:
            data = json.load(f)
        n_tiles = int(data.get('tiles_inspected', 0))
        n_anomalies = sum(1 for r in data.get('results', []) if r.get('anomaly'))
    except Exception as ex:
        rc.get_logger().warn(f'Could not read tile_anomaly_results.json: {ex}')

    # Open-loop escape from the cell before handing back to nav2: 3 s reverse,
    # then a 90° right turn + 2 s forward via raw /cmd_vel. Bypasses costmap
    # goal planning so a "stuck against the wall" state can't reject the moves.
    rc.get_logger().info('Post-anomaly: driving backwards 3 s (raw cmd_vel)...')
    bwd_end = time.time() + 3.0
    while time.time() < bwd_end:
        tw = TwistStamped()
        tw.header.stamp = rc.get_clock().now().to_msg()
        tw.twist.linear.x = -0.15
        rc.cmd_vel_pub.publish(tw)
        rclpy.spin_once(rc, timeout_sec=0.05)
    stop = TwistStamped()
    stop.header.stamp = rc.get_clock().now().to_msg()
    rc.cmd_vel_pub.publish(stop)

    rc.get_logger().info('Post-anomaly: turning right 90° (raw cmd_vel)...')
    turn_speed = 1.0
    turn_end = time.time() + (math.pi / 2.0) / turn_speed
    while time.time() < turn_end:
        tw = TwistStamped()
        tw.header.stamp = rc.get_clock().now().to_msg()
        tw.twist.angular.z = -turn_speed
        rc.cmd_vel_pub.publish(tw)
        rclpy.spin_once(rc, timeout_sec=0.05)
    stop = TwistStamped()
    stop.header.stamp = rc.get_clock().now().to_msg()
    rc.cmd_vel_pub.publish(stop)

    rc.get_logger().info('Post-anomaly: driving forward 2 s (raw cmd_vel)...')
    fwd_end = time.time() + 2.0
    while time.time() < fwd_end:
        tw = TwistStamped()
        tw.header.stamp = rc.get_clock().now().to_msg()
        tw.twist.linear.x = 0.15
        rc.cmd_vel_pub.publish(tw)
        rclpy.spin_once(rc, timeout_sec=0.05)
    stop = TwistStamped()
    stop.header.stamp = rc.get_clock().now().to_msg()
    rc.cmd_vel_pub.publish(stop)

    rc.get_logger().info(f'Returning to {label_prefix} to report...')
    _navigate(rc, gx, gy, float(q[2]), float(q[3]), f'Return to {label_prefix}')

    say_anomalies(n_anomalies, n_tiles)
    report_line = (
        f'[REPORT] {label_prefix} {cell_color} cell: '
        f'{n_anomalies} anomaly tile(s) of {n_tiles} inspected '
        f'(color_match={rc._color_match}).'
    )
    print(report_line)
    rc.get_logger().info(report_line)


def _handle_face_at_standoff(rc, face_index, gx, gy, q, label_prefix,
                             recognizer, anomaly_positions):
    """At a face standoff: re-sample the close-up class, greet by gender,
    listen for a task, play the response, and trigger the anomaly trip if asked.

    Caller must have already navigated to (gx, gy, q); `face_index` indexes
    into rc._confirmed['face'].
    """
    # Re-classify up close before any audio fires — the original median was
    # taken from several metres out and is often noisier than this reading.
    print(f'[FACE] Arrived at {label_prefix}. Sampling face class for 2.5 s...')
    new_class = rc.sample_face_class(duration=2.5, face_index=face_index)
    face_class = new_class or rc._confirmed['face'][face_index][4]
    gender = _resolve_gender(face_class)
    print(f'[FACE] class={face_class!r}  gender={gender or "unknown"}')

    if gender == 'man':
        print('[SPEECH] Greeting man')
        play('hi_man')
    elif gender == 'woman':
        print('[SPEECH] Greeting woman')
        play('hi_woman')
    else:
        print('[SPEECH] Generic greeting')
        play('Greet')

    print(f'[SPEECH] Waiting for task command (gender={gender or "unknown"})...')
    confirmed_task = _listen_for_task(recognizer, gender)
    print(f'[SPEECH] Task confirmed: {confirmed_task}')
    _respond_to_task(rc, confirmed_task)

    rc.face_requests.append({
        'identity': parse_face_class(face_class),
        'task': confirmed_task,
    })

    if confirmed_task in ('ANOMALY_RED', 'ANOMALY_GREEN'):
        cell_color = 'red' if confirmed_task == 'ANOMALY_RED' else 'green'
        _run_anomaly_trip(rc, cell_color, anomaly_positions, label_prefix,
                          gx, gy, q)


def main():
    """Full mission flow: waypoints + horizontal-barrel detours + faces +
    speech + anomaly + PDF + blue-line follow."""
    rclpy.init()
    rc = RobotCommander()

    # Wait until arm_mover_actions has subscribed (its sub uses depth=1, so an
    # early publish gets dropped if the match hasn't happened yet).
    while rc.arm_pub.get_subscription_count() == 0:
        rc.get_logger().info('Waiting for /arm_command subscriber...')
        rclpy.spin_once(rc, timeout_sec=0.5)
    rc.arm_pub.publish(String(data='ring'))
    rc.get_logger().info('Arm → ring')

    rc.wait_until_nav2_active()

    positions = [
        (-2.6569306611436088,   0.16201769689870127,  0.1107997044644789,   0.9938427569241445),
        (-3.816358212773971,  -0.9086231768172194,    0.6172894726642737,   0.786736110101642),
        (-3.956416238895057,    -0.3969542758607385, -0.699399301519097,    0.7147311501778828), 
        #(-2.9503815926671053,   -3.2369627304892323,  0.4752767559043263,   0.8798363514296619),
        (-1.9658334873514014,  -3.248015988877124,    0.35379317043051733,  0.9353236833079354),
        #(-1.2682009393012044,  -1.7584973254372267,   0.6963050240259081,   0.7177459951238178),
        (0.2534043595121116,  -4.266760998946057,    -0.07499761435754064,  0.9971837131846256),
        (-1.3306695404030213,   -4.268996079835222,  -0.999950799039901,    0.009919652184605814),
        (-1.255948004244705, -3.5475625400361914,    0.8650829533991125,    0.501628830648986),
        ( 0.7975906528003365,  -2.803675605193642,    0.024640921542908818, 0.9996963663960754),
        (0.6111489652726179,    -1.2718060081019265, -0.19227055728330686,  0.9813419550808814 ),
        #( 0.024602486672440412, -0.6436527675802777,  0.5724657221444425,   0.8206314120945361),
        (1.625114060765717,   0.22486767544404362,   0.9621911044781152,    0.272375253030052),
        (0.33009879706401773,   0.1367671688995752,    0.9242716147220383,  0.38173548724098993), 
    ]

    anomaly_positions = {
        'red':   ( 0.23561884813549774, -3.9968579196473355, -0.6708838823574673, 0.7415624157095423),
        'green': (-4.474548391702069,   -2.442423935915113,   0.9998693915777004, 0.016161676461284226),
    }


    # ── Phase 1: regular waypoints (with reactive horizontal-barrel detours) ──
    # Each detour does: drive to the standoff, call /detect_spill, snapshot+record on hit.
    visited_horizontal_barrels: set[int] = set()
    for i, (x, y, qz, qw) in enumerate(positions):
        if not _navigate(rc, x, y, qz, qw, f'Waypoint {i + 1}/{len(positions)}'):
            rc.get_logger().error('Navigation failed, aborting')
            break
        time.sleep(0.5)
        _drain_horizontal_barrels(rc, visited_horizontal_barrels)
    # Catch barrels confirmed during the last detour itself.
    _drain_horizontal_barrels(rc, visited_horizontal_barrels)

    # Phase 1 is done — pause all detection ingestion except anomaly. They'll
    # be re-enabled right before the blue-line phase below.
    rc.detections_enabled = False
    rc.get_logger().info('Detections (rings/cyl/face/spill) disabled for anomaly phase.')

    #── Phase 2: face positions ───────────────────────────────────────────────
    face_entries = rc._confirmed['face']
    rc.get_logger().info(f'Navigating to {len(face_entries)} confirmed face(s).')
    recognizer = sr.Recognizer()
    for i, face_entry in enumerate(face_entries):
        cx, cy, gx, gy, _ = face_entry
        yaw = math.atan2(cy - gy, cx - gx)
        q = quaternion_from_euler(0, 0, yaw)
        _navigate(rc, gx, gy, float(q[2]), float(q[3]), f'Face {i + 1}')
        _handle_face_at_standoff(rc, i, gx, gy, q, f'face {i + 1}',
                                 recognizer, anomaly_positions)

    # ── Inspection report ─────────────────────────────────────────────────────
    try:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(REPORT_DIR, f'inspection_report_{stamp}.pdf')
        generate_report(rc, report_path)
        rc.get_logger().info(f'Inspection report written: {report_path}')
    except Exception as ex:
        rc.get_logger().error(f'Could not write inspection report: {ex}')

    # Re-enable detections before the blue-line phase.
    rc.detections_enabled = True
    rc.get_logger().info('Detections re-enabled for blue-line phase.')

    # ── Phase 4: blue-line follow until close to a face ───────────────────────
    blue_start = (
        2.753821154322128, -0.3198221794622692,
        -0.7250343149504763, 0.6887127428357148,
    )
    _navigate(rc, *blue_start, 'Blue line start')

    rc.arm_pub.publish(String(data='down'))
    rc.get_logger().info('Arm → down; waiting 3.5 s for it to settle...')
    settle_end = time.time() + 3.5
    while time.time() < settle_end:
        rclpy.spin_once(rc, timeout_sec=0.1)

    FACE_APPROACH_DIST = 1.0  # metres — stop follower when robot is this close to a face

    baseline_faces = len(rc._confirmed['face'])
    new_face = rc.follow_line_until_face(baseline_faces, FACE_APPROACH_DIST)

    if new_face is not None:
        cx, cy, gx, gy, _ = new_face
        rc.get_logger().info('Line-follow face detected — playing report audio.')
        play('here_is_the_report')

    rc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
