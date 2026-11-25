"""FoundationPose/LEAP feedback and versioned manipulation/preparation commands."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import struct
import threading
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .Hand import vector
from .Pose import Pose
from ContactModelStudy.Utils.Quaternions import matToQuat

DEFAULT_URL = "udpm://239.255.76.67:7667?ttl=1"
DELTA_ACTION_CHANNEL = "LEAP_ACTION_DELTA"
TRACKED_ROTATION_TOLERANCE = 1e-3


class FeedbackTimingError(RuntimeError):
    """Camera/synchronization or plan timing failure; fresh joints may still hold."""


class FeedbackSkewError(FeedbackTimingError):
    """Individually fresh streams that currently fail the synchronization limit."""
    def __init__(self, metadata, max_skew):
        self.metadata = dict(metadata,
            skew_s=abs(metadata['robot_timestamp']-metadata['pose_timestamp'])*1e-6,
            max_skew_s=max_skew)
        super().__init__('Robot/object timestamps exceed maximum skew '
            f"({self.metadata['skew_s']:.6f}s > {max_skew:.6f}s; "
            f"robot age {metadata['robot_age']:.6f}s, object age {metadata['pose_age']:.6f}s)")


@dataclass
class RobotState:
    timestamp: int = 0
    qpos: np.ndarray = field(default_factory=lambda: np.zeros(16))
    qvel: np.ndarray = field(default_factory=lambda: np.zeros(16))

    def encode(self):
        return (struct.pack("<q", self.timestamp) + vector(self.qpos, "qpos").astype("<f4").tobytes()
                + vector(self.qvel, "qvel").astype("<f4").tobytes())

    @classmethod
    def decode(cls, data):
        if len(data) != 136:
            raise ValueError(f"LEAP_STATE must be 136 bytes, got {len(data)}")
        values = np.frombuffer(data, dtype="<f4", offset=8).astype(float)
        return cls(struct.unpack_from("<q", data)[0], vector(values[:16], "qpos"),
                   vector(values[16:], "qvel"))


@dataclass
class RobotAction:
    """Legacy absolute target packet; not used for manipulation delta commands."""
    timestamp: int = 0
    action: np.ndarray = field(default_factory=lambda: np.zeros(16))

    def encode(self):
        return struct.pack("<q", self.timestamp) + vector(self.action, "action").astype("<f4").tobytes()

    @classmethod
    def decode(cls, data):
        if len(data) != 72:
            raise ValueError(f"LEAP_ACTION must be 72 bytes, got {len(data)}")
        return cls(struct.unpack_from("<q", data)[0],
                   vector(np.frombuffer(data, dtype="<f4", offset=8).astype(float), "action"))


@dataclass
class RobotDeltaAction:
    """Versioned 80-byte delta packet, distinct from legacy absolute actions."""
    timestamp: int = 0
    delta: np.ndarray = field(default_factory=lambda: np.zeros(16))
    MAGIC = b"LEAPDLT1"

    def encode(self):
        return self.MAGIC + RobotAction(self.timestamp, vector(self.delta, "delta")).encode()

    @classmethod
    def decode(cls, data):
        if len(data) != 80 or data[:8] != cls.MAGIC:
            raise ValueError("Expected a versioned LEAP_ACTION_DELTA packet (80 bytes); not an absolute target")
        value = RobotAction.decode(data[8:])
        return cls(value.timestamp, value.action)


@dataclass
class RobotPositionAction(RobotAction):
    """Versioned absolute waypoint for preparation/reset, independent of MPPI bounds."""
    MAGIC = b"LEAPPOS1"

    def encode(self):
        return self.MAGIC + super().encode()

    @classmethod
    def decode(cls, data):
        if len(data) != 80 or data[:8] != cls.MAGIC:
            raise ValueError(f'Expected a versioned {cls.MAGIC.decode()} position packet (80 bytes)')
        value = RobotAction.decode(data[8:])
        return cls(value.timestamp, value.action)


@dataclass
class RobotTargetState(RobotPositionAction):
    """Bridge's held target, timestamped when reported after any pending write."""
    MAGIC = b'LEAPTGT1'


@dataclass
class RobotHoldAction:
    """Latch the bridge's measured joints once; later packets renew the hold."""
    timestamp: int = 0
    MAGIC = b"LEAPHLD1"

    def encode(self):
        return self.MAGIC + struct.pack("<q", self.timestamp)

    @classmethod
    def decode(cls, data):
        if len(data) != 16 or data[:8] != cls.MAGIC:
            raise ValueError(f"Expected a versioned {cls.__name__} packet (16 bytes)")
        return cls(struct.unpack_from("<q", data, 8)[0])


@dataclass
class RobotStopAction(RobotHoldAction):
    """Request a normal bridge shutdown, including verified torque disable."""
    MAGIC = b"LEAPSTP1"


def rigid_transform(matrix, name="transform"):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return matrix


def tracked_transform(matrix):
    """Remove small numerical rotation drift; reject scale, reflections and NaNs.

    Calibration files retain rigid_transform's stricter validation. The live
    tracker accumulates float32 rotation error, so project only near-rotations
    onto SO(3), leaving translation and the homogeneous row unchanged.
    """
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)):
        raise ValueError(f'tracked pose must be a finite homogeneous 4x4 transform: {matrix.tolist()}')
    rotation = matrix[:3, :3]
    u, singular, vt = np.linalg.svd(rotation)
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0 or np.max(np.abs(singular - 1)) > TRACKED_ROTATION_TOLERANCE:
        raise ValueError(f'tracked pose rotation is invalid: determinant={determinant:.8g}, '
                         f'singular_values={singular.tolist()}, matrix={matrix.tolist()}')
    corrected = matrix.copy()
    corrected[:3, :3] = u @ vt
    return rigid_transform(corrected, 'normalized tracked pose')


def load_pose_transform(path):
    config = json.loads(Path(path).expanduser().read_text())
    if config.get("extrinsics_status") == "pending_tag_measurement":
        raise ValueError("Camera extrinsics are pending: save the AprilTag calibration before hardware control")
    adjustment = np.eye(4)
    adjustment[:3, :3] = Rotation.from_euler("xyz", config["rotation_adjust_deg"], degrees=True).as_matrix()
    adjustment[:3, 3] = config["translation_adjust_m"]
    before = rigid_transform(adjustment) @ rigid_transform(config["T_model_from_published"])
    # Accept the original cube-only spelling in existing calibration files.
    after = rigid_transform(config.get("T_tracked_from_object", config.get("T_tracked_from_cube", np.eye(4))))
    return before, after, config


class FeedbackBuffer:
    """Assemble the measured hand plus free-joint object; never simulate feedback."""
    def __init__(self, model, before=None, after=None):
        self.model = model
        self.before = rigid_transform(np.eye(4) if before is None else before)
        self.after = rigid_transform(np.eye(4) if after is None else after)
        self.lock = threading.Lock()
        self.robot = self.pose = None
        self.robot_received = self.pose_received = 0.0
        self.object_velocity = np.zeros(6)
        self.error = None
        self.robot_error = self.pose_error = None
        self.rejected_poses = 0
        self.target = None
        self.target_received = 0.0
        self.target_error = None

    def on_target(self, channel, data):
        try:
            message = RobotTargetState.decode(data)
            if message.timestamp <= 0:
                raise ValueError('Target timestamp must be epoch microseconds')
            with self.lock:
                if self.target is None or message.timestamp > self.target.timestamp:
                    self.target = message
                    self.target_received = time.monotonic()
                    self.target_error = None
        except Exception as exc:
            with self.lock:
                self.target_error = exc

    def snapshot_target(self, max_target_age=0.15):
        with self.lock:
            error = self.error or self.target_error
            if error is not None:
                raise RuntimeError(f'LCM target feedback error: {error}') from error
            if self.target is None:
                raise RuntimeError('Waiting for bridge target feedback')
            age = time.time() - self.target.timestamp * 1e-6
            if age < -0.1 or age > max_target_age or time.monotonic() - self.target_received > max_target_age:
                raise RuntimeError(f'Stale bridge target feedback ({age:.3f}s)')
            return self.target.action.copy(), dict(target_timestamp=self.target.timestamp, target_age=age)

    def on_robot(self, channel, data):
        try:
            message = RobotState.decode(data)
            if message.timestamp <= 0:
                raise ValueError("Robot timestamp must be epoch microseconds")
            with self.lock:
                if self.robot is None or message.timestamp > self.robot.timestamp:
                    self.robot = message
                    self.robot_received = time.monotonic()
                    self.robot_error = None
        except Exception as exc:
            with self.lock:
                self.robot_error = exc

    def on_pose(self, channel, data):
        try:
            if len(data) != 80:
                raise ValueError("POSE must contain the generated Pose fingerprint and 16 floats")
            message = Pose.decode(data)
            if message.timestamp <= 0:
                raise ValueError("Pose timestamp must be epoch microseconds")
            matrix = self.before @ tracked_transform(np.asarray(message.pose_matrix).reshape(4, 4)) @ self.after
            pose = np.concatenate((matrix[:3, 3], matToQuat(matrix[:3, :3])))
            with self.lock:
                if self.pose is not None:
                    old_stamp, old_pose = self.pose
                    if message.timestamp <= old_stamp:
                        return
                    dt = (message.timestamp - old_stamp) * 1e-6
                    if 1e-4 < dt < 0.5:
                        previous = np.concatenate((np.zeros(16), old_pose))
                        current = np.concatenate((np.zeros(16), pose))
                        velocity = np.zeros(22)
                        mujoco.mj_differentiatePos(self.model, velocity, dt, previous, current)
                        self.object_velocity = velocity[16:]
                    else:
                        self.object_velocity = np.zeros(6)
                self.pose = (message.timestamp, pose)
                self.pose_received = time.monotonic()
                self.pose_error = None
        except Exception as exc:
            with self.lock:
                self.pose_error = exc
                self.rejected_poses += 1

    def snapshot_robot(self, max_robot_age=0.15):
        """Fresh joints for preparation/hold, without requiring an object in view."""
        with self.lock:
            error = self.error or self.robot_error
            if error is not None:
                raise RuntimeError(f"LCM feedback error: {error}") from error
            if self.robot is None:
                raise RuntimeError("Waiting for LEAP_STATE")
            age = time.time() - self.robot.timestamp * 1e-6
            if age < -0.1 or age > max_robot_age or time.monotonic() - self.robot_received > max_robot_age:
                raise RuntimeError(f"Stale robot feedback ({age:.3f}s)")
            return self.robot.qpos.copy(), self.robot.qvel.copy()

    def snapshot(self, max_robot_age=0.15, max_pose_age=0.3, max_skew=0.15, require_sync=True):
        """Full planning state by default; command rebasing needs fresh joints
        but not a new synchronized camera frame. Individual checks always apply.
        """
        with self.lock:
            error = self.error or self.robot_error or self.pose_error
            if error is not None:
                raise RuntimeError(f"LCM feedback error: {error}") from error
            if self.robot is None or self.pose is None:
                raise RuntimeError("Waiting for both LEAP_STATE and POSE")
            now_wall, now_mono = time.time(), time.monotonic()
            pose_stamp, pose = self.pose
            robot_age = now_wall - self.robot.timestamp * 1e-6
            pose_age = now_wall - pose_stamp * 1e-6
            for name, age, received, limit in (
                ("robot", robot_age, self.robot_received, max_robot_age),
                ("object", pose_age, self.pose_received, max_pose_age)):
                if age < -0.1 or age > limit or now_mono - received > limit:
                    error_type = FeedbackTimingError if name == "object" else RuntimeError
                    raise error_type(f"Stale {name} feedback ({age:.3f}s); check publisher timestamps")
            metadata = dict(robot_timestamp=self.robot.timestamp, pose_timestamp=pose_stamp,
                            robot_age=robot_age, pose_age=pose_age, rejected_poses=self.rejected_poses)
            if require_sync and abs(self.robot.timestamp - pose_stamp) * 1e-6 > max_skew:
                raise FeedbackSkewError(metadata, max_skew)
            q = np.concatenate((self.robot.qpos, pose))
            v = np.concatenate((self.robot.qvel, self.object_velocity))
            return q, v, metadata


class LCMFeedback:
    def __init__(self, model, url=DEFAULT_URL, robot_channel="LEAP_STATE", pose_channel="POSE",
                 action_channel=DELTA_ACTION_CHANNEL, before=None, after=None, connection=None,
                 target_feedback=False):
        if connection is None:
            import lcm
            connection = lcm.LCM(url)
        self.connection = connection
        self.buffer = FeedbackBuffer(model, before, after)
        self.action_channel = action_channel
        self.subscriptions = [connection.subscribe(robot_channel, self.buffer.on_robot),
                              connection.subscribe(pose_channel, self.buffer.on_pose)]
        if target_feedback:
            self.subscriptions.append(connection.subscribe(robot_channel + '_TARGET', self.buffer.on_target))
        for subscription in self.subscriptions:
            subscription.set_queue_capacity(1)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._receive, name="leap-lcm-feedback", daemon=True)
        self.thread.start()

    def _receive(self):
        while not self.stop.is_set():
            try:
                self.connection.handle_timeout(20)
            except Exception as exc:
                self.buffer.error = exc
                return

    def wait(self, timeout, require_target=False, **limits):
        deadline = time.monotonic() + timeout
        while True:
            try:
                state = self.buffer.snapshot(**limits)
                if require_target:
                    self.buffer.snapshot_target(limits.get('max_robot_age', 0.15))
                return state
            except RuntimeError:
                if self.buffer.error is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def publish_delta(self, delta):
        message = RobotDeltaAction(int(time.time() * 1e6), vector(delta, "delta"))
        self.connection.publish(self.action_channel, message.encode())

    def publish_position(self, position):
        message = RobotPositionAction(int(time.time() * 1e6), vector(position, 'position'))
        self.connection.publish(self.action_channel, message.encode())

    def publish_hold(self):
        self.connection.publish(self.action_channel, RobotHoldAction(int(time.time() * 1e6)).encode())

    def publish_stop(self):
        self.connection.publish(self.action_channel, RobotStopAction(int(time.time() * 1e6)).encode())

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        for subscription in self.subscriptions:
            self.connection.unsubscribe(subscription)
