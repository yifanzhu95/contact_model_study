"""Policy/world conventions, evaluation metrics and accepted-target transport."""
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
PERIOD = .064


def confined(value):
    p = Path(value).expanduser()
    p = (PACKAGE / p if not p.is_absolute() else p).resolve()
    if not p.is_relative_to(PACKAGE):
        raise ValueError(f'RL output/config path must remain inside {PACKAGE}: {p}')
    return p


def quat_matrix(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Expected finite nonzero WXYZ quaternion')
    return Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()


def matrix_quat(R):
    q = Rotation.from_matrix(R).as_quat()
    return np.r_[q[3], q[:3]]


def rigid(T):
    T = np.asarray(T, float)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1)):
        raise ValueError('Expected proper rigid transform')
    return T


def world_registration(interface, hardware, hand_xml, frame_mode):
    """Coordinate registration only: does not alter real gravity/mount dynamics."""
    if frame_mode == 'identity':
        return np.eye(4)
    palm = next(x for x in interface['static_training_scene_root_bodies'] if x['name'] == 'palm')
    root = ET.parse(hand_xml).getroot()
    body = root.find(".//worldbody/body[@name='palm']")
    if body is None:
        raise ValueError('Cannot find fixed palm in calibrated hand')
    current = np.eye(4)
    current[:3, :3] = quat_matrix(hardware['palm_quat_wxyz'])
    current[:3, 3] = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
    if not np.allclose(quat_matrix(np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')), current[:3, :3], atol=1e-6):
        raise ValueError('Hardware config and calibrated XML palm rotations differ')
    train = np.eye(4)
    train[:3, :3] = quat_matrix(palm['quaternion_wxyz'])
    train[:3, 3] = palm['position_world_m']
    return rigid(train @ np.linalg.inv(current))


def goal_for_episode(interface, sign, episode, T_training_from_world):
    actual_sign = (-1 if episode % 2 == 0 else 1) if sign == 'alternate' else int(sign)
    name = 'minus90' if actual_sign == -1 else 'plus90'
    position = np.asarray(interface['trained_goal_position_world_m'], float)
    quaternion = np.asarray(interface['trained_goal_quaternions_wxyz'][name], float)
    inv = np.linalg.inv(rigid(T_training_from_world))
    return dict(sign=actual_sign, training_position=position, training_quaternion=quaternion,
                world_position=inv[:3, :3] @ position + inv[:3, 3],
                world_quaternion=matrix_quat(inv[:3, :3] @ quat_matrix(quaternion)))


def policy_state(q, v, accepted_target, goal, step, T_training_from_world):
    q, v, u = np.asarray(q, float), np.asarray(v, float), np.asarray(accepted_target, float)
    if q.shape != (23,) or v.shape != (22,) or u.shape != (16,) or not all(np.isfinite(x).all() for x in (q, v, u)):
        raise ValueError('Invalid measured hand/object/target state')
    T = rigid(T_training_from_world)
    R = quat_matrix(q[19:23])
    # mj_differentiatePos returns free-joint angular velocity in BODY axes.
    # The package encoder expects WORLD angular velocity, then training axes.
    return dict(joint_position_rad=q[:16].tolist(), joint_velocity_rad_s=v[:16].tolist(),
        object_position_world_m=(T[:3, :3] @ q[16:19] + T[:3, 3]).tolist(),
        object_quaternion_wxyz=matrix_quat(T[:3, :3] @ R).tolist(),
        object_linear_velocity_world_m_s=(T[:3, :3] @ v[16:19]).tolist(),
        object_angular_velocity_world_rad_s=(T[:3, :3] @ R @ v[19:22]).tolist(),
        goal_position_world_m=goal['training_position'].tolist(),
        goal_quaternion_wxyz=goal['training_quaternion'].tolist(),
        previous_target_rad=u.tolist(), elapsed_steps=step)


def errors(q, v, goal):
    quat = np.asarray(q[19:23], float)
    quat = quat / np.linalg.norm(quat)
    desired = goal['world_quaternion'] / np.linalg.norm(goal['world_quaternion'])
    dot = np.clip(abs(np.dot(quat, desired)), 0., 1.)
    return dict(pos=float(np.linalg.norm(q[16:19] - goal['world_position'])),
                quat=float(1 - dot*dot), vel=float(np.linalg.norm(v[16:])))


def successful(error, thresholds):
    return all(error[k] < float(value) for k, value in thresholds.items())


def feedback_model(interface):
    """Layout solely for quaternion differentiation; never steps any physics."""
    import mujoco
    root = ET.Element('mujoco', model='rl_feedback_layout_only')
    world = ET.SubElement(root, 'worldbody')
    for j in interface['joints']:
        body = ET.SubElement(world, 'body', name=j['joint_name'] + '_layout')
        ET.SubElement(body, 'joint', name=j['joint_name'], type='hinge', axis='0 0 1')
        ET.SubElement(body, 'geom', type='sphere', size='.001', mass='.001', contype='0', conaffinity='0')
    body = ET.SubElement(world, 'body', name='obj')
    ET.SubElement(body, 'freejoint', name='obj_joint')
    ET.SubElement(body, 'geom', type='sphere', size='.001', mass='.001', contype='0', conaffinity='0')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    if (model.nq, model.nv) != (23, 22):
        raise ValueError('Unexpected feedback layout')
    return model


def actuator_bounds(hand_xml, interface):
    actuators = list(ET.parse(hand_xml).getroot().find('actuator'))
    expected = [j['actuator_name'] for j in interface['joints']]
    if [r.get('name') for r in actuators] != expected:
        raise ValueError('Calibrated actuator ordering differs from policy')
    bounds = np.array([np.fromstring(r.get('ctrlrange', ''), sep=' ') for r in actuators])
    if bounds.shape != (16, 2) or not np.isfinite(bounds).all():
        raise ValueError('Need explicit calibrated actuator target ranges')
    training = np.asarray([j['actuator_target_range_rad'] for j in interface['joints']])
    lower, upper = np.maximum(bounds[:, 0], training[:, 0]), np.minimum(bounds[:, 1], training[:, 1])
    if np.any(lower >= upper):
        raise ValueError('Empty intersection of calibrated and training actuator ranges')
    return lower, upper


def command_target(proposed, previous, lower, upper, down, up):
    """Apply existing bridge increment bounds and both actuator target ranges."""
    proposed = np.asarray(proposed, float)
    if proposed.shape != (16,) or not np.isfinite(proposed).all():
        raise ValueError('Invalid decoded RL target')
    if np.any(previous < lower - 2e-6) or np.any(previous > upper + 2e-6):
        raise RuntimeError('Previous accepted target is outside policy/calibrated actuator ranges')
    clipped = np.clip(np.asarray(proposed, float), lower, upper)
    return np.clip(clipped, np.maximum(previous - down, lower), np.minimum(previous + up, upper))


def wait_for_target(feedback, desired, published_us, timeout, max_age, monotonic=time.monotonic, sleep=time.sleep):
    """Use existing bridge target broadcasts; no additional serial reads."""
    deadline = monotonic() + timeout
    while True:
        actual, metadata = feedback.buffer.snapshot_target(max_age)
        if metadata['target_timestamp'] >= published_us and np.allclose(actual, desired, rtol=0, atol=2e-6):
            return actual, metadata
        if monotonic() >= deadline:
            from ContactModelStudy.Hardware.LCM import FeedbackTimingError
            raise FeedbackTimingError('Bridge did not acknowledge the requested target in time')
        sleep(.001)


class PolicyClock:
    """Nominal 64-ms decision starts; never execute catch-up bursts after an overrun."""
    def __init__(self, period=PERIOD, monotonic=time.monotonic, sleep=time.sleep):
        self.period, self.monotonic, self.sleep = period, monotonic, sleep
        self.deadline = monotonic()

    def finish_period(self):
        self.deadline += self.period
        now = self.monotonic()
        overrun = now >= self.deadline
        if overrun:
            self.deadline = now + self.period
        waiting = max(0., self.deadline - now)
        while self.monotonic() < self.deadline:
            self.sleep(min(.01, self.deadline - self.monotonic()))
        return waiting, overrun
