#!/usr/bin/env python3
"""Publish LEAP_STATE and apply LEAP_ACTION_DELTA through the LEAP hardware API.

Each new delta is added ONCE to the configured reference: the previous target
or measured joints. The final target is bounded around that reference by
ctrl_range. The held target is reported in previous-target mode.
The resulting absolute target is written at the maximum serial loop rate,
without sleeps or velocity/slew clamps. Optional measured joint limits clamp
delta targets after ctrl_range. Cached packets never
accumulate another delta. State packets retain the original bridge format;
delta packets are versioned and use a separate channel. Device faults and an
action timeout stop the bridge and disable torque.
Versioned hold packets capture measured q once; heartbeats renew the watchdog.
Versioned position packets carry gradual preparation/reset waypoints directly;
MPPI ctrl_range bounds apply only to delta packets.
"""
import argparse
import json
from pathlib import Path
import time
import signal

import mujoco
import numpy as np

from ContactModelStudy.Hardware.Hand import HardwareHand, check_device_feedback, clip_joint_target, control_limits, delta_limits, delta_reference
from ContactModelStudy.Hardware.LCM import (
    DEFAULT_URL, DELTA_ACTION_CHANNEL, RobotDeltaAction, RobotPositionAction,
    RobotHoldAction, RobotStopAction, RobotState, RobotTargetState,
    RobotKeepTargetAction, RobotRecoveryPositionAction,
)
from ContactModelStudy.Hardware.Scene import DEFAULT_SCENE
from ContactModelStudy.Hardware.BridgeTelemetry import BridgeTelemetry
from ContactModelStudy.Hardware.JointLimits import load_joint_limits

ROOT = Path(__file__).resolve().parent


class LeapHandLCMBridge:
    def __init__(self, hand, connection, model, config, state_channel="LEAP_STATE",
                 action_channel=DELTA_ACTION_CHANNEL, action_timeout=0.5, state_only=False,
                 telemetry=None):
        if not np.isfinite(action_timeout) or action_timeout <= 0:
            raise ValueError("Use a finite positive action timeout")
        self.hand, self.connection, self.config = hand, connection, config
        self.state_channel = state_channel
        self.timeout = action_timeout
        self.state_only = state_only
        self.telemetry = telemetry
        self.latest_received_us = None
        self.lower, self.upper = control_limits(model)
        self.delta_lower, self.delta_upper = delta_limits(config)
        self.delta_reference = delta_reference(config)
        self.joint_limits = load_joint_limits(config)
        self.hold_on_failure = config.get('hold_on_non_motor_failure', False)
        if not isinstance(self.hold_on_failure, bool):
            raise ValueError('hold_on_non_motor_failure must be boolean')
        self.fault_hold = False
        self.fault_status_file = None
        self.latest = None
        self.received = 0.0
        self.expired_actions = 0
        self.stop_requested = False
        self.subscription = None
        if not state_only:
            self.subscription = connection.subscribe(action_channel, self.on_action)
            self.subscription.set_queue_capacity(1)

    def on_action(self, channel, data):
        packet_type = {RobotHoldAction.MAGIC: RobotHoldAction, RobotStopAction.MAGIC: RobotStopAction,
                       RobotPositionAction.MAGIC: RobotPositionAction,
                       RobotKeepTargetAction.MAGIC: RobotKeepTargetAction,
                       RobotRecoveryPositionAction.MAGIC: RobotRecoveryPositionAction}.get(
            data[:8], RobotDeltaAction)
        action = packet_type.decode(data)
        if self.fault_hold and not isinstance(action, (RobotKeepTargetAction, RobotRecoveryPositionAction, RobotStopAction)):
            return  # A delayed/restarted planner must not resume motion after a fault.
        if self.latest is not None and action.timestamp <= self.latest.timestamp:
            return
        received_wall = time.time()
        age = received_wall - action.timestamp * 1e-6
        if age < -0.1:
            raise RuntimeError(f"Future-dated action received ({age:.3f}s); check publisher clock")
        if age > self.timeout:
            self.expired_actions += 1
            if self.expired_actions == 1:
                print(f"Discarding expired queued action ({age:.3f}s old); watchdog unchanged", flush=True)
            return
        if isinstance(action, RobotKeepTargetAction):
            self.enter_fault_hold('software recovery requested')
        self.latest = action
        self.latest_received_us = int(received_wall*1e6)
        # Transport/queue time counts toward the same action deadline. An old
        # packet must not acquire another full watchdog interval on arrival.
        self.received = time.monotonic() - max(age, 0.0)

    def enter_fault_hold(self, reason):
        if self.fault_hold:
            return
        self.fault_hold = True
        print(f'SOFTWARE HOLD: {reason}; retaining last target. Motor/USB checks remain active.',flush=True)
        if self.fault_status_file is not None:
            path=Path(self.fault_status_file)
            pending=path.with_suffix('.tmp')
            pending.write_text(json.dumps(dict(stage='software_hold',reason=reason,updated_at=time.time()))+'\n')
            pending.replace(path)

    def run(self):
        if self.stop_requested:
            return
        if self.state_only:
            self.hand.connect_read_only()
        else:
            self.hand.connect()
        if self.stop_requested:
            return
        q, v, current = self.hand.read()
        check_device_feedback(q, v, current, self.config)
        if self.stop_requested:
            return
        if not self.state_only:
            self.hand.enable(q)
        command = q.copy()
        if self.delta_reference == 'previous_target' and not self.state_only:
            self.publish_target(command)
        applied_timestamp = 0
        holding = False
        last_health = time.monotonic()
        health_motor = 0
        health_register_index = 0
        split_health = callable(getattr(self.hand, 'health_register', None))
        health_period = 1 / 32 if split_health else 1 / 16
        cycle = 0
        last_write_us = None
        target_since_us = None
        command_type = 'startup_hold'
        print("State-only bridge: no torque/register writes or action subscriptions" if self.state_only else
              f"Bridge started: unthrottled serial loop; delta_reference={self.delta_reference}; absolute preparation waypoints", flush=True)
        if self.joint_limits is not None and not self.state_only:
            print(f'MPPI target clamp: ctrl_range then measured joint limits from {self.joint_limits.model}', flush=True)
        while not self.stop_requested:
            trace = None
            if self.telemetry is not None:
                trace = dict(event='cycle', cycle=cycle, read_started_us=int(time.time()*1e6),
                    held_target_before_read=None if self.state_only else command.copy(),
                    held_action_timestamp_us=applied_timestamp,
                    held_command_type=command_type, held_target_since_us=target_since_us,
                    previous_write_finished_us=last_write_us, write_succeeded=False)
                read_started = time.monotonic()
            cycle += 1
            try:
                q, v, current = self.hand.read()
                if trace is not None:
                    trace.update(read_finished_us=int(time.time()*1e6),
                        read_duration_s=time.monotonic()-read_started,
                        q=q.copy(), qvel=v.copy(), current_raw_native=current.copy())
                if self.stop_requested:
                    break
                check_device_feedback(q, v, current, self.config)
                if not self.state_only:
                    for _ in range(32):
                        if not self.connection.handle_timeout(0):
                            break
                if self.stop_requested:
                    break
                state_stamp = int(time.time()*1e6)
                self.connection.publish(self.state_channel, RobotState(state_stamp, q, v).encode())
                if trace is not None:
                    trace.update(state_timestamp_us=state_stamp,
                        latest_action_timestamp_us=self.latest.timestamp if self.latest else None,
                        handler_received_us=self.latest_received_us)
                now = time.monotonic()
                if not self.state_only and self.latest is not None:
                    if not self.fault_hold and now - self.received > self.timeout:
                        if not self.hold_on_failure:
                            raise RuntimeError("Action timeout; disabling torque")
                        self.enter_fault_hold('controller action timeout')
                        command_type = 'fault_hold'
                        print('Action timeout: holding the last target; motor/USB checks remain active. '
                              'Waiting for operator-confirmed recovery or stop.', flush=True)
                    new_action = (self.latest.timestamp > applied_timestamp and
                                  (not self.fault_hold or isinstance(self.latest,
                                   (RobotKeepTargetAction, RobotRecoveryPositionAction, RobotStopAction))))
                    if trace is not None:
                        trace['new_action'] = new_action
                    if new_action:
                        if isinstance(self.latest, RobotStopAction):
                            print("Bridge stop requested; disabling torque.", flush=True)
                            return  # main's finally closes the hand and verifies torque off
                        elif isinstance(self.latest, RobotKeepTargetAction):
                            self.fault_hold = True
                            holding = False
                            command_type = 'fault_hold'
                        elif isinstance(self.latest, RobotPositionAction):
                            command = self.latest.action.copy()
                            holding = False
                            command_type = ('recovery_position' if isinstance(self.latest, RobotRecoveryPositionAction)
                                            else 'position')
                        elif isinstance(self.latest, RobotHoldAction):
                            if not holding:
                                command = q.copy()
                            holding = True
                            command_type = 'hold'
                        else:
                            base = command if self.delta_reference == 'previous_target' else q
                            requested_target = base + self.latest.delta
                            command = clip_joint_target(requested_target, base, self.delta_lower, self.delta_upper)
                            delta_clipped_target = command.copy() if trace is not None else None
                            if self.joint_limits is not None:
                                command = np.clip(command, self.joint_limits.lower, self.joint_limits.upper)
                            holding = False
                            command_type = 'delta'
                            if trace is not None:
                                trace.update(delta=self.latest.delta.copy(), application_base=base.copy(),
                                    requested_target=requested_target.copy(),
                                    delta_clipped_target=delta_clipped_target,
                                    joint_limit_clipped_joints=int(np.count_nonzero(abs(command-delta_clipped_target)>1e-7)))
                        applied_timestamp = self.latest.timestamp
                    if self.stop_requested:
                        break
                    if trace is not None:
                        trace.update(write_started_us=int(time.time()*1e6),
                            written_action_timestamp_us=applied_timestamp,
                            written_target=command.copy(), command_type=command_type)
                        write_started = time.monotonic()
                    self.hand.write(command)
                    if trace is not None:
                        last_write_us = int(time.time()*1e6)
                        if target_since_us is None or not np.array_equal(command, trace['held_target_before_read']):
                            target_since_us = last_write_us
                        trace.update(write_finished_us=last_write_us, write_succeeded=True,
                            write_duration_s=time.monotonic()-write_started,
                            written_target_since_us=target_since_us)
                # Target feedback and monitoring keep their existing order/rate.
                if self.delta_reference == 'previous_target' and not self.state_only:
                    self.publish_target(command)
                if now - last_health >= health_period:
                    health_started = time.monotonic() if trace is not None else None
                    if split_health:
                        self.hand.health_register(health_motor, health_register_index)
                    else:
                        self.hand.health(health_motor)
                    if trace is not None:
                        trace.update(health_motor=health_motor,
                            health_register_index=health_register_index if split_health else None,
                            health_duration_s=time.monotonic()-health_started)
                    if split_health:
                        health_register_index = 1 - health_register_index
                    if not split_health or health_register_index == 0:
                        health_motor = (health_motor + 1) % 16
                    last_health = now
            except BaseException as exc:
                if trace is not None:
                    trace['error'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                if trace is not None:
                    trace['cycle_finished_us'] = int(time.time()*1e6)
                    self.telemetry.submit(trace)
        print('Bridge shutdown requested; disabling torque.', flush=True)

    def publish_target(self, command):
        self.connection.publish(self.state_channel + '_TARGET',
            RobotTargetState(int(time.time()*1e6), command).encode())

    def close(self):
        try:
            if self.subscription is not None:
                self.connection.unsubscribe(self.subscription)
        finally:
            try:
                if self.state_only:
                    self.hand.close_read_only()
                else:
                    self.hand.close()
            finally:
                if self.telemetry is not None:
                    self.telemetry.close()


def main(argv=None):
    stop_requested = False
    bridge = None
    def interrupt(sig, frame):
        # Never raise inside the upstream serial transaction: its bare except
        # can swallow KeyboardInterrupt and leave the port busy during cleanup.
        nonlocal stop_requested
        stop_requested = True
        if bridge is not None:
            bridge.stop_requested = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-api", type=Path,
                        default=Path.home() / "leap_agentic_sysid/third_party/LEAP_Hand_Hardware")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs/leap_hardware.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--port")
    parser.add_argument("--lcm-url", default=DEFAULT_URL)
    parser.add_argument("--state-channel", default="LEAP_STATE")
    parser.add_argument("--action-channel", default=DELTA_ACTION_CHANNEL)
    parser.add_argument("--action-timeout", type=float, default=0.5)
    parser.add_argument('--telemetry-log', type=Path,
                        help='optional asynchronous JSONL log of reads, applied targets and write times')
    parser.add_argument('--status-file',type=Path,help='write a software-hold notification for the launcher')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--state-only", action="store_true",
                        help="publish measured joints without changing torque/registers or subscribing to actions")
    mode.add_argument('--disable-torque-only', action='store_true',
                      help='disable and verify torque on every motor, then exit; no target or enable commands')
    args = parser.parse_args(argv)
    if not np.isfinite(args.action_timeout) or args.action_timeout <= 0:
        parser.error("action timeout must be finite and positive")
    config = json.loads(args.hardware_config.expanduser().read_text())
    hand = HardwareHand(args.hardware_api, config, args.port)
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, interrupt)
    try:
        if args.disable_torque_only:
            hand.connect_read_only()
        else:
            import lcm
            model = mujoco.MjModel.from_xml_path(str(args.model.expanduser()))
            bridge = LeapHandLCMBridge(hand, lcm.LCM(args.lcm_url), model, config,
                                      args.state_channel, args.action_channel, args.action_timeout, args.state_only)
            bridge.fault_status_file=args.status_file
            if args.telemetry_log is not None:
                bridge.telemetry = BridgeTelemetry(args.telemetry_log, dict(
                    joint_order=[model.joint(i).name for i in range(16)],
                    current_order='native motor IDs 0..15', delta_reference=bridge.delta_reference,
                    state_only=args.state_only,
                    delta_lower=bridge.delta_lower.tolist(), delta_upper=bridge.delta_upper.tolist(),
                    joint_limits=bridge.joint_limits.metadata() if bridge.joint_limits is not None else None,
                    hardware_config=config, timestamp_units='epoch microseconds',
                    write_semantics='SDK write returned successfully; no extra motor register readback',
                    read_semantics='measurement occurred within read_started_us/read_finished_us interval'))
            bridge.stop_requested = stop_requested
            bridge.run()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if bridge is not None:
                bridge.close()
            else:
                hand.close()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    main()
