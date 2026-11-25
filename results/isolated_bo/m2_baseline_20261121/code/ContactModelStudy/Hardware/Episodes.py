"""Hardware episode preparation and supervised holds, separate from MPPI timing."""
import json
from pathlib import Path
import select
import sys
import time

import numpy as np

from .Hand import vector
from .Scene import JOINTS


class HardwareEpisodes:
    def __init__(self, feedback, config, max_robot_age, status_file=None, logging_enabled=True):
        self.feedback, self.max_robot_age = feedback, max_robot_age
        self.reference = vector(config["robot_qref"], "robot_qref").copy()
        preparation = config.get("episode_preparation", {})
        self.speed = float(preparation.get("speed_rad_s", 0.3))
        self.tolerance = float(preparation.get("tolerance_rad", 0.06))
        self.joint_tolerances = preparation.get("joint_tolerance_rad", {})
        if (not isinstance(self.joint_tolerances, dict)
                or set(self.joint_tolerances) - set(JOINTS)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not np.isfinite(value) or value <= 0 for value in self.joint_tolerances.values())):
            raise ValueError("joint_tolerance_rad must map known joint names to finite positive radians")
        self.settle_time = float(preparation.get("settle_time_s", 0.3))
        self.settle_velocity = float(preparation.get("settle_velocity_rad_s", 0.08))
        self.timeout = float(preparation.get("timeout_s", 0))
        self.interrupt_timeout = float(preparation.get("interrupt_timeout_s", 30))
        rate = float(preparation.get("command_hz", 50))
        if not all(np.isfinite(value) and value > 0 for value in
                   (self.speed, self.tolerance, rate, self.settle_time, self.settle_velocity)):
            raise ValueError("episode_preparation values must be finite and positive")
        if not np.isfinite(self.timeout) or self.timeout < 0:
            raise ValueError("episode_preparation.timeout_s must be nonnegative (0 waits for settling)")
        if not np.isfinite(self.interrupt_timeout) or self.interrupt_timeout <= 0:
            raise ValueError("episode_preparation.interrupt_timeout_s must be finite and positive")
        self.period = 1 / rate
        if self.period >= 0.25:
            raise ValueError("episode_preparation.command_hz must exceed 4 to renew the bridge watchdog")
        self.status_file = None if status_file is None else Path(status_file)
        self.logging_enabled = logging_enabled
        self.last_target = None
        self.hold_on_failure = config.get('hold_on_non_motor_failure', False)
        self.recovering = False
        self.recovery_guard = None

    def status(self, stage, episode, prompt, **details):
        print(prompt, flush=True)
        if self.status_file is not None:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            pending = self.status_file.with_suffix(".tmp")
            pending.write_text(json.dumps(dict(stage=stage, episode=episode, prompt=prompt, **details)) + "\n")
            pending.replace(self.status_file)

    def motion_progress(self, episode, stage, label, target, q, v, elapsed, state):
        error = target - q
        index = int(np.argmax(np.abs(error)))
        tolerances = self.position_tolerances()
        limiting = int(np.argmax(np.abs(error) / tolerances))
        details = dict(motion_state=state, elapsed_s=float(elapsed),
                       joint_order=list(JOINTS), target=target.tolist(), measured=q.tolist(),
                       velocity=v.tolist(), error=error.tolist(), tolerance_rad=float(tolerances[index]),
                       joint_tolerances_rad=tolerances.tolist(), timeout_s=self.timeout,
                       limiting_joint=JOINTS[limiting], limiting_error_rad=float(abs(error[limiting])),
                       limiting_tolerance_rad=float(tolerances[limiting]),
                       settle_time_s=self.settle_time, settle_velocity_rad_s=self.settle_velocity,
                       worst_joint=JOINTS[index], max_error_rad=float(abs(error[index])))
        prefix = f"Preparing episode {episode + 1}" if stage == 'preparing' else f"Reset after episode {episode + 1}"
        prompt = (f"{prefix}: {label} {state}, {elapsed:.1f}s; "
                  f"max error {abs(error[index]):.4f} rad ({np.degrees(abs(error[index])):.1f} deg) "
                  f"at {JOINTS[index]}; tolerance {tolerances[index]:.4f} rad.")
        self.status(stage, episode, prompt, **details)
        if self.status_file is not None and self.logging_enabled:
            trace = self.status_file.with_name('preparation_trace.jsonl')
            with trace.open('a') as stream:
                stream.write(json.dumps(dict(episode=episode, stage=stage, **details)) + '\n')
        return details

    def position_tolerances(self):
        return np.array([self.joint_tolerances.get(name, self.tolerance) for name in JOINTS])

    def command_target(self, target):
        """Publish an absolute preparation waypoint, independent of MPPI delta limits."""
        publish = self.feedback.publish_recovery_position if self.recovering else self.feedback.publish_position
        publish(vector(target, 'preparation target'))
        self.last_target = vector(target, 'preparation target').copy()

    def joints(self):
        if self.recovery_guard is not None:
            self.recovery_guard()
        q, v = self.feedback.buffer.snapshot_robot(max_robot_age=self.max_robot_age)
        return vector(q, "measured position"), vector(v, "measured velocity")

    def prepare(self, episode):
        self.move_to(self.reference, episode, "preparing", "robot_qref")

    def move_to(self, target, episode, stage, label):
        prefix = f"Preparing episode {episode + 1}" if stage == 'preparing' else f"Reset after episode {episode + 1}"
        self.status(stage, episode, f"{prefix}: gradually moving to {label}. MPPI is stopped.")
        initial, _ = self.joints()
        duration = np.max(np.abs(target - initial)) / self.speed
        started = time.monotonic()
        last_report = -float('inf')
        settled_since = None
        while True:
            q, v = self.joints()
            error = target - q
            now = time.monotonic()
            elapsed = now - started
            settled = (elapsed >= duration and np.all(np.abs(error) <= self.position_tolerances())
                       and np.max(np.abs(v)) <= self.settle_velocity)
            if settled:
                if settled_since is None:
                    settled_since = now
            else:
                settled_since = None
            if settled_since is not None and now - settled_since >= self.settle_time:
                # Keep the reference command, including its steady servo error;
                # capturing measured q here would lower the target and relax it.
                self.command_target(target)
                self.motion_progress(episode, stage, label, target, q, v, elapsed, 'reached')
                return
            if self.timeout > 0 and elapsed >= self.timeout:
                info = self.motion_progress(episode, stage, label, target, q, v, elapsed, 'timeout')
                if np.all(np.abs(error) <= self.position_tolerances()):
                    raise RuntimeError(f"{label} position reached but hand did not settle before episode preparation timeout")
                raise RuntimeError(f"{label} pose was not reached before episode preparation timeout; "
                                   f"{info['limiting_joint']} error {info['limiting_error_rad']:.4f} rad "
                                   f"exceeds {info['limiting_tolerance_rad']:.4f} rad tolerance")
            if elapsed - last_report >= 1:
                state = 'holding target; waiting for settling' if elapsed >= duration else 'moving'
                self.motion_progress(episode, stage, label, target, q, v, elapsed, state)
                last_report = elapsed
            # Move an absolute waypoint gradually. Integrating tiny increments
            # on measured q alone can stall against friction/gravity forever.
            fraction = min((now - started) / duration, 1) if duration else 1
            waypoint = initial + fraction * (target - initial)
            self.command_target(waypoint)
            time.sleep(self.period)

    @staticmethod
    def enter_pressed(timeout):
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return False
        line = sys.stdin.readline()
        if line == "":
            raise RuntimeError("Episode confirmation needs an interactive stdin; received EOF")
        return True

    def wait_for_enter(self, episode, stage, prompt, target=None, preserve_target=False):
        self.status(stage, episode, prompt)
        def heartbeat():
            if preserve_target:self.feedback.publish_keep_target()
            elif target is None:self.feedback.publish_hold()
            else:self.command_target(target)
        while True:
            try:
                self.joints()  # retain robot freshness checks throughout a hold
            except RuntimeError as exc:
                if not str(exc).startswith('Stale robot feedback'):
                    raise
                # This is a stationary operator gate, never an MPPI or motion
                # step. Renew only the existing hold/waypoint while allowing
                # 100 ms for a NEW fresh state after a receive/scheduling gap.
                # A stale state is never returned or used to start an episode.
                deadline=time.monotonic()+.1
                while True:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise exc
                    heartbeat()
                    time.sleep(min(self.period,remaining))
                    try:
                        self.joints()
                        print(f'Episode {episode+1}: fresh joint feedback recovered during stationary hold.',flush=True)
                        break
                    except RuntimeError as retry_error:
                        if not str(retry_error).startswith('Stale robot feedback'):raise
            heartbeat()
            if self.enter_pressed(self.period):
                return

    def before_episode(self, episode):
        self.prepare(episode)
        self.wait_for_enter(episode, "ready",
            f"Episode {episode + 1}: reference pose reached. Place the object, then press Enter to start MPPI.",
            target=self.reference)

    def hold_for(self, seconds, target=None, on_tick=None):
        """Stationary video padding, with fresh joints and watchdog heartbeats."""
        deadline = time.monotonic() + seconds
        while True:
            self.joints()
            if target is None:
                self.feedback.publish_hold()
            else:
                self.command_target(target)
            if on_tick is not None:
                on_tick()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(self.period, remaining))

    def wait_for_tracking(self, episode, timeout, **limits):
        self.status("waiting_for_tracking", episode, f"Episode {episode + 1}: waiting for fresh joint/object poses.")
        deadline = time.monotonic() + timeout
        while True:
            self.joints()
            self.command_target(self.reference)
            try:
                state = self.feedback.buffer.snapshot(**limits)
            except RuntimeError:
                if self.feedback.buffer.error is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(self.period)
                continue
            self.status("running", episode, f"Episode {episode + 1}: MPPI starting.")
            return state

    def after_success(self, episode, final):
        self.feedback.publish_hold()
        action = "return to zero and disable torque" if final else "return to zero before the next episode"
        self.wait_for_enter(episode, "success_hold",
            f"Episode {episode + 1}: success. Holding the reached joint pose. Press Enter to {action}.")

    def after_unsuccessful(self, episode, reason):
        self.wait_for_enter(episode, "episode_hold",
            f"Episode {episode + 1}: {reason}. Holding the hand. Press Enter to return to zero.")

    def return_to_zero(self, episode, stage="returning_to_zero"):
        self.move_to(np.zeros(16), episode, stage, "zero")
        q, _ = self.joints()
        self.status(stage if stage == "feedback_returning_to_zero" else "zero_reached",
                    episode, f"Episode {episode + 1}: zero pose reached.")
        return q

    def shutdown_after_feedback_error(self, episode, error):
        """Hold the last target until Enter, then reset using joint feedback only."""
        if self.hold_on_failure:
            return self.shutdown_after_error(episode, error)
        timeout = self.timeout
        try:
            q, _ = self.joints()  # A camera failure must not hide a robot failure.
            target = q if self.last_target is None else self.last_target.copy()
            self.command_target(target)  # Renew the watchdog before prompting.
            try:
                self.wait_for_enter(episode, "feedback_hold",
                    f"Feedback timing error: {error}. MPPI stopped; holding the last target with torque enabled. "
                    "Press Enter to return gradually to zero and disable torque (Ctrl-C also returns to zero).",
                    target=target)
            except KeyboardInterrupt:
                pass  # The first Ctrl-C confirms reset; another aborts the move.
            self.timeout = self.interrupt_timeout
            q = self.return_to_zero(episode, stage="feedback_returning_to_zero")
            self.status("feedback_complete", episode,
                        "Feedback error: zero pose reached. Disabling torque; experiment remains aborted.")
            return q
        except BaseException:
            self.status("feedback_failed", episode,
                        "Feedback-error hold/reset interrupted or failed. Disabling torque immediately.")
            raise
        finally:
            self.timeout = timeout
            self.feedback.publish_stop()

    def shutdown_on_interrupt(self, episode):
        """Stop planning, return using fresh joints only, then request torque off."""
        timeout = self.timeout
        self.timeout = self.interrupt_timeout
        try:
            q = self.return_to_zero(episode)
            self.status("interrupt_complete", episode,
                        "Ctrl-C: zero pose reached. Disabling torque.")
            return q
        except BaseException:
            self.status("interrupt_failed", episode,
                        "Ctrl-C: zero return interrupted or failed. Disabling torque immediately.")
            raise
        finally:
            self.timeout = timeout
            self.feedback.publish_stop()

    def shutdown_after_error(self, episode, error):
        """Software-fault recovery: preserve exact bridge target until explicit Enter."""
        timeout, previous_recovering = self.timeout, self.recovering
        summary=' '.join(str(error).split())[:180]
        try:
            self.joints()  # Never turn a robot-state failure into a blind hold/reset.
            self.feedback.publish_keep_target()
            self.recovering = True
            try:
                self.wait_for_enter(episode, 'software_hold',
                    f'Software failure: {summary}. MPPI stopped; holding the last bridge target. '
                    'Remove the object, then press Enter to return gradually to zero and disable torque. '
                    'Ctrl-C confirms reset; another interrupt stops immediately.', preserve_target=True)
            except KeyboardInterrupt:
                pass
            self.timeout = self.interrupt_timeout
            q = self.return_to_zero(episode, stage='software_returning_to_zero')
            self.status('software_complete', episode,
                        'Software failure: zero pose reached. Disabling torque; run remains aborted.')
            return q
        except BaseException:
            self.status('software_failed', episode,
                        'Software-error hold/reset failed or interrupted. Disabling torque immediately.')
            raise
        finally:
            self.timeout, self.recovering = timeout, previous_recovering
            self.feedback.publish_stop()
