"""Launcher takeover after a software process fails; never opens the serial port."""
import json
import signal
import time


def can_hold_after_failure(experiment, hardware_config, state, code):
    if (not code or not hardware_config.get('hold_on_non_motor_failure', False)
            or state.get('stop') or state.get('force_stop')):
        return False
    if not all(name in experiment.children for name in ('bridge', 'controller')):
        return False
    if experiment.children['bridge'].process.poll() is not None:
        return False
    try:
        stage = json.loads((experiment.directory/'episode_status.json').read_text()).get('stage')
    except (OSError, ValueError):
        stage = None
    return stage not in ('complete', 'interrupt_complete', 'feedback_complete', 'software_complete',
                         'feedback_failed', 'software_failed', 'interrupt_failed')


def recover_after_failure(experiment, config, hardware_config, commands, state, reason,
                          feedback_factory=None, episode_factory=None):
    from .LCM import LCMFeedback
    from .Episodes import HardwareEpisodes
    feedback_factory = feedback_factory or LCMFeedback
    episode_factory = episode_factory or HardwareEpisodes

    def guard():
        if state.get('force_stop') or state.get('stop') == 'SIGTERM':
            raise RuntimeError('Immediate stop requested during software recovery')
        if experiment.children['bridge'].process.poll() is not None:
            raise RuntimeError('Bridge stopped during software recovery')
        for name in ('viewer', 'mujoco'):
            if name in experiment.children and experiment.children[name].process.poll() == 0:
                raise RuntimeError('Viewer closed by operator during software recovery')

    # Prevent any remaining planner thread/process from publishing commands.
    # The bridge independently latches its last target on watchdog expiry while
    # this takeover initializes; it continues every motor/USB/current check.
    controller = experiment.children['controller'].process
    if controller.poll() is None:
        experiment.send('controller', signal.SIGTERM)
        deadline = time.monotonic()+1.
        while controller.poll() is None and time.monotonic()<deadline:
            guard()
            time.sleep(.02)
        if controller.poll() is None:
            experiment.send('controller', signal.SIGKILL)
            deadline = time.monotonic()+1.
            while controller.poll() is None and time.monotonic()<deadline:
                guard()
                time.sleep(.02)
        if controller.poll() is None:
            raise RuntimeError('Controller did not stop; refusing concurrent recovery commands')
    # Parse just the shared safety setting, without importing/warming a planner.
    flags=commands.get('controller', [])
    max_age=.15
    for i, value in enumerate(flags):
        if value=='--max-robot-age':max_age=float(flags[i+1])
        elif value.startswith('--max-robot-age='):max_age=float(value.partition('=')[2])
    feedback=feedback_factory(None, config['lcm_url'], config['state_channel'], None,
                              config['action_channel'])
    try:
        deadline=time.monotonic()+1.
        while True:
            guard()
            feedback.publish_keep_target()  # Retain the exact servo target; no new motion.
            try:
                feedback.buffer.snapshot_robot(max_robot_age=max_age)
                break
            except RuntimeError:
                if feedback.buffer.error is not None or time.monotonic()>=deadline:
                    raise
                time.sleep(.02)
        episode=0
        try:
            episode=int(json.loads((experiment.directory/'episode_status.json').read_text()).get('episode',0))
        except (OSError, ValueError):
            pass
        handler=episode_factory(feedback, hardware_config, max_age,
                                experiment.directory/'episode_status.json', experiment.logging_enabled)
        handler.recovery_guard=guard
        original_enter=handler.enter_pressed
        def entered(timeout):
            guard()
            return state.get('stop')=='SIGINT' or original_enter(timeout)
        handler.enter_pressed=entered
        q=handler.shutdown_after_error(episode, reason)
        return dict(zero_reached=True, qpos=q.tolist(), original_reason=reason,
                    recovery='launcher joint-only takeover; original run remains failed')
    finally:
        try:
            feedback.publish_stop()
        finally:
            feedback.close()
