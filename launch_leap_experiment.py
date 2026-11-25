#!/usr/bin/env python3
"""Reuse persistent tracking while running supervised LEAP bridge and MPPI.

Ctrl-C stops MPPI, returns to zero, then disables torque. A second Ctrl-C,
Q/Escape or closing a viewer stops immediately. Configured software-failure
recovery holds the last target for operator-confirmed zero return; motor and USB
faults still disable torque immediately. Tracking, its receiver and container
remain running between runs. During a controller software-failure hold/reset, stale camera frames or a stopped
tracker do not end the joint-only hold; viewer closure and bridge faults still do.
The first launch captures a mask; --restart-tracking explicitly recaptures and
reinitializes, and --stop-tracking shuts down the saved vision services only.
An owned SAM-HQ server is stopped after mask acceptance, before registration.
The receiver must display fresh frames
before the bridge/controller start. Tracking correctness is judged by the
operator, not by this launcher. --dry-run starts no experiment services;
GPU selection only queries nvidia-smi.

    python launch_leap_experiment.py --bbox 420 200 180 200
    python launch_leap_experiment.py --mask mask.jpg --observe-only
    python launch_leap_experiment.py --bbox 420 200 180 200 -- --rollout-model M3 --hand-acc low
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from ContactModelStudy.Hardware.UsbLatency import UsbLatencyTimer, MANAGED_ENV

from ContactModelStudy.Hardware.TrackingSession import (
    TrackingSession, VISION_SERVICES, process_alive, process_identity, tracking_spec)
from ContactModelStudy.Hardware.ExperimentLogs import ConfigurationLogs
from ContactModelStudy.Hardware.GPU import select_gpu
from ContactModelStudy.Hardware.FailureSummary import short_failure
from ContactModelStudy.Hardware.Recovery import can_hold_after_failure, recover_after_failure
from ContactModelStudy.Hardware import Continuation

ROOT = Path(__file__).resolve().parent
# Physical visual meshes from the reference eval scenes, in the obj_joint frame.
# Rollout approximations (including duck FOAM spheres) must not change tracking.
TRACKING_MESHES = {
    "cube_reorient": ("scenes/leap/objects/iris_cube_tracking.obj", 0.035),
    "ball_reorient": ("scenes/leap/objects/uv_sphere.obj", 0.0485),
    "duck_reorient": ("scenes/leap/objects/rubber_duck.stl", 1.0),
}


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def stop_tracking_directory(directory,explicit=False):
    """Find the isolated calibration session when the default registry is empty.

    Explicit/custom registry selections are always honored. Normal experiment
    startup is unchanged; this fallback is only for the stop-only command.
    """
    directory=local_path(directory)
    if not explicit and directory==local_path('.runtime/leap_tracking') and not (directory/'session.json').exists():
        calibration=ROOT/'camera_hand_calibration/runtime/tracking'
        if (calibration/'session.json').exists():return calibration
    return directory


def interpreter(value):
    value = sys.executable if value is None else os.path.expanduser(str(value))
    resolved = shutil.which(value)
    if resolved is None:
        raise ValueError(f"Python interpreter not found/executable: {value}")
    return resolved


def docker_executable(config):
    value = config.get("docker", {}).get("executable", "docker")
    return str(local_path(value)) if "/" in value else value


def string_args(value, name):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON list of argument strings")
    return value


def controller_selection(controller_args):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--task", default="cube_reorient", choices=sorted(TRACKING_MESHES))
    selector.add_argument("--model")
    selector.add_argument("--calibrated-hand", default="scenes/leap/hardware/calibrated_hand.xml")
    selector.add_argument("--max-robot-age", type=float, default=0.15)
    selector.add_argument("--max-pose-age", type=float, default=0.3)
    selector.add_argument("--max-state-skew", type=float, default=0.15)
    selected, _ = selector.parse_known_args(controller_args)
    return selected


def tracking_mesh(config, selected):
    """Select the physical object using the controller's final task option."""
    mesh, scale = config.get("mesh"), config.get("mesh_scale")
    if mesh is None:
        if selected.model is not None:
            raise ValueError("A custom --model requires an explicit --mesh in its object body frame")
        override = config.get('tracking_meshes', {}).get(selected.task)
        if override is not None:
            mesh, default_scale = override['mesh'], override.get('mesh_scale', 1.0)
        else:
            mesh, default_scale = TRACKING_MESHES[selected.task]
        scale = default_scale if scale is None else scale
    elif scale is None:
        scale = 1.0
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("mesh_scale must be finite and positive")
    path = local_path(mesh)
    if not path.is_file():
        raise ValueError(f"mesh file not found: {path}")
    return str(path), scale


def build_commands(config, directory, controller_args=()):
    """Build argv arrays without shell interpolation; shared channel settings win."""
    camera = str(local_path(config["camera_config"]))
    planning_camera=camera
    if config.get('continued_camera_config') is not None:
        planning_camera = str(local_path(config['continued_camera_config']))
    elif config.get('camera_calibration') is not None:
        from camera_hand_calibration.selection import selected_camera_config
        planning_camera=str(selected_camera_config(config['camera_calibration'],camera))
    mask = str(local_path(config["init_mask_path"]))
    hardware = str(local_path(config["hardware_config"]))
    python = config.get("interpreters", {})
    docker = config.get("docker", {})
    in_container = docker.get("enabled", False)
    vision = docker.get("python", "python") if in_container else interpreter(python.get("vision"))
    control = interpreter(python.get("control"))
    viewer = interpreter(python.get("viewer") or (None if in_container else python.get("vision")))
    tracker_args = string_args(config.get("tracker_args", []), "tracker_args")
    viewer_args = string_args(config.get("viewer_args", []), "viewer_args")
    bridge_args = string_args(config.get("bridge_args", []), "bridge_args")
    mppi_args = string_args(config.get("controller_args", []), "controller_args") + list(controller_args)
    gpu = config.get('gpu_assignment')
    tracking_gpu = config.get('tracking_gpu_assignment') or gpu
    if gpu:
        for value in mppi_args:
            if value.split('=', 1)[0] in ('--device', '--fingertip-device'):
                raise ValueError('Set the physical GPU with --mppi-gpu or --gpu/--device before --; child CUDA devices are local cuda:0')
    selected = controller_selection(mppi_args)
    mesh, mesh_scale = tracking_mesh(config, selected)
    calibration = local_path(selected.calibrated_hand)
    if not calibration.is_file():
        raise ValueError(f"Calibrated hand file not found: {calibration}")
    if any(value.split("=", 1)[0] == "--model" for value in bridge_args):
        raise ValueError("Configure the bridge's hand model through controller --calibrated-hand")
    # Reject conflicting per-process settings instead of silently running streams
    # with different calibrations, channels, meshes or endpoints.
    reserved = {"--camera-config", "--pose-transform", "--camera-calibration", "--lcm-url", "--state-channel", "--pose-channel",
                "--action-channel", "--hardware-config", "--viewer-bind", "--endpoint", "--ready-file",
                "--mesh_path", "--init_mask_path", "--mesh", "--mesh-scale", "--apply_scale",
                "--hardware", "--lcm", "--observe-only", "--capture-trigger", "--trigger_path", "--preview",
                "--reuse-mask"}
    reserved.update({"--mask-confirm-trigger", "--review-mask", "--registration-frame-path", "--goal-status-file",
                     "--logging", "--no-logging", '--overwrite-logs', '--fingertip-path-file',
                     '--record-video', '--no-record-video', '--video-fps', '--video-padding-seconds',
                     '--telemetry-log', '--bridge-telemetry', '--no-bridge-telemetry'})
    for values in (tracker_args, viewer_args, bridge_args, mppi_args):
        for value in values:
            if value.split("=", 1)[0] in reserved:
                raise ValueError(f"Configure shared setting {value} through the launcher/config")
    lcm = ["--lcm-url", config["lcm_url"]]
    channels = ["--state-channel", config["state_channel"], "--action-channel", config["action_channel"]]
    scripts = ROOT / "iris-foundationpose-plusplus"
    commands = {
        "tracker": [vision, "-u", str(scripts / "src/online_tracking2ctrl.py"), *tracker_args,
            "--registration-frame-path", str(directory / "registration_frame.npz"),
            "--camera-config", camera, "--mesh_path", mesh, "--init_mask_path", mask,
            "--apply_scale", str(mesh_scale), "--camera_index", str(config["camera_index"]),
            "--viewer-bind", config["viewer_bind"], "--pose-channel", config["pose_channel"], *lcm],
        "viewer": [viewer, "-u", str(scripts / "viewer/receiver.py"), *viewer_args,
            "--camera-config", camera, "--endpoint", config["viewer_endpoint"], "--draw-axes",
            "--mesh", mesh, "--mesh-scale", str(mesh_scale),
            "--ready-file", str(directory / "viewer_heartbeat.json")],
        "bridge": [control, "-u", str(ROOT / "leap_hand_lcm_bridge.py"), *bridge_args,
            "--model", str(calibration), "--hardware-config", hardware,
            '--status-file',str(directory/'bridge_status.json'), *channels, *lcm],
        "controller": [control, "-u", str(ROOT / "run_leap_hardware.py"),
            "--episode-status-file", str(directory / "episode_status.json"),
            "--goal-status-file", str(directory / "goal.json"),
            "--results", str(directory / "controller.json"), *mppi_args, "--hardware",
            "--camera-config", planning_camera, "--hardware-config", hardware,
            "--pose-channel", config["pose_channel"], *channels, *lcm],
    }
    if config.get('bridge_telemetry', False):
        if not config.get('logging', True):
            raise ValueError('--bridge-telemetry requires logging')
        commands['bridge'].extend(['--telemetry-log', str(directory/'bridge_telemetry.jsonl')])
    if config.get("observe_only", False):
        commands["controller"].append("--observe-only")
    commands['controller'].append('--logging' if config.get('logging', True) else '--no-logging')
    if config.get('record_video', False):
        ffmpeg = shutil.which(config.get('video_ffmpeg', 'ffmpeg'))
        if ffmpeg is None:
            raise ValueError('Video recording requires FFmpeg on the host; install ffmpeg or omit --record-video')
        if not config.get('observe_only', False):
            commands['controller'].extend(['--video-padding-seconds', '1.0'])
        commands['video'] = [viewer, '-u', str(ROOT/'record_leap_video.py'),
            '--endpoint', config['viewer_endpoint'], '--output-dir', str(directory/'video'),
            '--fps', str(config.get('video_fps', 30)), '--ffmpeg', ffmpeg,
            '--episode-windows-file', str(directory/'goal.json')]
        ionice = shutil.which('ionice')
        if ionice:
            commands['video'] = [ionice, '-c', '3', *commands['video']]
    if gpu:
        commands['controller'].extend(['--device', 'cuda:0'])
        if config.get('fingertip_paths', False):
            commands['controller'].extend(['--fingertip-device', 'cuda:0'])
    if config.get("mujoco_viewer", True):
        commands["mujoco"] = [control, "-u", str(ROOT / "view_leap_tracking.py"),
            "--task", selected.task, "--calibrated-hand", str(calibration),
            "--camera-config", planning_camera, "--state-channel", config["state_channel"],
            "--pose-channel", config["pose_channel"], *lcm,
            "--mesh", mesh, "--mesh-scale", str(mesh_scale),
            "--max-robot-age", str(selected.max_robot_age),
            "--max-pose-age", str(selected.max_pose_age),
            "--max-state-skew", str(selected.max_state_skew),
            "--status-file", str(directory / "mujoco_status.json")]
        commands["mujoco"].extend(["--goal-file", str(directory / "goal.json")])
        if config.get('fingertip_paths', False):
            paths = ['--fingertip-path-file', str(directory/'fingertip_paths.json')]
            commands['controller'].extend(paths)
            commands['mujoco'].extend(paths)
        if selected.model:
            commands["mujoco"].extend(["--model", str(local_path(selected.model))])
    sam = config.get("sam_hq", {})
    if config.get("generate_mask", False):
        checkpoint = ["--checkpoint_path", sam["checkpoint_path"]] if sam.get("checkpoint_path") else []
        if sam.get("start_server", True):
            commands["sam"] = [vision, "-u", str(scripts / "src/WebAPI/hq_sam_api.py"),
                                "--port", str(sam["port"]), *checkpoint]
    commands["mask"] = [vision, "-u", str(scripts / "src/utils/camera_mask.py"),
        "--once", "--preview", "--viewer-bind", config["viewer_bind"],
        "--trigger_path", str(directory / "capture_mask.trigger"),
        "--review-mask", "--mask-confirm-trigger", str(directory / "accept_mask.trigger"),
        "--registration-frame-path", str(directory / "registration_frame.npz"),
        "--camera-config", camera, "--camera_index", str(config["camera_index"]),
        "--bbox_xywh", json.dumps(config["bbox_xywh"]), "--output_mask_path", mask,
        "--snapshot-path", str(directory / "mask_frame.png"),
        "--web_api_url", sam.get("api_url", "http://127.0.0.1:9069/hq_sam")]
    commands["viewer"].extend(["--capture-trigger", str(directory / "capture_mask.trigger")])
    commands["viewer"].extend(["--mask-confirm-trigger", str(directory / "accept_mask.trigger")])
    if not config.get("generate_mask", False):
        commands["mask"].append("--reuse-mask")
        commands["viewer"].append("--reuse-mask")
    if in_container:
        executable = docker_executable(config)
        name = "leap_vision_" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
        commands["container"] = ["bash", str(scripts / "docker/run_container.sh"), "--detach",
            "--repo-root", str(ROOT), "--docker-executable", executable,
            "--image", docker["image"], "--name", name]
        mounts = {directory, Path(camera).parent, Path(mesh).parent, Path(mask).parent}
        mounts.update(local_path(path) for path in docker.get("extra_mounts", []))
        for path in sorted(mounts):
            if not path.is_relative_to(ROOT):
                commands["container"].extend(["--mount-path", str(path)])
        commands["container"].extend(["--", "sleep", "infinity"])
        for service in ("sam", "mask", "tracker"):
            if service in commands:
                if service == 'sam':
                    commands[service] = [vision, '-u', str(ROOT/'leap_owned_service.py'),
                        'run', '--pid-file', str(directory/'sam.pid.json'), '--', *commands[service]]
                environment = []
                visible = tracking_gpu['cuda_visible_devices'] if tracking_gpu else docker.get('cuda_visible_devices')
                if visible is not None:
                    environment += ['-e', 'CUDA_VISIBLE_DEVICES=' + str(visible)]
                if service == 'tracker' and docker.get('foundationpose_root'):
                    environment += ['-e', 'FOUNDATIONPOSE_ROOT=' + docker['foundationpose_root']]
                commands[service] = [executable, "exec", *environment, name, *commands[service]]
    return commands


def server_ready(url):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def fresh_display(path, timeout, now=None, stage=None):
    try:
        heartbeat = json.loads(Path(path).read_text())
        if stage is not None and heartbeat.get("stage") != stage:
            return False
        now = time.time() if now is None else now
        ages = [now - float(heartbeat[key]) for key in ("displayed_at", "frame_timestamp")]
        return all(math.isfinite(age) and -0.1 <= age <= timeout for age in ages)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def fresh_mujoco_display(path, timeout, now=None):
    """Window heartbeat, including while it waits for the bridge's first state."""
    try:
        record = json.loads(Path(path).read_text())
        age = (time.time() if now is None else now) - float(record["displayed_at"])
        return math.isfinite(age) and -0.1 <= age <= timeout
    except (OSError, ValueError, TypeError, KeyError):
        return False


class TrackingReadiness:
    """Require a steady stream, excluding slow registration/first-track frames."""
    def __init__(self, frames=5, max_age=0.3):
        self.required, self.max_age = frames, max_age
        self.count, self.last_timestamp = 0, None

    def update(self, path, now=None):
        if not fresh_display(path, self.max_age, now=now, stage="tracking"):
            self.count = 0
            return False
        try:
            stamp = float(json.loads(Path(path).read_text())["frame_timestamp"])
        except (OSError, ValueError, TypeError, KeyError):
            self.count = 0
            return False
        if self.last_timestamp is None or stamp > self.last_timestamp:
            if self.last_timestamp is not None and stamp - self.last_timestamp > self.max_age:
                self.count = 0
            self.count += 1
            self.last_timestamp = stamp
        elif stamp < self.last_timestamp:
            self.count = 0
        return self.count >= self.required


@dataclass
class Child:
    name: str
    process: subprocess.Popen
    log: object


class Experiment:
    def __init__(self, directory, logging_enabled=True, gpu_assignment=None, tracking_gpu_assignment=None):
        self.directory = directory
        self.logging_enabled = logging_enabled
        self.children = {}
        self.completed_services = set()
        self.container = None
        self.tracking_directory = directory
        self.tracking_session = None
        self.tracking_spec = None
        self.keep_tracking = False
        self.reusing_tracking = False
        self.gpu_assignment = gpu_assignment
        self.tracking_gpu_assignment = tracking_gpu_assignment or gpu_assignment
        self.video_warnings = []

    def start(self, name, command):
        stream = ((self.directory / f"{name}.log").open("w") if self.logging_enabled
                  else open(os.devnull, 'w'))
        try:
            environment = None
            gpu = self.tracking_gpu_assignment if name in VISION_SERVICES else self.gpu_assignment
            if gpu:
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['cuda_visible_devices'])
            if name == 'video':
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                                   OPENBLAS_NUM_THREADS='1')
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=environment)
        except BaseException:
            stream.close()
            raise
        self.children[name] = Child(name, process, stream)
        destination = f"log: {self.directory / (name + '.log')}" if self.logging_enabled else 'logging disabled'
        print(f"Started {name} (pid {process.pid}); {destination}", flush=True)

    def feedback_recovery_active(self):
        try:
            status = json.loads((self.directory / "episode_status.json").read_text())
            return status.get("stage") in ("feedback_hold", "feedback_returning_to_zero",
                                            'software_hold', 'software_returning_to_zero')
        except (OSError, ValueError):
            return False

    def finished(self):
        recovering = self.feedback_recovery_active()
        for name, child in self.children.items():
            if name in self.completed_services:
                continue
            code = child.process.poll()
            if code is not None:
                if name == 'video':
                    if not self.video_warnings:
                        warning = f'Video recorder exited with code {code}; manipulation continues. See video.log/video/recording.json.'
                        self.video_warnings.append(warning)
                        print(warning, flush=True)
                    continue
                if recovering and name in ("tracker", "sam"):
                    continue  # Joint-only hold/reset does not need vision.
                if name in ("container", "mask") and code == 0:
                    continue
                if name == "bridge" and code == 0 and "controller" in self.children:
                    try:
                        status = json.loads((self.directory / "episode_status.json").read_text())
                        if status.get("stage") in ("complete", "interrupt_complete", "feedback_complete", 'software_complete'):
                            # The controller explicitly stopped the bridge after
                            # its final zero return; wait for its result/log exit.
                            continue
                    except (OSError, ValueError):
                        pass
                if name == "controller" and code == 0:
                    return "complete", 0
                if name == "viewer" and code == 0:
                    return "viewer stopped by operator", 0
                if name == "mujoco" and code == 0:
                    return "MuJoCo viewer stopped by operator", 0
                detail = f"see {name}.log" if self.logging_enabled else 'logging disabled'
                return f"{name} exited with code {code}; {detail}", 1
        if self.reusing_tracking:
            skip = set(self.children) | ({"tracker", "sam"} if recovering else set())
            result=self.tracking_session.finished(skip=skip)
            if result is not None:return result
        if not recovering:
            try:
                status=json.loads((self.directory/'bridge_status.json').read_text())
                if status.get('stage')=='software_hold':
                    return 'Bridge software hold: '+status.get('reason','controller failure'),1
            except (OSError,ValueError):
                pass
        return None

    def send(self, name, sig):
        child = self.children[name]
        try:
            os.killpg(child.process.pid, sig)
        except ProcessLookupError:
            pass

    def stop_owned_sam(self, commands, grace):
        """Release only our SAM server before tracker registration allocates CUDA memory."""
        child = self.children.get('sam')
        if child is None:
            return  # Existing/shared SAM server was not started by this launcher.
        print('Mask accepted. Stopping owned SAM-HQ server before tracking registration...', flush=True)
        if self.container is not None:
            # Killing docker-exec's host process does not reliably signal the
            # Python/CUDA server inside the container. Stop it in its namespace.
            helper = str(ROOT/'leap_owned_service.py')
            command = commands['sam']
            index = command.index(helper)
            result = subprocess.run([*command[:index], helper, 'stop', '--pid-file',
                str(self.directory/'sam.pid.json'), '--timeout', str(grace)],
                capture_output=True, text=True, timeout=grace+6, start_new_session=True)
            if result.returncode:
                raise RuntimeError('SAM-HQ shutdown unconfirmed; tracker not started: '
                                   + (result.stderr or result.stdout).strip())
        elif child.process.poll() is None:
            self.send('sam', signal.SIGINT)
        # Also reap the host child / docker-exec client. The container helper
        # already verified the CUDA-owning server exited before this point.
        for sig, timeout in ((None, grace), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
            if sig is not None and child.process.poll() is None:
                self.send('sam', sig)
            try:
                child.process.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                continue
        if child.process.poll() is None:
            raise RuntimeError('SAM-HQ process did not stop; tracker not started')
        self.completed_services.add('sam')
        print('SAM-HQ stopped; its GPU allocation is released. Starting tracking.', flush=True)

    def return_to_zero_on_interrupt(self, grace, state):
        """Keep feedback/bridge alive until the controller finishes its reset."""
        if not all(name in self.children and self.children[name].process.poll() is None
                   for name in ("controller", "bridge")):
            return None
        print("Ctrl-C: stopping MPPI and returning the hand to zero, then disabling torque. "
              "Press Ctrl-C again for immediate stop.", flush=True)
        self.send("controller", signal.SIGINT)
        deadline, last_prompt = time.monotonic() + grace, None
        while True:
            if state.get("force_stop"):
                return "Immediate stop requested during zero return", 1
            try:
                status = json.loads((self.directory / "episode_status.json").read_text())
            except (OSError, ValueError):
                status = {}
            if status.get("prompt") and status["prompt"] != last_prompt:
                print(status["prompt"], flush=True)
                last_prompt = status["prompt"]
            code = self.children["controller"].process.poll()
            if code is not None:
                if status.get("stage") in ("feedback_complete", 'software_complete'):
                    return "Software/feedback error: zero return completed; experiment aborted", 1
                if code == 0 and status.get("stage") in ("complete", "interrupt_complete"):
                    return "Ctrl-C: zero return completed; torque disable requested", 0
                return "Ctrl-C: controller did not complete zero return; disabling torque", 1
            if self.children["bridge"].process.poll() is not None and status.get("stage") not in ("complete", "interrupt_complete"):
                return "Bridge stopped during zero return; disabling torque", 1
            if any(name in self.children and self.children[name].process.poll() is not None
                   for name in ("viewer", "mujoco")):
                return "Viewer closed during zero return; disabling torque", 1
            skip = set(self.children) | ({"tracker", "sam"} if self.feedback_recovery_active() else set())
            if self.reusing_tracking and self.tracking_session.finished(skip=skip):
                return "Tracking stopped during zero return; disabling torque", 1
            if time.monotonic() >= deadline:
                return "Ctrl-C: zero-return deadline exceeded; disabling torque", 1
            time.sleep(0.05)

    def shutdown(self, grace):
        # Stop command publication and request bridge torque cleanup immediately.
        # Signal every group before waiting, so vision never delays motor cleanup.
        order = [name for name in ("controller", "bridge", "tracker", "viewer", "mujoco", "mask", "sam", "container", "video")
                 if name in self.children and name not in self.completed_services
                 and not (self.keep_tracking and name in VISION_SERVICES)]
        for name in order:
            # Only an explicit first Ctrl-C requests controller reset. Faults,
            # window closes and forced shutdown must never start a new move.
            self.send(name, signal.SIGTERM if name == "controller" else signal.SIGINT)
        # docker exec clients do not forward Unix signals reliably. Stop only
        # our uniquely named container, independently of bridge torque cleanup.
        docker_stop = None
        docker_stop_failed = False
        if self.container is not None and not self.keep_tracking:
            executable, name = self.container
            try:
                docker_stop = subprocess.Popen([executable, "stop", "--time", "2", name],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                docker_stop_failed = True
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if all(self.children[name].process.poll() is not None for name in order):
                break
            time.sleep(0.05)
        forced = []
        for name in order:
            if self.children[name].process.poll() is None:
                forced.append(name)
                self.send(name, signal.SIGTERM)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and any(self.children[name].process.poll() is None for name in order):
            time.sleep(0.05)
        for name in order:
            self.send(name, signal.SIGKILL)
        for name, child in self.children.items():
            if name in order:
                child.process.wait()
            child.log.close()
        if docker_stop is not None:
            try:
                stop_code = docker_stop.wait(timeout=5)
            except subprocess.TimeoutExpired:
                docker_stop.kill()
                docker_stop.wait()
                stop_code = -1
            docker_stop_failed = stop_code != 0
        if docker_stop_failed:
            forced.append("container stop unconfirmed")
        return forced


def wait_stage(experiment, state, timeout, predicate, description):
    deadline = time.monotonic() + timeout
    while True:
        finished = experiment.finished()
        if state["stop"] or finished is not None:
            return (state["stop"], 0) if state["stop"] else finished
        if predicate():
            return None
        if time.monotonic() >= deadline:
            return f"Timed out: {description}", 1
        time.sleep(0.05)


def prepare_tracking(experiment, commands, config, state):
    timeout = config["startup_timeout"]
    ready = experiment.tracking_directory / "viewer_heartbeat.json"
    if "container" in commands:
        command = commands["container"]
        experiment.container = (docker_executable(config),
                                command[command.index("--name") + 1])
        experiment.start("container", command)
        result = wait_stage(experiment, state, timeout,
            lambda: experiment.children["container"].process.poll() is not None, "starting vision container")
        if result is not None:
            return result
    if config.get("generate_mask", False):
        if "sam" in commands:
            experiment.start("sam", commands["sam"])
        print("Waiting for SAM-HQ model/server readiness...", flush=True)
        result = wait_stage(experiment, state, timeout,
            lambda: server_ready(config["sam_hq"]["health_url"]), "waiting for SAM-HQ")
        if result is not None:
            return result
    experiment.start("viewer", commands["viewer"])
    print("Opening live object-placement preview with the configured bounding box...", flush=True)
    experiment.start("mask", commands["mask"])
    result = wait_stage(experiment, state, timeout,
        lambda: fresh_display(ready, config["supervision_timeout"], stage="placement"),
        "waiting for bounding-box placement preview")
    if result is not None:
        return result
    action = "capture the mask" if config.get("generate_mask", False) else "start tracking with the existing mask"
    print(f"Place the object inside the yellow box. Press Space/Enter in the receiver to {action}; "
          "Q/Escape cancels.", flush=True)
    trigger = experiment.tracking_directory / "capture_mask.trigger"
    deadline = time.monotonic() + config.get("placement_timeout", 0)
    while not trigger.is_file():
        finished = experiment.finished()
        if state["stop"] or finished is not None:
            return (state["stop"], 0) if state["stop"] else finished
        if experiment.children["mask"].process.poll() is not None:
            return "Mask capture process exited before placement was confirmed", 1
        if not fresh_display(ready, config["supervision_timeout"], stage="placement"):
            return "Object-placement preview stopped receiving fresh frames", 1
        if config.get("placement_timeout", 0) > 0 and time.monotonic() >= deadline:
            return "Timed out waiting for object placement", 1
        time.sleep(0.05)
    print("Review the green mask in the receiver: Enter accepts; R returns to placement for a new mask.", flush=True)
    result = wait_stage(experiment, state, timeout,
        lambda: experiment.children["mask"].process.poll() is not None, "generating object mask")
    if result is not None:
        return result
    if not local_path(config["init_mask_path"]).is_file():
        return "Mask process completed without producing the object mask", 1
    if not (experiment.tracking_directory / "accept_mask.trigger").is_file():
        return "Mask capture finished without operator mask acceptance", 1
    if not (experiment.tracking_directory / "registration_frame.npz").is_file():
        return "Mask capture finished without its paired RGB-D registration frame", 1
    experiment.stop_owned_sam(commands, config['shutdown_timeout'])
    experiment.start("tracker", commands["tracker"])
    if "viewer" not in experiment.children:
        experiment.start("viewer", commands["viewer"])
    if experiment.tracking_session is not None:
        experiment.tracking_session.register(experiment, commands, experiment.tracking_spec)
        experiment.keep_tracking = True
    return None


def run_managed(experiment, commands, config, state):
    timeout = config['startup_timeout']
    ready = experiment.tracking_directory / 'viewer_heartbeat.json'
    if experiment.reusing_tracking:
        print(f'Reusing live tracking from {experiment.tracking_directory}; no mask capture or tracker restart.', flush=True)
        session = experiment.tracking_session
        if not process_alive(session.record['processes'].get('viewer')):
            ready.unlink(missing_ok=True)
            experiment.start('viewer', session.record['commands']['viewer'])
            session.replace_viewer(experiment.children['viewer'])
    else:
        result = prepare_tracking(experiment, commands, config, state)
        if result is not None:
            return result
    if "mujoco" in commands:
        experiment.start("mujoco", commands["mujoco"])
    readiness = TrackingReadiness(config.get("tracking_ready_frames", 5),
                                  min(config["supervision_timeout"], config.get("tracking_ready_max_age", 0.3)))
    print(f"Waiting for {readiness.required} fresh tracked frames and display readiness...", flush=True)
    result = wait_stage(experiment, state, timeout,
        lambda: readiness.update(ready) and ("mujoco" not in commands or fresh_mujoco_display(
            experiment.directory / "mujoco_status.json", config["supervision_timeout"])),
        "waiting for steady fresh tracking and display readiness")
    if result is not None:
        if experiment.reusing_tracking and result[1]:
            return result[0] + '; tracking was preserved. Use --restart-tracking to recapture if tracking is lost.', result[1]
        return result
    if 'video' in commands:
        try:
            experiment.start('video', commands['video'])
            print(f"Episode video recording enabled: {experiment.directory/'video/episode_XX.mkv'} (JPEG stream copy; 1s stationary lead-in/tail).", flush=True)
        except OSError as exc:
            warning = f'Video recorder could not start: {exc}; manipulation continues.'
            experiment.video_warnings.append(warning)
            print(warning, flush=True)
    experiment.start("bridge", commands["bridge"])
    experiment.start("controller", commands["controller"])
    print("Experiment services running. MPPI waits for Enter after reference preparation. "
          "Ctrl-C returns to zero and disables torque; Q/Escape or closing either viewer stops immediately.", flush=True)
    last_prompt = None
    while True:
        finished = experiment.finished()
        if state["stop"] or finished is not None:
            return (state["stop"], 0) if state["stop"] else finished
        if (not experiment.feedback_recovery_active()
                and not fresh_display(ready, config["supervision_timeout"], stage="tracking")):
            return "Supervision display stopped receiving fresh frames", 1
        if "mujoco" in commands and not fresh_mujoco_display(
                experiment.directory / "mujoco_status.json", config["supervision_timeout"]):
            return "MuJoCo display stopped updating", 1
        try:
            status = json.loads((experiment.directory / "episode_status.json").read_text())
            prompt = (status["episode"], status["stage"], status["prompt"])
            if prompt != last_prompt:
                print(status["prompt"], flush=True)
                last_prompt = prompt
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.05)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/leap_experiment.json")
    parser.add_argument("--mask", type=Path, help="reuse an existing object mask instead of running SAM-HQ")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--sam-checkpoint", help="SAM-HQ checkpoint path inside the vision environment/container")
    parser.add_argument("--no-docker", action="store_true", help="run SAM/mask/tracker with vision Python on host")
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--mesh-scale", type=float, help="uniform tracking mesh scale; automatic for built-in tasks")
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument('--camera-calibration',choices=['original','latest'],
                        help='world calibration: original with -15 mm correction, or latest Method 5 manual fit without it; tracker stays running')
    parser.add_argument("--vision-python")
    parser.add_argument("--control-python")
    parser.add_argument("--viewer-python")
    parser.add_argument('--gpu', '--device', dest='gpu', metavar='INDEX|auto',
                        help='shared physical GPU override (also accepts cuda:N); auto selects separate tracking/MPPI GPUs')
    parser.add_argument('--tracking-gpu', metavar='INDEX|auto',
                        help='physical GPU for SAM/mask/tracking; auto keeps an active tracker, otherwise selects lowest load')
    parser.add_argument('--mppi-gpu', metavar='INDEX|auto',
                        help='physical GPU for MPPI/fingertip prediction; auto selects lowest load excluding tracking GPU')
    parser.add_argument("--observe-only", action="store_true", default=None)
    parser.add_argument("--mujoco-viewer", action=argparse.BooleanOptionalAction, default=None,
                        help="show live measured robot/object poses in MuJoCo (enabled by default)")
    parser.add_argument('--fingertip-paths', action=argparse.BooleanOptionalAction, default=None,
                        help='show weighted MPPI fingertip paths in MuJoCo (disabled by default)')
    parser.add_argument('--bridge-telemetry', action=argparse.BooleanOptionalAction, default=None,
                        help='record bridge applied targets/read/write times asynchronously (default: off)')
    parser.add_argument("--n-episodes", type=int, help="number of hardware episodes (default: controller setting)")
    parser.add_argument('--continue-run', type=Path, metavar='RUN_DIRECTORY',
                        help='continue a logged batch with its saved settings; completed episodes count, aborted attempts are retried in a new run folder')
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument('--log-dir', type=Path, help='parent folder for a new timestamped experiment directory')
    parser.add_argument('--logging', action=argparse.BooleanOptionalAction, default=None,
                        help='save experiment results and process logs; --no-logging uses temporary files only')
    parser.add_argument('--overwrite-logs', action='store_true',
                        help='replace cumulative logs for this configuration; default appends runs')
    parser.add_argument('--record-video', action=argparse.BooleanOptionalAction, default=None,
                        help='record episode clips with 1s stationary padding before/after manipulation (default: off; requires logging and host ffmpeg)')
    parser.add_argument('--video-fps', type=float,
                        help='recorded video frame rate, with repeated/subsampled JPEG packets (default: 30)')
    parser.add_argument('--tracking-dir', type=Path, help='persistent vision session registry folder')
    tracking = parser.add_mutually_exclusive_group()
    tracking.add_argument('--restart-tracking', action='store_true',
                          help='stop owned tracking, recapture the mask and initialize tracking again')
    tracking.add_argument('--stop-tracking', action='store_true',
                          help='stop the saved vision services without starting hardware')
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("controller_args", nargs=argparse.REMAINDER, help="after --, pass MPPI/task options")
    args = parser.parse_args(argv)
    if args.n_episodes is not None and args.n_episodes <= 0:
        parser.error("--n-episodes must be positive")
    continuation = None
    if args.continue_run is not None:
        if (args.controller_args or args.n_episodes is not None or args.overwrite_logs
                or args.stop_tracking or args.observe_only is not None
                or args.config != ROOT/'configs/leap_experiment.json'
                or any(getattr(args, key) is not None for key in (
                    'camera_calibration', 'camera_config', 'mesh', 'mesh_scale',
                    'bbox', 'mask', 'sam_checkpoint', 'logging', 'record_video',
                    'bridge_telemetry', 'video_fps')) or args.no_docker):
            parser.error('--continue-run restores the saved settings/count; omit controller options, --n-episodes and experiment-setting overrides. GPU, interpreter, viewer, tracking restart and output-location options remain available.')
        try:
            continuation, config, args.controller_args = Continuation.load(args.continue_run, ROOT)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f'Cannot continue run: {exc}')
        print(f"Continuing batch: {continuation['completed_episodes']}/{continuation['target_episodes']} completed; "
              f"{continuation['remaining_episodes']} episodes remaining. Aborted attempts do not count.", flush=True)
        if not continuation['remaining_episodes']:
            print('Batch already complete. No experiment services started.', flush=True)
            return 0
        args.n_episodes = continuation['remaining_episodes']
        if args.log_dir is None and args.run_dir is None:
            args.log_dir = Path(continuation['continued_from']).parent
    else:
        config = json.loads(args.config.expanduser().read_text())
    if args.camera_calibration is not None:config['camera_calibration']=args.camera_calibration
    tracking_directory = local_path(args.tracking_dir or config.get('tracking_dir', '.runtime/leap_tracking'))
    if args.stop_tracking:
        tracking_directory=stop_tracking_directory(tracking_directory,explicit=args.tracking_dir is not None)
    tracking_session = TrackingSession(tracking_directory)
    if args.stop_tracking:
        if args.dry_run:
            print(f'Stop owned tracking session: {tracking_session.path}')
            return 0
        try:
            tracking_session.acquire()
            saved=tracking_session.load()
            if saved is None:
                print(f'No saved tracking session at {tracking_session.path}. Nothing stopped.',flush=True)
                return 0
            print(f'Stopping tracking registry: {tracking_session.path}',flush=True)
            tracking_session.stop(config['shutdown_timeout'])
        finally:
            tracking_session.release()
        print('Tracking services stopped. No hardware processes started.', flush=True)
        return 0
    if args.mask is not None:
        config["generate_mask"] = False
    if args.bbox is not None:
        config["bbox_xywh"] = args.bbox
    if args.sam_checkpoint is not None:
        config["sam_hq"]["checkpoint_path"] = args.sam_checkpoint
    if args.no_docker:
        config["docker"]["enabled"] = False
    for flag, key in (("mask", "init_mask_path"), ("mesh", "mesh"), ("camera_config", "camera_config"),
                      ("observe_only", "observe_only"), ("mujoco_viewer", "mujoco_viewer"), ('logging', 'logging'),
                      ('fingertip_paths', 'fingertip_paths'), ('record_video', 'record_video'), ('video_fps', 'video_fps'),
                      ('bridge_telemetry', 'bridge_telemetry')):
        if getattr(args, flag) is not None:
            config[key] = str(getattr(args, flag)) if isinstance(getattr(args, flag), Path) else getattr(args, flag)
    config.setdefault("interpreters", {})
    for flag, key in (("vision_python", "vision"), ("control_python", "control"), ("viewer_python", "viewer")):
        if getattr(args, flag) is not None:
            config["interpreters"][key] = getattr(args, flag)
    if args.mesh_scale is not None:
        config["mesh_scale"] = args.mesh_scale
    for key in ("camera_config", "hardware_config"):
        if not local_path(config[key]).is_file():
            parser.error(f"{key} file not found: {config[key]}")
    for key in ("startup_timeout", "supervision_timeout", "shutdown_timeout"):
        if not isinstance(config[key], (float, int)) or not math.isfinite(config[key]) or config[key] <= 0:
            parser.error(f"{key} must be finite and positive")
    frames = config.get("tracking_ready_frames", 5)
    age = config.get("tracking_ready_max_age", 0.3)
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
        parser.error("tracking_ready_frames must be an integer >= 2")
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or age <= 0:
        parser.error("tracking_ready_max_age must be finite and positive")
    if not math.isfinite(config.get("placement_timeout", 0)) or config.get("placement_timeout", 0) < 0:
        parser.error("placement_timeout must be finite and nonnegative (0 waits for the operator)")
    logging_enabled = config.get('logging', True)
    if not isinstance(logging_enabled, bool):
        parser.error('logging must be true or false')
    if args.overwrite_logs and not logging_enabled:
        parser.error('--overwrite-logs requires logging; omit --no-logging')
    if not isinstance(config.get('record_video', False), bool):
        parser.error('record_video must be true or false')
    if not isinstance(config.get('bridge_telemetry', False), bool):
        parser.error('bridge_telemetry must be true or false')
    fps = config.get('video_fps', 30)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 0 < fps <= 120:
        parser.error('video_fps must be finite and in (0, 120]')
    if config.get('record_video', False) and not logging_enabled:
        parser.error('--record-video requires logging; omit --no-logging')
    if args.run_dir is not None and args.log_dir is not None:
        parser.error('give --run-dir or --log-dir, not both')
    if not logging_enabled and (args.run_dir is not None or args.log_dir is not None):
        parser.error('--no-logging uses a temporary workspace; omit --run-dir/--log-dir')
    name = f"leap_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    if logging_enabled:
        base = args.log_dir.expanduser().resolve() if args.log_dir else ROOT/'results'
        directory = args.run_dir.expanduser().resolve() if args.run_dir else base/name
    else:
        directory = Path(tempfile.gettempdir())/(name + f'_{os.getpid()}')
    bbox = config["bbox_xywh"]
    if len(bbox) != 4 or any(not isinstance(n, int) for n in bbox) or min(bbox[:2]) < 0 or min(bbox[2:]) <= 0:
        parser.error("bbox_xywh must be [nonnegative x, nonnegative y, positive width, positive height]")
    if config.get("generate_mask", False):
        config["init_mask_path"] = str(directory / "mask.png")
    elif not config.get("init_mask_path") or not local_path(config["init_mask_path"]).is_file():
        parser.error("Supply --mask with an existing mask, or enable generate_mask in the experiment config")
    forwarded = args.controller_args[1:] if args.controller_args[:1] == ["--"] else args.controller_args
    if args.n_episodes is not None:
        forwarded += ["--n-episodes", str(args.n_episodes)]
    try:
        selection = args.gpu if args.gpu is not None else config.get('gpu')
        tracking_selection = args.tracking_gpu if args.tracking_gpu is not None else (
            selection if args.gpu is not None else config.get('tracking_gpu', selection))
        mppi_selection = args.mppi_gpu if args.mppi_gpu is not None else (
            selection if args.gpu is not None else config.get('mppi_gpu', selection))
        if tracking_selection is not None or mppi_selection is not None:
            tracking_selection = 'auto' if tracking_selection is None else tracking_selection
            mppi_selection = 'auto' if mppi_selection is None else mppi_selection
            config.update(gpu=selection, tracking_gpu=tracking_selection, mppi_gpu=mppi_selection)
            saved_gpu = tracking_session.load() if str(tracking_selection).lower() == 'auto' and not args.restart_tracking else None
            preferred = None
            if saved_gpu and process_alive(saved_gpu['processes'].get('tracker')):
                preferred = saved_gpu['spec'].get('cuda_visible_devices')
            tracking_gpu = (select_gpu(tracking_selection, preferred_uuid=preferred) if preferred
                            else select_gpu(tracking_selection))
            mppi_gpu = (select_gpu(mppi_selection, exclude_uuids=(tracking_gpu['uuid'],))
                        if str(mppi_selection).lower() == 'auto' else select_gpu(mppi_selection))
            config['tracking_gpu_assignment'] = tracking_gpu
            config['gpu_assignment'] = mppi_gpu
            for role, gpu in (('Tracking/SAM', tracking_gpu), ('MPPI', mppi_gpu)):
                print(f"{role}: GPU {gpu['index']} ({gpu['uuid']}): sampled load {gpu['utilization_percent']}%, "
                      f"{gpu['memory_used_mib']} MiB used; local cuda:0.", flush=True)
            if preferred:
                print('Keeping the active tracker GPU; MPPI is selected independently. --restart-tracking selects tracking again.', flush=True)
        commands = build_commands(config, directory, forwarded)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        saved = tracking_session.load()
        print('tracking: ' + ('restart and recapture' if args.restart_tracking else
              'reuse saved session' if saved else 'initial start; persist tracking after manipulation'))
        for name, command in commands.items():
            if saved and not args.restart_tracking and name in VISION_SERVICES:
                continue
            print(f"{name}: {shlex.join(command)}")
        print(f'tracking registry: {tracking_session.path}')
        return 0
    try:
        hardware_config = json.loads(local_path(config["hardware_config"]).read_text())
        interrupt_grace = float(hardware_config.get("episode_preparation", {}).get("interrupt_timeout_s", 30))
        if not math.isfinite(interrupt_grace) or interrupt_grace <= 0:
            raise ValueError("interrupt_timeout_s must be finite and positive")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(f"Invalid hardware interrupt configuration: {exc}")
    directory.mkdir(parents=True, exist_ok=False)
    experiment = Experiment(directory, logging_enabled=logging_enabled, gpu_assignment=config.get('gpu_assignment'),
                            tracking_gpu_assignment=config.get('tracking_gpu_assignment'))
    experiment.tracking_session = tracking_session
    selected = controller_selection(string_args(config.get('controller_args', []), 'controller_args') + forwarded)
    mesh, scale = tracking_mesh(config, selected)
    experiment.tracking_spec = tracking_spec({**config, 'camera_config': str(local_path(config['camera_config']))}, mesh, scale)
    state = {"stop": None, "force_stop": False}
    def request_stop(sig, frame):
        if state["stop"] is not None or sig == signal.SIGTERM:
            state["force_stop"] = True
        state["stop"] = signal.Signals(sig).name
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    launch_started = time.monotonic()
    record = dict(config=config, commands=commands, status="starting", argv=sys.argv[1:] if argv is None else list(argv))
    record['launcher_process'] = process_identity(os.getpid())
    if continuation is not None:
        record['continuation'] = continuation
        Continuation.write_json(directory/'continuation.json', continuation)
    record['logging_mode'] = 'overwrite_configuration' if args.overwrite_logs else 'append_configuration'
    if logging_enabled:
        # Preserve command/settings even if the launcher is killed before its
        # finalizer. Continuations never overwrite their source run.
        Continuation.write_json(directory/'launch.json', record)
    reason, code = "startup failed", 1
    usb_latency = None
    previous_usb_environment = os.environ.get(MANAGED_ENV)
    def release_usb_latency():
        try:
            if usb_latency is not None:
                usb_latency.close()
                record['usb_latency_timer'] = usb_latency.metadata
        finally:
            if previous_usb_environment is None:
                os.environ.pop(MANAGED_ENV, None)
            else:
                os.environ[MANAGED_ENV] = previous_usb_environment
    try:
        tracking_session.acquire()
        if hardware_config.get('usb_latency_timer_ms', 1) is not None:
            # Authenticate a sysfs-only helper before detached bridge children
            # lose the launch terminal's sudo credential scope. The bridge
            # inherits a verified lease and performs no privileged operation.
            port_parser = argparse.ArgumentParser(add_help=False)
            port_parser.add_argument('--port', default=hardware_config['port'])
            port, _ = port_parser.parse_known_args(commands.get('bridge', []))
            usb_latency = UsbLatencyTimer(port.port, hardware_config.get('usb_latency_timer_ms', 1),
                                          hardware_config.get('usb_latency_restore_ms', 16))
            usb_latency.start()
            os.environ[MANAGED_ENV] = usb_latency.inherited_environment()
            record['usb_latency_timer'] = usb_latency.metadata
        saved = tracking_session.load()
        if saved and args.restart_tracking:
            print('Restarting tracking: stopping the saved vision services before mask recapture.', flush=True)
            tracking_session.stop(config['shutdown_timeout'])
            saved = None
        if saved:
            tracking_session.check_compatible(experiment.tracking_spec)
            if args.mask is not None or args.bbox is not None or args.sam_checkpoint is not None:
                print('Reusing active tracking; mask/bounding-box/SAM capture options apply only with --restart-tracking.', flush=True)
            experiment.reusing_tracking = experiment.keep_tracking = True
            experiment.tracking_directory = Path(saved['directory'])
            record['commands'] = {**commands, **saved['commands']}
            if logging_enabled:
                for name in ('mask.png', 'mask_frame.png', 'mask.png.overlay.png', 'registration_frame.npz'):
                    source = experiment.tracking_directory/name
                    if source.is_file():
                        shutil.copy2(source, directory/name)
                snapshot = saved.get('mask_snapshot')
                if snapshot and Path(snapshot).is_file() and not (directory/Path(snapshot).name).exists():
                    shutil.copy2(snapshot, directory/Path(snapshot).name)
        if logging_enabled:
            record['input_snapshots'] = dict(hardware_config=hardware_config,
                camera_config=json.loads(local_path(config['camera_config']).read_text()),
                calibrated_hand_sha256=hashlib.sha256(local_path(selected.calibrated_hand).read_bytes()).hexdigest())
            from ContactModelStudy.Hardware.JointLimits import load_joint_limits
            measured_limits = load_joint_limits(hardware_config)
            if measured_limits is not None:
                record['input_snapshots']['joint_limits'] = measured_limits.metadata()
            if config.get('camera_calibration') is not None:
                command=commands['controller'];selected_camera=Path(command[command.index('--camera-config')+1])
                record['input_snapshots']['selected_camera_config']=json.loads(selected_camera.read_text())
        reason, code = (state['stop'], 0) if state['stop'] else run_managed(experiment, commands, config, state)
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        print(f"Stopping experiment: {reason}", flush=True)
        if code:
            record['crash_summary'] = short_failure(directory, reason)
            print(record['crash_summary'], flush=True)
        try:
            try:
                if reason == "SIGINT" and not state["force_stop"] and not config.get("observe_only", False):
                    result = experiment.return_to_zero_on_interrupt(interrupt_grace + config["shutdown_timeout"], state)
                    if result is not None:
                        reason, code = result
                        print(reason, flush=True)
                elif (not config.get('observe_only',False)
                      and can_hold_after_failure(experiment, hardware_config, state, code)):
                    print('Software failure: keeping the bridge alive for operator-confirmed hold/reset.',flush=True)
                    record['software_error_return'] = recover_after_failure(
                        experiment,config,hardware_config,commands,state,reason)
                    reason,code = f'{reason}; operator-confirmed zero return completed',1
            except Exception as exc:
                reason, code = f"Hold/reset shutdown failed: {type(exc).__name__}: {exc}", 1
                print(reason, flush=True)
            finally:
                try:
                    forced = experiment.shutdown(config["shutdown_timeout"])
                finally:
                    try:
                        release_usb_latency()
                    except Exception as exc:
                        code = 1
                        record['usb_latency_error'] = f'{type(exc).__name__}: {exc}'
                        print(f'USB latency cleanup failed: {exc}', flush=True)
            exits = {name: child.process.returncode for name, child in experiment.children.items()}
            if "bridge" in forced or exits.get("bridge", 0) not in (0, None) or "container stop unconfirmed" in forced:
                code = 1
            record.update(status="complete" if reason == "complete" else "stopped" if code == 0 else "failed",
                          reason=reason, forced_shutdown=forced,
                          exit_codes=exits, wall_time_s=time.monotonic() - launch_started)
            record['tracking'] = dict(reused=experiment.reusing_tracking, kept_running=experiment.keep_tracking,
                                      registry=str(tracking_session.path), session=tracking_session.record)
            record['tracking']['kept_running'] = bool(tracking_session.record and
                process_alive(tracking_session.record['processes'].get('tracker')))
            record['started_services'] = list(experiment.children)
            record['completed_services'] = sorted(experiment.completed_services)
            if config.get('record_video', False):
                record['video_recording'] = dict(enabled=True, directory=str(directory/'video'),
                    warnings=experiment.video_warnings)
                try:
                    record['video_recording']['summary'] = json.loads((directory/'video/recording.json').read_text())
                except (OSError, ValueError):
                    record['video_recording']['summary'] = dict(status='unavailable')
            if logging_enabled:
                results = directory/'controller.json'
                if continuation is not None:
                    try:
                        batch = {**continuation, **Continuation.progress(continuation, directory)}
                        record['continuation'] = batch
                        Continuation.write_json(directory/'continuation.json', batch)
                        print(f"Batch progress: {batch['completed_episodes']}/{batch['target_episodes']} completed; "
                              f"{batch['remaining_episodes']} remaining. Continuation: {directory}", flush=True)
                    except (OSError, ValueError, KeyError) as exc:
                        record['continuation_error'] = str(exc)
                        print(f'Could not update continuation progress: {exc}', flush=True)
                if results.is_file():
                    try:
                        record['configuration_log'] = ConfigurationLogs(directory.parent).record(
                            results, record, overwrite=args.overwrite_logs)
                        print(f"Configuration logs ({record['configuration_log']['mode']}): {record['configuration_log']['directory']}", flush=True)
                    except Exception as exc:
                        code = 1
                        record.update(status='failed', configuration_log_error=f'{type(exc).__name__}: {exc}')
                        print(f"Configuration logging failed: {exc}. Raw run results are in {directory}", flush=True)
                record['wall_time_s'] = time.monotonic() - launch_started
                (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
            if "bridge" in forced:
                detail = f"inspect {directory / 'bridge.log'}" if logging_enabled else 'logging disabled'
                print(f"Bridge cleanup did not finish normally; {detail}", flush=True)
            elif record["exit_codes"].get("bridge", 0) not in (0, None):
                detail = f"inspect {directory / 'bridge.log'} for the cause and torque-cleanup status" if logging_enabled else 'logging disabled'
                print(f"Bridge exited with an error; {detail}", flush=True)
        finally:
            # Also covers a failure before service shutdown was entered.
            try:
                release_usb_latency()
            except Exception as exc:
                print(f'USB latency cleanup failed: {exc}', flush=True)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            try:
                retained = (experiment.keep_tracking and tracking_session.record is not None
                            and Path(tracking_session.record['directory']) == directory)
                if not logging_enabled and not retained:
                    shutil.rmtree(directory)
            finally:
                tracking_session.release()
        if experiment.keep_tracking:
            alive = process_alive(tracking_session.record['processes'].get('tracker'))
            status = 'Tracking left running' if alive else 'Tracking session retained; its tracker has stopped'
            print(f'{status}; registry: {tracking_session.path}. Use --restart-tracking to recapture, or --stop-tracking to stop it.', flush=True)
        print(f"Experiment logs: {directory}" if logging_enabled else 'Testing run complete; only active tracking operational files are retained.', flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
