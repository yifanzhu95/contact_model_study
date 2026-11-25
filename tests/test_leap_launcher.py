"""Process orchestration and interruption, using ordinary CPU subprocesses."""
import json
import sys
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import launch_leap_experiment as launcher
from ContactModelStudy.Hardware.TrackingSession import TrackingSession, process_alive


@pytest.fixture(autouse=True)
def cpu_only_gpu_selection(monkeypatch):
    # Orchestration tests spawn ordinary CPU children; GPU inventory/selection
    # is covered independently without depending on the test host's devices.
    def select(value, preferred_uuid=None, exclude_uuids=()):
        index = 3 if str(value) == 'auto' else int(str(value).removeprefix('cuda:'))
        if preferred_uuid:
            index = int(preferred_uuid.removeprefix('GPU-test-'))
        elif f'GPU-test-{index}' in exclude_uuids:
            index = 0
        return dict(requested=str(value), index=index, uuid=f'GPU-test-{index}',
                    cuda_visible_devices=f'GPU-test-{index}', local_device='cuda:0',
                    utilization_percent=0, memory_used_mib=0, memory_total_mib=24564)
    monkeypatch.setattr(launcher, 'select_gpu', select)


def configuration(tmp_path):
    config = json.loads((launcher.ROOT / "configs/leap_experiment.json").read_text())
    config.pop('gpu', None)  # CPU subprocess fixtures must not query real GPUs.
    config.pop('tracking_gpu', None)
    config.pop('mppi_gpu', None)
    config["docker"]["enabled"] = False
    config.update(startup_timeout=3, supervision_timeout=1, shutdown_timeout=2,
                  tracking_dir=str(tmp_path/'tracking'))
    # Legacy orchestration fixtures do not speak LCM. Fault takeover is tested
    # separately using a fake joint-feedback client, never the live channel.
    hardware=json.loads((launcher.ROOT/'configs/leap_hardware.json').read_text())
    hardware.pop('hold_on_non_motor_failure',None)
    hardware['usb_latency_timer_ms']=None  # CPU process fixtures never touch host USB/sysfs.
    hardware_path=tmp_path/'hardware.json';hardware_path.write_text(json.dumps(hardware))
    config['hardware_config']=str(hardware_path)
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config))
    return path, config


@pytest.mark.parametrize('fail',[False,True])
def test_launcher_usb_guard_spans_shutdown_and_restores_parent_environment(tmp_path,monkeypatch,fail):
    path,config=configuration(tmp_path)
    hardware_path=Path(config['hardware_config']);hardware=json.loads(hardware_path.read_text())
    hardware['usb_latency_timer_ms']=1;hardware['usb_latency_restore_ms']=16
    hardware_path.write_text(json.dumps(hardware))
    events=[]
    class Timer:
        def __init__(self,port,runtime,restore):
            assert runtime==1 and restore==16
            self.info=dict(runtime_ms=runtime,restore_ms=restore,active=False);self.closed=False
        @property
        def metadata(self):return dict(self.info)
        def start(self):events.append('latency 1');self.info['active']=True
        def inherited_environment(self):return 'fake lease; no USB processes'
        def close(self):
            if not self.closed:
                events.append('latency 16');self.closed=True
                self.info.update(active=False,restored_ms=16)
    monkeypatch.setattr(launcher,'UsbLatencyTimer',Timer)
    monkeypatch.delenv(launcher.MANAGED_ENV,raising=False)
    def managed(*a):
        assert launcher.os.environ[launcher.MANAGED_ENV]=='fake lease; no USB processes'
        events.append('services')
        if fail:raise RuntimeError('fake startup failure')
        return 'complete',0
    monkeypatch.setattr(launcher,'run_managed',managed)
    def shutdown(*a):events.append('torque cleanup');return []
    monkeypatch.setattr(launcher.Experiment,'shutdown',shutdown)
    run=tmp_path/'guarded_run'
    if fail:
        with pytest.raises(RuntimeError,match='fake startup failure'):
            launcher.main(['--config',str(path),'--run-dir',str(run)])
    else:
        assert launcher.main(['--config',str(path),'--run-dir',str(run)])==0
    assert events==['latency 1','services','torque cleanup','latency 16']
    assert launcher.MANAGED_ENV not in launcher.os.environ
    assert json.loads((run/'launch.json').read_text())['usb_latency_timer']['restored_ms']==16


def test_dry_run_never_starts_usb_latency_guard(tmp_path,monkeypatch):
    path,config=configuration(tmp_path)
    hardware_path=Path(config['hardware_config']);hardware=json.loads(hardware_path.read_text())
    hardware['usb_latency_timer_ms']=1;hardware_path.write_text(json.dumps(hardware))
    monkeypatch.setattr(launcher,'UsbLatencyTimer',lambda *a:pytest.fail('dry-run accessed USB'))
    assert launcher.main(['--config',str(path),'--dry-run'])==0


def test_stop_only_discovers_calibration_registry_when_default_is_empty(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    alternate=tmp_path/'camera_hand_calibration/runtime/tracking';alternate.mkdir(parents=True)
    (alternate/'session.json').write_text('{}')
    default=tmp_path/'.runtime/leap_tracking'
    assert launcher.stop_tracking_directory(default)==alternate
    assert launcher.stop_tracking_directory(default,explicit=True)==default
    custom=tmp_path/'custom'
    assert launcher.stop_tracking_directory(custom)==custom
    default.mkdir(parents=True);(default/'session.json').write_text('{}')
    assert launcher.stop_tracking_directory(default)==default


def test_stop_empty_registry_reports_no_session(tmp_path,capsys,monkeypatch):
    path,config=configuration(tmp_path)
    assert launcher.main(['--config',str(path),'--stop-tracking'])==0
    assert 'Nothing stopped' in capsys.readouterr().out


@pytest.fixture(autouse=True)
def cleanup_test_tracking(tmp_path):
    yield
    session = TrackingSession(tmp_path/'tracking')
    try:
        session.acquire()
        session.load()
        session.stop(1)
    finally:
        session.release()


def child_code(directory, name, body):
    # Every child records that its Python finally block ran after Ctrl-C.
    code = ("import json, signal, time\nfrom pathlib import Path\n"
            "def stop(sig, frame): raise SystemExit(0)\n"
            "signal.signal(signal.SIGINT, stop)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "try:\n" + "\n".join("    " + line for line in body.splitlines()) +
            f"\nfinally:\n    Path({str(directory / (name + '.stopped'))!r}).write_text('stopped')\n")
    return [sys.executable, "-u", "-c", code]


def fake_commands(config, directory, extra=(), finish="viewer", mask_failure=False, stall_tracking=False):
    long_running = "while True: time.sleep(0.02)"
    trigger = str(directory / "capture_mask.trigger")
    mask = ("raise SystemExit(7)" if mask_failure else
            f"while not Path({trigger!r}).exists(): time.sleep(0.02)\n"
            + (f"Path({config['init_mask_path']!r}).write_bytes(b'mask')" if config.get("generate_mask", False)
               else f"assert Path({config['init_mask_path']!r}).is_file()"))
    if not mask_failure:
        mask += (f"\nPath({str(directory/'registration_frame.npz')!r}).write_bytes(b'paired RGB-D')"
                 f"\nPath({str(directory/'accept_mask.trigger')!r}).write_text('operator accepted')")
    heartbeat = str(directory / "viewer_heartbeat.json")
    tracker_started = str(directory / "tracker.started")
    viewer = ("def display(stage):\n"
              f"    pending = Path({heartbeat!r}).with_suffix('.tmp')\n"
              "    pending.write_text(json.dumps(dict(displayed_at=time.time(), frame_timestamp=time.time(), stage=stage)))\n"
              f"    pending.replace({heartbeat!r})\n"
              "for i in range(5):\n"
              "    display('placement')\n"
              "    time.sleep(0.02)\n"
              f"Path({trigger!r}).write_text('capture')\n"
              f"while not Path({tracker_started!r}).exists():\n"
              "    display('placement')\n"
              "    time.sleep(0.02)\n"
              f"for i in range({25 if stall_tracking else 40}):\n"
              "    display('tracking')\n"
              "    time.sleep(0.02)\n")
    if finish != "viewer":
        viewer += (long_running if stall_tracking else "while True:\n    display('tracking')\n    time.sleep(.02)")
    controller = "time.sleep(0.12)" if finish == "controller" else long_running
    commands = {name: child_code(directory, name, body) for name, body in dict(
        sam=long_running, mask=mask, tracker=f"Path({tracker_started!r}).write_text('started')\n" + long_running, viewer=viewer,
        bridge=long_running, controller=controller).items()}
    if config.get('mujoco_viewer', True):
        commands['mujoco'] = child_code(directory, 'mujoco',
            f"status = Path({str(directory/'mujoco_status.json')!r})\n"
            "ticks = 0\n"
            "while True:\n"
            "    pending = status.with_suffix('.tmp')\n"
            "    pending.write_text(json.dumps(dict(displayed_at=time.time(), tracking_valid=False)))\n"
            "    pending.replace(status)\n"
            f"    if {finish == 'mujoco'} and Path({str(directory/'controller.log')!r}).exists():\n"
            "        ticks += 1\n"
            "        if ticks > 5: break\n"
            "    time.sleep(.02)")
    return commands


@pytest.mark.parametrize("finish", ["viewer", "controller", "mujoco"])
def test_one_launch_orders_mask_then_tracking_and_cleans_every_process(tmp_path, monkeypatch, finish):
    path, _ = configuration(tmp_path)
    run = tmp_path / "run with spaces"
    monkeypatch.setattr(launcher, "server_ready", lambda url: True)
    monkeypatch.setattr(launcher, "build_commands",
                        lambda c, d, a: fake_commands(c, d, a, finish=finish))
    assert launcher.main(["--config", str(path), "--run-dir", str(run)]) == 0
    record = json.loads((run / "launch.json").read_text())
    assert record["status"] == ("complete" if finish == "controller" else "stopped")
    for name in ("mask", "mujoco", "bridge", "controller"):
        assert (run / f"{name}.stopped").exists(), name
        assert record["exit_codes"][name] == 0
    assert record['tracking']['kept_running']
    assert process_alive(record['tracking']['session']['processes']['tracker'])
    assert not (run/'tracker.stopped').exists()
    assert (run/'sam.stopped').exists()
    assert record['completed_services'] == ['sam']
    assert 'sam' not in record['tracking']['session']['processes']
    assert (run/'sam.stopped').stat().st_mtime_ns <= (run/'tracker.started').stat().st_mtime_ns
    assert (run / "mask.png").stat().st_mtime <= (run / "viewer_heartbeat.json").stat().st_mtime


def test_mask_failure_never_starts_tracking_or_hardware(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path)
    run = tmp_path / "bad mask"
    monkeypatch.setattr(launcher, "server_ready", lambda url: True)
    monkeypatch.setattr(launcher, "build_commands",
                        lambda c, d, a: fake_commands(c, d, a, mask_failure=True))
    assert launcher.main(["--config", str(path), "--run-dir", str(run)]) == 1
    assert not (run / "tracker.log").exists()
    assert not (run / "bridge.log").exists()
    assert not (run / "controller.log").exists()
    record = json.loads((run / "launch.json").read_text())
    assert record["status"] == "failed" and "mask exited with code 7" in record["reason"]
    assert (run / "sam.stopped").exists()


def test_unreviewed_mask_never_starts_tracker_or_hardware(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path)
    run = tmp_path/'unreviewed'
    def commands(c, directory, extra):
        result = fake_commands(c, directory, extra)
        result['mask'] = child_code(directory, 'mask',
            f"while not Path({str(directory/'capture_mask.trigger')!r}).exists(): time.sleep(.02)\n"
            f"Path({c['init_mask_path']!r}).write_bytes(b'mask')\n"
            f"Path({str(directory/'registration_frame.npz')!r}).write_bytes(b'frame')")
        return result
    monkeypatch.setattr(launcher, 'server_ready', lambda url: True)
    monkeypatch.setattr(launcher, 'build_commands', commands)
    assert launcher.main(['--config', str(path), '--run-dir', str(run)]) == 1
    assert not (run/'tracker.log').exists() and not (run/'bridge.log').exists()
    assert 'mask acceptance' in json.loads((run/'launch.json').read_text())['reason']


def test_cancel_placement_never_starts_tracking_or_hardware(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path)
    run = tmp_path / "placement cancelled"
    def commands(c, directory, extra):
        result = fake_commands(c, directory, extra)
        result["viewer"] = child_code(directory, "viewer",
            f"Path({str(directory / 'viewer_heartbeat.json')!r}).write_text("
            "json.dumps(dict(displayed_at=time.time(), frame_timestamp=time.time(), stage='placement')))\n"
            "time.sleep(0.15)")  # operator closes the window before pressing Space
        return result
    monkeypatch.setattr(launcher, "server_ready", lambda url: True)
    monkeypatch.setattr(launcher, "build_commands", commands)
    assert launcher.main(["--config", str(path), "--run-dir", str(run)]) == 0
    assert not (run / "mask.png").exists()
    assert not (run / "tracker.log").exists() and not (run / "bridge.log").exists()
    assert not (run / "controller.log").exists()
    assert (run / "mask.stopped").exists()


def test_reused_mask_still_previews_without_starting_sam(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path)
    run = tmp_path / "reuse"
    existing = tmp_path / "existing-mask.png"
    existing.write_bytes(b"unchanged mask")
    monkeypatch.setattr(launcher, "server_ready", lambda url: pytest.fail("mask reuse started SAM"))
    monkeypatch.setattr(launcher, "build_commands", fake_commands)
    assert launcher.main(["--config", str(path), "--run-dir", str(run), "--mask", str(existing)]) == 0
    assert (run / "capture_mask.trigger").exists() and not (run / "sam.log").exists()
    assert existing.read_bytes() == b"unchanged mask"


def test_external_sam_server_is_not_stopped(tmp_path, monkeypatch):
    path, config = configuration(tmp_path)
    config['sam_hq']['start_server'] = False
    path.write_text(json.dumps(config))
    monkeypatch.setattr(launcher, 'server_ready', lambda url: True)
    def commands(c,d,a):
        result = fake_commands(c,d,a,finish='controller')
        result.pop('sam')
        return result
    monkeypatch.setattr(launcher, 'build_commands', commands)
    run=tmp_path/'external SAM'
    assert launcher.main(['--config',str(path),'--run-dir',str(run)]) == 0
    record=json.loads((run/'launch.json').read_text())
    assert record['completed_services'] == []
    assert not (run/'sam.log').exists()


def test_sam_shutdown_failure_blocks_tracker_and_hardware(tmp_path, monkeypatch):
    path,_=configuration(tmp_path)
    monkeypatch.setattr(launcher,'server_ready',lambda url:True)
    monkeypatch.setattr(launcher,'build_commands',fake_commands)
    def fail(*args):raise RuntimeError('SAM-HQ shutdown unconfirmed; tracker not started')
    monkeypatch.setattr(launcher.Experiment,'stop_owned_sam',fail)
    run=tmp_path/'SAM did not stop'
    with pytest.raises(RuntimeError,match='SAM-HQ shutdown unconfirmed'):
        launcher.main(['--config',str(path),'--run-dir',str(run)])
    assert not (run/'tracker.log').exists() and not (run/'bridge.log').exists()
    assert json.loads((run/'launch.json').read_text())['status']=='failed'


def test_container_sam_shutdown_runs_inside_same_namespace(tmp_path, monkeypatch):
    helper=str(launcher.ROOT/'leap_owned_service.py')
    command=['docker-wrapper','exec','-e','CUDA_VISIBLE_DEVICES=GPU-vision','owned-container',
             'python','-u',helper,'run','--pid-file',str(tmp_path/'sam.pid.json'),'--','python','sam.py']
    calls=[]
    process=SimpleNamespace(returncode=None,pid=123)
    def wait(**kw):process.returncode=130;return 130
    process.wait=wait;process.poll=lambda:process.returncode
    e=launcher.Experiment(tmp_path);e.container=('docker-wrapper','owned-container')
    e.children['sam']=launcher.Child('sam',process,None)
    monkeypatch.setattr(launcher.subprocess,'run',lambda argv,**kw:
        (calls.append(argv) or SimpleNamespace(returncode=0,stdout='',stderr='')))
    e.stop_owned_sam({'sam':command},.1)
    assert calls == [[*command[:command.index(helper)],helper,'stop','--pid-file',str(tmp_path/'sam.pid.json'),'--timeout','0.1']]
    assert e.completed_services == {'sam'} and e.finished() is None


def test_stale_viewer_cannot_start_controller(tmp_path, monkeypatch):
    path, config = configuration(tmp_path)
    config["startup_timeout"] = 0.4
    path.write_text(json.dumps(config))
    run = tmp_path / "stale"
    def commands(c, directory, extra):
        result = fake_commands(c, directory, extra)
        result["viewer"] = child_code(directory, "viewer",
            f"Path({str(directory / 'viewer_heartbeat.json')!r}).write_text("
            "json.dumps(dict(displayed_at=time.time(), frame_timestamp=1)))\n"
            "while True: time.sleep(0.02)")
        return result
    monkeypatch.setattr(launcher, "server_ready", lambda url: True)
    monkeypatch.setattr(launcher, "build_commands", commands)
    assert launcher.main(["--config", str(path), "--run-dir", str(run)]) == 1
    assert not (run / "bridge.log").exists() and not (run / "controller.log").exists()


def test_stalled_supervision_stops_an_active_controller(tmp_path, monkeypatch):
    path, config = configuration(tmp_path)
    config["supervision_timeout"] = 0.2
    path.write_text(json.dumps(config))
    run = tmp_path / "stalled display"
    def commands(c, directory, extra):
        return fake_commands(c, directory, extra, finish="none", stall_tracking=True)
    monkeypatch.setattr(launcher, "server_ready", lambda url: True)
    monkeypatch.setattr(launcher, "build_commands", commands)
    assert launcher.main(["--config", str(path), "--run-dir", str(run)]) == 1
    assert (run / "controller.stopped").exists() and (run / "bridge.stopped").exists()
    assert "stopped receiving fresh frames" in json.loads((run / "launch.json").read_text())["reason"]


def test_registration_then_warmup_gap_does_not_start_hardware(tmp_path, monkeypatch):
    path, config = configuration(tmp_path)
    config['supervision_timeout'] = .2
    path.write_text(json.dumps(config))
    run = tmp_path / 'cold tracker'
    def commands(c, directory, extra):
        result = fake_commands(c, directory, extra, finish='controller')
        body = ("def display(stage, age=0):\n"
                f"    p = Path({str(directory/'viewer_heartbeat.json')!r})\n"
                "    tmp = p.with_suffix('.tmp')\n"
                "    tmp.write_text(json.dumps(dict(displayed_at=time.time(), frame_timestamp=time.time()-age, stage=stage)))\n"
                "    tmp.replace(p)\n"
                "for i in range(5):\n    display('placement')\n    time.sleep(.02)\n"
                f"Path({str(directory/'capture_mask.trigger')!r}).write_text('capture')\n"
                f"while not Path({str(directory/'tracker.started')!r}).exists():\n"
                "    display('placement')\n    time.sleep(.02)\n"
                "display('tracking', .19)\ntime.sleep(.35)\n"
                f"assert not Path({str(directory/'bridge.log')!r}).exists()\n"
                f"Path({str(directory/'warmup_waited')!r}).write_text('no hardware during warmup')\n"
                "while True:\n    display('tracking')\n    time.sleep(.02)")
        result['viewer'] = child_code(directory, 'viewer', body)
        return result
    monkeypatch.setattr(launcher, 'server_ready', lambda url: True)
    monkeypatch.setattr(launcher, 'build_commands', commands)
    assert launcher.main(['--config', str(path), '--run-dir', str(run)]) == 0
    assert (run/'warmup_waited').exists()
    assert json.loads((run/'launch.json').read_text())['status'] == 'complete'


def test_shutdown_stops_only_its_owned_container(tmp_path, monkeypatch):
    experiment = launcher.Experiment(tmp_path)
    experiment.container = ("docker-test", "owned-vision-container")
    calls = []
    def popen(command, **kwargs):
        calls.append(command)
        return type("Finished", (), {"wait": lambda self, **kw: 0})()
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    assert experiment.shutdown(0.01) == []
    assert calls == [["docker-test", "stop", "--time", "2", "owned-vision-container"]]


@pytest.mark.parametrize("completed", [False, True])
def test_explicit_bridge_shutdown_is_expected_only_after_final_zero(tmp_path, completed):
    experiment = launcher.Experiment(tmp_path)
    experiment.children = {
        'bridge': SimpleNamespace(process=SimpleNamespace(poll=lambda: 0)),
        'controller': SimpleNamespace(process=SimpleNamespace(poll=lambda: None)),
    }
    (tmp_path/'episode_status.json').write_text(json.dumps(dict(stage='complete' if completed else 'running')))
    result = experiment.finished()
    assert result is None if completed else result[1] == 1


def test_launcher_episode_count_is_forwarded_to_controller(tmp_path, monkeypatch, capsys):
    path, _ = configuration(tmp_path)
    assert launcher.main(['--config', str(path), '--dry-run', '--n-episodes', '3']) == 0
    assert '--n-episodes 3' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        launcher.main(['--config', str(path), '--dry-run', '--n-episodes', '0'])


def test_video_is_optional_and_uses_existing_stream_on_host(tmp_path, monkeypatch):
    _, config = configuration(tmp_path)
    config['init_mask_path'] = str(tmp_path/'mask.png')
    assert 'video' not in launcher.build_commands(config,tmp_path)
    config.update(record_video=True,video_fps=15)
    commands=launcher.build_commands(config,tmp_path)
    command=commands['video']
    assert 'record_leap_video.py' in ' '.join(command)
    assert command[command.index('--endpoint')+1]==config['viewer_endpoint']
    assert command[command.index('--output-dir')+1]==str(tmp_path/'video')
    assert command[command.index('--fps')+1]=='15'
    assert command[command.index('--episode-windows-file')+1]==str(tmp_path/'goal.json')
    assert '--record-video' not in commands['controller']
    assert commands['controller'][commands['controller'].index('--video-padding-seconds')+1]=='1.0'
    assert '--video-padding-seconds' not in launcher.build_commands({**config,'record_video':False},tmp_path)['controller']
    assert commands['tracker']==launcher.build_commands({**config,'record_video':False},tmp_path)['tracker']
    config['video_ffmpeg']='definitely-missing-ffmpeg-for-test'
    with pytest.raises(ValueError,match='requires FFmpeg'):
        launcher.build_commands(config,tmp_path)


def test_optional_recorder_exit_does_not_stop_manipulation(tmp_path,capsys):
    experiment=launcher.Experiment(tmp_path)
    experiment.children={
        'video':SimpleNamespace(process=SimpleNamespace(poll=lambda:1)),
        'controller':SimpleNamespace(process=SimpleNamespace(poll=lambda:None)),
        'bridge':SimpleNamespace(process=SimpleNamespace(poll=lambda:None))}
    assert experiment.finished() is None
    assert experiment.finished() is None
    assert len(experiment.video_warnings)==1
    assert 'manipulation continues' in capsys.readouterr().out


def test_video_flag_is_launcher_owned_and_requires_saved_run(tmp_path,capsys):
    path,_=configuration(tmp_path)
    assert launcher.main(['--config',str(path),'--dry-run','--record-video','--video-fps','15'])==0
    assert 'record_leap_video.py' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        launcher.main(['--config',str(path),'--dry-run','--record-video','--no-logging'])
    with pytest.raises(SystemExit):
        launcher.main(['--config',str(path),'--dry-run','--record-video','--video-fps','0'])


def test_video_is_per_run_and_failure_does_not_end_reused_tracking(tmp_path,monkeypatch):
    path,_=configuration(tmp_path)
    monkeypatch.setattr(launcher,'server_ready',lambda url:True)
    count=[0]
    def commands(c,d,a):
        result=fake_commands(c,d,a,finish='controller')
        result['controller']=child_code(d,'controller',"time.sleep(.35)")
        result['video']=child_code(d,'video',"raise SystemExit(9)")
        count[0]+=1
        return result
    monkeypatch.setattr(launcher,'build_commands',commands)
    for n in [1,2]:
        run=tmp_path/f'run{n}'
        assert launcher.main(['--config',str(path),'--run-dir',str(run),'--record-video'])==0
        saved=json.loads((run/'launch.json').read_text())
        assert saved['video_recording']['enabled']
        assert saved['video_recording']['warnings']
        assert (run/'video.stopped').exists() and (run/'bridge.stopped').exists()
        assert saved['tracking']['kept_running']
        if n==2:assert saved['tracking']['reused']


def test_generated_commands_keep_vision_in_docker_and_viewer_on_host(tmp_path):
    from ContactModelStudy.Hardware.LCM import DELTA_ACTION_CHANNEL
    config = json.loads((launcher.ROOT / "configs/leap_experiment.json").read_text())
    config["init_mask_path"] = str(tmp_path / "mask.png")
    commands = launcher.build_commands(config, tmp_path, ["--rollout-model", "M3"])
    for name in ("sam", "mask", "tracker"):
        assert commands[name][:2] == [str(launcher.ROOT / "leap_docker.py"), "exec"]
        assert 'CUDA_VISIBLE_DEVICES=' + config['docker']['cuda_visible_devices'] in commands[name]
    for name in ("viewer", "mujoco", "bridge", "controller"):
        assert commands[name][0] == sys.executable
    assert "--once" in commands["mask"]
    assert "--preview" in commands["mask"]
    assert "--capture-trigger" in commands["viewer"]
    assert "--viewer-bind" in commands["tracker"]
    assert "--ready-file" in commands["viewer"]
    assert commands['mujoco'][2] == str(launcher.ROOT/'view_leap_tracking.py')
    for option in ('--camera-config', '--pose-channel', '--lcm-url'):
        assert commands['mujoco'][commands['mujoco'].index(option)+1] == commands['controller'][commands['controller'].index(option)+1]
    assert commands['mujoco'][commands['mujoco'].index('--state-channel')+1] == config['state_channel']
    assert '--action-channel' not in commands['mujoco'] and '--hardware' not in commands['mujoco']
    assert commands["container"][commands["container"].index("--image") + 1] == "irislab_fppp:leap_hardware"
    for name in ("bridge", "controller"):
        assert commands[name][commands[name].index("--action-channel") + 1] == DELTA_ACTION_CHANNEL
        assert "--rate" not in commands[name]
    assert ["--rollout-model", "M3"] == commands["controller"][
        commands["controller"].index("M3") - 1:commands["controller"].index("M3") + 1]


@pytest.mark.parametrize("task", launcher.TRACKING_MESHES)
def test_task_selects_same_physical_mesh_for_tracker_viewer_and_model(tmp_path, task):
    import mujoco
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    from ContactModelStudy.Drivers import run_episodes as reference
    assert set(launcher.TRACKING_MESHES) == set(reference.TASKS)
    _, config = configuration(tmp_path)
    config.pop('tracking_meshes', None)  # exercise the reference geometry defaults
    config["init_mask_path"] = str(tmp_path / "mask.png")
    args = ["--task", task, "--hand-acc", "low", "--obj-acc", "foam4" if task == "duck_reorient" else "high"]
    commands = launcher.build_commands(config, tmp_path, args)
    track, view = commands["tracker"], commands["viewer"]
    path = track[track.index("--mesh_path") + 1]
    scale = float(track[track.index("--apply_scale") + 1])
    assert view[view.index("--mesh") + 1] == path
    assert float(view[view.index("--mesh-scale") + 1]) == scale
    _, evaluation = reference.buildTasks(reference.parseArgs(args))
    model = evaluation.mjm
    geom = model.geom("obj_visual").id
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    expected = model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + model.geom_pos[geom]
    if path.endswith(".obj"):
        # Read geometry without Pillow, which the host control env need not have.
        actual = np.array([[float(x) for x in line.split()[1:4]]
                           for line in launcher.Path(path).read_text().splitlines()
                           if line.startswith("v ")]) * scale
    else:
        actual = trimesh.load(path, force="mesh", process=False).vertices * scale
    assert cKDTree(actual).query(expected)[0].max() < 1e-7
    assert cKDTree(expected).query(actual)[0].max() < 1e-7


def test_per_object_mesh_override_uses_native_scan_scale_without_affecting_other_tasks(tmp_path):
    _, config = configuration(tmp_path)
    config['init_mask_path'] = str(tmp_path / 'mask.png')
    cube = launcher.build_commands(config, tmp_path, ['--task', 'cube_reorient'])['tracker']
    assert cube[cube.index('--mesh_path') + 1].endswith('cube_70_mm_2_adjusted/cube_70_mm_2.obj')
    assert cube[cube.index('--apply_scale') + 1] == '1.0'
    ball = launcher.build_commands(config, tmp_path, ['--task', 'ball_reorient'])['tracker']
    assert ball[ball.index('--mesh_path') + 1].endswith('uv_sphere.obj')
    assert float(ball[ball.index('--apply_scale') + 1]) == 0.0485


def test_custom_scene_requires_explicit_tracking_mesh(tmp_path):
    _, config = configuration(tmp_path)
    config["init_mask_path"] = str(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="custom --model requires an explicit --mesh"):
        launcher.build_commands(config, tmp_path, ["--model", "custom.xml"])
    config["mesh"] = "scenes/leap/hardware/cube_tracking.obj"
    commands = launcher.build_commands(config, tmp_path, ["--model", "custom.xml"])
    assert commands["tracker"][commands["tracker"].index("--apply_scale") + 1] == "1.0"


def test_bridge_and_controller_use_the_same_selected_hand_calibration(tmp_path):
    _, config = configuration(tmp_path)
    config["init_mask_path"] = str(tmp_path / "mask.png")
    hand = tmp_path / "custom_hand.xml"
    hand.write_text("<mujoco/>")  # argv construction only; runner validates its model
    commands = launcher.build_commands(config, tmp_path, ["--calibrated-hand", str(hand)])
    assert commands["bridge"][commands["bridge"].index("--model") + 1] == str(hand)
    assert commands["controller"][commands["controller"].index("--calibrated-hand") + 1] == str(hand)
    config["bridge_args"] = ["--model", "some_other_hand.xml"]
    with pytest.raises(ValueError, match="controller --calibrated-hand"):
        launcher.build_commands(config, tmp_path)


def test_dry_run_never_starts_container_or_processes(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run launched a process")
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    run = tmp_path / "dry"
    assert launcher.main(["--dry-run", "--tracking-dir", str(tmp_path/'tracking'),
                          "--run-dir", str(run), "--bbox", "10", "20", "30", "40"]) == 0
    output = capsys.readouterr().out
    assert "camera_mask.py" in output and "hq_sam_api.py" in output and 'persist tracking' in output
    assert not run.exists()


def test_display_requires_fresh_capture_and_fresh_render(tmp_path):
    path = tmp_path / "heartbeat.json"
    path.write_text(json.dumps(dict(displayed_at=100, frame_timestamp=95)))
    assert not launcher.fresh_display(path, 2, now=100)
    path.write_text(json.dumps(dict(displayed_at=99.8, frame_timestamp=99.5)))
    assert launcher.fresh_display(path, 2, now=100)
    path.write_text(json.dumps(dict(displayed_at=100, frame_timestamp=float('nan'))))
    assert not launcher.fresh_display(path, 2, now=100)


def test_preview_heartbeat_cannot_arm_tracking(tmp_path):
    path = tmp_path / "heartbeat.json"
    path.write_text(json.dumps(dict(displayed_at=100, frame_timestamp=100, stage="placement")))
    assert launcher.fresh_display(path, 2, now=100, stage="placement")
    assert not launcher.fresh_display(path, 2, now=100, stage="tracking")


def test_tracking_readiness_requires_distinct_fresh_frames_and_resets_after_gap(tmp_path):
    path = tmp_path/'heartbeat.json'
    readiness = launcher.TrackingReadiness(frames=3, max_age=.3)
    def update(stamp, now):
        path.write_text(json.dumps(dict(stage='tracking', displayed_at=now, frame_timestamp=stamp)))
        return readiness.update(path, now=now)
    assert not update(98.05, 100)  # reproduces the slow registration frame
    assert not update(100., 100.02)
    assert not update(100., 100.04)  # re-displaying an image is not a new frame
    assert not update(100.05, 100.07)
    assert not update(100.5, 100.52)  # a first-track/JIT pause restarts the count
    assert readiness.count == 1
    assert not update(100.55, 100.57)
    assert update(100.6, 100.62)
    assert not readiness.update(path, now=101)  # stall after readiness revokes it


def test_ctrl_c_to_launcher_cleans_child_process_groups(tmp_path):
    path, _ = configuration(tmp_path)
    run = tmp_path / "interrupted"
    code = ("import sys\n"
            f"sys.path.insert(0, {str(launcher.ROOT / 'tests')!r})\n"
            "import test_leap_launcher as fixtures\n"
            "import launch_leap_experiment as launcher\n"
            "launcher.server_ready = lambda url: True\n"
            "def commands(c,d,a):\n"
            "    result = fixtures.fake_commands(c,d,a,finish='none')\n"
            "    body = ('def reset(sig, frame):\\n'\n"
            "            '    time.sleep(.15)\\n'\n"
            "            '    assert not Path(' + repr(str(d/'bridge.stopped')) + ').exists()\\n'\n"
            "            '    Path(' + repr(str(d/'zero.reached')) + ').write_text(\"zero\")\\n'\n"
            "            '    Path(' + repr(str(d/'episode_status.json')) + ').write_text(json.dumps(dict(stage=\"interrupt_complete\")))\\n'\n"
            "            '    raise SystemExit(0)\\n'\n"
            "            'signal.signal(signal.SIGINT, reset)\\n'\n"
            "            'while True: time.sleep(.02)')\n"
            "    result['controller'] = fixtures.child_code(d, 'controller', body)\n"
            "    return result\n"
            "launcher.build_commands = commands\n"
            f"raise SystemExit(launcher.main(['--config', {str(path)!r}, '--run-dir', {str(run)!r}]))\n")
    process = subprocess.Popen([sys.executable, "-c", code], cwd=launcher.ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (run / "controller.log").exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(process.communicate()[0])
            time.sleep(0.02)
        assert (run / "controller.log").exists()
        time.sleep(0.1)  # allow newly spawned Python children to install cleanup handlers
        process.send_signal(signal.SIGINT)
        output = process.communicate(timeout=5)[0]
        assert process.returncode == 0, output
        assert 'zero return completed' in json.loads((run / "launch.json").read_text())["reason"]
        assert (run/'zero.reached').stat().st_mtime <= (run/'bridge.stopped').stat().st_mtime
        assert (run / "bridge.stopped").exists()
        assert (run / "controller.stopped").exists()
        record = json.loads((run/'launch.json').read_text())
        assert record['tracking']['kept_running']
        assert process_alive(record['tracking']['session']['processes']['tracker'])
        assert process_alive(record['tracking']['session']['processes']['viewer'])
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_launcher_mirrors_prompts_and_enter_reaches_controller_stdin(tmp_path):
    path, config = configuration(tmp_path)
    config['supervision_timeout'] = 5
    path.write_text(json.dumps(config))
    run = tmp_path / 'interactive'
    code = ("import sys\n"
            f"sys.path.insert(0, {str(launcher.ROOT / 'tests')!r})\n"
            "import test_leap_launcher as fixtures\n"
            "import launch_leap_experiment as launcher\n"
            "def commands(c, d, a):\n"
            "    result = fixtures.fake_commands(c, d, a, finish='none')\n"
            "    body = ('from ContactModelStudy.Hardware.Episodes import HardwareEpisodes\\n'\n"
            "            'for stage in [\"ready\", \"success_hold\"]:\\n'\n"
            "            '    p = Path(' + repr(str(d / 'episode_status.json')) + ')\\n'\n"
            "            '    p.write_text(json.dumps(dict(episode=0, stage=stage, prompt=stage + \" Enter gate\")))\\n'\n"
            "            '    while not HardwareEpisodes.enter_pressed(.02): pass')\n"
            "    result['controller'] = fixtures.child_code(d, 'controller', body)\n"
            "    return result\n"
            "launcher.server_ready = lambda url: True\n"
            "launcher.build_commands = commands\n"
            f"raise SystemExit(launcher.main(['--config', {str(path)!r}, '--run-dir', {str(run)!r}]))\n")
    terminal = tmp_path / 'terminal.log'
    with terminal.open('w') as output:
        process = subprocess.Popen([sys.executable, '-c', code], cwd=launcher.ROOT,
            stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, text=True)
        try:
            for stage in ('ready', 'success_hold'):
                deadline = time.monotonic() + 5
                while stage + ' Enter gate' not in terminal.read_text() and time.monotonic() < deadline:
                    assert process.poll() is None, terminal.read_text()
                    time.sleep(.02)
                assert stage + ' Enter gate' in terminal.read_text(), terminal.read_text()
                assert process.poll() is None  # no implicit/automatic confirmation
                process.stdin.write('\n')
                process.stdin.flush()
            assert process.wait(timeout=5) == 0, terminal.read_text()
            assert json.loads((run / 'launch.json').read_text())['status'] == 'complete'
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)
            process.stdin.close()


def test_mujoco_viewer_shares_task_mesh_and_controller_freshness_limits(tmp_path):
    _, config = configuration(tmp_path)
    config['init_mask_path'] = str(tmp_path/'mask.png')
    commands = launcher.build_commands(config, tmp_path, ['--task', 'duck_reorient',
        '--max-robot-age', '.12', '--max-pose-age', '.25', '--max-state-skew', '.09'])
    view = commands['mujoco']
    assert view[view.index('--task')+1] == 'duck_reorient'
    assert view[view.index('--mesh')+1] == commands['viewer'][commands['viewer'].index('--mesh')+1]
    for key, value in (('--max-robot-age', .12), ('--max-pose-age', .25), ('--max-state-skew', .09)):
        assert float(view[view.index(key)+1]) == value
    assert view[view.index('--calibrated-hand')+1] == str(launcher.ROOT/'scenes/leap/hardware/calibrated_hand.xml')


def test_mujoco_window_must_render_before_hardware_starts(tmp_path, monkeypatch):
    path, config = configuration(tmp_path)
    config['startup_timeout'] = .5
    path.write_text(json.dumps(config))
    run = tmp_path/'no-mujoco-display'
    def commands(c, directory, extra):
        result = fake_commands(c, directory, extra)
        result['mujoco'] = child_code(directory, 'mujoco', 'while True: time.sleep(.02)')
        return result
    monkeypatch.setattr(launcher, 'server_ready', lambda url: True)
    monkeypatch.setattr(launcher, 'build_commands', commands)
    assert launcher.main(['--config', str(path), '--run-dir', str(run)]) == 1
    assert not (run/'bridge.log').exists() and not (run/'controller.log').exists()


def test_mujoco_frozen_display_stops_experiment(tmp_path, monkeypatch):
    path, config = configuration(tmp_path)
    config['supervision_timeout'] = .2
    path.write_text(json.dumps(config))
    run = tmp_path/'frozen-mujoco'
    def commands(c, directory, extra):
        result = fake_commands(c, directory, extra, finish='none')
        result['mujoco'] = child_code(directory, 'mujoco',
            f"status = Path({str(directory/'mujoco_status.json')!r})\n"
            f"while not Path({str(directory/'bridge.log')!r}).exists():\n"
            "    pending = status.with_suffix('.tmp')\n"
            "    pending.write_text(json.dumps(dict(displayed_at=time.time())))\n"
            "    pending.replace(status)\n    time.sleep(.02)\n"
            "while True: time.sleep(.02)")
        return result
    monkeypatch.setattr(launcher, 'server_ready', lambda url: True)
    monkeypatch.setattr(launcher, 'build_commands', commands)
    assert launcher.main(['--config', str(path), '--run-dir', str(run)]) == 1
    assert (run/'bridge.stopped').exists() and (run/'controller.stopped').exists()
    assert 'MuJoCo display stopped updating' in json.loads((run/'launch.json').read_text())['reason']


def test_camera_display_staleness_does_not_end_supervised_feedback_hold(tmp_path, monkeypatch):
    path, config = configuration(tmp_path)
    config['supervision_timeout'] = .3
    path.write_text(json.dumps(config))
    run = tmp_path/'feedback-hold'
    def commands(c, directory, extra):
        result = fake_commands(c, directory, extra, finish='none')
        phase = str(directory/'episode_status.json')
        heartbeat = str(directory/'viewer_heartbeat.json')
        result['controller'] = child_code(directory, 'controller',
            f"Path({phase!r}).write_text(json.dumps(dict(stage='feedback_hold', episode=0, prompt='Waiting for Enter')))\n"
            # Simulate lost camera frames after the controller enters its hold.
            f"Path({heartbeat!r}).write_text(json.dumps(dict(displayed_at=time.time()-10, frame_timestamp=time.time()-10, stage='tracking')))\n"
            "time.sleep(.7)\n"
            f"Path({str(directory/'hold.survived')!r}).write_text('held')")
        # Keep the receiver window alive without refreshing its frame timestamp.
        result['viewer'] = child_code(directory, 'viewer',
            "def display(stage):\n"
            f"    Path({heartbeat!r}).write_text(json.dumps(dict(displayed_at=time.time(), frame_timestamp=time.time(), stage=stage)))\n"
            "display('placement')\n"
            f"Path({str(directory/'capture_mask.trigger')!r}).write_text('capture')\n"
            f"while not Path({str(directory/'tracker.started')!r}).exists(): time.sleep(.02)\n"
            f"while not Path({phase!r}).exists():\n    display('tracking')\n    time.sleep(.02)\n"
            "while True: time.sleep(.02)")
        return result
    monkeypatch.setattr(launcher, 'server_ready', lambda url: True)
    monkeypatch.setattr(launcher, 'build_commands', commands)
    assert launcher.main(['--config', str(path), '--run-dir', str(run)]) == 0
    assert (run/'hold.survived').exists()


@pytest.mark.parametrize('stage', ['feedback_hold', 'feedback_returning_to_zero'])
def test_feedback_hold_ignores_tracker_exit_but_preserves_bridge_and_viewer_faults(tmp_path, stage):
    experiment = launcher.Experiment(tmp_path)
    (tmp_path/'episode_status.json').write_text(json.dumps(dict(stage=stage)))
    codes = dict(tracker=1, controller=None, bridge=None, viewer=None)
    experiment.children = {name: SimpleNamespace(process=SimpleNamespace(poll=lambda name=name: codes[name]))
                           for name in codes}
    assert experiment.finished() is None
    codes['bridge'] = 1
    assert 'bridge exited' in experiment.finished()[0]
    codes['bridge'], codes['viewer'] = None, 0
    assert experiment.finished() == ('viewer stopped by operator', 0)
    codes['viewer'] = None
    del experiment.children['tracker']  # A reused tracker is outside this launch's children.
    experiment.reusing_tracking = True
    experiment.tracking_session = SimpleNamespace(finished=lambda skip: None if 'tracker' in skip else ('tracker failed', 1))
    assert experiment.finished() is None


def test_mujoco_viewer_can_be_disabled(tmp_path, capsys):
    path,_=configuration(tmp_path)
    assert launcher.main(['--config',str(path),'--dry-run', '--no-mujoco-viewer']) == 0
    assert 'view_leap_tracking.py' not in capsys.readouterr().out


def test_mujoco_render_heartbeat_does_not_require_a_robot_before_bridge_start(tmp_path):
    path = tmp_path/'mujoco_status.json'
    path.write_text(json.dumps(dict(displayed_at=100, tracking_valid=False)))
    assert launcher.fresh_mujoco_display(path, .3, now=100.1)
    assert not launcher.fresh_mujoco_display(path, .3, now=101)


@pytest.mark.parametrize('stop_kind', ['second_interrupt', 'deadline'])
def test_interrupt_reset_can_be_aborted_without_leaving_bridge_running(tmp_path, stop_kind):
    experiment = launcher.Experiment(tmp_path)
    state = {'force_stop': False}
    experiment.start('bridge', child_code(tmp_path, 'bridge', 'while True: time.sleep(.02)'))
    experiment.start('controller', child_code(tmp_path, 'controller',
        "def reset(sig, frame): pass\n"
        "signal.signal(signal.SIGINT, reset)\n"
        f"Path({str(tmp_path/'ready')!r}).write_text('ready')\n"
        "while True: time.sleep(.02)"))
    try:
        deadline = time.monotonic()+3
        while not (tmp_path/'ready').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path/'ready').exists()
        if stop_kind == 'second_interrupt':
            original = experiment.send
            def second_interrupt(name, sig):
                original(name, sig)
                if name == 'controller' and sig == signal.SIGINT:
                    state['force_stop'] = True
            experiment.send = second_interrupt
        reason, code = experiment.return_to_zero_on_interrupt(.1, state)
        assert code == 1
        assert ('Immediate stop' if stop_kind == 'second_interrupt' else 'deadline exceeded') in reason
    finally:
        assert experiment.shutdown(1) == []
    assert (tmp_path/'bridge.stopped').exists() and (tmp_path/'controller.stopped').exists()


def test_controller_and_viewer_use_same_goal_status_file(tmp_path):
    _, config = configuration(tmp_path)
    config['init_mask_path'] = str(tmp_path/'mask.png')
    config['fingertip_paths'] = True
    flags = ['--pace-control', '--cube-friction', '0.8', '0.005', '0.0001', '--fingertip-friction', '0.4', '0.005', '0.0001',
             '--palm-friction', '0.6', '0.005', '0.0001', '--finger-friction', '0.7', '0.005', '0.0001']
    commands = launcher.build_commands(config, tmp_path, flags)
    command = commands['controller']
    assert '--pace-control' in command
    assert '--pace-control' not in commands['bridge']
    index = command.index('--cube-friction')
    assert command[index+1:index+4] == ['0.8', '0.005', '0.0001']
    assert '--cube-friction' not in commands['tracker']
    for flag in ('--fingertip-friction', '--palm-friction', '--finger-friction'):
        index = command.index(flag)
        expected = flags.index(flag)
        assert command[index+1:index+4] == flags[expected+1:expected+4]
        assert flag not in commands['tracker']
    assert commands['controller'][commands['controller'].index('--goal-status-file')+1] == str(tmp_path/'goal.json')
    assert commands['mujoco'][commands['mujoco'].index('--goal-file')+1] == str(tmp_path/'goal.json')
    for name in ('controller', 'mujoco'):
        assert commands[name][commands[name].index('--fingertip-path-file')+1] == str(tmp_path/'fingertip_paths.json')


@pytest.mark.parametrize('option', ['--no-fingertip-paths', '--no-mujoco-viewer'])
def test_fingertip_preview_can_be_disabled_without_starting_any_process(tmp_path, capsys, option):
    path, _ = configuration(tmp_path)
    assert launcher.main(['--config', str(path), '--dry-run', option]) == 0
    assert '--fingertip-path-file' not in capsys.readouterr().out


@pytest.mark.parametrize('exit_code', [0, 1])
def test_logging_off_uses_and_removes_only_a_temporary_workspace(tmp_path, monkeypatch, exit_code):
    path, _ = configuration(tmp_path)
    seen = []
    def managed(experiment, commands, config, state):
        assert not experiment.logging_enabled
        assert '--no-logging' in commands['controller']
        assert '--goal-file' in commands['mujoco']  # operational IPC remains available
        seen.append(experiment.directory)
        (experiment.directory/'mask.png').write_bytes(b'temporary capture')
        return ('complete' if exit_code == 0 else 'mock fault'), exit_code
    monkeypatch.setattr(launcher, 'run_managed', managed)
    assert launcher.main(['--config', str(path), '--no-logging']) == exit_code
    assert len(seen) == 1 and not seen[0].exists()
    assert path.exists()  # never remove the user's input/config files


def test_logging_disabled_children_have_no_process_log_files(tmp_path):
    experiment = launcher.Experiment(tmp_path, logging_enabled=False)
    experiment.start('controller', child_code(tmp_path, 'controller', 'time.sleep(.1)'))
    experiment.children['controller'].process.wait(timeout=3)
    assert experiment.shutdown(1) == []
    assert not (tmp_path/'controller.log').exists()


def test_log_directory_is_parent_of_a_new_run(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path)
    output = tmp_path/'chosen log folder'
    seen = []
    def managed(experiment, *a):
        seen.append(experiment.directory)
        return 'complete', 0
    monkeypatch.setattr(launcher, 'run_managed', managed)
    assert launcher.main(['--config', str(path), '--logging', '--log-dir', str(output)]) == 0
    assert seen[0].parent == output
    assert (seen[0]/'launch.json').is_file()
    with pytest.raises(SystemExit):
        launcher.main(['--config', str(path), '--no-logging', '--log-dir', str(output)])


@pytest.mark.parametrize('model,horizon,steps', [('M1', .192, 3), ('M2', .32, 5),
    ('M3', .384, 6), ('M4', .256, 4), ('M6', .384, 6)])
def test_planned_hardware_experiments_pass_explicit_settings_without_overriding_weights(tmp_path, capsys, model, horizon, steps):
    from ContactModelStudy.Drivers import run_episodes as reference
    from ContactModelStudy.Utils.ContactModelPresets import getContactModel
    import run_leap_hardware as controller
    path, config = configuration(tmp_path)
    config['init_mask_path'] = str(tmp_path/'mask.png')
    options = ['--task', 'cube_reorient', '--rollout-model', model, '--hand-acc', 'high',
        '--obj-acc', 'high', '--n-samples', '256', '--n-iterations', '1', '--ctrl-time-step', '.064',
        '--time-horizon', str(horizon), '--noise-sigma', '.05', '--temperature', '1.0']
    assert launcher.main(['--config', str(path), '--dry-run', '--logging', '--n-episodes', '10', '--', *options]) == 0
    command = launcher.build_commands(config, tmp_path, [*options, '--n-episodes', '10'])['controller']
    args = controller.parseArgs(command[3:])
    task, _ = reference.buildTasks(args)
    cfg = getContactModel(model).makeConfig(task, **reference.rolloutSimOverrides(args, task))
    assert args.hardware and args.n_episodes == 10 and args.temperature == 1.0
    assert args.hand_acc == args.obj_acc == 'high'
    assert args.n_samples == 256 and args.n_iterations == 1 and args.noise_sigma == .05
    assert args.cost_weights is None and args.contact_params == {}
    assert cfg.control_timestep == .064 and cfg.resolved_horizon == steps


def setup_reusable_run(tmp_path, monkeypatch, **options):
    path, config = configuration(tmp_path)
    config.update(options)
    path.write_text(json.dumps(config))
    monkeypatch.setattr(launcher, 'server_ready', lambda url: True)
    monkeypatch.setattr(launcher, 'build_commands', lambda c,d,a: fake_commands(c,d,a,finish='controller'))
    first = tmp_path/'first'
    assert launcher.main(['--config', str(path), '--run-dir', str(first)]) == 0
    session = TrackingSession(config['tracking_dir'])
    record = session.load()
    assert process_alive(record['processes']['tracker'])
    return path, config, first, record


def test_default_second_launch_reuses_tracker_and_receiver_without_capturing_again(tmp_path, monkeypatch):
    path, config, first, original = setup_reusable_run(tmp_path, monkeypatch)
    mask_time = (first/'mask.png').stat().st_mtime_ns
    second = tmp_path/'second'
    # A different MPPI/contact model must not restart the same physical object's tracker.
    assert launcher.main(['--config', str(path), '--run-dir', str(second), '--', '--rollout-model', 'M3']) == 0
    current = TrackingSession(config['tracking_dir']).load()
    assert current['processes']['tracker'] == original['processes']['tracker']
    assert current['processes']['viewer'] == original['processes']['viewer']
    assert (first/'mask.png').stat().st_mtime_ns == mask_time
    for service in ('mask', 'tracker', 'sam', 'viewer'):
        assert not (second/f'{service}.log').exists()
    assert (second/'bridge.stopped').exists() and (second/'controller.stopped').exists()
    assert json.loads((second/'launch.json').read_text())['tracking']['reused']
    assert (second/'registration_frame.npz').read_bytes() == (first/'registration_frame.npz').read_bytes()


def test_restart_flag_stops_old_vision_then_recaptures_and_starts_a_new_tracker(tmp_path, monkeypatch):
    path, config, first, original = setup_reusable_run(tmp_path, monkeypatch)
    second = tmp_path/'restarted'
    assert launcher.main(['--config', str(path), '--restart-tracking', '--run-dir', str(second)]) == 0
    current = TrackingSession(config['tracking_dir']).load()
    assert current['processes']['tracker'] != original['processes']['tracker']
    assert not process_alive(original['processes']['tracker'])
    assert process_alive(current['processes']['tracker'])
    assert (first/'tracker.stopped').exists() and (first/'viewer.stopped').exists()
    assert (second/'mask.log').exists() and (second/'accept_mask.trigger').exists()
    assert not json.loads((second/'launch.json').read_text())['tracking']['reused']


def stop_identity(identity):
    import os
    os.killpg(identity['pgid'], signal.SIGINT)
    deadline = time.monotonic()+3
    while process_alive(identity) and time.monotonic() < deadline:
        time.sleep(.02)
    assert not process_alive(identity)


def test_dead_tracker_requires_explicit_restart_and_never_starts_hardware(tmp_path, monkeypatch):
    path, config, first, original = setup_reusable_run(tmp_path, monkeypatch)
    stop_identity(original['processes']['tracker'])
    second = tmp_path/'dead'
    with pytest.raises(RuntimeError, match='--restart-tracking'):
        launcher.main(['--config', str(path), '--run-dir', str(second)])
    assert not (second/'bridge.log').exists() and not (second/'mask.log').exists()
    assert process_alive(original['processes']['viewer'])


def test_changed_camera_or_mesh_requires_explicit_restart(tmp_path, monkeypatch):
    path, config, first, original = setup_reusable_run(tmp_path, monkeypatch)
    second = tmp_path/'changed'
    with pytest.raises(RuntimeError, match='--restart-tracking'):
        launcher.main(['--config', str(path), '--run-dir', str(second), '--', '--task', 'duck_reorient'])
    assert not (second/'bridge.log').exists() and not (second/'mask.log').exists()
    assert process_alive(original['processes']['tracker'])


def test_closed_receiver_is_reopened_without_restarting_tracking(tmp_path, monkeypatch):
    path, config, first, original = setup_reusable_run(tmp_path, monkeypatch)
    stop_identity(original['processes']['viewer'])
    second = tmp_path/'newreceiver'
    assert launcher.main(['--config', str(path), '--run-dir', str(second)]) == 0
    current = TrackingSession(config['tracking_dir']).load()
    assert current['processes']['tracker'] == original['processes']['tracker']
    assert current['processes']['viewer'] != original['processes']['viewer']
    assert (second/'viewer.log').exists() and not (second/'mask.log').exists()
    assert process_alive(current['processes']['viewer'])


def test_stop_tracking_flag_stops_saved_vision_without_starting_hardware(tmp_path, monkeypatch):
    path, config, first, original = setup_reusable_run(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, 'build_commands', lambda *a: pytest.fail('Stop-only must not build/start an experiment'))
    assert launcher.main(['--config', str(path), '--stop-tracking']) == 0
    assert not (Path(config['tracking_dir'])/'session.json').exists()
    assert not process_alive(original['processes']['tracker'])
    assert not process_alive(original['processes']['viewer'])


def test_stale_reused_display_never_restarts_tracking_or_enables_hardware(tmp_path, monkeypatch):
    path, config, first, original = setup_reusable_run(tmp_path, monkeypatch)
    # Keep the tracker alive, but stop its receiver's updates while retaining
    # the receiver PID. The reuse gate must demand genuinely new frames.
    import os
    viewer = original['processes']['viewer']
    os.killpg(viewer['pgid'], signal.SIGSTOP)
    config.update(startup_timeout=.4, supervision_timeout=.1)
    path.write_text(json.dumps(config))
    second = tmp_path/'stalereuse'
    time.sleep(.15)
    try:
        assert launcher.main(['--config', str(path), '--run-dir', str(second)]) == 1
        assert not (second/'bridge.log').exists() and not (second/'mask.log').exists()
        assert process_alive(original['processes']['tracker'])
    finally:
        os.killpg(viewer['pgid'], signal.SIGCONT)


def test_testing_run_keeps_only_operational_tracking_files_until_explicit_stop(tmp_path, monkeypatch):
    path, config = configuration(tmp_path)
    monkeypatch.setattr(launcher, 'server_ready', lambda url: True)
    monkeypatch.setattr(launcher, 'build_commands', lambda c,d,a: fake_commands(c,d,a,finish='controller'))
    assert launcher.main(['--config', str(path), '--no-logging']) == 0
    saved = TrackingSession(config['tracking_dir']).load()
    workspace = Path(saved['directory'])
    try:
        assert workspace.exists() and (workspace/'registration_frame.npz').exists()
        assert not (workspace/'controller.log').exists() and not (workspace/'launch.json').exists()
        assert process_alive(saved['processes']['tracker'])
        assert launcher.main(['--config', str(path), '--stop-tracking']) == 0
        assert not workspace.exists()
    finally:
        session = TrackingSession(config['tracking_dir'])
        try:
            session.acquire()
            session.load()
            session.stop(1)
        finally:
            session.release()


def test_capture_options_do_not_implicitly_restart_a_running_tracker(tmp_path, monkeypatch, capsys):
    path, config, first, original = setup_reusable_run(tmp_path, monkeypatch)
    second = tmp_path/'newbbox'
    assert launcher.main(['--config', str(path), '--run-dir', str(second), '--bbox', '10', '20', '30', '40']) == 0
    saved = TrackingSession(config['tracking_dir']).load()
    assert saved['processes']['tracker'] == original['processes']['tracker']
    assert not (second/'mask.log').exists()
    assert 'capture options apply only with --restart-tracking' in capsys.readouterr().out


def test_launcher_appends_configuration_results_and_explicitly_overwrites(tmp_path, monkeypatch):
    import run_leap_hardware as runner
    from ContactModelStudy.Hardware.RunLogging import RunLogger, utc_now

    path, _ = configuration(tmp_path)
    output = tmp_path/'experiment logs'

    def managed(experiment, commands, config, state):
        # Exercise the real controller result writer and launcher finalization,
        # without starting any processes, opening a camera or moving hardware.
        args = runner.parseArgs(commands['controller'][3:])
        report = dict(mode='lcm', status='complete', started_at_utc=utc_now(),
            args={key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}, episodes=[dict(
                episode=0, completed=True, finish_reason='success', steps=[dict(
                    step=0, planning_time=.08, action_published=True)],
                manipulation_wall_time_s=.1, cycle_wall_time_s=.2)], wall_time_s=.3)
        RunLogger(args.results).close(report)
        return 'complete', 0

    monkeypatch.setattr(launcher, 'run_managed', managed)
    common = ['--config', str(path), '--log-dir', str(output), '--n-episodes', '10']
    assert launcher.main(common) == 0
    assert launcher.main(common) == 0
    runs = sorted(output.glob('leap_experiment_*'))
    first = json.loads((runs[0]/'launch.json').read_text())['configuration_log']
    second = json.loads((runs[1]/'launch.json').read_text())['configuration_log']
    assert first['configuration_id'] == second['configuration_id']
    assert first['mode'] == second['mode'] == 'append'
    assert second['runs'] == 2
    aggregate = Path(second['directory'])
    summary = json.loads((aggregate/'summary.json').read_text())
    assert summary['successful_episodes'] == 2
    assert summary['mppi_time']['count'] == 2

    assert launcher.main(common + ['--', '--rollout-model', 'M3']) == 0
    other = sorted(output.glob('leap_experiment_*'))[-1]
    other_log = json.loads((other/'launch.json').read_text())['configuration_log']
    assert other_log['configuration_id'] != first['configuration_id']

    assert launcher.main(common + ['--overwrite-logs']) == 0
    latest = sorted(output.glob('leap_experiment_*'))[-1]
    overwritten = json.loads((latest/'launch.json').read_text())['configuration_log']
    assert overwritten['configuration_id'] == first['configuration_id']
    assert overwritten['mode'] == 'overwrite' and overwritten['runs'] == 1
    assert json.loads((aggregate/'summary.json').read_text())['successful_episodes'] == 1
    assert json.loads((Path(other_log['directory'])/'summary.json').read_text())['runs'] == 1
    assert all((run/'controller.json').is_file() for run in runs)


def test_overwrite_option_requires_logging_and_belongs_to_launcher(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path)
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **k: pytest.fail('validation started a process'))
    with pytest.raises(SystemExit) as error:
        launcher.main(['--config', str(path), '--dry-run', '--no-logging', '--overwrite-logs'])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        launcher.main(['--config', str(path), '--dry-run', '--', '--overwrite-logs'])
    assert error.value.code == 2
    assert not (tmp_path/'tracking').exists()
