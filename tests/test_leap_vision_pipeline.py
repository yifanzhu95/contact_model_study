"""Supervision image protocol, responsive viewer quit, and one-shot SAM capture."""
import ast
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "iris-foundationpose-plusplus"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publisher_receiver_round_trip_without_network_or_gui(monkeypatch):
    publisher = load_file("supervision_publisher", VISION / "src/utils/publisher.py")
    receiver = load_file("supervision_receiver", VISION / "viewer/receiver.py")
    messages = []
    class Socket:
        def setsockopt(self, *a): pass
        def setsockopt_string(self, *a): pass
        def bind(self, *a): pass
        def connect(self, *a): pass
        def close(self, **k): pass
        def send_multipart(self, parts, **k): messages.append(parts)
        def poll(self, *a): return bool(messages)
        def recv_multipart(self, *a): return messages.pop(0)
    def context():
        return SimpleNamespace(socket=lambda kind: Socket(), term=lambda: None)
    monkeypatch.setattr(publisher.zmq, "Context", context)
    pub, sub = publisher.Publisher(endpoint="inproc://test"), receiver.Subscriber(endpoint="inproc://test")
    pose = np.eye(4)
    pose[2, 3] = 0.5
    bgr = np.full((32, 48, 3), [10, 20, 200], dtype=np.uint8)
    assert pub.publish(pose, bgr, 1234.5)
    actual_pose, color, stamp = sub.receive()
    np.testing.assert_array_equal(actual_pose, pose.astype(np.float32))
    np.testing.assert_allclose(color[10, 10], bgr[10, 10], atol=3)
    assert stamp == 1234.5 and sub.receive() is None
    pub.close()
    sub.close()


@pytest.mark.parametrize("with_frame", [False, True])
def test_viewer_quits_even_when_no_frames_arrive(tmp_path, monkeypatch, with_frame):
    import time
    receiver = load_file("supervision_quit", VISION / "viewer/receiver.py")
    closed = []
    pose = np.eye(4)
    pose[2, 3] = 0.5
    frame = (pose, np.zeros((480, 640, 3), dtype=np.uint8), time.time())
    messages = [frame, None] if with_frame else [None]
    monkeypatch.setattr(receiver, "Subscriber", lambda **k: SimpleNamespace(
        receive=lambda: messages.pop(0), close=lambda: closed.append(True)))
    keys = iter([-1, ord('q')]) if with_frame else iter([27])
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(receiver.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(receiver.cv2, "waitKey", lambda *a: next(keys))
    monkeypatch.setattr(receiver.cv2, "getWindowProperty", lambda *a: 1)
    ready = tmp_path / "heartbeat.json"
    # The selected physical cube is textured; the host viewer must be able to
    # read its bounds without the vision container's image/material libraries.
    assert receiver.main(["--ready-file", str(ready), "--mesh",
        str(ROOT / "scenes/leap/objects/iris_cube.obj"), "--mesh-scale", "0.035"]) == 0
    assert closed == [True]
    assert ready.exists() == with_frame
    if with_frame:
        assert json.loads(ready.read_text())["frame_timestamp"] == frame[2]
        assert json.loads(ready.read_text())["stage"] == "tracking"


@pytest.mark.parametrize("api_success", [False, True])
@pytest.mark.parametrize("preview,review", [(False, None), (True, None), (True, "accept"), (True, "retry")])
@pytest.mark.parametrize("reuse", [False, True])
def test_one_shot_mask_uses_shared_paths_and_releases_camera(tmp_path, monkeypatch, api_success, preview, review, reuse):
    camera_instances, requests = [], []
    trigger = tmp_path / "capture.trigger"
    previews, preview_closed, reviews = [], [], []
    confirm = tmp_path / "accept.trigger"
    registration = tmp_path / "registration.npz"
    class Camera:
        def __init__(self, **kwargs):
            self.stopped = False
            self.pipeline = SimpleNamespace(stop=lambda: setattr(self, "stopped", True))
            camera_instances.append(self)
        def getFrame(self):
            return np.full((8, 8, 3), [255, 0, 0], dtype=np.uint8), np.full((8, 8), .5, dtype=np.float32), 123
    def post(url, json, timeout):
        assert not reuse, "reusing an existing mask must not call SAM"
        if preview:
            assert trigger.exists() and len(previews) % 3 == 0
        requests.append(json)
        if api_success:
            image = np.zeros((8, 8), dtype=np.uint8)
            image[2:6, 2:6] = 255
            cv2.imwrite(json["output_mask_path"], image)
        return SimpleNamespace(status_code=200 if api_success else 500, text="failure")
    package = ModuleType("utils")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "utils", package)
    monkeypatch.setitem(sys.modules, "utils.camera", SimpleNamespace(
        Camera=Camera, discover_camera_devices=lambda: [{"serial": "test"}]))
    class Publisher:
        def __init__(self, **kwargs):
            assert preview, "headless one-shot must not open a publisher"
        def publish(self, pose, color, timestamp):
            if pose[2, 3] < 0:
                assert review
                assert color[3, 3, 1] > 0 and color[0, 0, 1] == 0
                reviews.append(color.copy())
                flag = Path(str(confirm) + '.retry') if review == 'retry' and len(reviews) == 1 else confirm
                flag.write_text('operator reviewed mask')
                return
            np.testing.assert_array_equal(pose, np.eye(4))
            np.testing.assert_array_equal(color[1, 1], [0, 255, 255])  # configured yellow placement box
            previews.append(color.copy())
            if len(previews) % 3 == 0:
                trigger.write_text("operator pressed Space")
        def close(self):
            preview_closed.append(True)
    monkeypatch.setitem(sys.modules, "utils.publisher", SimpleNamespace(Publisher=Publisher))
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))
    capture = load_file("one_shot_camera_mask", VISION / "src/utils/camera_mask.py")
    output, snapshot = tmp_path / "mask.png", tmp_path / "frame.png"
    extra = ["--preview", "--trigger_path", str(trigger)] if preview else []
    if review:
        extra += ["--review-mask", "--mask-confirm-trigger", str(confirm), "--registration-frame-path", str(registration)]
    if reuse:
        extra.append("--reuse-mask")
        image = np.zeros((8, 8), dtype=np.uint8)
        image[2:6, 2:6] = 255
        cv2.imwrite(str(output), image)
        original_mask = output.read_bytes()
    calibration = json.loads((ROOT / 'configs/leap_pose_transform.json').read_text())
    calibration['camera_serial'] = 'test'
    camera_config = tmp_path / 'camera.json'
    camera_config.write_text(json.dumps(calibration))
    monkeypatch.setattr(sys, "argv", ["camera_mask.py", "--once", *extra, "--bbox_xywh", "[1,1,6,6]",
        "--output_mask_path", str(output), "--snapshot-path", str(snapshot),
        "--camera-config", str(camera_config)])
    if api_success or reuse:
        capture.main()
        assert output.is_file()
        if review:
            assert len(reviews) == (2 if review == 'retry' else 1)
            assert confirm.exists() and registration.exists()
            from ContactModelStudy.Hardware.Camera import load_registration_frame
            rgb, depth, stamp = load_registration_frame(registration, (8, 8), 'test')
            np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])
            np.testing.assert_allclose(depth, .5)
            assert stamp > 123  # epoch capture time, not device milliseconds
    else:
        with pytest.raises(RuntimeError, match="SAM-HQ mask request failed"):
            capture.main()
    assert camera_instances[0].stopped
    assert preview_closed == ([True] if preview else [])
    if reuse:
        assert not requests and not snapshot.exists()
        assert output.read_bytes() == original_mask
    else:
        assert requests[0]["frame_path"] == str(snapshot)
        assert requests[0]["output_mask_path"] == str(output)
        # Capture is RGB; the saved image must have the same visible red color.
        np.testing.assert_array_equal(cv2.cvtColor(cv2.imread(str(snapshot)), cv2.COLOR_BGR2RGB)[0, 0], [255, 0, 0])


@pytest.mark.parametrize("depth,stage,capture", [(0, "placement", True), (0.5, "tracking", False)])
def test_space_captures_only_during_placement(tmp_path, monkeypatch, depth, stage, capture):
    import time
    receiver = load_file("placement_receiver", VISION / "viewer/receiver.py")
    pose = np.eye(4)
    pose[2, 3] = depth
    messages = [(pose, np.zeros((480, 640, 3), dtype=np.uint8), time.time()), None]
    monkeypatch.setattr(receiver, "Subscriber", lambda **k: SimpleNamespace(
        receive=lambda: messages.pop(0), close=lambda: None))
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(receiver.cv2, name, lambda *a, **k: None)
    keys = iter([32, ord('q')])
    monkeypatch.setattr(receiver.cv2, "waitKey", lambda *a: next(keys))
    monkeypatch.setattr(receiver.cv2, "getWindowProperty", lambda *a: 1)
    axes = []
    def draw_axes(image, *args, **kwargs):
        axes.append(1)
        return image
    monkeypatch.setattr(receiver, "draw_xyz_axis", draw_axes)
    trigger, heartbeat = tmp_path / "capture.trigger", tmp_path / "heartbeat.json"
    assert receiver.main(["--capture-trigger", str(trigger), "--ready-file", str(heartbeat), "--draw-axes"]) == 0
    assert trigger.exists() == capture
    assert len(axes) == (0 if capture else 1)
    assert json.loads(heartbeat.read_text())["stage"] == stage


def test_sam_endpoint_passes_rgb_to_model_and_saves_mask(tmp_path):
    # Load the actual endpoint without importing Torch/model weights or starting
    # HTTP: the test covers the file/color contract between capture and SAM-HQ.
    path = VISION / "src/WebAPI/hq_sam_api.py"
    endpoint = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "segment_image")
    endpoint.decorator_list = []
    seen = []
    def segment(image, bbox):
        seen.append((image, bbox))
        return np.ones(image.shape[:2], dtype=bool)
    # PIL's contract is an RGB image; emulate that loader in this CPU test env.
    image_loader = SimpleNamespace(open=lambda path: cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    namespace = dict(Image=image_loader, np=np, cv2=cv2, os=__import__('os'), Message=object,
        sam_model=SimpleNamespace(segment=segment),
        Response=lambda **kw: SimpleNamespace(**kw), JSONResponse=lambda **kw: SimpleNamespace(**kw))
    exec(compile(ast.Module(body=[endpoint], type_ignores=[]), str(path), 'exec'), namespace)
    frame, output = tmp_path / "red.png", tmp_path / "mask.png"
    cv2.imwrite(str(frame), np.full((8, 8, 3), [0, 0, 255], dtype=np.uint8))
    response = asyncio.run(namespace['segment_image'](SimpleNamespace(
        frame_path=str(frame), output_mask_path=str(output), bbox_xywh=[1, 1, 6, 6])))
    assert response.status_code == 200 and output.is_file()
    np.testing.assert_array_equal(seen[0][0][0, 0], [255, 0, 0])
    assert seen[0][1] == [1, 1, 7, 7]


@pytest.mark.parametrize('key,accepted,retry', [(13, True, False), (ord('r'), False, True), (32, False, False)])
def test_mask_review_accept_and_retry_never_request_tracking_pose(tmp_path, monkeypatch, key, accepted, retry):
    import time
    receiver = load_file('review_receiver', VISION / 'viewer/receiver.py')
    marker = np.eye(4)
    marker[2, 3] = -1
    messages = [(marker, np.zeros((480, 640, 3), np.uint8), time.time()-5), None]
    monkeypatch.setattr(receiver, 'Subscriber', lambda **k: SimpleNamespace(
        receive=lambda: messages.pop(0), close=lambda: None))
    for name in ('namedWindow', 'resizeWindow', 'imshow', 'destroyAllWindows'):
        monkeypatch.setattr(receiver.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(receiver, 'draw_xyz_axis', lambda *a, **k: pytest.fail('Mask review is not a tracking pose'))
    keys = iter([key, ord('q')])
    monkeypatch.setattr(receiver.cv2, 'waitKey', lambda *a: next(keys))
    monkeypatch.setattr(receiver.cv2, 'getWindowProperty', lambda *a: 1)
    confirm, capture, heartbeat = tmp_path/'accept', tmp_path/'capture', tmp_path/'heartbeat.json'
    assert receiver.main(['--mask-confirm-trigger', str(confirm), '--capture-trigger', str(capture),
                          '--ready-file', str(heartbeat), '--draw-axes']) == 0
    assert confirm.exists() == accepted and Path(str(confirm)+'.retry').exists() == retry
    assert not capture.exists()
    assert json.loads(heartbeat.read_text())['stage'] == 'mask_review'
