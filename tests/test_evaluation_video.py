"""Video clocks, scratch poses and cleanup without GL, ffmpeg or optimization."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np
import pytest
import torch

from ContactModelStudy.Drivers.EvaluationVideo import EpisodeVideo, VideoConfig
from ContactModelStudy.Renderers.MujocoVideoRenderer import MujocoVideoRenderer, apply_body_poses
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.Tasks.Registry import task_from_environment


@pytest.mark.parametrize('values', [{'fps': 0}, {'fps': float('nan')}, {'fps': True}, {'width': 3}, {'height': -2}])
def test_invalid_video_settings(values):
    with pytest.raises(ValueError):
        VideoConfig(**values)


def test_actual_body_visual_override_preserves_qpos_and_goal():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="moving"><freejoint/><geom size=".02" pos=".1 0 0"/>
      <site pos="0 .1 0"/></body><body name="goal" mocap="true"><geom size=".01"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    qpos, goal = data.qpos.copy(), data.mocap_quat.copy()
    pose = np.array([1., 2., 3., 2**-.5, 0., 0., 2**-.5])
    apply_body_poses(model, data, {1: pose})
    assert np.allclose(data.geom_xpos[0], [1., 2.1, 3.])
    assert np.allclose(data.site_xpos[0], [.9, 2., 3.])
    assert np.array_equal(data.qpos, qpos) and np.array_equal(data.mocap_quat, goal)
    with pytest.raises(ValueError, match='exclude world and goal'):
        apply_body_poses(model, data, {2: pose})


def test_image_api_does_not_buffer_and_legacy_state_api_still_does():
    renderer = MujocoVideoRenderer.__new__(MujocoVideoRenderer)
    renderer.mjm = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
    renderer.mjd = mujoco.MjData(renderer.mjm)
    renderer._camera, renderer._closed, renderer._frames = -1, False, []
    renderer._renderer = SimpleNamespace(update_scene=lambda *args, **kwargs: None,
        render=lambda: np.zeros((2, 2, 3), dtype=np.uint8))
    image = renderer.RenderImage(renderer.mjm.qpos0)
    assert image.shape == (2, 2, 3) and renderer.frame_count == 0
    renderer.RenderState(renderer.mjm.qpos0)
    assert renderer.frame_count == 1 and np.array_equal(renderer._frames[0], image)


@pytest.fixture
def recording(tmp_path, monkeypatch):
    state = [0]
    config = EnvConfig(num_envs=1)
    task = task_from_environment(config)
    bank = CaseBank.development(count=1, seed=2, difficulty=8)
    case = bank.cases[0]
    frames, renderers = [], []
    class Renderer:
        def __init__(self, task, config):
            self.mjm = mujoco.MjModel.from_xml_path(task.getModelPath())
            self.mjd = mujoco.MjData(self.mjm)
            self.shape = (config.height, config.width, 3)
            self.closed = False; renderers.append(self)
        def RenderImage(self, qpos, *, body_poses=None):
            return np.full(self.shape, state[0], dtype=np.uint8)
        def Close(self): self.closed = True
    class Writer:
        def __init__(self, path, **kwargs): self.path = Path(path)
        def __enter__(self): return self
        def add_image(self, image): frames.append(image.copy())
        def close(self): self.path.write_bytes(b'test-encoder')
    monkeypatch.setitem(sys.modules, 'mediapy', SimpleNamespace(VideoWriter=Writer))
    monkeypatch.setattr(importlib.import_module('ContactModelStudy.Renderers.MujocoVideoRenderer'), 'MujocoVideoRenderer', Renderer)
    monkeypatch.setattr('ContactModelStudy.Drivers.EvaluationVideo.shutil.which', lambda *args: '/test/ffmpeg')
    env = SimpleNamespace(task=task, config=config, backend=SimpleNamespace(),
                          read_frame=lambda: SimpleNamespace(qpos=torch.tensor([case.qpos])))
    recorder = EpisodeVideo(VideoConfig(fps=20, width=8, height=8), tmp_path/'videos', env)
    return recorder, case, state, frames, renderers


def test_streamed_frames_follow_sim_time_and_case_goal(recording):
    video, case, state, frames, renderers = recording
    video.begin(case, 0)
    for step in range(1, 5):
        state[0] = step
        video.capture(step, step*.016)
    result = video.finish()
    video.close()
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 3, 4]
    records = [json.loads(line) for line in (video.output/'case_00000.frames.jsonl').read_text().splitlines()]
    assert [r['sample_time'] for r in records] == [0., .048, .064]
    assert records[-1]['terminal']
    assert result['sim_seconds'] == .064 and result['completed']
    assert (video.output/'case_00000.mp4').exists()
    assert not (video.output/'case_00000.partial.mp4').exists()
    mid = renderers[0].mjm.body('goal').mocapid
    assert np.allclose(renderers[0].mjd.mocap_quat[mid], case.goal_quaternion)
    assert np.allclose(renderers[0].mjd.mocap_pos[mid], case.goal_position)
    assert renderers[0].closed and video._image is None


def test_interrupted_video_retains_partial_label_and_closes(recording):
    video, case, state, frames, renderers = recording
    video.begin(case, 0)
    video.close(); video.close()
    assert (video.output/'case_00000.partial.mp4').exists()
    assert not (video.output/'case_00000.mp4').exists()
    assert not json.loads((video.output/'case_00000.json').read_text())['completed']
    assert renderers[0].closed


def test_video_uses_actual_body_readout_when_backend_provides_it(recording):
    video, case, state, frames, renderers = recording
    actual = {1: np.array([0., 0., 0., 1., 0., 0., 0.])}
    video.env.backend.render_body_poses = lambda world: actual
    received = []
    original = video.renderer.RenderImage
    def draw(qpos, *, body_poses=None):
        received.append(body_poses)
        return original(qpos, body_poses=body_poses)
    video.renderer.RenderImage = draw
    video.begin(case, 0)
    result = video.finish(); video.close()
    assert received == [actual] and result['state_source'] == 'actual_maximal_body_poses'


def test_goal_marker_history_and_frame_indices_follow_switch_without_extra_time(recording):
    from dataclasses import replace
    video, case, state, frames, renderers = recording
    video.begin(case, 0)
    state[0] = 1; video.capture(1, .016)
    changed = replace(case, goal_quaternion=(0., 1., 0., 0.))
    video.set_goal(changed, 1, .016, 1)
    for step in range(2, 5):
        state[0] = step; video.capture(step, step*.016)
    result = video.finish(); video.close()
    records = [json.loads(line) for line in (video.output/'case_00000.frames.jsonl').read_text().splitlines()]
    assert [r['goal_index'] for r in records] == [0, 1, 1]
    assert [r['sample_time'] for r in records] == [0., .048, .064]
    assert result['sim_seconds'] == .064 and len(frames) == 3
    assert result['goal_history'][1]['control_step'] == 1
    assert result['goal_history'][1]['goal_quaternion_wxyz'] == list(changed.goal_quaternion)
    mid = renderers[0].mjm.body('goal').mocapid
    np.testing.assert_allclose(renderers[0].mjd.mocap_quat[int(mid[0])], changed.goal_quaternion)
