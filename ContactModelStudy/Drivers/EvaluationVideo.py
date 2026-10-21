"""Optional streaming case videos; no changes to simulator stepping or controls."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil

import mujoco

from ContactModelStudy.RL.RunIO import save_json


@dataclass(frozen=True)
class VideoConfig:
    fps: float = 30.
    width: int = 640
    height: int = 480
    camera: str | None = None

    def __post_init__(self):
        if (isinstance(self.fps, bool) or not isinstance(self.fps, (int, float))
                or not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError('Video fps must be finite and positive')
        for value in (self.width, self.height):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2 or value % 2:
                raise ValueError('MP4 width and height must be positive even integers')


def add_video_arguments(parser):
    group = parser.add_argument_group('evaluation video (off by default)')
    group.add_argument('--video', action='store_true', help='Stream one MP4 per case under output/videos')
    group.add_argument('--fps', type=float)
    group.add_argument('--width', type=int)
    group.add_argument('--height', type=int)
    group.add_argument('--camera', help="Scene camera; 'none' uses the free camera")


def video_from_arguments(args, parser):
    if not args.video:
        if any(getattr(args, name) is not None for name in ('fps', 'width', 'height', 'camera')):
            parser.error('Video options require --video')
        return None
    try:
        return VideoConfig(**{name: getattr(args, name) for name in ('fps', 'width', 'height', 'camera')
                              if getattr(args, name) is not None})
    except ValueError as exc:
        parser.error(str(exc))


class EpisodeVideo:
    """Keep two images, stream encode, and record frame/sample times explicitly."""
    def __init__(self, config, output, env):
        try:
            import mediapy
        except ImportError as exc:
            raise RuntimeError("Video requires the optional rl-video extra and ffmpeg") from exc
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('Video requires ffmpeg on PATH')
        from ContactModelStudy.Renderers.MujocoVideoRenderer import MujocoVideoRenderer
        from ContactModelStudy.Renderers.RendererBase import VideoRendererBaseConfig
        self.config, self.output, self.env = config, Path(output), env
        self.output.mkdir(exist_ok=False)
        renderer_config = VideoRendererBaseConfig(fps=config.fps, width=config.width, height=config.height)
        env.task.alignRendererConfigWithTask(renderer_config)
        if config.camera is not None:
            renderer_config.cam_name = None if config.camera.lower() == 'none' else config.camera
        self.renderer = MujocoVideoRenderer(env.task, renderer_config)
        self._writer_class = mediapy.VideoWriter
        self._writer = self._trace = None
        self._image = None
        self._closed = False

    def begin(self, case, index):
        if self._writer is not None:
            raise RuntimeError('Finish the previous case video first')
        self._stem = self.output/f'case_{index:05d}'
        self._partial = self._stem.with_suffix('.partial.mp4')
        self._case, self._count, self._last_written = case, 0, None
        self._goal_index = self._sample_goal_index = 0
        self._goal_history = []
        self._set_marker(case, 0, 0., 0)
        self._trace = self._stem.with_suffix('.frames.jsonl').open('x')
        writer = self._writer_class(self._partial, shape=(self.config.height, self.config.width), fps=self.config.fps)
        self._writer = writer
        self._sample_time, self._sample_step = 0., 0
        self._pose_source = 'not_captured'
        writer.__enter__()
        self._image = self._render()
        self._emit(self._image, 0., 0)

    def _set_marker(self, case, step, sim_time, goal_index):
        body = mujoco.mj_name2id(self.renderer.mjm, mujoco.mjtObj.mjOBJ_BODY, 'goal')
        mid = int(self.renderer.mjm.body_mocapid[body]) if body >= 0 else -1
        if mid < 0:
            raise ValueError('The selected video scene has no goal mocap marker')
        self.renderer.mjd.mocap_pos[mid] = case.goal_position
        self.renderer.mjd.mocap_quat[mid] = case.goal_quaternion
        self._goal_history.append({'goal_index': goal_index, 'control_step': step, 'sim_seconds': sim_time,
                                   'goal_position': list(case.goal_position),
                                   'goal_quaternion_wxyz': list(case.goal_quaternion)})

    def set_goal(self, case, step, sim_time, goal_index):
        if self._writer is None or step != self._sample_step or abs(sim_time-self._sample_time) > 1e-10:
            raise ValueError('Video goal switches must follow the current captured control state')
        self._set_marker(case, step, sim_time, goal_index)
        self._goal_index = self._sample_goal_index = goal_index
        # Replace only the held image from this instant forward, never frames
        # before the switch. No simulation stepping or duplicated video time.
        self._image = self._render()

    def _render(self):
        frame = self.env.read_frame()
        actual = getattr(self.env.backend, 'render_body_poses', None)
        self._pose_source = 'actual_maximal_body_poses' if actual is not None else 'canonical_qpos_kinematics'
        return self.renderer.RenderImage(frame.qpos[0].cpu().numpy(), body_poses=actual(0) if actual else None)

    def _emit(self, image, sample_time, step, terminal=False, goal_index=None):
        self._writer.add_image(image)
        self._trace.write(json.dumps({'frame': self._count, 'video_time': self._count/self.config.fps,
                                     'sample_time': sample_time, 'control_step': step, 'terminal': terminal,
                                     'goal_index': self._goal_index if goal_index is None else goal_index})+'\n')
        self._trace.flush()
        self._count += 1
        self._last_written = sample_time

    def capture(self, step, sim_time):
        image = self._render()
        # Frames between control samples hold the preceding real state. No
        # interpolation, additional physics steps or wall-time playback clock.
        while self._count/self.config.fps < sim_time-1e-10:
            self._emit(self._image, self._sample_time, self._sample_step, goal_index=self._sample_goal_index)
        if self._count/self.config.fps <= sim_time+1e-10:
            self._emit(image, sim_time, step)
        self._image, self._sample_time, self._sample_step = image, sim_time, step
        self._sample_goal_index = self._goal_index

    def finish(self, *, completed=True):
        if self._writer is None:
            return None
        if completed and self._last_written != self._sample_time:
            self._emit(self._image, self._sample_time, self._sample_step, terminal=True)
        writer, self._writer = self._writer, None
        try:
            writer.close()
        finally:
            self._trace.close(); self._trace = None
        path = self._partial
        if completed:
            path = self._stem.with_suffix('.mp4')
            self._partial.replace(path)
        report = {'case_id': self._case.case_id, 'path': str(path.relative_to(self.output.parent)),
                  'completed': completed, 'frames': self._count, **asdict(self.config),
                  'sim_seconds': self._sample_time, 'encoded_seconds': self._count/self.config.fps,
                  'state_source': self._pose_source, 'goal_position': list(self._case.goal_position),
                  'goal_quaternion_wxyz': list(self._case.goal_quaternion),
                  'goal_history': self._goal_history,
                  'timing': 'control_sample_zero_order_hold_with_final_state_frame',
                  'control_dt': self.env.config.control_dt}
        save_json(self._stem.with_suffix('.json'), report)
        self._image = None
        return report

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.finish(completed=False)
        finally:
            if self._trace is not None:
                self._trace.close(); self._trace = None
            self.renderer.Close()
