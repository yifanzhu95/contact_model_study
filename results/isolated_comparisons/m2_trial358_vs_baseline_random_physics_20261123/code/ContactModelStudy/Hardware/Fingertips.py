"""Read-only rollout of MPPI's weighted mean controls for fingertip previews."""
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import queue
import signal
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp

from .Workflow import SceneTaskMixin, build_rollout, reference
from ContactModelStudy.Tasks.LeapReorient import FINGERTIP_SITES


def write_prediction(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, allow_nan=False) + '\n')
    temporary.replace(path)


class MeanTrajectory:
    """Simulate the final weighted CONTROL sequence, then FK its predicted states.

    In pos_relative mode future commands depend on evolving predicted joints;
    last_action_seq's static-q approximation is therefore not used here.
    """
    def __init__(self, simulator, mode):
        if mode not in ('absolute', 'ctrl_relative', 'pos_relative'):
            raise ValueError(f'Unknown prediction control mode: {mode}')
        self.sim, self.mode = simulator, mode
        self.model = simulator.mjm
        self.fk = mujoco.MjData(self.model)
        self.sites = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                      for name in FINGERTIP_SITES]
        if min(self.sites) < 0:
            raise ValueError('Fingertip prediction requires all four fingertip sites')
        device = simulator.config.device
        self.q = wp.empty(simulator.nq, dtype=wp.float32, device=device)
        self.v = wp.empty(simulator.nv, dtype=wp.float32, device=device)
        self.u = wp.empty(simulator.nu, dtype=wp.float32, device=device)

    def _points(self, q):
        if not np.isfinite(q).all():
            raise RuntimeError('Non-finite state in weighted-mean prediction')
        self.fk.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.fk)
        return self.fk.site_xpos[self.sites].copy()

    def predict(self, mean, q, v, u, activation):
        mean = np.asarray(mean, dtype=float)
        if mean.shape != (self.sim.horizon, self.sim.nu) or not np.isfinite(mean).all():
            raise ValueError('Weighted mean must be a finite (horizon, joints) control sequence')
        self.q.assign(np.asarray(q, dtype=np.float32))
        self.v.assign(np.asarray(v, dtype=np.float32))
        self.u.assign(np.asarray(u, dtype=np.float32))
        self.sim.ClearControlSequence()
        self.sim.activation.assign(np.asarray(activation, dtype=np.float32))
        self.sim.BroadcastState(self.q, self.v, self.u)
        state = self.sim.GetState()[0][0].copy()
        points = [self._points(state)]
        previous = np.asarray(u, dtype=float).copy()
        for row in mean:
            if self.mode == 'pos_relative':
                command = state[:self.sim.nu] + row
            elif self.mode == 'ctrl_relative':
                command = previous + row
            else:
                command = row.copy()
            self.sim.SetControl(command)
            self.sim.Step_GPU(self.sim.config.resolved_substeps)
            state = self.sim.GetState()[0][0].copy()
            points.append(self._points(state))
            previous = command
        return np.asarray(points)


def _prediction_worker(selected, scene, task_config, lower, upper, path, initial,
                       pending, lock, generation, stopping, ready):
    """Spawned process: Warp's default device/stream must not be shared with MPPI."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sim = None
    warmed = False
    try:
        args = SimpleNamespace(**selected)
        with wp.ScopedDevice(args.device):
            base = reference.TASKS[args.task]
            cls = type('Prediction' + base.__name__, (SceneTaskMixin, base), {})
            task = cls(scene, task_config)
            sim = build_rollout(args, task, lower, upper)
            trajectory = MeanTrajectory(sim, args.control_mode)
            trajectory.predict(*initial)
            ready.send(dict(device=str(sim.config.device), n_worlds=sim.N,
                            rollout_config=asdict(sim.config), solver_iterations=sim.mjm.opt.iterations))
            warmed = True
            ready.close()
            while not stopping.is_set():
                try:
                    job = pending.get(timeout=.2)
                except queue.Empty:
                    continue
                with lock:
                    if stopping.is_set() or job['generation'] != generation.value:
                        continue
                started = time.monotonic()
                record = dict(source='mppi_weighted_mean_rollout',
                    planned_at=job['planned_at'], episode=job['episode'], step=job['step'],
                    rollout_model=args.rollout_model, control_mode=trajectory.mode,
                    device=str(sim.config.device), site_names=list(FINGERTIP_SITES),
                    control_timestep=sim.config.control_timestep)
                try:
                    points = trajectory.predict(job['mean'], job['q'], job['v'], job['u'], job['activation'])
                    record.update(status='ok', points=points.tolist(),
                        time_offsets_s=(np.arange(len(points))*sim.config.control_timestep).tolist())
                except Exception as exc:
                    record.update(status='error', error=f'{type(exc).__name__}: {exc}')
                record.update(generated_at=time.time(), computation_time_s=time.monotonic()-started)
                with lock:
                    if not stopping.is_set() and job['generation'] == generation.value:
                        write_prediction(path, record)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if not warmed:
            ready.send(dict(error=error))
        else:
            with lock:
                if not stopping.is_set():
                    write_prediction(path, dict(status='error', error=error))
    finally:
        ready.close()
        if sim is not None:
            with wp.ScopedDevice(selected['device']):
                sim.Close()


class FingertipPrediction:
    """Rate-limited, coalesced preview in an isolated one-world process."""
    def __init__(self, args, task, lower, upper, path):
        self.path = Path(path)
        self.period = 1 / args.fingertip_path_hz
        self.last_submit = -float('inf')
        # CUDA contexts cannot safely be inherited through fork. Spawn also
        # keeps Warp's global device/stream settings outside the control process.
        context = mp.get_context('spawn')
        self.pending = context.Queue(maxsize=1)
        self.lock = context.Lock()
        self.generation = context.Value('q', 0, lock=False)
        self.stopping = context.Event()
        self.process = None
        self.closed = False
        self.selected = dict(vars(args), n_samples=1, device=args.fingertip_device or args.device)
        self.device = self.selected['device']
        self.recipe = (str(task.getModelPath()), task.config, lower, upper)
        self.context = context
        self.metadata = None

    def start(self, mean, q, v, u, activation):
        # Build/compile/warm the preview before the first reference move/action.
        receiving, sending = self.context.Pipe(duplex=False)
        initial = tuple(np.array(value, copy=True) for value in (mean, q, v, u, activation))
        self.process = self.context.Process(target=_prediction_worker,
            args=(self.selected, *self.recipe, self.path, initial, self.pending, self.lock,
                  self.generation, self.stopping, sending), name='leap-fingertips', daemon=True)
        try:
            self.process.start()
            sending.close()
            if not receiving.poll(120):
                raise RuntimeError('Fingertip preview initialization timed out')
            self.metadata = receiving.recv()
            if 'error' in self.metadata:
                raise RuntimeError(self.metadata['error'])
        except BaseException:
            self.close()
            raise
        finally:
            receiving.close()
            sending.close()

    def submit(self, mean, q, v, u, activation, planned_at, episode, step):
        now = time.monotonic()
        if mean is None or now - self.last_submit < self.period or self.closed:
            return
        self.last_submit = now
        job = dict(mean=np.array(mean, copy=True), q=np.array(q, copy=True),
                   v=np.array(v, copy=True), u=np.array(u, copy=True),
                   activation=np.array(activation, copy=True), planned_at=planned_at,
                   episode=episode, step=step)
        with self.lock:
            job['generation'] = self.generation.value
            # Never wait for a busy prediction worker in the control loop.
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            try:
                self.pending.put_nowait(job)
            except queue.Full:
                pass  # the queue feeder has not exposed the old job yet

    def clear(self):
        with self.lock:
            self.generation.value += 1
            self.path.unlink(missing_ok=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stopping.set()
        self.clear()
        if self.process is not None and self.process.pid is not None:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
        self.pending.cancel_join_thread()
        self.pending.close()
