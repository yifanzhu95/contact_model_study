"""test_allegro.py

Forward simulation debugger for contact_study.
Runs any contact model backend against any registered task or raw XML,
with optional rendering and two control modes.

Usage:
    # Headless forward sim, comfree backend, grasp_reorient task
    python tests/test_allegro.py --task grasp_reorient

    # Live viewer, XPBD, user controls sim via viewer actuator panel
    python tests/test_allegro.py --backend xpbd --render viewer --ctrl_mode user

    # Save video, MuJoCo soft contact, periodic random perturbations
    python tests/test_allegro.py --backend mjwarp --render video --ctrl_mode perturb

    # Raw XML (no task), batched, all backends compared
    python tests/test_allegro.py --xml scenes/test_data/allegro/env_allegro_cube.xml \\
                                  --nworld 512 --backend all

    # Full diagnostics
    python tests/test_allegro.py --debug --backend comfree
"""

from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"   # must be set before importing mujoco

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp
import mediapy as media

from contact_study.contact_models.config import ContactModelConfig
from contact_study.contact_models import api

# Direct import for viewer sync — XPBDData proxy can confuse world_id overload
import comfree_warp.mujoco_warp as _mjwarp

# Ensure all tasks are registered
import contact_study.tasks  # noqa: F401
from contact_study.tasks.base import get_task
from contact_study.tasks.config import DEFAULT_SCENE_VARIANT

wp.init()


# ---------------------------------------------------------------------------
# Contact model table (mirrors run_experiment.py / test_mppi.py)
# ---------------------------------------------------------------------------

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}

BACKEND_TO_MODEL = {
    "mjwarp":      "M2",
    "mjwarp_hard": "M1",
    "comfree":     "M3",
    "xpbd":        "M4",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_inner_data(d):
    return d._d if hasattr(d, "_d") else d


def _get_inner_model(m):
    return m._m if hasattr(m, "_m") else m


def _make_cfg(backend: str, comfree_stiffness: float, comfree_damping: float) -> ContactModelConfig:
    cfg = MODEL_FACTORIES[BACKEND_TO_MODEL[backend]]()
    if backend == "comfree":
        cfg.comfree.stiffness = comfree_stiffness
        cfg.comfree.damping   = comfree_damping
    return cfg


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    task_name:         str | None = None,
    xml_path:          str        = "scenes/test_data/allegro/env_allegro_cube.xml",
    backend:           str        = "comfree",
    nworld:            int        = 1,
    nconmax:           int        = 64,
    njmax:             int        = 200,
    num_steps:         int        = 1000,
    comfree_stiffness: float      = 0.1,
    comfree_damping:   float      = 0.001,
    ctrl_noise:        float      = 0.1,
    ctrl_update_every: int        = 20,
    ctrl_mode:         str        = "perturb",  # "perturb" | "user"
    render_mode:       str        = "none",
    settle_seconds:    float      = 1.0,
    warmup_steps:      int        = 50,
    debug:             bool       = False,
    seed:              int        = 42,
    geometry:          str        = "accurate",
) -> dict:
    """Run forward simulation and return throughput stats.

    Parameters
    ----------
    task_name:
        Registered task name (e.g. grasp_reorient).  If None, xml_path is
        loaded directly and no success/failure checking is done.
    ctrl_mode:
        "perturb" — apply random noise to controls every ctrl_update_every steps.
        "user"    — read ctrl from the viewer's actuator panel each step so the
                    user drives the simulation live.  Requires render_mode="viewer".
    render_mode:
        "none", "viewer" (live window), or "video" (save mp4).
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Validate / coerce argument combos
    # ------------------------------------------------------------------
    if ctrl_mode == "user" and render_mode != "viewer":
        print("  [warn] ctrl_mode=user requires render_mode=viewer — switching render_mode to viewer")
        render_mode = "viewer"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    task = None
    if task_name is not None:
        geo  = geometry
        task = get_task(task_name, geometry=geo)
        mjm, _ = task.load()
        task._mjm = mjm
    else:
        mjm = mujoco.MjSpec.from_file(xml_path).compile()

    mjd = mujoco.MjData(mjm)

    if mjm.nkey > 0:
        key = mjm.key(0)
        mjd.qpos[:] = key.qpos
        mjd.qvel[:] = key.qvel
        mjd.ctrl[:] = key.ctrl
    mujoco.mj_forward(mjm, mjd)

    ref_ctrl = mjd.ctrl.copy()

    model_key = BACKEND_TO_MODEL[backend]
    cfg       = _make_cfg(backend, comfree_stiffness, comfree_damping)

    print(f"\n{'='*60}")
    print(f"  backend     : {backend}  ({model_key})")
    print(f"  task        : {task_name or xml_path}")
    print(f"  nworld      : {nworld}    steps: {num_steps}")
    print(f"  ctrl_mode   : {ctrl_mode}")
    print(f"  render_mode : {render_mode}")
    print(f"{'='*60}")
    print(f"  nq={mjm.nq}  nv={mjm.nv}  nu={mjm.nu}")
    print(f"  integrator  = {mjm.opt.integrator}")
    print(f"  timestep    = {mjm.opt.timestep}")
    print(f"  dof_damping : min={mjm.dof_damping.min():.2e}  max={mjm.dof_damping.max():.2e}")

    # ------------------------------------------------------------------
    # Build device model + data
    # ------------------------------------------------------------------
    m = api.put_model(mjm, cfg)
    d = api.put_data(mjm, mjd, m, nworld=nworld, nconmax=nconmax, njmax=njmax)

    # ------------------------------------------------------------------
    # Debug: print initial diagnostics
    # ------------------------------------------------------------------
    if debug:
        inner_d = _get_inner_data(d)
        inner_m = _get_inner_model(m)
        wp.synchronize()
        print(f"\n  [DEBUG] njmax={getattr(inner_d, 'njmax', '?')}  "
              f"naconmax={getattr(inner_d, 'naconmax', '?')}")
        print(f"  [DEBUG] d type: {type(d).__name__}  inner_d type: {type(inner_d).__name__}")
        print(f"  [DEBUG] body_mass     = {mjm.body_mass}")
        print(f"  [DEBUG] dof_M0 min/max = {mjm.dof_M0.min():.3e} / {mjm.dof_M0.max():.3e}")
        from contact_study.contact_models.xpbd_backend import print_constraint_types
        print_constraint_types()

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------
    step_fn = lambda: api.step(m, d)
    print("Compiling CUDA graph...")
    step_fn()
    step_fn()
    with wp.ScopedCapture() as capture:
        step_fn()
    graph = capture.graph
    wp.synchronize()
    print("Done.")

    # ------------------------------------------------------------------
    # Settle (host-side, world 0 only — gives a clean initial state)
    # ------------------------------------------------------------------
    if settle_seconds > 0.0:
        settle_steps = int(settle_seconds / mjm.opt.timestep)
        print(f"  Settling {settle_steps} steps ({settle_seconds:.1f}s)…")
        for _ in range(settle_steps):
            mujoco.mj_step(mjm, mjd)
        # Re-upload settled state to GPU
        d2 = api.put_data(mjm, mjd, m, nworld=nworld, nconmax=nconmax, njmax=njmax)
        d  = d2
        # Rebuild graph with new data handle
        step_fn = lambda: api.step(m, d)
        with wp.ScopedCapture() as capture2:
            step_fn()
        graph = capture2.graph
        wp.synchronize()
        ref_ctrl = mjd.ctrl.copy()
        print("  Settled.")

    # ------------------------------------------------------------------
    # Rendering setup
    # ------------------------------------------------------------------
    mjd_view = mujoco.MjData(mjm)
    mjd_view.qpos[:] = mjd.qpos
    mjd_view.qvel[:] = mjd.qvel
    mjd_view.ctrl[:] = mjd.ctrl
    mujoco.mj_forward(mjm, mjd_view)

    v        = None
    renderer = None
    frames   = []

    if render_mode == "viewer":
        v = mujoco.viewer.launch_passive(mjm, mjd_view)
        time.sleep(0.3)
    elif render_mode == "video":
        renderer = mujoco.Renderer(mjm, height=480, width=640)

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------
    step_times = []
    success    = False

    try:
        for step_i in range(num_steps):

            # ── Control update ──────────────────────────────────────────
            if ctrl_mode == "user" and v is not None:
                # User adjusts actuators via the viewer's actuator panel;
                # launch_passive writes changes into mjd_view.ctrl.
                d.ctrl.assign(
                    np.tile(mjd_view.ctrl.astype(np.float32), (nworld, 1))
                )
            elif ctrl_mode == "perturb" and mjm.nu > 0 and step_i % ctrl_update_every == 0:
                noise = rng.uniform(-ctrl_noise, ctrl_noise, ref_ctrl.shape)
                d.ctrl.assign(
                    np.tile((ref_ctrl + noise).astype(np.float32), (nworld, 1))
                )

            # ── GPU step ────────────────────────────────────────────────
            t0 = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize()
            step_times.append(time.perf_counter() - t0)

            # ── Viewer / video sync (world 0) ───────────────────────────
            if render_mode in ("viewer", "video"):
                inner_d = _get_inner_data(d)
                _mjwarp.get_data_into(mjd_view, mjm, inner_d, world_id=0)

            if render_mode == "viewer" and v is not None:
                v.sync()
                if not v.is_running():
                    print("  Viewer closed by user.")
                    break
            elif render_mode == "video" and renderer is not None:
                renderer.update_scene(mjd_view)
                frames.append(renderer.render())

            # ── Debug output ────────────────────────────────────────────
            if debug and step_i % 100 == 0:
                inner_d = _get_inner_data(d)
                qpos = inner_d.qpos.numpy()
                qvel = inner_d.qvel.numpy()
                qacc = inner_d.qacc.numpy()
                print(f"  [step {step_i:5d}]  "
                      f"qpos[:4]={qpos[0,:min(4,qpos.shape[1])]!r}  "
                      f"|qvel|={np.linalg.norm(qvel[0]):.4f}  "
                      f"|qacc|={np.linalg.norm(qacc[0]):.4f}")
                if hasattr(d, "qfrc_total"):
                    nefc  = inner_d.nefc.numpy()
                    nacon = inner_d.nacon.numpy()
                    print(f"           nefc={nefc[0]}  nacon={nacon[0]}")

            # ── Task success / failure (world 0) ────────────────────────
            if task is not None:
                # Sync host mjd for task checks
                inner_d = _get_inner_data(d)
                _mjwarp.get_data_into(mjd, mjm, inner_d, world_id=0)

                if task.is_success(mjd) and not success:
                    success = True
                    print(f"  ✓  Task succeeded at step {step_i}")
                if task.has_failed(mjd):
                    print(f"  ✗  Task failed at step {step_i}")
                    break

    finally:
        wp.synchronize()
        del graph
        if v is not None:
            v.close()

    # ------------------------------------------------------------------
    # Save video
    # ------------------------------------------------------------------
    if render_mode == "video" and frames:
        Path("videos").mkdir(exist_ok=True)
        video_path = f"videos/video_{task_name or 'raw'}_{backend}.mp4"
        media.write_video(video_path, frames, fps=int(1.0 / mjm.opt.timestep))
        print(f"  Saved video to {video_path}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    dts = np.array(step_times[warmup_steps:]) if len(step_times) > warmup_steps else np.array(step_times)
    throughput = nworld / dts if dts.size else np.array([0.0])

    stats = {
        "backend":           backend,
        "model_key":         model_key,
        "task":              task_name or xml_path,
        "nworld":            nworld,
        "mean_throughput":   float(throughput.mean()),
        "std_throughput":    float(throughput.std()),
        "mean_step_time_ms": float(dts.mean() * 1e3),
        "success":           success,
    }

    print(f"\n  Mean throughput : {stats['mean_throughput']:.2e} steps/sec")
    print(f"  Mean step time  : {stats['mean_step_time_ms']:.4f} ms")
    if task is not None:
        print(f"  Task success    : {success}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Forward simulation debugger — integrates with contact_study tasks and models."
    )
    parser.add_argument("--task",    type=str, default="grasp_reorient",
                        help="Registered task name (e.g. grasp_reorient). "
                             "Omit to load a raw XML instead.")
    parser.add_argument("--xml",     type=str,
                        default=None,
                        help="Raw XML path. Used when --task is not set.")
    parser.add_argument("--backend", type=str, default="comfree",
                        choices=["mjwarp", "comfree", "mjwarp_hard", "xpbd", "all"])
    parser.add_argument("--geometry", type=str, default=DEFAULT_SCENE_VARIANT,
                        help="Scene variant: '<object>' or "
                             "'<object>_<hand_acc>_<obj_acc>' (e.g. duck_low_high). "
                             "Legacy geometry names map to the default scene.")
    parser.add_argument("--nworld",  type=int,   default=1)
    parser.add_argument("--nconmax", type=int,   default=64)
    parser.add_argument("--njmax",   type=int,   default=200)
    parser.add_argument("--steps",   type=int,   default=1000)
    parser.add_argument("--stiffness", type=float, default=0.1)
    parser.add_argument("--damping",   type=float, default=0.001)
    parser.add_argument("--ctrl_noise",        type=float, default=0.1,
                        help="Std of random control perturbation (ctrl_mode=perturb)")
    parser.add_argument("--ctrl_update_every", type=int,   default=20,
                        help="Steps between control updates (ctrl_mode=perturb)")
    parser.add_argument("--ctrl_mode", type=str, default="perturb",
                        choices=["perturb", "user"],
                        help="perturb: apply random noise periodically. "
                             "user: read ctrl from viewer actuator panel.")
    parser.add_argument("--render", type=str, default="video",
                        choices=["none", "viewer", "video"],
                        help="Rendering mode: none, viewer (live), or video (save mp4)")
    parser.add_argument("--settle", type=float, default=1.0,
                        help="Seconds to let physics settle before the main loop")
    parser.add_argument("--warmup", type=int,   default=50,
                        help="Steps to skip when computing throughput stats")
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--debug",  action="store_true",
                        help="Print per-step diagnostics every 100 steps")
    args = parser.parse_args()

    task_name = None if (args.task is None or args.task.lower() == "none") else args.task

    backends = (
        ["mjwarp", "comfree", "mjwarp_hard", "xpbd"]
        if args.backend == "all"
        else [args.backend]
    )

    all_stats = []
    for i, backend in enumerate(backends):
        # Only open viewer for the first backend when sweeping all
        current_render = args.render
        if args.render == "viewer" and i > 0:
            current_render = "none"

        stats = run(
            task_name         = task_name,
            xml_path          = args.xml,
            backend           = backend,
            nworld            = args.nworld,
            nconmax           = args.nconmax,
            njmax             = args.njmax,
            num_steps         = args.steps,
            comfree_stiffness = args.stiffness,
            comfree_damping   = args.damping,
            ctrl_noise        = args.ctrl_noise,
            ctrl_update_every = args.ctrl_update_every,
            ctrl_mode         = args.ctrl_mode,
            render_mode       = current_render,
            settle_seconds    = args.settle,
            warmup_steps      = args.warmup,
            debug             = args.debug,
            seed              = args.seed,
            geometry          = args.geometry,
        )
        all_stats.append(stats)

    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print(f"  Summary  (task={task_name or 'raw xml'})")
        print(f"{'='*60}")
        print(f"  {'backend':<16}  {'throughput':>14}  {'step_ms':>9}")
        print(f"  {'-'*16}  {'-'*14}  {'-'*9}")
        for s in all_stats:
            print(f"  {s['backend']:<16}  "
                  f"{s['mean_throughput']:>12.2e}/s  "
                  f"{s['mean_step_time_ms']:>8.3f}ms")


if __name__ == "__main__":
    main()
