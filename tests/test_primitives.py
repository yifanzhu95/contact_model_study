"""test_viewer.py

Real-time viewer + throughput test for contact_study backends.

Adapted from the original mjwarp/comfree test_viewer.py to dispatch
through contact_study.contact_models.api, so any Mk config (M1..M4)
can be loaded against any scene. Mirrors test_forward.py's CLI style
but adds real-time pacing (sleeps so playback matches mjm.opt.timestep)
and per-step latency prints, which is what you usually want when
eyeballing a scene in the viewer.

Usage:
    # Default: comfree backend, primitives scene, viewer on
    python test_primitives.py

    # XPBD backend on the allegro cube scene, headless
    python test_primitives.py --backend xpbd \
        --xml scenes/test_data/allegro/env_allegro_cube.xml --no-viewer

    # MuJoCo soft contact, batched
    python test_primitives.py --backend mujoco_soft --nworld 16
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp

from contact_study.contact_models.config import ContactModelConfig
from contact_study.contact_models import api

# Direct import for the world-0 viewer sync (XPBDData proxy can confuse
# the upstream get_data_into type checks — same trick as test_forward.py)
import comfree_warp.mujoco_warp as _mjwarp

wp.set_device("cuda:0")


def _get_inner_data(d):
    """Unwrap XPBDData (or any wrapper with _d) to get raw MJWarp Data."""
    return d._d if hasattr(d, "_d") else d


def _make_cfg(backend: str,
              comfree_stiffness: float,
              comfree_damping: float) -> ContactModelConfig:

    if backend == "mjwarp_hard":
        return ContactModelConfig.M1()
    if backend in ("mjwarp"):
        return ContactModelConfig.M2()
    if backend == "comfree":
        cfg = ContactModelConfig.M3()
        cfg.comfree.stiffness = comfree_stiffness
        cfg.comfree.damping = comfree_damping
        return cfg
    if backend == "xpbd":
        return ContactModelConfig.M4()
    raise ValueError(f"unknown backend: {backend}")


def run(
    xml_path: str = "scenes/test_data/primitives.xml",
    backend: str = "comfree",
    nworld: int = 1,
    nconmax: int = 1000,
    njmax: int = 5000,
    num_steps: int = 10000,
    comfree_stiffness: float = 0.1,
    comfree_damping: float = 0.001,
    viewer: bool = True,
    realtime: bool = True,
    warmup_steps: int = 50,
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  backend : {backend}")
    print(f"  xml     : {xml_path}")
    print(f"  nworld  : {nworld}")
    print(f"  steps   : {num_steps}")
    print(f"  viewer  : {viewer}  (realtime={realtime})")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Load scene
    # ------------------------------------------------------------------
    mjm = mujoco.MjSpec.from_file(xml_path).compile()
    mjm.opt.ccd_iterations = 50
    mjd = mujoco.MjData(mjm)

    if mjm.nkey > 0:
        key = mjm.key(0)
        mjd.qpos[:] = key.qpos
        mjd.qvel[:] = key.qvel
        mjd.ctrl[:] = key.ctrl

    mujoco.mj_forward(mjm, mjd)
    ref_ctrl = mjd.ctrl.copy()
    print(f"  timestep   = {mjm.opt.timestep}")
    print(f"  integrator = {mjm.opt.integrator}")

    # ------------------------------------------------------------------
    # Build device model + data via contact_study api
    # ------------------------------------------------------------------
    cfg = _make_cfg(backend, comfree_stiffness, comfree_damping)
    m = api.put_model(mjm, cfg)
    d = api.put_data(mjm, mjd, m, nworld=nworld, nconmax=nconmax, njmax=njmax)
    step_fn = lambda: api.step(m, d)

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------
    print("Compiling CUDA graph...")
    step_fn()
    step_fn()
    with wp.ScopedCapture() as capture:
        step_fn()
    graph = capture.graph
    wp.synchronize()
    print("Done.")

    # ------------------------------------------------------------------
    # Optional viewer
    # ------------------------------------------------------------------
    v = mujoco.viewer.launch_passive(mjm, mjd) if viewer else None
    time.sleep(0.5)

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------
    step_times = []
    dt = float(mjm.opt.timestep)

    try:
        for step_i in range(num_steps):

            t0 = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize()
            elapsed = time.perf_counter() - t0
            step_times.append(elapsed)

            # Real-time pacing
            if realtime and elapsed < dt:
                time.sleep(dt - elapsed)

            # Viewer sync (world 0)
            if v is not None:
                inner_d = _get_inner_data(d)
                _mjwarp.get_data_into(mjd, mjm, inner_d, world_id=0)
                v.sync()

            if step_i % 100 == 0:
                print(f"Step {step_i:5d}  compute={elapsed * 1e3:7.3f} ms")

            if v is not None and not v.is_running():
                print("viewer closed by user")
                break

    finally:
        wp.synchronize()
        del graph
        if v is not None:
            v.close()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    dts = np.array(step_times[warmup_steps:]) if len(step_times) > warmup_steps else np.array(step_times)
    throughput = nworld / dts if dts.size else np.array([0.0])

    stats = {
        "backend": backend,
        "nworld": nworld,
        "mean_throughput": float(throughput.mean()),
        "std_throughput": float(throughput.std()),
        "mean_step_time_ms": float(dts.mean() * 1e3),
    }

    print(f"\n  Mean throughput : {stats['mean_throughput']:.2e} steps/sec")
    print(f"  Mean step time  : {stats['mean_step_time_ms']:.4f} ms")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="scenes/test_data/primitives.xml")
    parser.add_argument("--backend", default="comfree",
                        choices=["mjwarp", "comfree", "mjwarp_hard", "xpbd", "all"])
    parser.add_argument("--nworld", type=int, default=1)
    parser.add_argument("--nconmax", type=int, default=1000)
    parser.add_argument("--njmax", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--stiffness", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=0.001)
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false",
                        help="Disable real-time pacing (run as fast as possible)")
    parser.add_argument("--warmup", type=int, default=50)
    parser.set_defaults(viewer=True, realtime=True)
    args = parser.parse_args()

    backends = (
        ["mjwarp", "comfree", "mjwarp_hard", "xpbd"]
        if args.backend == "all"
        else [args.backend]
    )

    all_stats = []
    for i, backend in enumerate(backends):
        stats = run(
            xml_path=args.xml,
            backend=backend,
            nworld=args.nworld,
            nconmax=args.nconmax,
            njmax=args.njmax,
            num_steps=args.steps,
            comfree_stiffness=args.stiffness,
            comfree_damping=args.damping,
            viewer=args.viewer and (i == 0),
            realtime=args.realtime,
            warmup_steps=args.warmup,
        )
        all_stats.append(stats)

    if len(all_stats) > 1:
        print(f"\n{'=' * 60}\n  Summary\n{'=' * 60}")
        for s in all_stats:
            print(f"  {s['backend']:20s}  {s['mean_throughput']:.2e} steps/sec  "
                  f"({s['mean_step_time_ms']:.3f} ms/step)")


if __name__ == "__main__":
    main()