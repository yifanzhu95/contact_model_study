"""test_forward.py

Forward simulation debugger for contact_study.
Runs any contact model backend against any XML scene, with optional
MuJoCo viewer showing world 0.

Usage:
    # Headless, comfree backend
    python test_forward.py

    # With viewer, MuJoCo soft contact
    python test_forward.py --viewer --backend mujoco_soft

    # Batched, custom XML
    python test_forward.py --xml scenes/tasks/peg_in_hole_accurate.xml \
                            --nworld 512 --backend comfree --steps 500

    # All backends, no viewer
    python test_forward.py --backend all
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp

from contact_study.contact_models.config import ContactModelConfig, Backend
from contact_study.contact_models import api

# get_data_into for viewer sync still needs world_id, call upstream directly
import comfree_warp.mujoco_warp as _mjwarp

wp.set_device("cuda:0")


def run(
    xml_path:          str   = "scenes/test_data/allegro/env_allegro_cube.xml",
    backend:           str   = "comfree",      # "mjwarp" | "comfree" | "mujoco_soft" | "mujoco_anitescu"
    nworld:            int   = 1,
    nconmax:           int   = 64,
    njmax:             int   = 200,
    num_steps:         int   = 1000,
    comfree_stiffness: float = 0.1,
    comfree_damping:   float = 0.001,
    ctrl_noise:        float = 0.1,           # std of random control perturbation
    ctrl_update_every: int   = 20,            # steps between control updates
    viewer:            bool  = False,
    warmup_steps:      int   = 50,
) -> dict:
    """Run forward simulation and return throughput stats."""

    print(f"\n{'='*60}")
    print(f"  backend : {backend}")
    print(f"  xml     : {xml_path}")
    print(f"  nworld  : {nworld}")
    print(f"  steps   : {num_steps}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    mjm = mujoco.MjSpec.from_file(xml_path).compile()
    mjd = mujoco.MjData(mjm)

    if mjm.nkey > 0:
        key = mjm.key(0)
        mjd.qpos[:] = key.qpos
        mjd.qvel[:] = key.qvel
        mjd.ctrl[:] = key.ctrl

    ref_ctrl = mjd.ctrl.copy()

    # ------------------------------------------------------------------
    # Build ContactModelConfig for the requested backend
    # ------------------------------------------------------------------
    if backend == "comfree":
        cfg = ContactModelConfig.M3()
        cfg.comfree.stiffness = comfree_stiffness
        cfg.comfree.damping   = comfree_damping
    elif backend == "mujoco_soft":
        cfg = ContactModelConfig.M2()
    elif backend == "mujoco_anitescu":
        cfg = ContactModelConfig.M1()
    else:  # mjwarp — same as mujoco_soft, no solver patch
        cfg = ContactModelConfig.M2()

    # ------------------------------------------------------------------
    # Build device model + data via contact_study api
    # ------------------------------------------------------------------
    m = api.put_model(mjm, cfg)
    d = api.make_data(mjm, m, nworld=nworld, nconmax=nconmax, njmax=njmax)
    step_fn = lambda: api.step(m, d)

    # Broadcast initial state to all worlds
    d.qpos.assign(np.tile(mjd.qpos, (nworld, 1)).astype(np.float32))
    d.qvel.assign(np.tile(mjd.qvel, (nworld, 1)).astype(np.float32))

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
    v = None
    if viewer:
        v = mujoco.viewer.launch_passive(mjm, mjd)

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------
    step_times = []

    try:
        for step in range(num_steps):

            # Perturb controls periodically
            if mjm.nu > 0 and step % ctrl_update_every == 0:
                noise = np.random.uniform(-ctrl_noise, ctrl_noise, ref_ctrl.shape)
                ctrl  = np.tile((ref_ctrl + noise).astype(np.float32), (nworld, 1))
                d.ctrl.assign(ctrl)

            t0 = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize()
            step_times.append(time.perf_counter() - t0)

            # Sync viewer with world 0
            if v is not None:
                _mjwarp.get_data_into(mjd, mjm, d, world_id=0)
                v.sync()

    finally:
        wp.synchronize()  # drain all pending CUDA work
        del graph         # release CUDA graph before OpenGL context dies
        if v is not None:
            v.close()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    dts       = np.array(step_times[warmup_steps:])
    throughput = nworld / dts

    stats = {
        "backend":            backend,
        "nworld":             nworld,
        "mean_throughput":    float(throughput.mean()),
        "std_throughput":     float(throughput.std()),
        "mean_step_time_ms":  float(dts.mean() * 1e3),
    }

    print(f"  Mean throughput : {stats['mean_throughput']:.2e} steps/sec")
    print(f"  Mean step time  : {stats['mean_step_time_ms']:.4f} ms")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml",     default="scenes/test_data/allegro/env_allegro_cube.xml")
    parser.add_argument("--backend", default="comfree",
                        choices=["mjwarp", "comfree", "mujoco_soft", "mujoco_anitescu", "all"])
    parser.add_argument("--nworld",  type=int,   default=1)
    parser.add_argument("--nconmax", type=int,   default=64)
    parser.add_argument("--njmax",   type=int,   default=200)
    parser.add_argument("--steps",   type=int,   default=1000)
    parser.add_argument("--stiffness", type=float, default=0.1)
    parser.add_argument("--damping",   type=float, default=0.001)
    parser.add_argument("--ctrl_noise",        type=float, default=0.1)
    parser.add_argument("--ctrl_update_every", type=int,   default=20)
    parser.add_argument("--viewer",  action="store_true")
    parser.add_argument("--warmup",  type=int,   default=50)
    args = parser.parse_args()

    backends = (
        ["mjwarp", "comfree", "mujoco_soft", "mujoco_anitescu"]
        if args.backend == "all"
        else [args.backend]
    )

    # --all + --viewer only shows viewer for first backend to avoid
    # opening multiple windows
    all_stats = []
    for i, backend in enumerate(backends):
        stats = run(
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
            viewer            = args.viewer and (i == 0),
            warmup_steps      = args.warmup,
        )
        all_stats.append(stats)

    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print(f"  Summary")
        print(f"{'='*60}")
        for s in all_stats:
            print(f"  {s['backend']:20s}  {s['mean_throughput']:.2e} steps/sec  "
                  f"({s['mean_step_time_ms']:.3f} ms/step)")


if __name__ == "__main__":
    main()
