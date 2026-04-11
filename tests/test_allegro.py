"""test_allegro.py

Forward simulation debugger for contact_study.
Runs any contact model backend against any XML scene, with optional
MuJoCo viewer showing world 0.

Usage:
    # Headless, comfree backend
    python test_allegro.py

    # With viewer, MuJoCo soft contact
    python test_allegro.py --viewer --backend mujoco_soft

    # Batched, custom XML
    python test_allegro.py --xml scenes/tasks/peg_in_hole_accurate.xml \
                            --nworld 512 --backend comfree --steps 500

    # All backends, no viewer
    python test_allegro.py --backend all
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


def _get_inner_data(d):
    """Unwrap XPBDData (or any wrapper with _d) to get raw MJWarp Data."""
    return d._d if hasattr(d, '_d') else d


def _get_inner_model(m):
    """Unwrap XPBDModel (or any wrapper with _m) to get raw MJWarp Model."""
    return m._m if hasattr(m, '_m') else m


def run(
    xml_path:          str   = "scenes/test_data/allegro/env_allegro_cube.xml",
    backend:           str   = "comfree",      # "mjwarp" | "comfree" | "mujoco_anitescu"
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
    debug:             bool  = False,
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

    print(f"  integrator       = {mjm.opt.integrator}")   # 0=Euler, 1=RK4, 2=implicit, 3=implicitfast
    print(f"  dof_armature min = {mjm.dof_armature.min():.2e}")
    print(f"  dof_armature max = {mjm.dof_armature.max():.2e}")
    print(f"  dof_damping min  = {mjm.dof_damping.min():.2e}")

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
    elif backend == "mujoco_anitescu":
        cfg = ContactModelConfig.M1()
    elif backend == "xpbd":
        cfg = ContactModelConfig.M4()
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
    # Debug: print initial state info
    # ------------------------------------------------------------------
    if debug:
        wp.synchronize()
        inner_d = _get_inner_data(d)
        inner_m = _get_inner_model(m)
        print(f"\n  [DEBUG] nv={inner_m.nv}, nq={mjm.nq}, nu={mjm.nu}")
        print(f"  [DEBUG] njmax={inner_d.njmax}, naconmax={inner_d.naconmax}")
        print(f"  [DEBUG] integrator={mjm.opt.integrator}, cone={mjm.opt.cone}")
        print(f"  [DEBUG] d type: {type(d).__name__}, inner_d type: {type(inner_d).__name__}")
        if hasattr(d, 'qfrc_total'):
            print(f"  [DEBUG] XPBD scratch arrays present (qfrc_total, qvel_pred, qfrc_constraint)")
        print()

        # --- #3: mass matrix conditioning ---
        # Host-side: inertias as MuJoCo compiled them
        print(f"\n  [DEBUG] body_mass     = {mjm.body_mass}")
        print(f"  [DEBUG] body_inertia min/max = "
            f"{mjm.body_inertia.min():.3e} / {mjm.body_inertia.max():.3e}")
        print(f"  [DEBUG] dof_M0 (joint-space mass diag, pre-crb) =")
        print(f"          min={mjm.dof_M0.min():.3e}  max={mjm.dof_M0.max():.3e}")
        print(f"          full: {mjm.dof_M0}")

        # Device-side: the actual qM after factor_m runs once
        # Run one forward so qM is populated, then read it back
        api.step(m, d)
        wp.synchronize()
        qM = inner_d.qM.numpy()    # shape (nworld, nM) — sparse upper triangle
        print(f"  [DEBUG] qM[0] min/max = {qM[0].min():.3e} / {qM[0].max():.3e}")
        inner_d = _get_inner_data(d)
        print(f"  [comfree sanity] nacon after 1 step = {inner_d.nacon.numpy()[0]}")
        print(f"  [comfree sanity] nefc  after 1 step = {inner_d.nefc.numpy()[0]}")

        # --- #6: efc.D vs efc.efc_mass convention ---
        # (api.step above already ran make_constraint once)
        nefc0 = int(inner_d.nefc.numpy()[0])
        print(f"\n  [DEBUG] nefc[0] = {nefc0}")
        if nefc0 > 0:
            efc_D = inner_d.efc.D.numpy()[0, :nefc0]
            print(f"  [DEBUG] efc.D[:nefc]        = {efc_D}")
            # efc_mass only exists on the comfree fork — guard it
            if hasattr(inner_d.efc, 'efc_mass'):
                efc_mass = inner_d.efc.efc_mass.numpy()[0, :nefc0]
                print(f"  [DEBUG] efc.efc_mass[:nefc] = {efc_mass}")
                print(f"  [DEBUG] D * efc_mass        = {efc_D * efc_mass}")
                print(f"  [DEBUG] 1/D                 = {1.0/efc_D}")

        from contact_study.contact_models.xpbd_backend import print_constraint_types
        print_constraint_types()
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
        for step_i in range(num_steps):

            # Perturb controls periodically
            if mjm.nu > 0 and step_i % ctrl_update_every == 0:
                noise = np.random.uniform(-ctrl_noise, ctrl_noise, ref_ctrl.shape)
                ctrl  = np.tile((ref_ctrl + noise).astype(np.float32), (nworld, 1))
                d.ctrl.assign(ctrl)

            t0 = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize()
            step_times.append(time.perf_counter() - t0)

            # Debug: print state every 100 steps
            if debug and step_i < 500 and step_i % 100 == 0:
                inner_d = _get_inner_data(d)
                qpos = inner_d.qpos.numpy()
                qvel = inner_d.qvel.numpy()
                qacc = inner_d.qacc.numpy()
                qacc_s = inner_d.qacc_smooth.numpy()
                print(f"  [DEBUG] step {step_i}:")
                print(f"    qpos[0,:5]       = {qpos[0,:min(5,qpos.shape[1])]}")
                print(f"    qvel[0] norm     = {np.linalg.norm(qvel[0]):.6f}")
                print(f"    qacc[0] norm     = {np.linalg.norm(qacc[0]):.6f}")
                print(f"    qacc_smooth norm = {np.linalg.norm(qacc_s[0]):.6f}")
                if hasattr(d, 'qfrc_total'):
                    qfrc_s = inner_d.qfrc_smooth.numpy()
                    qfrc_c = d.qfrc_constraint.numpy()
                    qfrc_t = d.qfrc_total.numpy()
                    print(f"    qfrc_smooth norm = {np.linalg.norm(qfrc_s[0]):.6f}")
                    print(f"    qfrc_const norm  = {np.linalg.norm(qfrc_c[0]):.6f}")
                    print(f"    qfrc_total norm  = {np.linalg.norm(qfrc_t[0]):.6f}")
                    nefc = inner_d.nefc.numpy()
                    nacon = inner_d.nacon.numpy()
                    print(f"    nefc={nefc[0]}, nacon={nacon[0]}")
                    if np.allclose(qacc[0], qacc_s[0], atol=1e-10):
                        print(f"    ⚠️  qacc == qacc_smooth")
                    if np.allclose(qacc[0], 0, atol=1e-10):
                        print(f"    ⚠️  qacc is all zeros!")

            # Sync viewer with world 0
            if v is not None:
                # Unwrap to raw MJWarp Data for the viewer sync —
                # XPBDData.__getattr__ proxy doesn't always work with
                # _mjwarp.get_data_into which may do internal type checks.
                inner_d = _get_inner_data(d)
                _mjwarp.get_data_into(mjd, mjm, inner_d, world_id=0)
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
                        choices=["mjwarp", "comfree", "mujoco_soft", "mujoco_anitescu", "xpbd", "all"])
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
    parser.add_argument("--debug",   action="store_true",
                        help="Print per-step diagnostics (first 500 steps)")
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
            debug             = args.debug,
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