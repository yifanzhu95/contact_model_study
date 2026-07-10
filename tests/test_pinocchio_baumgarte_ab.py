"""Pinocchio + ADMM eval "fling to infinity": a 2x2 study of two fixes.

Runs the same seeded grasp under a 2x2 of simulator variants and logs, per
physics substep, the max contact penetration (mm), the cube's linear speed
(m/s), and its height z (m).

Axis 1 — how the joint PD enters the contact solve
    plain-M     : the current code. The ADMM Delassus is built from the bare mass
                  matrix M, so the solver treats the PD-held fingers as free-
                  floating at their tiny (~3e-6) inertia — inconsistent with the
                  integration, and badly conditioned against the 0.14 kg cube.
    armature-PD : fold the implicit-PD impedance dt*(kv + dt*kp) into
                  `model.armature`, so crba AND the constraint Cholesky (=> the
                  Delassus the ADMM solve uses) both see A = M + that diagonal.
                  The solve then "sees" the stiff, PD-held fingers. Implemented
                  natively via the sim's explicit_pd path + armature (verified:
                  both crba and aba honor armature, and it shrinks the Delassus).

Axis 2 — the Baumgarte penetration corrector
    raw    : g[2::3] -= beta*pen/dt   (uncapped; the diagnosed fling source).
    clamp  : cap the separating speed at --clamp m/s (with a --slop deadband), by
             clipping the penetration fed to the corrector to clamp*dt/beta. This
             removes overlap gently instead of in a single step. The TRUE
             penetration is still logged (only the corrector sees the clamp).

What to look for
    * raw columns (plain-M / armature) pump cube speed and eject it (the fling).
    * clamp columns keep the cube slow and in the grasp.
    * plain-M vs armature-PD isolates whether folding PD into the solver's inertia
      changes the fling on its own — prediction: little on its own, and because a
      consistent operator delivers the Baumgarte kick more faithfully, it may NOT
      reduce it; the cap is what actually fixes it.

The cube is seeded a few mm into the thumb-tip mesh (--pre_penetrate) so the
beta*pen/dt kick fires deterministically.

Outputs
    videos/pinocchio_baumgarte_ab_<slug>.mp4    (one per condition)
    results/pinocchio_baumgarte_ab.png          (3-panel comparison plot)
    results/pinocchio_baumgarte_ab.npz          (raw per-substep logs)

Run (needs pinocchio; rendering also needs panda3d + EGL, like test_pinochio.py):
    python tests/test_pinocchio_baumgarte_ab.py                       # full 2x2
    python tests/test_pinocchio_baumgarte_ab.py --no-video --seconds 0.5
    python tests/test_pinocchio_baumgarte_ab.py --modes armature --columns raw clamp
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # allow `python tests/...` from any cwd

from contact_study.contact_models.pinocchio_sim import (  # noqa: E402
    PinocchioSimulator,
    PinocchioJointChannel,
    PinocchioFreeBodyChannel,
    PinocchioPdActuation,
    PinocchioContactConfig,
)

SCENE_PATH = REPO_ROOT / "scenes/leap_hand/leap_hand_right_w_sites.xml"
VIDEOS_DIR = REPO_ROOT / "videos"
RESULTS_DIR = REPO_ROOT / "results"

# Hand + cube initial state / grasp command. Mirrors _INIT_QPOS / _INIT_CTRL in
# contact_study/tasks/grasp_reorient.py (the same state the eval driver resets
# to). Layout: qpos = [16 hand joints, obj pos(3), obj quat wxyz(4)]  (nq = 23);
# ctrl = [16 hand joint position targets] (nu = 16). Kept here so the diagnostic
# stays self-contained (no warp/CUDA/task import).
_INIT_QPOS = np.array([
    0.74346777, -0.56903687, 0.91440081, 0.5741493,
    -0.010605284, -0.08351411, 0.70321997, 1.0184264,
    0.80782262, 0.61122899, 0.92718954, 0.61047876,
    0.69887738, 1.438706, 1.3375555, 0.19482527,
    0.018495468, 0.033628956, 0.083264539,
    0.93823638, 0.12995374, 0.31377877, 0.066086313,
], dtype=np.float64)

_INIT_CTRL = np.array([
    0.765751, -0.568012, 0.916951, 0.573897,
    -0.0191225, -0.0837503, 0.709056, 1.01884,
    0.830768, 0.610365, 0.929305, 0.610097,
    0.69912, 1.44581, 1.33179, 0.192794,
], dtype=np.float64)

_TIP_SITES = ["if_tip", "mf_tip", "rf_tip", "th_tip"]

# --- eval timestep / camera (mirrors grasp_reorient.py TaskConfig) -------------
_TIMESTEP = 1e-4
_CAM_FPS = 60.0
_CAM_POS = (0.2, 0.02, 0.4)
_cam_right = np.array([0.0, 1.0, 0.0]); _cam_right /= np.linalg.norm(_cam_right)
_cam_up = np.array([-1.0, 0.0, 0.5]); _cam_up /= np.linalg.norm(_cam_up)
_cam_fwd = -np.cross(_cam_right, _cam_up)
_cam_down = -_cam_up
_CAM_ROTMAT = tuple(
    tuple(float(v) for v in row)
    for row in np.column_stack([_cam_right, _cam_down, _cam_fwd])
)


def _make_config() -> SimpleNamespace:
    """The handful of fields PinocchioSimulator reads off its `config`."""
    return SimpleNamespace(
        timestep=_TIMESTEP,
        cam_fps=_CAM_FPS,
        cam_pos=_CAM_POS,
        cam_rotmat=_CAM_ROTMAT,
    )


def build_channels(mjm):
    """Replicate GraspReorientTask._make_pinocchio_simulator's identity channel
    map: every 1-DOF joint is a hand joint (MuJoCo name == Pinocchio name), and
    the free `obj_joint` is the cube. Returns everything the sim + logger need."""
    hand_jids = [
        j for j in range(mjm.njnt)
        if mjm.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    joint_channels = [
        PinocchioJointChannel(
            pin_name=mjm.joint(j).name,
            q_adr=int(mjm.jnt_qposadr[j]),
            v_adr=int(mjm.jnt_dofadr[j]),
        )
        for j in hand_jids
    ]
    obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
    obj_qadr = int(mjm.jnt_qposadr[obj_jnt])
    obj_vadr = int(mjm.jnt_dofadr[obj_jnt])
    free_channels = [
        PinocchioFreeBodyChannel(pin_name="obj_joint", q_adr=obj_qadr, v_adr=obj_vadr)
    ]
    ctrl_joint_names = [
        mjm.joint(int(mjm.actuator(a).trnid[0])).name for a in range(mjm.nu)
    ]
    return joint_channels, free_channels, ctrl_joint_names, obj_qadr, obj_vadr


def seed_penetration(mjm, qpos, obj_qadr, depth, tip):
    """Nudge the cube `depth` metres toward one fingertip so it starts a few mm
    inside the tip meshes, making the β*pen/dt kick fire deterministically on the
    first substep. Seeding toward the *centroid* of the four tips is degenerate
    (they cage the cube, so the centroid ~ the cube center); driving toward a
    single opposing tip (the thumb by default) reliably produces deep,
    multi-contact penetration (empirically ~7 mm at 8 mm for th_tip)."""
    if depth <= 0.0:
        return qpos
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = qpos
    mujoco.mj_forward(mjm, mjd)
    tip_pos = mjd.site_xpos[
        mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, tip)
    ].copy()
    cube = qpos[obj_qadr:obj_qadr + 3].copy()
    d = tip_pos - cube
    n = np.linalg.norm(d)
    if n < 1e-9:
        return qpos
    qpos = qpos.copy()
    qpos[obj_qadr:obj_qadr + 3] = cube + depth * (d / n)
    return qpos


@dataclass(frozen=True)
class Condition:
    """One cell of the 2x2. `mode` selects how PD enters the contact solve;
    `clamp` (m/s) caps the Baumgarte separating speed (None = uncapped 'raw')."""
    mode: str              # "plainM" | "armature"
    beta: float            # Baumgarte gain
    clamp: float | None    # separating-speed cap (m/s); None = uncapped
    slop: float = 0.0      # penetration deadband (m) before the corrector engages

    @property
    def clamped(self) -> bool:
        return self.clamp is not None

    @property
    def label(self) -> str:
        m = "armature-PD" if self.mode == "armature" else "plain-M"
        c = f", cap {self.clamp:g} m/s" if self.clamped else ""
        return f"{m}, β={self.beta:g}{c}"

    @property
    def slug(self) -> str:
        c = f"cap{self.clamp:g}" if self.clamped else "raw"
        return f"{self.mode}_b{self.beta:g}_{c}".replace(".", "p")


# Entity-stable colors: color follows the condition, not its position in the run
# (dataviz non-negotiable). Warm = uncapped corrector (flings), cool = capped.
_COND_COLOR = {
    ("plainM", False):   "#e34948",  # red    — current baseline (flings)
    ("armature", False): "#eb6834",  # orange — PD folded into the operator, still uncapped
    ("plainM", True):    "#2a78d6",  # blue   — capped corrector
    ("armature", True):  "#1baf7a",  # aqua   — PD folded in + capped
}


def _cond_color(cond: Condition) -> str:
    return _COND_COLOR.get((cond.mode, cond.clamped), "#4a3aa7")


def run_condition(cond, mjm, channels, args):
    """Build the sim for this Condition, drive the held grasp, and log per-substep
    penetration / cube speed / height. Returns a log dict."""
    joint_channels, free_channels, ctrl_joint_names, obj_qadr, obj_vadr = channels

    want_video = not args.no_video
    video_path = (str(VIDEOS_DIR / f"pinocchio_baumgarte_ab_{cond.slug}.mp4")
                  if want_video else None)

    # Both axes are driven through the *native* pinocchio_sim config, so this test
    # exercises the production code paths (not a mock): the Baumgarte cap lives in
    # PinocchioContactConfig.baumgarte_max_vel/slop, and armature-PD in
    # PinocchioPdActuation.armature_pd.
    contact_cfg = PinocchioContactConfig(
        friction=args.friction, use_mesh_geoms=True, baumgarte_gain=cond.beta,
        baumgarte_max_vel=(cond.clamp if cond.clamped else 0.0),
        baumgarte_slop=(cond.slop if cond.clamped else 0.0),
    )
    pid = PinocchioPdActuation(
        ctrl_joint_names=ctrl_joint_names, use_direct_gains=True, kp=3.0, kd=0.01,
        armature_pd=(cond.mode == "armature"),
    )
    sim = PinocchioSimulator(
        model_path=str(SCENE_PATH), config=_make_config(),
        nq=mjm.nq, nv=mjm.nv, pid=pid,
        joint_channels=joint_channels, free_channels=free_channels,
        contact_cfg=contact_cfg, video_path=video_path, render=want_video,
    )

    # Log the coal penetration each substep (the corrector's cap is applied
    # natively inside _substep, so what we log here is the true overlap).
    probe = {"pens": None, "ncon": 0}
    _orig_detect = sim._detect_contacts

    def _detect_and_log():
        cms, cds, pens = _orig_detect()
        probe["pens"] = pens
        probe["ncon"] = len(cms)
        return cms, cds, pens

    sim._detect_contacts = _detect_and_log

    # Initial state: grasp pose with the cube seeded a few mm into the tips.
    qpos0 = seed_penetration(mjm, _INIT_QPOS.copy(), obj_qadr,
                             args.pre_penetrate, args.seed_tip)
    sim.reset(qpos0, np.zeros(mjm.nv))
    ctrl_hi = mjm.actuator_ctrlrange[:, 1]
    sim.apply_control(_INIT_CTRL + args.squeeze * (ctrl_hi - _INIT_CTRL))

    n_steps = int(round(args.seconds / _TIMESTEP))
    t, pen_mm, speed, wspeed, zpos, ncon = [], [], [], [], [], []
    flung_at = None
    for i in range(n_steps):
        sim.step(1)
        st = sim.get_state()
        pos = st.qpos[obj_qadr:obj_qadr + 3]
        pens = probe["pens"]
        pmax = float(np.max(pens)) * 1e3 if pens is not None and pens.size else 0.0

        t.append((i + 1) * _TIMESTEP)
        pen_mm.append(pmax)
        speed.append(float(np.linalg.norm(st.qvel[obj_vadr:obj_vadr + 3])))
        wspeed.append(float(np.linalg.norm(st.qvel[obj_vadr + 3:obj_vadr + 6])))
        zpos.append(float(pos[2]))
        ncon.append(int(probe["ncon"]))

        if want_video:
            sim.render()

        # Stop once the cube has clearly been flung (or the state blew up).
        if not np.all(np.isfinite(st.qpos)) or np.linalg.norm(pos) > 2.0:
            flung_at = t[-1]
            break

    if want_video:
        sim.save_video(video_path)

    peak_v = max(speed) if speed else 0.0
    print(f"  {cond.label:26s}  steps={len(t):5d}  peak_speed={peak_v:7.3f} m/s  "
          f"final_pen={pen_mm[-1] if pen_mm else 0.0:6.2f} mm  "
          f"mean_contacts={np.mean(ncon) if ncon else 0.0:.1f}  "
          + (f"FLUNG @ {flung_at*1e3:.1f} ms" if flung_at else "no fling"))
    if want_video:
        print(f"      video -> {video_path}")

    return {
        "label": cond.label, "slug": cond.slug, "color": _cond_color(cond),
        # linestyle encodes the mode so the (often near-identical) capped pair
        # stays visible where the colors overlap: plain-M solid, armature dashed.
        "ls": "-" if cond.mode == "plainM" else (0, (5, 2)),
        "t": np.asarray(t), "pen_mm": np.asarray(pen_mm),
        "speed": np.asarray(speed), "wspeed": np.asarray(wspeed),
        "z": np.asarray(zpos), "ncon": np.asarray(ncon),
        "peak_speed": peak_v,
        "flung_at": flung_at if flung_at is not None else np.nan,
    }


# --- plot (dataviz palette: CVD-safe categorical hues, recessive chrome) --------
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"


def _style_axis(ax, ylabel, title):
    ax.set_facecolor(_SURFACE)
    ax.set_ylabel(ylabel, color=_INK2, fontsize=10)
    ax.set_title(title, color=_INK, fontsize=11, loc="left", pad=6)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_INK2, labelsize=9)


def make_plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9.5, 10.0))
    fig.patch.set_facecolor(_SURFACE)

    panels = [
        ("pen_mm", "max penetration (mm)",
         "Max contact penetration  —  capped correctors (cool) remove overlap gently, not in one step"),
        ("speed", "cube speed (m/s)",
         "Cube linear speed  —  the fling signal: an uncapped corrector (warm) pumps it; capping holds it down"),
        ("z", "cube height z (m)",
         "Cube height  —  runaway = flung, flat = held in the grasp"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for r in results:
            ax.plot(r["t"], r[key], color=r["color"], linewidth=1.8,
                    linestyle=r["ls"], label=r["label"])
        _style_axis(ax, ylabel, title)

    axes[-1].set_xlabel("time (s)", color=_INK2, fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    ncol = 2 if len(results) > 2 else len(results)
    leg = fig.legend(handles, labels, loc="upper center", ncol=ncol,
                     frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.997))
    for txt in leg.get_texts():
        txt.set_color(_INK)

    fig.suptitle("Fixing the ADMM fling:  joint-PD-in-solver (armature)  ×  capped Baumgarte corrector",
                 color=_INK, fontsize=12.5, y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"  plot  -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=["plainM", "armature"],
                   choices=["plainM", "armature"],
                   help="PD-in-solver axis (default: both).")
    p.add_argument("--columns", nargs="+", default=["raw", "clamp"],
                   choices=["raw", "clamp"],
                   help="Baumgarte-corrector axis (default: both).")
    p.add_argument("--beta", type=float, default=0.2,
                   help="Baumgarte gain used across the matrix (default 0.2).")
    p.add_argument("--clamp", type=float, default=0.05,
                   help="Separating-speed cap (m/s) for the 'clamp' column "
                        "(default 0.05).")
    p.add_argument("--slop", type=float, default=0.0005,
                   help="Penetration deadband (m) for the 'clamp' column "
                        "(default 0.0005 = 0.5 mm).")
    p.add_argument("--seconds", type=float, default=0.5,
                   help="Sim seconds per condition (default 0.5; dt=1e-4).")
    p.add_argument("--pre_penetrate", type=float, default=0.010,
                   help="Metres to seed the cube into the tip meshes at reset "
                        "(default 0.010; 0 = the bare grasp, which already carries "
                        "~6 contacts at ~0.9 mm).")
    p.add_argument("--seed_tip", type=str, default="th_tip", choices=_TIP_SITES,
                   help="Fingertip to drive the cube toward when seeding "
                        "penetration (default th_tip; the thumb gives the deepest "
                        "multi-contact overlap).")
    p.add_argument("--squeeze", type=float, default=0.0,
                   help="Fraction [0,1] to firm the grasp command toward each "
                        "actuator's closing limit (default 0 = hold _INIT_CTRL).")
    p.add_argument("--friction", type=float, default=0.5)
    p.add_argument("--no-video", action="store_true",
                   help="Skip rendering (no panda3d/EGL needed); still logs+plots.")
    args = p.parse_args()

    if not args.no_video:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    mjm = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    if _INIT_QPOS.shape[0] != mjm.nq or _INIT_CTRL.shape[0] != mjm.nu:
        raise ValueError(
            f"_INIT_QPOS ({_INIT_QPOS.shape[0]}) / _INIT_CTRL ({_INIT_CTRL.shape[0]}) "
            f"do not match model nq={mjm.nq} / nu={mjm.nu}."
        )
    channels = build_channels(mjm)

    # Build the requested cells of the 2x2 (mode x corrector).
    conditions = [
        Condition(mode=mode, beta=args.beta,
                  clamp=(args.clamp if col == "clamp" else None), slop=args.slop)
        for mode, col in itertools.product(args.modes, args.columns)
    ]

    print(f"2x2 fling study  |  scene={SCENE_PATH.name}  seconds={args.seconds}  "
          f"β={args.beta}  clamp={args.clamp} m/s  slop={args.slop*1e3:.1f}mm  "
          f"pre_penetrate={args.pre_penetrate*1e3:.1f}mm  squeeze={args.squeeze}")
    results = [run_condition(c, mjm, channels, args) for c in conditions]

    make_plot(results, str(RESULTS_DIR / "pinocchio_baumgarte_ab.png"))

    npz = {}
    for r in results:
        for k in ("t", "pen_mm", "speed", "wspeed", "z", "ncon"):
            npz[f"{r['slug']}_{k}"] = r[k]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(str(RESULTS_DIR / "pinocchio_baumgarte_ab.npz"), **npz)
    print(f"  logs  -> {RESULTS_DIR / 'pinocchio_baumgarte_ab.npz'}")

    # Headline table: peak cube speed per cell (higher = closer to a fling).
    print("\nSummary (peak cube speed, m/s — lower is better):")
    for r in sorted(results, key=lambda r: r["peak_speed"], reverse=True):
        flung = "" if np.isnan(r["flung_at"]) else f"  (flung @ {r['flung_at']*1e3:.0f} ms)"
        print(f"  {r['label']:26s}  {r['peak_speed']:7.3f}{flung}")


if __name__ == "__main__":
    main()
