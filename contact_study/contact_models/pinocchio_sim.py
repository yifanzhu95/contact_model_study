"""Pinocchio-backed eval simulator.

Wraps a Pinocchio model + ADMM contact solver as the high-fidelity "real"
environment, exposing the EvalSimulator interface so it is interchangeable with
MujocoSimulator and DrakeSimulator. The simulation scheme mirrors
tests/replay_pinocchio_controls.py exactly:

  * multi-root MJCF (hand fingers + free object) split into single-root MJCFs,
    parsed separately, and merged via pin.appendModel into one model;
  * each fine step: detect coal contacts, build a PointContactConstraintModel per
    contact with a native Baumgarte corrector, and resolve them with the ADMM
    solver;
  * the hand joints are position-controlled with an explicit forward-Euler PD:
    fixed stiffness kp, per-joint critically-damped kd = 2*zeta*sqrt(kp*M_ii).

Like DrakeSimulator, Pinocchio stores state in its own joint layout, so the
caller supplies channels mapping it onto the MuJoCo-ordered qpos/qvel the
EvalSimulator interface promises:
  * `joint_channels` — 1-DOF revolute/prismatic joints -> MuJoCo qpos/qvel addr.
  * `free_channels`  — floating bodies -> MuJoCo freejoint (7 qpos [pos, quat
                       wxyz] + 6 qvel [lin(world), ang(body-local)]).

Two convention conversions are handled internally (never via raw q slices):
  * Pinocchio free-joint quaternion is (x,y,z,w); MuJoCo's is (w,x,y,z).
  * A free joint's body world pose is oMi[jid]; its q is jointPlacement^-1 * pose
    (the parser bakes the body offset into jointPlacement), and its velocity is
    expressed in the joint/body frame (MuJoCo's linear part is world-frame).

pinocchio (and panda3d for rendering) are imported lazily so importing this
module does not require them on the GPU-rollout machines.
"""

from __future__ import annotations

import datetime
import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from contact_study.sim.base import (
    EvalSimulator, EvalState, FrameClock, camera_pose_from_config, resolve_video_path,
)

# A geom whose in-plane box half-extents both exceed this is treated as the floor
# and excluded from all collision pairs.
_FLOOR_HALFEXTENT_THRESH = 0.5

# Debug flag: when True, each substep prints every detected contact's coal
# penetration depth (m) and whether the ADMM solve converged. Off by default —
# this fires every substep, so it is a lot of stdout at real step rates.
PRINT_CONTACT_DEBUG = False


# ---------------------------------------------------------------------------
# Channel / actuation configuration (mirrors drake_sim.py)
# ---------------------------------------------------------------------------
@dataclass
class PinocchioJointChannel:
    """Maps one Pinocchio 1-DOF joint (by name) to a MuJoCo qpos/qvel address."""
    pin_name: str
    q_adr: int
    v_adr: int


@dataclass
class PinocchioFreeBodyChannel:
    """Maps a Pinocchio free-flyer joint (by name) to a MuJoCo freejoint
    (7 qpos [px,py,pz, qw,qx,qy,qz] / 6 qvel [vx,vy,vz, wx,wy,wz] ang body-local)."""
    pin_name: str
    q_adr: int
    v_adr: int


@dataclass
class PinocchioPdActuation:
    """Position control via an explicit forward-Euler PD on the hand joints.

    ctrl_joint_names is in MuJoCo *control* order: apply_control(ctrl)[k] is the
    desired position for Pinocchio joint ctrl_joint_names[k]. The stiffness `kp`
    is used directly (it sets the closed-loop time constant).

    By default (use_direct_kd=False) the damping is derived per-joint from the
    mass-matrix diagonal so the loop stays critically damped at that stiffness:
    kd = 2*zeta*sqrt(kp*M_ii). Set use_direct_kd=True to instead apply a fixed
    `kd` directly to every controlled joint, bypassing the mass-matrix/zeta
    derivation entirely (zeta is then unused).

    `armature` is a per-DOF rotor inertia added to each controlled joint's mass
    (model.armature). The LEAP finger inertias are tiny (M_ii ~ 3e-6), so the
    contact solve treats the PD-held fingers as near-massless next to the cube; a
    large armature raises their effective inertia (better-conditioned Delassus,
    steadier grasp). It also enters the critically-damped kd via M_ii above
    (when use_direct_kd=False).

    `force_limit` mirrors MuJoCo's per-joint actuator-force saturation
    (jnt_actfrcrange): a (n_ctrl, 2) array of [lo, hi] torque bounds in ctrl
    order (same order as ctrl_joint_names). Each substep the commanded PD torque
    is clamped to this range before it drives the dynamics, exactly as MuJoCo
    clamps qfrc_actuator to jnt_actfrcrange. Without it the PD applies the full
    (unbounded) kp*error torque, which diverges from MuJoCo whenever the raw PD
    torque exceeds the joint's limit (e.g. the LEAP hand's +/-0.95 N*m cap).
    None => unlimited (no clamp), matching a model with no actuatorfrcrange."""
    ctrl_joint_names: list[str]
    kp: float = 3.0
    zeta: float = 1.0
    gravity_comp: bool = False
    armature: float = 0.001
    use_direct_kd: bool = True
    kd: float = 0.11
    force_limit: "np.ndarray | None" = None


@dataclass
class PinocchioJointConstraintConfig:
    """Joint-level constraints (position limits + dry friction), resolved in the
    same ADMM solve as the contacts.

    Pinocchio's forward dynamics (aba) applies NEITHER of these on its own —
    model.lower/upperPositionLimit and model.friction are inert metadata, exactly
    like model.damping. They only take effect as explicit constraint models, which
    is what this config switches on.

    enforce_limits: build a JointLimitConstraintModel over the 1-DOF joints named
        in joint_channels. Bounds come from the model's own lower/upperPositionLimit,
        which Pinocchio's MJCF parser DOES import correctly from <joint range>. The
        constraint is unilateral: each substep only the joints within `limit_margin`
        of (or past) a bound are activated, so free joints are unaffected.
    limit_margin: activate a bound this far (rad) before it is reached. 0.0 means
        the joint must cross the limit before the constraint pushes back (a small
        overshoot, resolved within a step or two). A small positive margin engages
        earlier and reduces that overshoot.
    frictionloss: pin joint name -> dry-friction torque (N*m), i.e. MuJoCo's
        <joint frictionloss>. Pass the MuJoCo values explicitly: Pinocchio's MJCF
        parser does NOT inherit frictionloss from a <default> block (same gap as
        contype/conaffinity), so model.friction reads 0 for a defaults-driven scene.
        Omitted/zero-valued joints get no friction constraint.

        NOTE the solver works in IMPULSE space, so these torques are scaled by the
        timestep (bound = frictionloss * dt) when the constraint is built; passing
        the raw torque as the bound would lock the joint permanently.
    """
    enforce_limits: bool = True #IDK IF THIS MAKES SENSE TO KEPP ON TBH.
    limit_margin: float = 0.0
    frictionloss: "dict[str, float] | None" = None


@dataclass
class PinocchioContactConfig:
    """Frictional point-contact knobs. Collision pairs are all non-adjacent,
    non-floor geom pairs (cube<->hand and hand<->hand). Each contact carries a
    native Baumgarte corrector: its drift gets a push-back Kp*position_error/dt
    (+ Kd*velocity_error/dt), read straight off the constraint in the solve loop
    (mirroring results/g1-constraint-simulation.py). Kp=0 disables it."""
    friction: float = 0.5
    use_mesh_geoms: bool = True
    # Replace each triangle-mesh (BVH) collision geom with its convex hull so coal
    # uses the analytic convex-convex (GJK/EPA) contact path — one stable normal +
    # accurate penetration depth per step — instead of per-triangle BVH queries
    # whose faceted, frame-to-frame-jittery normals drove the cube penetrating /
    # "sticking" to the mesh fingertips. The LEAP tips are already convex to within
    # triangulation noise, so the hull reproduces their shape almost exactly. Only
    # affects collision geoms; the visual meshes are untouched.
    use_convex_tips: bool = True
    baumgarte_kp: float = 100.0
    baumgarte_kd: float = 50.0
    admm_max_iterations: int = 5000


# ---------------------------------------------------------------------------
# Model build / contact helpers (self-contained; mirrors replay_pinocchio_controls.py)
# ---------------------------------------------------------------------------
def _quat_mul(a, b):
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)


def _euler_to_quat_wxyz(angles, seq="xyz"):
    """MuJoCo default (intrinsic, radian) euler -> wxyz quaternion: R = R0·R1·R2
    for seq[0],seq[1],seq[2], each an intrinsic rotation about the moving axis."""
    axes = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for ch, ang in zip(seq, angles):
        ax, ay, az = axes[ch]
        h = 0.5 * ang
        s = np.sin(h)
        q = _quat_mul(q, np.array([np.cos(h), ax * s, ay * s, az * s]))
    return q


def _root_body_pose(body):
    """Placement (pos[3], quat_wxyz[4]) to REAPPLY to a worldbody root <body> when
    merging its subtree, or identity when none is needed.

    Pinocchio's MJCF parser only drops the placement of a root body that has NO
    joint — a body welded straight to the world (e.g. the UR5e base with
    quat="0 0 0 -1", which otherwise leaves the whole arm rotated 180 deg about
    z). That dropped pose is read here from the body's pos + quat/euler and
    reapplied by merge_models.

    A root body that carries a joint keeps its placement baked into that joint's
    jointPlacement, so reapplying it would double-count (it left the LEAP fingers
    offset by their mount translation and rotated by their mount quat). Return
    identity for those:
      - a free joint: pose comes from the free-joint qpos we write each step;
      - a hinge/slide joint: the parser already placed the joint at the mount."""
    has_joint = (body.find("freejoint") is not None or
                 len(body.findall("joint")) > 0)
    if has_joint:
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ", dtype=float)
    if pos.shape[0] != 3:
        pos = np.zeros(3)
    if body.get("quat") is not None:
        quat = np.fromstring(body.get("quat"), sep=" ", dtype=float)
        quat = quat / (np.linalg.norm(quat) or 1.0)
    elif body.get("euler") is not None:
        quat = _euler_to_quat_wxyz(np.fromstring(body.get("euler"), sep=" ", dtype=float))
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    return pos, quat


# Name of the synthetic wrapper <body> that split_into_single_root_mjcfs uses to
# carry the loose <worldbody> geoms at the world frame. Canonicalized to "world"
# by _canonical_body_name so the scene's <exclude body1="world" .../> tags match
# the wrapped geoms.
_WORLD_GEOMS_BODY = "world_geoms"

# MJCF top-level sections that MuJoCo *merges* (rather than replaces) when the
# same section appears in an included file: their children accumulate into a
# single section. Everything else (compiler, default, option, statistic, visual,
# ...) is left as separate sibling elements — the split/exclude parsers read
# those via find()/findall() and don't need them coalesced.
_MJCF_MERGE_SECTIONS = frozenset({
    "asset", "worldbody", "contact", "actuator", "sensor",
    "equality", "tendon", "keyframe",
})


def load_mjcf_root_with_includes(mjcf_path):
    """Parse an MJCF and recursively splice every <include file="..."/> inline,
    returning a single flattened <mujoco> root with no <include> elements.

    Pinocchio's MJCF machinery here (split_into_single_root_mjcfs,
    parse_mjcf_excludes, ...) reads the scene with ElementTree, which — unlike
    MuJoCo's own loader — does not follow <include>. Scenes like
    scenes/leap/env_leap_*.xml keep the hand in a separate file and pull it in
    with <include file="./leap_right_hand.xml"/>; without this the hand bodies,
    assets, contacts, and actuators are invisible to the Pinocchio path.

    Included children are merged the way MuJoCo merges them: the container
    sections in _MJCF_MERGE_SECTIONS (asset/worldbody/contact/...) coalesce into
    one element so downstream find("worldbody")/find("asset") see the union;
    other sections are appended as siblings. <include> paths are resolved
    relative to the file that contains them, and nested includes are followed."""
    root = ET.parse(mjcf_path).getroot()
    base_dir = os.path.dirname(os.path.abspath(mjcf_path))

    merged = ET.Element(root.tag, dict(root.attrib))
    sections = {}  # tag -> element, for the mergeable container sections

    def _append(child):
        tag = child.tag
        if tag in _MJCF_MERGE_SECTIONS:
            existing = sections.get(tag)
            if existing is None:
                sections[tag] = child
                merged.append(child)
            else:
                for sub in list(child):
                    existing.append(sub)
                # carry over any section-level attributes (rare) without clobber
                for k, v in child.attrib.items():
                    existing.attrib.setdefault(k, v)
        else:
            merged.append(child)

    def _splice(node, node_dir):
        for child in list(node):
            if child.tag == "include":
                inc_path = os.path.join(node_dir, child.get("file"))
                inc_root = load_mjcf_root_with_includes(inc_path)
                _splice(inc_root, os.path.dirname(os.path.abspath(inc_path)))
            else:
                _append(child)

    _splice(root, base_dir)
    return merged


def split_into_single_root_mjcfs(mjcf_path, scene_dir):
    """Write one temp MJCF per <worldbody> root <body> (with the loose worldbody
    geoms folded into the first), so Pinocchio's first-body-only MJCF parser can
    read each independent root into its own model. Returns (temp_paths,
    root_poses), where root_poses[i] is the (pos, quat_wxyz) fixed world placement
    of subtree i's root body (Pinocchio drops it; merge_models reapplies it)."""
    root = load_mjcf_root_with_includes(mjcf_path)
    compiler = root.find("compiler")
    asset = root.find("asset")
    # <default> class/childclass definitions must travel with the split models:
    # bodies/geoms/joints that reference a class (e.g. the UR5e arm's
    # childclass="ur5e", class="size3"/"visual") make Pinocchio's MJCF parser
    # throw `unordered_map::at` on an undefined-class lookup otherwise. Scenes
    # with no class references (e.g. the LEAP hand) are unaffected.
    defaults = root.findall("default")
    # Pinocchio's MJCF parser throws `unordered_map::at` on a <material> that
    # carries a `class` attribute (it mishandles the material-class lookup even
    # when the class is defined). Materials are visual-only and these split
    # MJCFs are Pinocchio-only (never MuJoCo), so drop the attribute — the geoms
    # still resolve the material by name; only the class-inherited specular/
    # shininess (cosmetic) is lost.
    if asset is not None:
        for mat in asset.findall("material"):
            mat.attrib.pop("class", None)
    worldbody = root.find("worldbody")

    loose_geoms = [el for el in worldbody if el.tag == "geom"]
    bodies = [el for el in worldbody if el.tag == "body"]

    # Instance-specific suffix so concurrent runs (e.g. one Pinocchio eval per
    # HPC node/process) don't write, read, and delete the same shared temp files
    # in scene_dir and clobber each other.
    token = (f"{datetime.datetime.now():%Y%m%d_%H%M%S}_"
             f"{os.getpid()}_{uuid.uuid4().hex[:8]}")

    def _write_split(children, label):
        new_root = ET.Element("mujoco", root.attrib)
        if compiler is not None:
            new_root.append(compiler)
        if asset is not None:
            new_root.append(asset)
        for d in defaults:
            new_root.append(d)
        new_worldbody = ET.SubElement(new_root, "worldbody")
        for child in children:
            new_worldbody.append(child)
        tmp_path = os.path.join(scene_dir, f"_tmp_pin_sim_split_{token}_{label}.xml")
        ET.ElementTree(new_root).write(tmp_path)
        return tmp_path

    # One split per root <body>, each reapplied at its own fixed world pose by
    # merge_models (Pinocchio drops the root placement; see _root_body_pose).
    tmp_paths = []
    root_poses = []
    for i, body in enumerate(bodies):
        tmp_paths.append(_write_split([body], str(i)))
        root_poses.append(_root_body_pose(body))

    # Loose <worldbody> geoms (palm plates, floor) belong to the WORLD frame, not
    # to any finger. Give them their own identity-placed wrapper <body> split
    # instead of folding them into the first finger: folding made merge_models
    # reapply that finger's mount rotation to them, tilting the palm plates and
    # floor (a ~120 deg rotation for this LEAP hand). The wrapper's name
    # canonicalizes to "world" (see _canonical_body_name) so <exclude
    # body1="world" .../> tags still match the wrapped geoms.
    if loose_geoms:
        wrapper = ET.Element("body", {"name": _WORLD_GEOMS_BODY, "pos": "0 0 0"})
        for geom in loose_geoms:
            wrapper.append(geom)
        tmp_paths.append(_write_split([wrapper], "worldgeoms"))
        root_poses.append((np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    return tmp_paths, root_poses


def merge_models(pin, parts, root_poses):
    """Append single-root (model, collision, visual) tuples into one combined
    model, each placed at its root body's fixed world pose (root_poses[i] =
    (pos, quat_wxyz)). We seed from an EMPTY model rather than parts[0] so the
    FIRST subtree's root pose is applied too — Pinocchio's MJCF parser drops
    every root <body>'s placement, including the seed's (this is what left the
    UR5e arm rotated 180 deg). appendModel merges one geometry model per call, so
    run it twice per subtree against the same pre-merge snapshot to keep the
    collision and visual geometry index-consistent. Returns (model, coll, vis)."""
    model = pin.Model()
    coll = pin.GeometryModel()
    vis = pin.GeometryModel()
    for (mB, collB, visB), (pos, quat) in zip(parts, root_poses):
        w, x, y, z = quat
        aMb = pin.SE3(pin.Quaternion(w, x, y, z).matrix(), np.asarray(pos, dtype=float))
        prev = model
        model, coll = pin.appendModel(prev, mB, coll, collB, 0, aMb)
        _, vis = pin.appendModel(prev, mB, vis, visB, 0, aMb)
    return model, coll, vis


def ensure_obj_material_stubs(visual_model) -> int:
    """Write a neutral stub .mtl next to any visual .obj mesh that references a
    `mtllib` file which doesn't exist on disk. panda3d's OBJ loader (assimp)
    aborts the whole load when it can't find a referenced .mtl ("failed to
    locate material" -> "No object detected" -> the model fails as invalid),
    whereas MuJoCo/coal ignore the missing .mtl entirely. The UR5e meshes here
    were exported (Blender) with `mtllib Black.mtl` etc. but no .mtl files. The
    panda3d viewer overrides every material anyway (overrideMaterial=True), so
    the stub's content is cosmetic — it only needs to exist for assimp to parse
    the geometry. Idempotent: skips any .mtl that already exists. Returns the
    number of stubs written."""
    written = 0
    seen_objs = set()
    for go in visual_model.geometryObjects:
        mesh_path = getattr(go, "meshPath", "") or ""
        if not mesh_path.lower().endswith(".obj") or mesh_path in seen_objs:
            continue
        seen_objs.add(mesh_path)
        if not os.path.isfile(mesh_path):
            continue
        mesh_dir = os.path.dirname(mesh_path)
        try:
            with open(mesh_path) as f:
                libs = [ln.split(None, 1)[1].strip()
                        for ln in f if ln.startswith("mtllib") and len(ln.split()) > 1]
        except OSError:
            continue
        for lib in libs:
            mtl_path = os.path.join(mesh_dir, lib)
            if os.path.isfile(mtl_path):
                continue
            # Name the material after the .mtl stem so a `usemtl Foo` in the .obj
            # resolves; a neutral gray is fine (the viewer overrides it anyway).
            stem = os.path.splitext(os.path.basename(lib))[0]
            try:
                with open(mtl_path, "w") as f:
                    f.write(f"newmtl {stem}\nKa 0 0 0\nKd 0.6 0.6 0.6\nKs 0 0 0\nd 1\n")
                written += 1
            except OSError:
                pass
    return written


def _box_half_extents(go):
    """Return the box half-side as a length-3 array, or None for non-box geoms."""
    try:
        return np.asarray(go.geometry.halfSide, dtype=float)
    except Exception:
        return None


def _coal_module():
    """Return the coal (a.k.a. hppfcl) module Pinocchio builds its geometry with,
    or None if neither import is available."""
    try:
        import coal
        return coal
    except Exception:
        pass
    try:
        import hppfcl
        return hppfcl
    except Exception:
        return None


def convexify_mesh_geoms(geom_model):
    """Replace every triangle-mesh (BVH) collision geom with its convex hull, in
    place. coal loads MJCF `<geom type="mesh">` as a BVHModelOBBRSS triangle soup,
    whose contact query returns a single per-triangle face normal that flips
    between adjacent facets as bodies slide — the source of the noisy normals /
    erratic penetration depth that made the cube penetrate and stick to the mesh
    fingertips. A coal.Convex uses the analytic GJK/EPA support-mapping path
    (like Box/Sphere): one stable normal + accurate signed penetration per step,
    while keeping the true tip shape (the LEAP tips are convex to triangulation
    noise). Returns the number of geoms converted."""
    coal = _coal_module()
    if coal is None:
        return 0
    n_converted = 0
    for go in geom_model.geometryObjects:
        g = go.geometry
        if not type(g).__name__.startswith("BVHModel"):
            continue  # already a primitive / convex; nothing to do
        try:
            verts = np.asarray(g.vertices(), dtype=float)
        except Exception:
            continue
        if verts.ndim != 2 or verts.shape[0] < 4:
            continue
        pts = coal.StdVec_Vec3s()
        for row in verts:
            pts.append(row)
        # keepTriangles=True needs qhull's "Qt" (triangulated output).
        go.geometry = coal.Convex.convexHull(pts, True, "Qt")
        n_converted += 1
    return n_converted


def _is_adjacent(model, j1, j2):
    """Same joint or a direct parent-child relationship between two non-universe
    joints (a contact there would be a spurious, permanently-active constraint).
    Universe (j=0) is the computational root, not a physical body, so the
    parent-child test is skipped whenever either joint is universe (otherwise
    every palm<->finger / palm<->object contact would be filtered out)."""
    if j1 == j2:
        return True
    if j1 > 0 and j2 > 0:
        return model.parents[j1] == j2 or model.parents[j2] == j1
    return False


def parse_mjcf_excludes(mjcf_path) -> set[frozenset[str]]:
    """Read <contact><exclude body1=".." body2=".."/></contact> pairs from the
    original (pre-split) MJCF. These are body-name pairs the model author has
    explicitly marked as never-colliding (mirroring MuJoCo's own semantics),
    used to patch cases _is_adjacent's joint-topology heuristic can't see —
    e.g. a body welded straight to the world (no <joint>) followed by a child
    joint: the child's parent joint id is universe (0), which _is_adjacent
    deliberately treats as "not adjacent" (see its docstring) since that same
    joint-parent-is-universe shape also covers legitimate independent contacts
    (separate fingers, hand<->object) that must NOT be excluded."""
    contact = load_mjcf_root_with_includes(mjcf_path).find("contact")
    if contact is None:
        return set()
    return {
        frozenset((exc.get("body1"), exc.get("body2")))
        for exc in contact.findall("exclude")
    }


def parse_body_box_geoms(mjcf_path, body_name):
    """Read the box <geom> children of the named <body> as
    (half_extents[3], local_pos[3], local_quat_wxyz[4], rgba[4]) tuples, so the
    body's on-screen appearance can be cloned elsewhere (e.g. a translucent goal
    marker built from the manipulated cube's own face plates). Poses/sizes are in
    the body frame, exactly as MuJoCo stores them (<geom size> is a box HALF-side;
    a missing quat is identity and a missing rgba is MuJoCo's 0.5 0.5 0.5 1
    default). Only box geoms are returned; other primitives/meshes are skipped."""
    body = None
    for b in load_mjcf_root_with_includes(mjcf_path).iter("body"):
        if b.get("name") == body_name:
            body = b
            break
    if body is None:
        return []
    geoms = []
    for g in body.findall("geom"):   # direct children only (the body's own shell)
        if g.get("type") != "box":
            continue
        size = np.fromstring(g.get("size", ""), sep=" ", dtype=float)
        if size.shape[0] != 3:
            continue
        pos = np.fromstring(g.get("pos", "0 0 0"), sep=" ", dtype=float)
        if pos.shape[0] != 3:
            pos = np.zeros(3)
        quat = np.fromstring(g.get("quat", "1 0 0 0"), sep=" ", dtype=float)
        if quat.shape[0] != 4:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        rgba = np.fromstring(g.get("rgba", "0.5 0.5 0.5 1"), sep=" ", dtype=float)
        if rgba.shape[0] != 4:
            rgba = np.array([0.5, 0.5, 0.5, 1.0])
        geoms.append((size, pos, quat, rgba))
    return geoms


def _canonical_body_name(name: str) -> str:
    """Map a body/frame name onto a simulator-neutral name so MJCF <exclude>
    tags match Pinocchio frames. MuJoCo calls the root body "world"; Pinocchio's
    MJCF parser calls the same frame "universe". Geoms declared loose on the
    <worldbody> (e.g. this LEAP scene's palm plates) hang off that frame in both,
    so an <exclude body1="world" .../> must match Pinocchio's "universe". Those
    loose geoms are now carried by the _WORLD_GEOMS_BODY wrapper body (see
    split_into_single_root_mjcfs), so its name canonicalizes to "world" too."""
    return "world" if name in ("universe", _WORLD_GEOMS_BODY) else name


def build_collision_pairs(pin, model, geom_model, use_mesh_geoms, excluded_body_pairs=None):
    """Wipe the parser's default pairs and add every non-adjacent, non-floor,
    non-excluded geom pair (cube<->hand and hand<->hand). `excluded_body_pairs`
    (from parse_mjcf_excludes) additionally drops any pair whose owning bodies
    match one of the model's own <contact><exclude> tags (compared under
    _canonical_body_name, so "world" tags match Pinocchio's "universe" frame).
    Returns the list of collidable geom ids."""
    excluded_body_pairs = excluded_body_pairs or set()
    geom_model.removeAllCollisionPairs()
    ids = []
    for gid, go in enumerate(geom_model.geometryObjects):
        half = _box_half_extents(go)
        is_floor = half is not None and (
            half[0] > _FLOOR_HALFEXTENT_THRESH and half[1] > _FLOOR_HALFEXTENT_THRESH
        )
        if is_floor:
            continue
        if half is None and not use_mesh_geoms:  # mesh geom
            continue
        ids.append(gid)

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ga, gb = ids[a], ids[b]
            ja = geom_model.geometryObjects[ga].parentJoint
            jb = geom_model.geometryObjects[gb].parentJoint
            if _is_adjacent(model, ja, jb):
                continue
            if excluded_body_pairs:
                fa = geom_model.geometryObjects[ga].parentFrame
                fb = geom_model.geometryObjects[gb].parentFrame
                body_pair = frozenset((
                    _canonical_body_name(model.frames[fa].name),
                    _canonical_body_name(model.frames[fb].name),
                ))
                if body_pair in excluded_body_pairs:
                    continue
            geom_model.addCollisionPair(pin.CollisionPair(ga, gb))
    return ids


def _rotation_from_normal(n):
    """3x3 rotation whose third column is the unit normal n (contact-frame z-axis
    = contact normal; x,y span the friction-tangent plane)."""
    n = np.asarray(n, dtype=float)
    nn = np.linalg.norm(n)
    if nn < 1e-9 or not np.all(np.isfinite(n)):
        return np.eye(3)
    z = n / nn
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = a - z * (a @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# Panda3D's ShowBase is a per-process singleton — constructing a second one
# (e.g. one PinocchioSimulator per episode during a sweep) raises "Attempt to
# spawn multiple ShowBase instances!". We build the offscreen viewer once and
# reattach every simulator's visualizer to it (see _setup_viewer).
_PANDA_VIEWER = None


class PinocchioSimulator(EvalSimulator):
    def __init__(
        self,
        model_path: str,
        config,
        nq: int,
        nv: int,
        pid: PinocchioPdActuation,
        joint_channels: list[PinocchioJointChannel] | None = None,
        free_channels: list[PinocchioFreeBodyChannel] | None = None,
        contact_cfg: PinocchioContactConfig | None = None,
        joint_cfg: PinocchioJointConstraintConfig | None = None,
        video_path: str | None = None,
        render: bool = True,
        goal_pose: "tuple[np.ndarray, np.ndarray] | None" = None,
        goal_opacity: float = 0.3,
        goal_body_name: str = "obj",
        goal_marker_body: str = "goal",
        msaa_samples: int = 8,
        render_size: "tuple[int, int] | None" = (400,300),#(1280, 960),
        use_mp4: bool = True,
    ):
        import pinocchio as pin

        self._pin = pin
        self._config = config
        self._model_path = model_path
        # Optional translucent goal marker: (world_pos[3], world_quat_wxyz[4]) of
        # the target pose, rendered as a clone of `goal_body_name`'s visual geoms
        # (see _add_goal_marker). None => no marker. Visual model only.
        self._goal_pose = goal_pose
        self._goal_opacity = float(goal_opacity)
        self._goal_body_name = goal_body_name
        # Name of the scene's static goal-marker <body> (MuJoCo mocap body). Its
        # visual geoms are re-placed by set_goal_quat whenever a new goal is
        # sampled — Pinocchio has no mocap concept, so the marker is a fixed
        # body whose geometry placements we rewrite directly (see
        # _index_goal_marker). None/absent body => set_goal_quat is a no-op.
        self._goal_marker_body = goal_marker_body
        # Offscreen-render anti-aliasing. Both only apply to the FIRST viewer
        # built in the process (the Panda3D ShowBase is a singleton — see
        # _PANDA_VIEWER — so later episodes reuse its framebuffer as-is).
        # msaa_samples: MSAA sample count (0 disables); render_size: offscreen
        # framebuffer/video resolution, None keeps panda3d's 800x600 default.
        self._msaa_samples = int(msaa_samples)
        self._render_size = render_size
        self.nq = nq
        self.nv = nv
        self._joint_channels = joint_channels or []
        self._free_channels = free_channels or []
        self._pid = pid
        self._contact_cfg = contact_cfg or PinocchioContactConfig()
        self._joint_cfg = joint_cfg or PinocchioJointConstraintConfig()
        self._timestep = float(config.timestep)
        self._video_path = video_path
        self._want_render = render or (video_path is not None)
        # Output container for save_video: .mp4 (default) or .gif. The flag wins
        # over whatever extension the caller passes.
        self._use_mp4 = bool(use_mp4)

        # --- build the combined Pinocchio model from the (multi-root) MJCF ----
        scene_dir = os.path.dirname(model_path)
        tmp_paths, root_poses = split_into_single_root_mjcfs(model_path, scene_dir)
        try:
            parts = [pin.buildModelsFromMJCF(p, contacts=False) for p in tmp_paths]
        finally:
            for p in tmp_paths:
                os.remove(p)
        model, coll, vis = merge_models(pin, parts, root_poses)
        model.gravity = pin.Motion(np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0]))

        self._model = model
        self._collision_model = coll
        self._visual_model = vis
        self._data = model.createData()
        # Swap triangle-mesh (BVH) fingertips for their convex hulls so contacts
        # use coal's analytic GJK/EPA path (stable normal + accurate depth); must
        # happen before GeometryData is built off the collision model.
        if self._contact_cfg.use_convex_tips:
            convexify_mesh_geoms(coll)
        excludes = parse_mjcf_excludes(model_path)
        build_collision_pairs(pin, model, coll, self._contact_cfg.use_mesh_geoms, excludes)
        self._geom_data = pin.GeometryData(coll)
        # Ask coal to populate the contact manifold (normal + witness points) per
        # pair; without this the collision results carry NaN normals.
        for req in self._geom_data.collisionRequests:
            req.enable_contact = True
            req.num_max_contacts = 16

        # Resolve channel joint names -> ids.
        self._joint_jid = {ch.pin_name: model.getJointId(ch.pin_name)
                           for ch in self._joint_channels}
        self._free_jid = {ch.pin_name: model.getJointId(ch.pin_name)
                          for ch in self._free_channels}

        # PD: pin q/v indices of the controlled joints, in ctrl order.
        self._ctrl_qadr = np.array(
            [model.joints[model.getJointId(n)].idx_q for n in pid.ctrl_joint_names],
            dtype=int,
        )
        self._ctrl_vadr = np.array(
            [model.joints[model.getJointId(n)].idx_v for n in pid.ctrl_joint_names],
            dtype=int,
        )

        # Per-controlled-joint actuator torque limits (MuJoCo jnt_actfrcrange),
        # in ctrl order. None => unlimited (_substep skips the clamp).
        if pid.force_limit is not None:
            fl = np.asarray(pid.force_limit, dtype=float).reshape(-1, 2)
            if fl.shape[0] != self._ctrl_vadr.shape[0]:
                raise ValueError(
                    f"force_limit has {fl.shape[0]} rows but there are "
                    f"{self._ctrl_vadr.shape[0]} controlled joints."
                )
            self._ctrl_frc_lo = fl[:, 0].copy()
            self._ctrl_frc_hi = fl[:, 1].copy()
        else:
            self._ctrl_frc_lo = None
            self._ctrl_frc_hi = None

        # Rotor inertia on the controlled joints: crba (=> the Delassus the ADMM
        # solve builds) and the mass-scaled kd both see A = M + armature, so the
        # contact solve doesn't treat the PD-held fingers as near-massless.
        if pid.armature:
            model.armature[self._ctrl_vadr] += pid.armature

        # --- joint-level constraints (position limits + dry friction) ----------
        # Both are inert in aba (see PinocchioJointConstraintConfig) and only bite
        # as constraint models appended to the same ADMM solve as the contacts.
        jcfg = self._joint_cfg
        chan_jids = [model.getJointId(ch.pin_name) for ch in self._joint_channels]

        # Position limits: one JointLimitConstraintModel over all channel joints.
        # It is unilateral and re-selected per substep against the live q (see
        # _joint_constraints), so it costs nothing while the joints are interior.
        self._limit_raw = None
        if jcfg.enforce_limits and chan_jids:
            jv = pin.StdVec_Index()
            for j in chan_jids:
                jv.append(j)
            self._limit_raw = pin.JointLimitConstraintModel(model, jv)
            if jcfg.limit_margin:
                # setPositionLimitAndMargin takes model.nq-sized vectors.
                margin = np.full(model.nq, float(jcfg.limit_margin))
                self._limit_raw.setPositionLimitAndMargin(
                    np.asarray(model.lowerPositionLimit, dtype=float),
                    np.asarray(model.upperPositionLimit, dtype=float),
                    margin,
                )

        # Dry friction. Two API details: the bounds are IMPULSES (the solver works
        # in impulse space, so scale the frictionloss torque by dt — passing the
        # raw torque locks the joint permanently), and lb/ub must be full model.nv
        # vectors indexed by each joint's idx_v, not one entry per selected joint.
        # The model is built per (excluded-joint) set and cached, because a joint
        # sitting at an active limit must be left out — see _friction_model.
        self._friction_spec = []   # (name, jid, idx_q, idx_v, impulse_bound)
        if jcfg.frictionloss:
            for ch in self._joint_channels:
                val = float(jcfg.frictionloss.get(ch.pin_name, 0.0))
                if val > 0.0:
                    jid = model.getJointId(ch.pin_name)
                    self._friction_spec.append((
                        ch.pin_name, jid,
                        model.joints[jid].idx_q, model.joints[jid].idx_v,
                        val * self._timestep,
                    ))
        self._friction_cache = {}

        # ADMM solver setup (mirrors replay_pinocchio_controls.py's make_solver).
        self._solver = pin.ADMMConstraintSolver()
        s = pin.ADMMSolverSettings()
        s.max_iterations = self._contact_cfg.admm_max_iterations
        s.absolute_feasibility_tol = 1e-10
        s.relative_feasibility_tol = 1e-12
        s.absolute_complementarity_tol = 1e-10
        s.relative_complementarity_tol = 1e-12
        s.admm_update_rule = pin.ADMMUpdateRule.SPECTRAL
        s.anderson_capacity = 10
        s.admm_proximal_rule = pin.ADMMProximalRule.AUTOMATIC
        s.stat_record = False
        s.solve_ncp = True
        self._settings = s
        self._result = pin.ADMMSolverResult()
        self._fext = [pin.Force.Zero() for _ in range(model.njoints)]

        # Running state (Pinocchio layout) + PD desired positions.
        self._q = pin.neutral(model)
        self._v = np.zeros(model.nv)
        self._q_des = self._q.copy()

        # Rendering: frames are captured from inside step() on the SIM clock (one
        # per 1/cam_fps of simulated time, at the substep nearest each deadline),
        # mirroring MujocoSimulator — see FrameClock.
        self._frames: list[np.ndarray] = []
        self._clock = FrameClock(config.cam_fps)
        self._viz = None
        self._goal_marker_M0 = None
        self._goal_marker_geoms = []
        if self._want_render:
            self._index_goal_marker()
            self._setup_viewer()

    # -- runtime goal marker -------------------------------------------------
    def _index_goal_marker(self):
        """Record the visual geoms belonging to the goal-marker body, together
        with their placements EXPRESSED IN THAT BODY'S FRAME, so set_goal_quat
        can spin the marker in place later.

        The marker is a jointless <body> (a MuJoCo mocap body), so Pinocchio
        anchors its geoms to the universe joint with their world placement baked
        in — there is no joint whose q we could write to move it. What we can do
        is rewrite each GeometryObject.placement directly: display() runs
        updateGeometryPlacements, which recomputes oMg = oMi[parentJoint] *
        placement every frame, so a new placement takes effect immediately.

        The body's own world pose is read off the model frame of the same name
        (merge_models reapplies the root placement there), and each geom's
        body-local offset is M_body^-1 * placement."""
        self._goal_marker_M0 = None
        self._goal_marker_geoms = []
        name = self._goal_marker_body
        if not name:
            return
        model = self._model
        if not model.existFrame(name):
            return
        M_body = model.frames[model.getFrameId(name)].placement
        M_body_inv = M_body.inverse()
        for gid, go in enumerate(self._visual_model.geometryObjects):
            if model.frames[go.parentFrame].name == name:
                self._goal_marker_geoms.append((gid, M_body_inv * go.placement))
        if self._goal_marker_geoms:
            self._goal_marker_M0 = M_body

    def set_goal_quat(self, quat_wxyz) -> None:
        """Re-orient the scene's goal-marker body about its own origin (its
        position is left at the scene's placement). Mirrors writing
        mjd.mocap_quat for the same body in MuJoCo. No-op when the scene has no
        such body or rendering is off."""
        if self._goal_marker_M0 is None:
            return
        pin = self._pin
        w, x, y, z = np.asarray(quat_wxyz, dtype=float)
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n == 0.0:
            return
        R = pin.Quaternion(w / n, x / n, y / n, z / n).matrix()
        M_body = pin.SE3(R, self._goal_marker_M0.translation)
        # The visualizer holds a reference to this same GeometryModel (verified:
        # viz.visual_model is self._visual_model), so writing here is enough —
        # the next display() re-derives oMg from these placements.
        for gid, local_M in self._goal_marker_geoms:
            self._visual_model.geometryObjects[gid].placement = M_body * local_M

    # -- viewer --------------------------------------------------------------
    def _add_goal_marker(self):
        """Clone the goal body's box visual geoms into the visual model as a
        static, translucent marker at the goal pose, so the render shows the
        target cube in the SAME colors/shape as the manipulated one. Each cloned
        geom is anchored to the universe joint (joint 0) at goal_pose * its body-
        frame placement, and its alpha is scaled by goal_opacity; the geoms live
        only in the visual model, so they never enter collision detection or the
        contact solve. No-op if no goal pose was given or coal is unavailable."""
        if self._goal_pose is None:
            return
        coal = _coal_module()
        if coal is None:
            return
        pin = self._pin
        pos, quat = self._goal_pose
        w, x, y, z = np.asarray(quat, dtype=float)
        goal_M = pin.SE3(pin.Quaternion(w, x, y, z).matrix(),
                         np.asarray(pos, dtype=float))
        alpha = self._goal_opacity
        geoms = parse_body_box_geoms(self._model_path, self._goal_body_name)
        for i, (half, gpos, gquat, rgba) in enumerate(geoms):
            gw, gx, gy, gz = gquat
            local_M = pin.SE3(pin.Quaternion(gw, gx, gy, gz).matrix(),
                              np.asarray(gpos, dtype=float))
            box = coal.Box(2.0 * half[0], 2.0 * half[1], 2.0 * half[2])
            go = pin.GeometryObject(f"goal_marker_{i}", 0, 0, goal_M * local_M, box)
            go.overrideMaterial = True
            go.meshColor = np.array([rgba[0], rgba[1], rgba[2], rgba[3] * alpha])
            self._visual_model.addGeometryObject(go)

    def _setup_viewer(self):
        global _PANDA_VIEWER
        # Panda3D's headless EGL pipe must be selected before the panda3d import.
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "load-display p3headlessgl")
        # Multisample anti-aliasing. panda3d_viewer's ViewerApp already requests
        # AntialiasAttrib.MAuto on the scene root, but that only takes effect if
        # the framebuffer actually has samples — which it does not by default, so
        # the offscreen render comes out with hard, stair-stepped edges. These
        # two prc settings are what ViewerConfig.enable_antialiasing writes; they
        # must be loaded before the ShowBase/framebuffer is created. Only applied
        # on the FIRST viewer (the ShowBase is process-wide, see _PANDA_VIEWER).
        if _PANDA_VIEWER is None and self._msaa_samples > 0:
            loadPrcFileData("", "framebuffer-multisample 1")
            loadPrcFileData("", f"multisamples {self._msaa_samples}")
        if _PANDA_VIEWER is None:
            # MSAA only smooths geometry silhouettes. The cube-face textures
            # alias on their own when minified (a 180px sticker drawn ~40px
            # wide), which shows up as crawling//sparkling letter edges. Trilinear
            # mipmapping plus anisotropic filtering is the fix for that half.
            loadPrcFileData("", "texture-minfilter linear_mipmap_linear")
            loadPrcFileData("", "texture-magfilter linear")
            loadPrcFileData("", "texture-anisotropic-degree 16")
            if self._render_size is not None:
                w, h = self._render_size
                loadPrcFileData("", f"win-size {int(w)} {int(h)}")
        from pinocchio.visualize import Panda3dVisualizer

        # Add the goal-cube overlay (if requested) BEFORE the visualizer loads the
        # visual model, so its geoms get uploaded with the rest of the scene.
        self._add_goal_marker()

        # Make sure every visual .obj has the .mtl it names on disk, or assimp
        # aborts the load (the UR5e meshes reference nonexistent Black.mtl etc.).
        ensure_obj_material_stubs(self._visual_model)

        # Apply MJCF rgba (stored in meshColor) instead of the default material.
        for go in self._visual_model.geometryObjects:
            go.overrideMaterial = True

        viz = Panda3dVisualizer(self._model, self._collision_model, self._visual_model)
        # Reuse the one process-wide ShowBase (created on the first episode);
        # later episodes reattach to it and swap in this episode's geometry via
        # the fixed "pin_eval" group name.
        if _PANDA_VIEWER is None:
            viz.initViewer(open=False)
            _PANDA_VIEWER = viz.viewer
        else:
            viz.initViewer(viewer=_PANDA_VIEWER, open=False)
        viz.loadViewerModel(group_name="pin_eval")
        viz.displayVisuals(True)

        # panda3d_viewer draws its own world-origin axes (red/green/blue rays)
        # and ground grid by default (ViewerApp reads show-axes/show-grid, both
        # defaulting True). They are debug scaffolding, not part of the scene —
        # the MJCF's own <geom name="floor"> is the real ground — so drop them
        # from the recorded video.
        viz.viewer.show_axes(False)
        viz.viewer.show_grid(False)

        # Camera from the shared TaskConfig pose (same framing as MuJoCo/Drake).
        R, eye = camera_pose_from_config(self._config)
        forward = np.asarray(R)[:, 2]
        dist = float(np.linalg.norm(eye)) or 1.0
        lookat = np.asarray(eye) + forward * dist
        viz.viewer._app.camLens.set_near(0.01)
        viz.viewer.reset_camera(pos=tuple(np.asarray(eye, dtype=float)),
                                look_at=tuple(lookat.astype(float)))
        self._viz = viz

    # -- per-channel read/write ---------------------------------------------
    def _write_joint(self, ch, qpos, qvel):
        j = self._model.joints[self._joint_jid[ch.pin_name]]
        self._q[j.idx_q] = qpos[ch.q_adr]
        self._v[j.idx_v] = qvel[ch.v_adr]

    def _read_joint(self, ch, qpos, qvel):
        j = self._model.joints[self._joint_jid[ch.pin_name]]
        qpos[ch.q_adr] = self._q[j.idx_q]
        qvel[ch.v_adr] = self._v[j.idx_v]

    def _write_free(self, ch, qpos, qvel):
        pin = self._pin
        jid = self._free_jid[ch.pin_name]
        j = self._model.joints[jid]
        a, va = ch.q_adr, ch.v_adr
        # Desired world body pose (MuJoCo quat is wxyz).
        quat = np.asarray(qpos[a + 3:a + 7], dtype=float)
        n = np.linalg.norm(quat)
        quat = quat / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])
        w, x, y, z = quat
        R = pin.Quaternion(w, x, y, z).matrix()
        M_des = pin.SE3(R, np.asarray(qpos[a:a + 3], dtype=float))
        # q is jointPlacement^-1 * world pose (parser bakes offset into placement).
        M_q = self._model.jointPlacements[jid].inverse() * M_des
        qq = pin.Quaternion(M_q.rotation)
        self._q[j.idx_q:j.idx_q + 3] = M_q.translation
        self._q[j.idx_q + 3:j.idx_q + 7] = np.array([qq.x, qq.y, qq.z, qq.w])
        # Velocity: MuJoCo [lin(world), ang(body-local)] -> joint-frame [lin, ang].
        self._v[j.idx_v:j.idx_v + 3] = R.T @ np.asarray(qvel[va:va + 3], dtype=float)
        self._v[j.idx_v + 3:j.idx_v + 6] = np.asarray(qvel[va + 3:va + 6], dtype=float)

    def _read_free(self, ch, qpos, qvel):
        pin = self._pin
        jid = self._free_jid[ch.pin_name]
        j = self._model.joints[jid]
        a, va = ch.q_adr, ch.v_adr
        X = self._data.oMi[jid]               # body world pose (free joint == body)
        R = X.rotation
        qpos[a:a + 3] = X.translation
        quat = pin.Quaternion(R)
        qpos[a + 3] = quat.w
        qpos[a + 4] = quat.x
        qpos[a + 5] = quat.y
        qpos[a + 6] = quat.z
        # joint-frame velocity -> MuJoCo [lin(world), ang(body-local)].
        v_lin = self._v[j.idx_v:j.idx_v + 3]
        qvel[va:va + 3] = R @ v_lin
        qvel[va + 3:va + 6] = self._v[j.idx_v + 3:j.idx_v + 6]

    # -- contact detection + dynamics ---------------------------------------
    def _detect_contacts(self):
        """Detect coal contacts at the current q and turn each into a frictional
        PointContactConstraintModel with a native Baumgarte corrector. The contact
        frame's z-axis is the contact normal, so lateral (pushing) forces resolve
        correctly. Returns (constraint_models, constraint_datas)."""
        pin = self._pin
        model, data = self._model, self._data
        gm, gd = self._collision_model, self._geom_data
        cfg = self._contact_cfg
        q = self._q
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, gm, gd, q)
        pin.computeCollisions(model, data, gm, gd, q, False)

        baumgarte = pin.BaumgarteCorrectorParameters(cfg.baumgarte_kp, cfg.baumgarte_kd)
        cms = []
        penetrations = []
        for k in range(len(gm.collisionPairs)):
            cr = gd.collisionResults[k]
            if not cr.isCollision():
                continue
            cp = gm.collisionPairs[k]
            j1 = gm.geometryObjects[cp.first].parentJoint
            j2 = gm.geometryObjects[cp.second].parentJoint
            c1 = np.asarray(gd.oMg[cp.first].translation, dtype=float)
            c2 = np.asarray(gd.oMg[cp.second].translation, dtype=float)

            normal = None
            try:
                nrm = np.asarray(cr.getContact(0).normal, dtype=float)
                if np.all(np.isfinite(nrm)) and np.linalg.norm(nrm) > 1e-9:
                    normal = nrm
            except Exception:
                pass
            if normal is None:
                normal = c2 - c1
            R_n = _rotation_from_normal(normal)

            # World contact points: finite coal witness points, else geom midpoint.
            world_points = []
            try:
                n_contacts = cr.numContacts()
            except Exception:
                n_contacts = 0
            for i in range(n_contacts):
                try:
                    contact_i = cr.getContact(i)
                    p = np.asarray(contact_i.pos, dtype=float)
                    if np.all(np.isfinite(p)):  # box-box manifolds can return NaN
                        world_points.append(p)
                        pen = float(contact_i.penetration_depth)
                        if np.isfinite(pen):
                            penetrations.append(pen)
                except Exception:
                    pass
            if not world_points:
                world_points.append(0.5 * (c1 + c2))

            for p_world in world_points:
                M_world = pin.SE3(R_n, p_world)
                plc1 = data.oMi[j1].inverse() * M_world
                plc2 = data.oMi[j2].inverse() * M_world
                cm = pin.PointContactConstraintModel(model, j1, plc1, j2, plc2)
                cm.setFriction(cfg.friction)
                cm.setBaumgarteCorrectorParameters(baumgarte)
                cms.append(pin.ConstraintModel(cm))
        cds = [cm.createData() for cm in cms]
        if PRINT_CONTACT_DEBUG:
            if penetrations:
                print(f"[pinocchio_sim] contacts={len(penetrations)} "
                      f"penetration(m): max={max(penetrations):.6f} "
                      f"mean={sum(penetrations) / len(penetrations):.6f} "
                      f"all={['%.6f' % p for p in penetrations]}")
            else:
                print("[pinocchio_sim] contacts=0 penetration(m): n/a")
        return cms, cds

    def _joints_at_limit(self, q):
        """Names of friction joints currently sitting at (or past) a position
        limit, using the same rule JointLimitConstraintModel's proximity filter
        applies. Those joints must be dropped from the friction constraint — see
        _friction_model."""
        model = self._model
        margin = float(self._joint_cfg.limit_margin)
        at = set()
        for name, _jid, idx_q, _idx_v, _b in self._friction_spec:
            lo = float(model.lowerPositionLimit[idx_q])
            hi = float(model.upperPositionLimit[idx_q])
            if q[idx_q] >= hi - margin or q[idx_q] <= lo + margin:
                at.add(name)
        return at

    def _friction_model(self, exclude):
        """Wrapped JointFrictionConstraintModel over every friction joint except
        `exclude`, cached per exclusion set (normally empty, so this is a dict hit).

        Joints at an active limit MUST be excluded: the limit constraint's row for
        that DOF is +/-e_i and the friction row is e_i — exactly linearly dependent,
        which makes the Delassus J*M^-1*J^T singular and the solve return NaN. No
        amount of Cholesky regularization fixes it (verified up to mu=1e-4), and
        dropping friction there is physically right anyway: a joint pinned against
        its stop is already held, and dry friction changes nothing."""
        key = frozenset(exclude)
        if key in self._friction_cache:
            return self._friction_cache[key]
        pin, model = self._pin, self._model
        jids = []
        lb = np.zeros(model.nv)
        ub = np.zeros(model.nv)
        for name, jid, _idx_q, idx_v, bound in self._friction_spec:
            if name in key:
                continue
            jids.append(jid)
            ub[idx_v] = bound
            lb[idx_v] = -bound
        cm = None
        if jids:
            jv = pin.StdVec_Index()
            for j in jids:
                jv.append(j)
            cm = pin.ConstraintModel(pin.JointFrictionConstraintModel(model, jv, lb, ub))
        self._friction_cache[key] = cm
        return cm

    def _joint_constraints(self, q):
        """Constraint models for joint position limits and dry friction at the
        current q, to be appended AFTER the contacts (the Baumgarte pass in
        _substep only spans the leading contact rows).

        The limit constraint is re-selected and re-wrapped every substep because
        (a) its active set depends on q — only joints near/past a bound engage,
        and makeSelectionFilteredByLimitProximity is what feeds q in — and (b)
        pin.ConstraintModel(...) *copies* the model, so a wrapper built earlier
        would freeze a stale active set. When no joint is near a limit the
        residual size is 0 and nothing is added."""
        pin = self._pin
        cms, cds = [], []
        at_limit = set()
        if self._limit_raw is not None:
            self._limit_raw.makeSelectionFilteredByLimitProximity(q)
            if self._limit_raw.residualSize() > 0:
                cm = pin.ConstraintModel(self._limit_raw)
                cms.append(cm)
                cds.append(cm.createData())
                at_limit = self._joints_at_limit(q)
        if self._friction_spec:
            fcm = self._friction_model(at_limit)
            if fcm is not None:
                cms.append(fcm)
                cds.append(fcm.createData())
        return cms, cds

    def _substep(self):
        pin = self._pin
        model, data = self._model, self._data
        q, v = self._q, self._v
        dt = self._timestep
        cms, cds = self._detect_contacts()
        # Contacts occupy the leading rows; the Baumgarte pass below relies on it.
        n_contact = len(cms)
        jcms, jcds = self._joint_constraints(q)
        cms = cms + jcms
        cds = cds + jcds

        # Explicit forward-Euler PD on the hand: fixed stiffness kp. Damping is
        # either a fixed kd applied directly (use_direct_kd=True) or derived
        # per-joint from the mass-matrix diagonal so the loop stays critically
        # damped at that stiffness: kd = 2*zeta*sqrt(kp*M_ii) (default).
        pin.crba(model, data, q, pin.Convention.WORLD)
        m_diag = np.diag(data.M)[self._ctrl_vadr]
        kp = self._pid.kp
        if self._pid.use_direct_kd:
            kd = np.full_like(m_diag, self._pid.kd)
        else:
            kd = 2.0 * self._pid.zeta * np.sqrt(kp * m_diag)
        # Actuator stiffness (+ optional gravity compensation) — the part MuJoCo
        # produces as qfrc_actuator and saturates to jnt_actfrcrange.
        act = kp * (self._q_des[self._ctrl_qadr] - q[self._ctrl_qadr])
        if self._pid.gravity_comp:
            act = act + pin.computeGeneralizedGravity(model, data, q)[self._ctrl_vadr]
        if self._ctrl_frc_lo is not None:
            act = np.clip(act, self._ctrl_frc_lo, self._ctrl_frc_hi)

        # Velocity damping is applied OUTSIDE the saturation. In MuJoCo the
        # dominant hand damping is passive joint damping (qfrc_passive), which is
        # not part of the clamped actuator force — so when the actuator saturates,
        # MuJoCo still brakes with -damping*v. Clamping the damping together with
        # the stiffness (the naive approach) drops that braking during saturation
        # and lets the Pinocchio joint run ahead of MuJoCo. (A tiny 0.01*v of this
        # kd is really the position actuator's kv, which MuJoCo does clamp; leaving
        # it outside is a <=0.01*v approximation, negligible next to the 0.1
        # passive term.)
        tau = np.zeros(model.nv)
        tau[self._ctrl_vadr] = act - kd * v[self._ctrl_vadr]

        v_free = v + dt * pin.aba(model, data, q, v, tau, self._fext)

        if len(cms) == 0:
            self._q = pin.integrate(model, q, v_free * dt)
            self._v = v_free
            return

        for cm, cd in zip(cms, cds):
            cm.calc(model, data, cd)
        chol = pin.ConstraintCholeskyDecomposition(model, data, cms, cds)
        chol.compute(model, data, cms, cds, 1e-10)
        delassus = chol.getDelassusOperatorCholeskyExpression()
        # Force 2-D: with a single constraint row (e.g. one joint-limit bound
        # active and nothing else) getConstraintsJacobian returns a flat (nv,)
        # array, so Jc @ v_free would collapse to a scalar and the solver would
        # reject g. Contacts alone never hit this (a point contact is 3 rows).
        Jc = np.asarray(pin.getConstraintsJacobian(model, data, cms, cds)).reshape(-1, model.nv)
        g = Jc @ v_free

        # Baumgarte stabilization: bias each contact's drift by its position error
        # scaled by the Kp set on the constraint (g1-style), g += Kp * perr / dt.
        # Restricted to the leading contact rows: the joint-limit/friction models
        # appended after them do not carry this state and raise on access
        # (JointFrictionConstraintModel has no baumgarte_corrector_parameters, and
        # JointLimitConstraintData has no constraint_position_error). They need no
        # Baumgarte anyway — the limit is unilateral and resolves its own
        # overshoot, and friction acts purely at the velocity level.
        idx = 0
        for cm, cd in zip(cms[:n_contact], cds[:n_contact]):
            size = cm.residualSize()
            kp_b = cm.baumgarte_corrector_parameters.Kp
            if kp_b != 0.0:
                g[idx:idx + size] += kp_b * cd.extract().constraint_position_error / dt
            idx += size

        converged = self._solver.solve(delassus, g, cms, cds, self._settings, self._result)
        if PRINT_CONTACT_DEBUG:
            print(f"[pinocchio_sim] ADMM converged={converged} "
                  f"iterations={self._result.iterations}")
        forces = (1.0 / dt) * np.asarray(self._result.retrieveConstraintImpulses()).ravel()
        v_new = v + dt * pin.aba(model, data, q, v, tau + Jc.T @ forces, self._fext)
        self._q = pin.integrate(model, q, v_new * dt)
        self._v = v_new

    # -- EvalSimulator interface --------------------------------------------
    def reset(self, qpos, qvel) -> None:
        self._clock.reset()
        self._frames = []
        self.set_state(qpos, qvel)

    def set_state(self, qpos, qvel) -> None:
        qpos = np.asarray(qpos, dtype=float)
        qvel = np.asarray(qvel, dtype=float)
        for ch in self._joint_channels:
            self._write_joint(ch, qpos, qvel)
        for ch in self._free_channels:
            self._write_free(ch, qpos, qvel)
        # Keep PD desired aligned to the new pose (no command yet -> hold).
        self._q_des = self._q.copy()
        self._pin.forwardKinematics(self._model, self._data, self._q)

    def get_state(self) -> EvalState:
        self._pin.forwardKinematics(self._model, self._data, self._q)
        qpos = np.zeros(self.nq)
        qvel = np.zeros(self.nv)
        for ch in self._joint_channels:
            self._read_joint(ch, qpos, qvel)
        for ch in self._free_channels:
            self._read_free(ch, qpos, qvel)
        return EvalState(qpos, qvel)

    def apply_control(self, ctrl) -> None:
        ctrl = np.asarray(ctrl, dtype=float)
        self._q_des[self._ctrl_qadr] = ctrl

    def step(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            self._substep()
            if self._clock.advance(self._timestep) and self._viz is not None:
                self._capture()

    def _capture(self) -> None:
        self._viz.display(self._q)
        self._frames.append(self._viz.captureImage())

    def save_video(self, path: str | None = None) -> str | None:
        if not self._frames:
            return None
        import mediapy as media
        out = path or self._video_path
        if out is None:
            return None
        out = resolve_video_path(out, self._use_mp4)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        kwargs = {"codec": "gif"} if out.lower().endswith(".gif") else {}
        # The frames are cam_fps apart in sim time, so writing at cam_fps replays
        # the episode in real time.
        media.write_video(out, self._frames, fps=float(self._config.cam_fps), **kwargs)
        return out

    @property
    def timestep(self) -> float:
        return self._timestep
