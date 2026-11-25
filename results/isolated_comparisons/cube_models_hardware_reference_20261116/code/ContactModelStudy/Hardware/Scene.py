"""Export an editable hand/cube MJCF while checking it against calibration."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = ROOT / "scenes/leap/hardware/cube.xml"
DEFAULT_ARCHIVE = Path.home() / "Downloads/20261110T033819Z_e01f2852_model_and_params.zip"
DEFAULT_HAND_SOURCE = Path.home() / "leap_agentic_sysid/leap/leap_right_hand.xml"
JOINTS = tuple(f"{finger}_{joint}" for finger in ("if", "mf", "rf")
               for joint in ("mcp", "rot", "pip", "dip")) + (
                   "th_cmc", "th_axl", "th_mcp", "th_ipl")
CALIBRATION_FIELDS = (
    "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
    "jnt_pos", "jnt_axis", "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss",
    "geom_pos", "geom_quat", "geom_size", "geom_contype", "geom_conaffinity", "geom_friction",
    "geom_solref", "geom_solimp", "actuator_gainprm", "actuator_biasprm", "actuator_dynprm",
    "actuator_gear", "actuator_dyntype", "actuator_ctrlrange", "actuator_forcerange",
    "actuator_ctrllimited", "actuator_forcelimited", "site_pos", "site_quat", "mesh_vert",
)
if hasattr(mujoco.MjModel, "dof_dampingpoly"):
    CALIBRATION_FIELDS += ("dof_dampingpoly",)


def joint_parameter(model, dof, field, *, for_xml=False):
    """Read joint dynamics, retaining MuJoCo 3.14's damping polynomial."""
    value = getattr(model, "dof_" + field)[dof]
    if field == "damping" and hasattr(model, "dof_dampingpoly"):
        # Newton 1.6 reads linear damping as a scalar. Preserve that XML form
        # when the calibrated polynomial has no nonlinear terms.
        if for_xml and not np.any(model.dof_dampingpoly[dof]):
            return value
        return np.r_[value, model.dof_dampingpoly[dof]]
    return value


def check_calibration(model, reference):
    if (model.nq, model.nv, model.nu, model.na) != (16, 16, 16, 16):
        raise ValueError("Expected a calibrated hand with 16 filtered position servos")
    for i, name in enumerate(JOINTS):
        if (model.joint(i).name != name or model.actuator(i).name != name + "_act"
                or model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_FILTEREXACT):
            raise ValueError(f"Unexpected calibrated layout at {name}")
    for field in CALIBRATION_FIELDS:
        if not np.allclose(getattr(model, field), getattr(reference, field), rtol=1e-10, atol=1e-12):
            raise ValueError(f"Export does not reproduce calibrated {field}")


def prepare_scene(archive_path=DEFAULT_ARCHIVE, source=DEFAULT_HAND_SOURCE, output=DEFAULT_SCENE,
                  cube_size=0.07, cube_mass=0.07):
    """Copy original mesh assets, transplant calibrated values, and add a cube."""
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    archive_path = Path(archive_path).expanduser().resolve()
    if cube_size <= 0 or cube_mass <= 0 or not np.isfinite([cube_size, cube_mass]).all():
        raise ValueError("Cube size and mass must be finite and positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as temporary:
        payload = archive.read("calibrated.mjb")
        binary = Path(temporary) / "calibrated.mjb"
        binary.write_bytes(payload)
        reference = mujoco.MjModel.from_binary_path(str(binary))
        parameters = json.loads(archive.read("optimized_parameters.json"))
    spec = mujoco.MjSpec.from_file(str(source))
    spec.option.timestep = reference.opt.timestep
    spec.option.gravity = reference.opt.gravity
    for name in ("integrator", "solver", "cone", "iterations", "tolerance", "impratio",
                 "ls_iterations", "ls_tolerance", "disableflags", "enableflags"):
        setattr(spec.option, name, getattr(reference.opt, name))
    for body in spec.bodies:
        if body.name and body.name != "world":
            body.pos = reference.body(body.name).pos
            body.quat = reference.body(body.name).quat
    for name in JOINTS:
        joint = spec.joint(name)
        index = reference.joint(name).id
        dof = reference.jnt_dofadr[index]
        for field in ("damping", "armature", "frictionloss"):
            setattr(joint, field, joint_parameter(reference, dof, field))
        joint.range = reference.jnt_range[index]
        actuator = spec.actuator(name + "_act")
        index = reference.actuator(name + "_act").id
        for field in ("gainprm", "biasprm", "dynprm", "gear", "dyntype", "gaintype", "biastype",
                      "ctrlrange", "forcerange", "actrange", "ctrllimited", "forcelimited",
                      "actlimited", "actearly"):
            value = getattr(reference, "actuator_" + field)[index]
            enum = {"dyntype": mujoco.mjtDyn, "gaintype": mujoco.mjtGain,
                    "biastype": mujoco.mjtBias}.get(field)
            if enum is not None:
                value = enum(int(value))
            elif np.isscalar(value):
                value = value.item()
            setattr(actuator, field, value)
        actuator.actdim = 1
    # Identification adds these sensors in memory, not to its source XML.
    for name in JOINTS:
        sensor_name = name + "_velocity_sensor"
        if spec.sensor(sensor_name) is None:
            spec.add_sensor(name=sensor_name, type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                            objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=name)
    check_calibration(spec.compile(), reference)
    assets = output.parent / "assets"
    assets.mkdir(exist_ok=True)
    for mesh in spec.meshes:
        asset = source.parent / spec.meshdir / mesh.file
        shutil.copy2(asset, assets / Path(mesh.file).name)
        mesh.file = Path(mesh.file).name
    license_path = source.parent / "leap_hand/LICENSE"
    if license_path.exists():
        shutil.copy2(license_path, assets / "LICENSE")
    spec.meshdir = str(assets)
    hand_path = output.parent / "calibrated_hand.xml"
    tree = ET.fromstring(spec.to_xml())
    tree.find("compiler").set("meshdir", "assets")
    def numbers(values):
        return " ".join(format(float(v), ".17g") for v in np.atleast_1d(values))
    # MjSpec's XML writer rounds to six significant digits. Keep identified
    # parameters and the mounting quaternion at full double precision.
    for body in tree.findall(".//worldbody//body"):
        for field in ("pos", "quat"):
            body.set(field, numbers(getattr(reference.body(body.get("name")), field)))
    for joint in tree.findall(".//worldbody//joint"):
        dof = reference.jnt_dofadr[reference.joint(joint.get("name")).id]
        for field in ("damping", "armature", "frictionloss"):
            joint.set(field, numbers(joint_parameter(reference, dof, field, for_xml=True)))
    for actuator in tree.findall("./actuator/*"):
        index = reference.actuator(actuator.get("name")).id
        for field in ("gainprm", "biasprm", "dynprm", "gear", "ctrlrange", "forcerange"):
            actuator.set(field, numbers(getattr(reference, "actuator_" + field)[index]))
    ET.indent(tree)
    hand_path.write_text(ET.tostring(tree, encoding="unicode") + "\n")
    check_calibration(mujoco.MjModel.from_xml_path(str(hand_path)), reference)
    # Box physics uses the reference cube dimensions. Visual mesh is separate so
    # the tracker and simulator share a centered object coordinate frame.
    half = cube_size / 2
    output.write_text(f'''<mujoco model="calibrated_leap_cube">
  <include file="calibrated_hand.xml"/>
  <statistic center="0 0 0.05" extent="0.3"/>
  <worldbody>
    <light pos="0 0 1"/>
    <camera name="demo-cam" pos="0.3 -0.3 0.3" xyaxes="0.7071 0.7071 0 -0.4082 0.4082 0.8165"/>
    <body name="obj" pos="0.02 0.035 0.085">
      <freejoint name="obj_joint"/>
      <geom name="obj" type="box" size="{half} {half} {half}" mass="{cube_mass}"
            friction="0.5 0.005 0.0001" rgba="0.9 0.6 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
''')
    combined = mujoco.MjModel.from_xml_path(str(output))
    if (combined.nq, combined.nv, combined.nu, combined.na) != (23, 22, 16, 16):
        raise ValueError("Hand/cube layout must be nq=23, nv=22, nu=16, na=16")
    (output.parent / "optimized_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (output.parent / "provenance.json").write_text(json.dumps(dict(
        archive=str(archive_path), archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(payload).hexdigest(), source_xml=str(source),
        mujoco_version=mujoco.__version__, cube_size_m=cube_size, cube_mass_kg=cube_mass,
        verified_fields=list(CALIBRATION_FIELDS)), indent=2) + "\n")
    # A centered, axis-aligned cube mesh for FoundationPose (coordinates in meters).
    vertices = [(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)]
    faces = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2),
             (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    (output.parent / "cube_tracking.obj").write_text(
        "# Centered cube; meters; same axes as obj_joint\n" +
        "".join("v %g %g %g\n" % v for v in vertices) +
        "".join("f %d %d %d %d\n" % f for f in faces))
    return output
