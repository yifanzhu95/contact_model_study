"""LEAP hardware access shared by the standalone LCM bridge."""
from pathlib import Path
import sys

import numpy as np

MOTORS = list(range(16))


def vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly 16 finite values")
    return result


def control_limits(model):
    """Absolute physical limits from the calibrated model, never delta bounds."""
    lower = np.maximum(model.jnt_range[:16, 0], model.actuator_ctrlrange[:, 0])
    upper = np.minimum(model.jnt_range[:16, 1], model.actuator_ctrlrange[:, 1])
    if np.any(lower >= upper):
        raise ValueError("Model has empty actuator/joint control intervals")
    return lower, upper


def delta_limits(config):
    """Positive downward/upward step magnitudes, in radians around measured q."""
    bounds = config.get("ctrl_range")
    if bounds is None:
        bounds = {}
    if not isinstance(bounds, dict) or set(bounds) - {"lower", "upper"}:
        raise ValueError("ctrl_range must be an object with lower and upper delta magnitudes")
    result = []
    for name in ("lower", "upper"):
        value = bounds.get(name)
        array = np.asarray(0.1 if value is None else value)
        if array.dtype.kind not in "fiu":
            raise ValueError(f"ctrl_range.{name} must be a nonnegative number or 16 nonnegative numbers")
        if array.shape == ():
            array = np.full(16, float(array))
        array = vector(array, f"ctrl_range.{name}")
        if np.any(array < 0):
            raise ValueError(f"ctrl_range.{name} is a nonnegative delta magnitude")
        result.append(array)
    return tuple(result)


def delta_reference(config):
    """Select the base for hardware MPPI increments; omission preserves legacy behavior."""
    value = config.get('delta_reference', 'measured_position')
    if value not in ('measured_position', 'previous_target'):
        raise ValueError('delta_reference must be measured_position or previous_target')
    return value


def clip_joint_target(target, reference, lower_margin, upper_margin):
    """Limit the final target around the selected increment reference."""
    target, reference = vector(target, 'joint target'), vector(reference, 'target reference')
    return np.clip(target, reference - lower_margin, reference + upper_margin)


class HardwareHand:
    """Use upstream communication and mappings without LeapNode's home command."""
    def __init__(self, api_root, config, port=None):
        source = Path(api_root).expanduser().resolve() / "python"
        if not (source / "leap_hand_utils/dynamixel_client.py").is_file():
            raise FileNotFoundError(f"LEAP hardware Python API missing: {source}")
        sys.path.insert(0, str(source))
        from leap_hand_utils import dynamixel_client, mj_leap_utils
        for module in (dynamixel_client, mj_leap_utils):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise RuntimeError("LEAP API already imported from another checkout")
        self.mapping = mj_leap_utils
        self.config = config
        self.client = dynamixel_client.DynamixelClient(
            MOTORS, port or config["port"], config["baudrate"], cur_scale=1.0)
        self.client.handle_packet_result = self._require_success

    def _require_success(self, comm_result, dxl_error=None, dxl_id=None, context=None):
        if comm_result != self.client.dxl.COMM_SUCCESS or dxl_error not in (None, 0):
            raise OSError(f"Dynamixel failure: {comm_result}, {dxl_error}, {dxl_id}, {context}")
        return True

    def register(self, motor, address, size):
        value, result, error = getattr(self.client.packet_handler, f"read{size}ByteTxRx")(
            self.client.port_handler, motor, address)
        self._require_success(result, error, motor)
        return int(value)

    def connect(self, *, torque_off=False):
        fields = {"model_number": (0, 2), "mode": (11, 1), "p": (84, 2),
                  "i": (82, 2), "d": (80, 2), "goal_current": (102, 2)}
        expected = {}
        for name, (_, size) in fields.items():
            values = np.asarray(self.config['expected_registers'][name])
            if (values.dtype.kind not in 'fiu' or values.shape not in ((), (16,))
                    or not np.isfinite(values).all() or np.any(values != np.floor(values))
                    or np.any(values < 0) or np.any(values >= 1 << (8 * size))):
                raise ValueError(f'expected_registers.{name} must contain unsigned {size}-byte integers (scalar or 16 values)')
            expected[name] = np.broadcast_to(values, (16,)).astype(int)
        self.client.connect()
        if torque_off:
            self.disable_motors(MOTORS)
        updates = []
        # Inspect the entire hand before writing anything. A wrong device,
        # operating mode or enabled torque must still fail startup.
        for motor in MOTORS:
            if self.register(motor, 64, 1) != 0:
                raise RuntimeError(f"Motor {motor}: start with torque disabled")
            for name, (address, size) in fields.items():
                actual = self.register(motor, address, size)
                target = int(expected[name][motor])
                if actual != target:
                    if name in ('model_number', 'mode'):
                        raise RuntimeError(f'Motor {motor}: {name}={actual}, expected {target}')
                    updates.append((motor, name, address, size, actual, target))
        self.health()
        # Load the configured runtime gains/current, rather than requiring a
        # previous program to have initialized them. This never writes a joint
        # target or enables torque; enable() preloads measured q later.
        for motor, name, address, size, actual, target in updates:
            result, error = getattr(self.client.packet_handler, f'write{size}ByteTxRx')(
                self.client.port_handler, motor, address, target)
            self._require_success(result, error, motor, context=f'initialize {name}')
            observed = self.register(motor, address, size)
            if observed != target:
                raise RuntimeError(f'Motor {motor}: {name} readback={observed}, expected {target} after initialization')
            print(f'Motor {motor}: initialized {name} {actual} -> {target} (verified; torque off)', flush=True)

    def connect_read_only(self):
        """Open the serial connection without changing torque or registers."""
        self.client.connect()

    def health(self, motor=None):
        """Check all motors at connection, or one motor in the live loop."""
        for motor in MOTORS if motor is None else (motor,):
            if self.register(motor, 70, 1) != 0:
                raise RuntimeError(f"Motor {motor}: hardware error")
            if self.register(motor, 146, 1) >= self.config["max_temperature"]:
                raise RuntimeError(f"Motor {motor}: temperature limit reached")

    def read(self):
        q, v, current = self.client.read_pos_vel_cur()
        reader = self.client._pos_vel_cur_reader
        for motor in MOTORS:
            if not reader.operation.isAvailable(motor, reader.address, reader.size):
                raise OSError(f"Fresh feedback missing for motor {motor}")
            if hasattr(reader.operation, "getError") and reader.operation.getError(motor):
                raise OSError(f"Motor {motor}: feedback device error")
        return (vector(self.mapping.map_leap_to_mujoco(np.asarray(q) - np.pi), "feedback q"),
                vector(self.mapping.map_leap_to_mujoco(v), "feedback v"),
                vector(current, "feedback current"))

    def write(self, command):
        pose = self.mapping.apply_offset(self.mapping.map_mujoco_to_leap(command))
        self.client.write_desired_pos(MOTORS, pose)

    @staticmethod
    def _selected_motors(motors):
        selected = np.asarray(motors)
        if (selected.ndim != 1 or not selected.size or selected.dtype.kind not in 'iu'
                or len(np.unique(selected)) != len(selected)
                or np.any(selected < 0) or np.any(selected >= len(MOTORS))):
            raise ValueError('motors must be distinct native IDs between 0 and 15')
        return selected.astype(int).tolist()

    def write_motors(self, command, motors):
        """Write only selected native motor IDs, mapping the full MuJoCo pose first."""
        motors = self._selected_motors(motors)
        pose = self.mapping.apply_offset(self.mapping.map_mujoco_to_leap(vector(command, 'finger target')))
        self.client.write_desired_pos(motors, np.asarray(pose)[motors])

    def disable_motors(self, motors):
        motors = self._selected_motors(motors)
        self.client.set_torque_enabled(motors, False, retries=0)
        for motor in motors:
            if self.register(motor, 64, 1) != 0:
                raise OSError(f'Motor {motor}: torque disable unconfirmed')

    def enable_motors(self, q, motors):
        """Preload the measured pose and enable only these motors, with readback."""
        motors = self._selected_motors(motors)
        for motor in motors:
            if self.register(motor, 64, 1) != 0:
                raise RuntimeError(f'Motor {motor}: finger must be torque-off before capture')
        self.write_motors(q, motors)
        self.client.set_torque_enabled(motors, True, retries=0)
        for motor in motors:
            if self.register(motor, 64, 1) != 1:
                raise OSError(f'Motor {motor}: torque enable unconfirmed')

    def enable(self, q):
        self.write(q)  # preload the measured pose BEFORE enabling torque
        self.client.set_torque_enabled(MOTORS, True, retries=0)
        for motor in MOTORS:
            if self.register(motor, 64, 1) != 1:
                raise OSError(f"Motor {motor}: torque enable failed")

    def close_read_only(self):
        self.close(disable_torque=False)

    def close(self, disable_torque=True):
        errors = []
        try:
            if self.client.is_connected and disable_torque:
                port = self.client.port_handler
                # All calls are synchronous and have unwound before close().
                # Recover an abandoned SDK flag/receive buffer before cleanup;
                # it is not a lock owned by another serial worker.
                if getattr(port, 'is_using', False):
                    port.is_using = False
                    try:
                        port.clearPort()
                        print('Recovered interrupted Dynamixel transaction for torque cleanup.', flush=True)
                    except Exception as exc:
                        errors.append(f'Serial cleanup recovery failed: {exc}')
                for motor in MOTORS:
                    try:
                        self.client.set_torque_enabled([motor], False, retries=0)
                        if self.register(motor, 64, 1) != 0:
                            raise OSError(f"Motor {motor}: torque disable unconfirmed")
                    except Exception as exc:
                        errors.append(str(exc))
                if not errors:
                    print('Torque disabled and verified on all 16 motors.', flush=True)
        finally:
            try:
                if self.client.is_connected:
                    self.client.port_handler.closePort()
            finally:
                self.client.OPEN_CLIENTS.discard(self.client)
        if errors:
            raise OSError("; ".join(errors))


def bounded_command(proposed, previous, measured, lower, upper, max_step, max_error,
                    delta_lower=None, delta_upper=None):
    proposed = vector(proposed, "planned action")
    lo = np.maximum(lower, np.maximum(previous - max_step, measured - max_error))
    hi = np.minimum(upper, np.minimum(previous + max_step, measured + max_error))
    if delta_lower is not None and delta_upper is not None:
        proposed = clip_joint_target(proposed, measured, delta_lower, delta_upper)
        lo = np.maximum(lo, measured - delta_lower)
        hi = np.minimum(hi, measured + delta_upper)
    if np.any(lo > hi):
        raise RuntimeError("No command satisfies position, rate and tracking limits")
    return np.clip(proposed, lo, hi)


def check_feedback(q, v, current, config, lower, upper):
    """Manual viewer's position/velocity envelope, in addition to device faults."""
    check_device_feedback(q, v, current, config)
    if np.any(q < lower) or np.any(q > upper):
        raise RuntimeError("Measured pose outside hardware envelope")
    if np.max(np.abs(v)) > config["max_velocity"]:
        raise RuntimeError("Measured velocity exceeds hardware limit")


def check_device_feedback(q, v, current, config):
    """Validate measurements/device current without shaping position commands."""
    for value, name in ((q, "q"), (v, "v"), (current, "current")):
        vector(value, name)
    currents = np.asarray(current, dtype=float)
    limit = config["max_current_raw"]
    exceeded = np.flatnonzero(np.abs(currents) > limit)
    if exceeded.size:
        # Current remains in native motor-ID order; q/v have already been
        # mapped to MuJoCo order by read(). Do not label this with q indices.
        readings = ', '.join(f'motor {MOTORS[i]}={currents[i]:g}' for i in exceeded)
        raise RuntimeError(
            f"Measured current exceeds hardware limit: {readings} raw; "
            f"allowed +/-{limit:g} raw; currents_by_motor_id_raw={currents.tolist()}")

