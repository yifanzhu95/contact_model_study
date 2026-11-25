"""Inspect deployment files/package metadata; never open USB/camera or start services."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def local(value, root=ROOT):
    root = Path(root).resolve()
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f'Deployment runtime paths must be checkout-relative: {value}')
    path = (root/path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Deployment path escapes the checkout: {value}')
    return path


def read(path):
    return json.loads(Path(path).read_text())


def inspect(profile=HERE/'experiment.json', *, root=ROOT, check_environment=False):
    root = Path(root).resolve()
    profile = Path(profile)
    if not profile.is_absolute():
        profile = root/profile
    errors, notes = [], []
    manifest = read(root/'deployment/leap_cube/bundle_manifest.json')
    for name, expected in manifest['sha256'].items():
        path = local(name, root)
        if not path.is_file():
            errors.append(f'Missing versioned reference asset: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f'Reference asset changed: {name}')
    config = read(profile)
    for field in ['hardware_config', 'camera_config']:
        if not local(config[field], root).is_file():
            errors.append(f'Missing {field}: {config[field]}')
    if errors:
        return errors, notes
    hardware = read(local(config['hardware_config'], root))
    camera = read(local(config['camera_config'], root))
    for name in [hardware['joint_limits']['model'], config['tracking_meshes']['cube_reorient']['mesh']]:
        if not local(name, root).is_file():
            errors.append(f'Missing model/mesh: {name}')
    args = config['controller_args']
    for flag in ['--calibrated-hand', '--joint-reference-file']:
        name = args[args.index(flag)+1]
        if not local(name, root).is_file():
            errors.append(f'Missing {flag}: {name}')
    if config.get('camera_calibration') is not None:
        errors.append('The standalone camera file already maps to world; leave camera_calibration null to avoid another transform.')
    if camera.get('object_translation_adjust_camera_m') != [0, 0, 0]:
        errors.append('This Method 5 deployment reference uses zero object correction.')
    if len(hardware.get('robot_qref', [])) != 16:
        errors.append('Preparation robot_qref must contain 16 joints.')
    notes += [f"USB path (not opened): {hardware['port']}",
              f"Camera serial (not queried): {camera.get('camera_serial')}",
              'Camera transform, joint limits and actuator calibration describe the source rig; verify or replace them for the destination.',
              'Passing checks files/configuration, not physical calibration or hardware operation.']
    if check_environment:
        versions = read(root/'deployment/leap_cube/host_versions.json')
        for name in ['mujoco','mujoco-warp','warp-lang','numpy','scipy','lcm','dynamixel-sdk',
                     'pyserial','pyrealsense2','opencv-python','pyzmq','trimesh']:
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f'Missing host distribution: {name}')
                continue
            if actual != versions[name]:
                notes.append(f'Host differs from source snapshot: {name} {actual} (source {versions[name]})')
        api_info = read(root/'deployment/leap_cube/external_assets.json')['leap_hardware_api']
        api = local(config['bridge_args'][config['bridge_args'].index('--hardware-api')+1], root)
        for name, expected in api_info['sha256'].items():
            path = api/name
            if not path.is_file():
                errors.append(f'Missing external LEAP API: {path}; clone the pinned checkout in README.md.')
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f'External LEAP API differs from recorded source: {path}')
        if config.get('record_video') and not shutil.which('ffmpeg'):
            errors.append('ffmpeg is required for record_video.')
        if config.get('docker', {}).get('enabled') and not shutil.which('docker'):
            errors.append('Docker CLI is missing; the prepared vision image and NVIDIA runtime are also required.')
        notes.append('Docker images, GPUs, serial permissions and device identities are not queried by this check.')
    return errors, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=HERE/'experiment.json')
    parser.add_argument('--check-environment', action='store_true',
                        help='also inspect installed package metadata, external API files and executables; starts no services')
    args = parser.parse_args(argv)
    errors, notes = inspect(args.profile, check_environment=args.check_environment)
    for message in notes:
        print(message)
    for message in errors:
        print('ERROR: '+message, file=sys.stderr)
    print('Deployment file check: '+('FAILED' if errors else 'PASS'))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
