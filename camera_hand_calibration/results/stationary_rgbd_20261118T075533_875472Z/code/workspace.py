"""Self-contained paths, provenance and snapshots for hand/camera calibration."""
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parent


def local(value):
    path=Path(value)
    path=(ROOT/path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path must stay inside {ROOT}: {value}')
    return path


def utc():return datetime.now(timezone.utc).isoformat()
def identifier():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=local(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def read_json(path):return json.loads(local(path).read_text())


def environment():
    result=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={})
    for name in ['numpy','scipy','mujoco','pyrealsense2','opencv-python','matplotlib','lcm','pytest']:
        try:result['packages'][name]=metadata.version(name)
        except metadata.PackageNotFoundError:result['packages'][name]=None
    return result


def transform(value):
    value=np.asarray(value,float)
    if (value.shape!=(4,4) or not np.isfinite(value).all() or not np.allclose(value[3],[0,0,0,1])
        or not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-5)
        or not np.isclose(np.linalg.det(value[:3,:3]),1,atol=1e-5)):
        raise ValueError('Expected a finite proper rigid 4x4 transform')
    return value


def bootstrap(model_source=None,camera_source=None,hardware_source=None):
    """Only this command reads source project inputs outside the workspace."""
    import mujoco
    from scipy.spatial.transform import Rotation
    for folder in ['configs','model/assets','datasets','results','logs','cache','tests']:
        local(folder).mkdir(parents=True,exist_ok=True)
    if local('configs/experiment.json').exists():
        raise ValueError('Workspace already initialized; existing snapshots were preserved')
    repo=ROOT.parent
    camera_source=Path(camera_source or repo/'configs/leap_pose_transform.json').expanduser().resolve()
    hand_source=Path(model_source or repo/'scenes/leap/hardware/calibrated_hand.xml').expanduser().resolve()
    hardware_source=Path(hardware_source or repo/'configs/leap_hardware.json').expanduser().resolve()
    cam=json.loads(camera_source.read_text())
    tree=ET.parse(hand_source);root=tree.getroot()
    if root.findall('.//include'):raise ValueError('Snapshot requires a standalone XML with includes expanded')
    compiler=root.find('compiler');meshdir=compiler.get('meshdir','');texturedir=compiler.get('texturedir','')
    assets={}
    for kind,directory in [('mesh',meshdir),('texture',texturedir)]:
        for element in root.findall(f'.//asset/{kind}'):
            filename=element.get('file')
            if not filename:continue
            source=(hand_source.parent/directory/filename).resolve()
            target=local('model/assets')/f'{kind}_{source.name}'
            if target.name in assets and assets[target.name]['sha256']!=digest(source):
                raise ValueError('Asset filename collision')
            shutil.copy2(source,target);element.set('file',target.name)
            assets[target.name]=dict(source=str(source),sha256=digest(target))
    compiler.set('meshdir','assets');compiler.set('texturedir','assets')
    tree.write(local('model/hand.xml'),encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(local('model/hand.xml')))
    mujoco.mj_saveModel(model,str(local('model/hand.mjb')))
    shutil.copy2(camera_source,local('configs/source_camera.json'))
    shutil.copy2(hardware_source,local('configs/source_hardware.json'))
    base=(transform(cam['T_base_from_tag'])@transform(cam['T_tag_from_camera'])
          if 'T_base_from_tag' in cam else transform(cam['camera_to_published']))
    adjustment=np.eye(4)
    adjustment[:3,:3]=Rotation.from_euler('xyz',cam.get('rotation_adjust_deg',[0,0,0]),degrees=True).as_matrix()
    adjustment[:3,3]=cam.get('translation_adjust_m',[0,0,0])
    effective=adjustment@transform(cam.get('T_model_from_published',np.eye(4)))@base
    write_json('configs/current_transform.json',dict(T_world_from_camera=effective.tolist(),
        convention='OpenCV camera right/down/forward -> MuJoCo model world, meters',
        source='frozen current calibration; refinement prior, not new evidence',
        object_only_offset_excluded=cam.get('object_translation_adjust_camera_m',[0,0,0])))
    names=[model.joint(i).name for i in range(model.njnt)]
    config=dict(camera=dict(serial=cam.get('camera_serial'),width=cam['image_size'][0],height=cam['image_size'][1],
        fps=cam.get('camera_fps',60),warmup_frames=60,enable_global_time=True,
        save_every_n_frames=12,depth_min_m=.1,depth_max_m=1.5),
        model='model/hand.xml',model_binary='model/hand.mjb',
        camera_prior='configs/current_transform.json',
        joints=dict(url='udpm://239.255.76.67:7667?ttl=1',channel='LEAP_STATE',
                    names=names,max_frame_skew_ms=30.,settle_window_s=.4,
                    max_motion_rad=.01,max_speed_rad_s=.03,min_settle_samples=5,max_sample_gap_s=.05),
        fit=dict(ransac_px=3.,robust_scale_px=2.,heldout_p95_limit_px=4.,seed=0,max_scaled_jacobian_condition=1e6),
        notes=['No motor commands or direct serial access are implemented.',
            'Joint LCM subscription is explicitly opt-in; await operator confirmation before a hardware experiment.',
            'No object-only translation correction is used for hand calibration.'])
    write_json('configs/experiment.json',config)
    write_json('model/manifest.json',dict(created_at_utc=utc(),source_xml=str(hand_source),
        source_xml_sha256=digest(hand_source),snapshot_xml_sha256=digest(local('model/hand.xml')),
        binary_sha256=digest(local('model/hand.mjb')),assets=assets,joint_names=names,
        camera_source_sha256=digest(camera_source),hardware_source_sha256=digest(hardware_source)))
    write_json('configs/environment.json',environment())
    versions=environment()['packages']
    local('requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items() if v)+'\n')
    return dict(workspace=str(ROOT),assets=len(assets),joints=names)


def verify_model():
    manifest=read_json('model/manifest.json')
    if digest(local('model/hand.xml'))!=manifest['snapshot_xml_sha256']:
        raise ValueError('Frozen hand XML has changed')
    for name,record in manifest['assets'].items():
        if digest(local('model/assets')/name)!=record['sha256']:raise ValueError(f'Frozen mesh changed: {name}')
    return manifest
