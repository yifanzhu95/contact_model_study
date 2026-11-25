"""Select a world calibration without changing the running tracker publication.

The tracker keeps its configured publication frame/offset. Consumers receive an
adapter that removes that mapping and applies exactly one selected calibration.
All generated selection snapshots remain inside the calibration workspace.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from ContactModelStudy.Hardware.Camera import load_camera_config, rigid_transform

WORKSPACE=Path(__file__).resolve().parent


def selected_camera_config(choice, publication_config, *, workspace=WORKSPACE):
    if choice not in ('original','latest'):raise ValueError('camera calibration must be original or latest')
    workspace=Path(workspace).resolve();publication_config=Path(publication_config).expanduser().resolve()
    incoming=load_camera_config(publication_config,require_extrinsics=True)
    original=workspace/'configs/source_camera.json'
    if choice=='original':
        source=original;desired=load_camera_config(source,require_extrinsics=True)
        target=desired['camera_to_published'];offset=desired['object_translation_adjust_camera_m']
    else:
        pointer=workspace/'method5/latest_fit.json'
        latest=json.loads(pointer.read_text());source=(workspace/latest['directory']/'candidate.json').resolve()
        if not source.is_relative_to(workspace):raise ValueError('Saved calibration points outside the calibration workspace')
        candidate=json.loads(source.read_text())
        if candidate.get('method')!=5:raise ValueError('Latest calibration must be a saved Method 5 manual fit')
        target=rigid_transform(candidate['T_world_from_camera'],'saved camera-to-world calibration');offset=np.zeros(3)
    shift=np.eye(4);shift[:3,3]=np.asarray(offset)-incoming['object_translation_adjust_camera_m']
    before=target@shift@np.linalg.inv(incoming['camera_to_published'])
    config=json.loads(publication_config.read_text())
    # The consumer uses the exact publication matrix, not tag-site metadata.
    for key in ('T_tag_from_camera','T_base_from_tag','tag_calibration'):config.pop(key,None)
    config.update(camera_to_published=incoming['camera_to_published'].tolist(),
        T_model_from_published=before.tolist(),translation_adjust_m=[0,0,0],rotation_adjust_deg=[0,0,0],
        extrinsics_status='selected_calibration',
        selected_calibration=dict(name=choice,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            publication_config=str(publication_config),publication_sha256=hashlib.sha256(publication_config.read_bytes()).hexdigest(),
            T_world_from_camera=np.asarray(target).tolist(),object_translation_adjust_camera_m=np.asarray(offset).tolist(),
            application='consumer adapter; tracker publication is unchanged'))
    payload=json.dumps(config,sort_keys=True,indent=2,allow_nan=False)+'\n'
    out=workspace/'runtime/selections'/(hashlib.sha256(payload.encode()).hexdigest()+'.json');out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        import os,tempfile
        fd,path=tempfile.mkstemp(prefix='.selection_',dir=out.parent)
        try:
            with os.fdopen(fd,'w') as stream:stream.write(payload)
            os.replace(path,out)
        finally:Path(path).unlink(missing_ok=True)
    return out
