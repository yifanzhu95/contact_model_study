import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import pytest
from camera_hand_calibration.selection import selected_camera_config
from ContactModelStudy.Hardware.Camera import load_camera_config
from ContactModelStudy.Hardware.LCM import load_pose_transform


def fixtures(tmp_path):
    workspace=tmp_path/'workspace';(workspace/'configs').mkdir(parents=True);(workspace/'method5/fit').mkdir(parents=True)
    base=json.loads((Path(__file__).resolve().parents[1]/'configs/source_camera.json').read_text())
    original=workspace/'configs/source_camera.json';original.write_text(json.dumps(base))
    target=np.eye(4);target[:3,:3]=Rotation.from_euler('xyz',[3,-7,4],degrees=True).as_matrix();target[:3,3]=[.4,.02,.5]
    (workspace/'method5/fit/candidate.json').write_text(json.dumps(dict(method=5,T_world_from_camera=target.tolist())))
    (workspace/'method5/latest_fit.json').write_text(json.dumps(dict(directory='method5/fit')))
    return workspace,original,target


def test_latest_removes_publication_and_old_offset_exactly_once(tmp_path):
    w,base,target=fixtures(tmp_path);incoming=load_camera_config(base);path=selected_camera_config('latest',base,workspace=w)
    before,after,config=load_pose_transform(path)
    raw=np.eye(4);raw[:3,:3]=Rotation.from_euler('xyz',[20,8,-90],degrees=True).as_matrix();raw[:3,3]=[.02,-.04,.62]
    shifted=raw.copy();shifted[:3,3]+=incoming['object_translation_adjust_camera_m'];published=incoming['camera_to_published']@shifted
    np.testing.assert_allclose(before@published@after,target@raw,atol=1e-12)
    np.testing.assert_allclose(config['selected_calibration']['object_translation_adjust_camera_m'],0)
    assert path==selected_camera_config('latest',base,workspace=w)
    assert load_camera_config(base)['object_translation_adjust_camera_m'][2]==-.015


def test_original_preserves_existing_corrected_publication(tmp_path):
    w,base,_=fixtures(tmp_path);path=selected_camera_config('original',base,workspace=w)
    before,after,config=load_pose_transform(path)
    np.testing.assert_allclose(before,np.eye(4),atol=1e-12)
    assert config['selected_calibration']['object_translation_adjust_camera_m']==[0,0,-.015]


def test_latest_missing_or_untrusted_pointer_fails(tmp_path):
    w,base,_=fixtures(tmp_path);pointer=w/'method5/latest_fit.json';pointer.unlink()
    with pytest.raises(FileNotFoundError):selected_camera_config('latest',base,workspace=w)
    pointer.write_text(json.dumps(dict(directory='../../elsewhere')))
    with pytest.raises(ValueError,match='outside'):selected_camera_config('latest',base,workspace=w)


def test_launcher_keeps_tracker_camera_and_shares_selected_consumer_config(tmp_path):
    import launch_leap_experiment as launcher
    config=json.loads((launcher.ROOT/'configs/leap_experiment.json').read_text());config['init_mask_path']=str(tmp_path/'mask.png')
    config['docker']['enabled']=False;config['camera_calibration']='latest'
    commands=launcher.build_commands(config,tmp_path)
    def camera(name):
        argv=commands[name];return Path(argv[argv.index('--camera-config')+1])
    assert camera('tracker')==launcher.local_path(config['camera_config'])
    assert camera('mask')==camera('tracker')==camera('viewer')
    assert camera('controller')==camera('mujoco') and camera('controller')!=camera('tracker')
    assert json.loads(camera('controller').read_text())['selected_calibration']['name']=='latest'
