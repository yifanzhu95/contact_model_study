import json
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from workspace import ROOT,local,write_json,read_json,digest,transform,identifier,verify_model
from geometry import project,fit_pose,point_in_world
from capture import frame_epoch,match_joints,Writer,JointSubscriber,load_frame


def test_paths_and_symlinks_cannot_escape_workspace(tmp_path):
    with pytest.raises(ValueError):local('/tmp/outside_calibration.json')
    with pytest.raises(ValueError):local('../configs/leap_pose_transform.json')
    link=tmp_path/'escape';link.symlink_to('/tmp',target_is_directory=True)
    with pytest.raises(ValueError):local(link/'outside.json')


def test_frozen_model_and_assets_are_verified():
    assert len(verify_model()['assets'])==10


def test_workspace_status_requires_no_camera_or_hand_modules(monkeypatch):
    monkeypatch.setitem(sys.modules,'pyrealsense2',None)
    monkeypatch.setitem(sys.modules,'lcm',None)
    from quality import workspace_status
    result=workspace_status()
    assert result['production_configuration_modified'] is False
    assert result['direct_motor_access_implemented'] is True


@pytest.mark.parametrize('initial',[None,'perturbed'])
def test_known_se3_recovered_from_natural_3d_2d_points(initial):
    rng=np.random.default_rng(17);points=rng.uniform(-.06,.06,(80,3));K=np.array([[606,0,317],[0,605,253],[0,0,1.]])
    C=np.eye(4);C[:3,:3]=Rotation.from_euler('xyz',[.2,-.1,.15]).as_matrix();C[:3,3]=[.02,-.03,.6]
    truth=np.linalg.inv(C);pixels,_=project(points,truth,K);pixels+=rng.normal(0,.2,pixels.shape);pixels[:5]+=30
    prior=truth.copy();prior[:3,3]+=[.01,-.01,.02]
    fitted,diag=fit_pose(points,pixels,K,initial=prior if initial else None)
    error=fitted@np.linalg.inv(truth)
    assert np.linalg.norm(error[:3,3])<.002
    assert Rotation.from_matrix(error[:3,:3]).magnitude()<np.deg2rad(.2)
    assert diag['jacobian_rank']==6 and sum(diag['inliers'])>=74


def test_degenerate_correspondences_and_invalid_intrinsics_rejected():
    points=np.column_stack([np.linspace(0,1,8),np.zeros((8,2))]);pixels=np.ones((8,2))
    with pytest.raises(ValueError,match='collinear'):fit_pose(points,pixels,np.eye(3))
    with pytest.raises(ValueError,match='intrinsics'):fit_pose(points,pixels,-np.eye(3))


def test_fixed_palm_needs_no_joint_measurement_but_fingers_do():
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(local('model/hand.xml')))
    assert point_in_world(model,'palm',[0,0,0]).shape==(3,)
    with pytest.raises(ValueError,match='measured joints'):point_in_world(model,'if_ds',[0,0,0])
    assert np.isfinite(point_in_world(model,'if_ds',[0,0,0],np.zeros(model.nq))).all()


def test_clock_domains_are_not_silently_equated_to_epoch():
    now=time.time()
    stamp,basis=frame_epoch(now*1000,'timestamp_domain.global_time',now+.01)
    assert stamp==pytest.approx(now,abs=1e-6) and basis=='sdk_epoch'
    assert frame_epoch(12000,'timestamp_domain.hardware_clock',now)==(now,'host_receipt_approximate')
    assert frame_epoch(12000,'timestamp_domain.global_time',now)==(now,'host_receipt_approximate')


def test_joint_matching_requires_static_synchronized_measurements():
    config=read_json('configs/experiment.json')['joints'];stamp=1000.
    rows=[dict(timestamp_epoch_s=t,q=np.ones(16).tolist(),v=np.zeros(16).tolist()) for t in np.linspace(999.6,1000.,41)]
    assert match_joints(rows,stamp,'sdk_epoch',config)['usable']
    assert not match_joints(rows,stamp,'host_receipt_approximate',config)['usable']
    assert not match_joints(rows,stamp+.1,'sdk_epoch',config)['usable']
    rows[-1]['q'][2]+=.1
    assert not match_joints(rows,stamp,'sdk_epoch',config)['usable']


def test_joint_stability_cannot_be_inferred_across_a_long_feedback_gap():
    config=read_json('configs/experiment.json')['joints']
    rows=[dict(timestamp_epoch_s=t,q=[0.]*16,v=[0.]*16) for t in [999.6,999.61,999.62,999.99,1000.]]
    assert not match_joints(rows,1000.,'sdk_epoch',config)['usable']


def test_joint_subscriber_has_no_publish_or_serial_path(tmp_path,monkeypatch):
    events=[]
    class Connection:
        def __init__(self,url):events.append(('connect',url))
        def subscribe(self,channel,callback):
            events.append(('subscribe',channel));return SimpleNamespace(set_queue_capacity=lambda n:None)
        def handle_timeout(self,timeout):time.sleep(.001)
        def unsubscribe(self,subscription):events.append(('unsubscribe',))
        def publish(self,*args):pytest.fail('Passive capture attempted to publish')
    monkeypatch.setitem(sys.modules,'lcm',SimpleNamespace(LCM=Connection))
    writer=Writer(tmp_path);collector=JointSubscriber(read_json('configs/experiment.json')['joints'],writer)
    try:
        collector.callback('LEAP_STATE',struct.pack('<q32f',int(time.time()*1e6),*range(32)))
        assert collector.samples[-1]['q']==list(range(16))
        assert collector.samples[-1]['v']==list(range(16,32))
    finally:collector.close();writer.close()
    assert json.loads((tmp_path/'joints.jsonl').read_text())['q']==list(range(16))


def test_end_to_end_synthetic_fit_is_clearly_not_a_real_calibration(tmp_path):
    from study import fit,model
    m=model();rng=np.random.default_rng(18)
    prior=np.asarray(read_json('configs/current_transform.json')['T_world_from_camera']);truth=prior.copy();truth[:3,3]+=[.004,-.003,.002]
    K=np.asarray(read_json('configs/source_camera.json')['camera_intrinsics'])
    observations=[]
    for pose in range(4):
        metadata=dict(camera_intrinsics=K.tolist(),camera_serial='SYNTHETIC_TEST_ONLY',
            distortion_coefficients=[0]*5,depth_scale_m=.001,joints=None,pose_id=f'synthetic_pose_{pose}')
        frame=tmp_path/f'frame{pose}.npz'
        np.savez_compressed(frame,color_rgb=np.zeros((480,640,3),np.uint8),depth_z16=np.zeros((480,640),np.uint16),metadata_json=np.asarray(json.dumps(metadata)))
        for j in range(20):
            xyz=rng.uniform([-.08,-.07,-.04],[0.,0.,.01]);world=point_in_world(m,'palm',xyz)
            uv,z=project(world[None],truth,K);assert z[0]>0
            observations.append(dict(frame=str(frame.relative_to(ROOT)),frame_sha256=digest(frame),
                model_sha256=digest(local('model/hand.xml')),landmark=dict(id=f'p{j}',body='palm',xyz_body_m=xyz.tolist()),
                world_point_m=world.tolist(),pixel=(uv[0]+rng.normal(0,.1,2)).tolist(),
                pose_id=metadata['pose_id'],split='validation' if pose==3 else 'train',depth_m=None,depth_reliable=False))
    source=tmp_path/'observations.json';write_json(source,dict(synthetic_test_only=True,observations=observations))
    result=fit(source,'scratch','synthetic_test_'+identifier())
    assert result['status']=='synthetic_test_only' and not result['production_configuration_modified']
    assert result['metrics']['validation']['after_pixel_error']['p95']<1
    estimate=np.asarray(read_json(local(result['directory'])/'candidate_transform.json')['T_world_from_camera'])
    assert np.linalg.norm(estimate[:3,3]-truth[:3,3])<.002
