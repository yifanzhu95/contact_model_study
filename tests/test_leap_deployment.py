"""Portable baseline contract and file-only checks; no hardware or camera calls."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from deployment.leap_cube import verify
import launch_leap_experiment as launcher
import run_leap_hardware as runner
from ContactModelStudy.Hardware.Camera import load_camera_config

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'deployment/leap_cube'


def test_reference_bundle_is_complete_and_checked_without_external_api():
    errors,notes=verify.inspect()
    assert errors==[]
    assert any('not opened' in note for note in notes)


def test_explicit_profile_matches_trial358_baseline_not_task_defaults():
    config=json.loads((HERE/'experiment.json').read_text())
    args=runner.parseArgs(config['controller_args']+['--hardware','--hardware-config',config['hardware_config']])
    assert (args.rollout_model,args.hand_acc,args.obj_acc)==('M2','high','high')
    assert args.n_samples==512 and args.n_iterations==1 and args.steps==1000
    assert args.time_horizon==.256 and args.ctrl_time_step==.064 and args.pace_control
    assert args.temperature==.1 and args.noise_sigma==.05 and args.warm_start=='shift'
    assert args.action_rate_weight==50 and args.joint_limit_weight==817.6259701873483
    assert args.plan_blend_alpha==1
    expected={'w_quat':.01,'w_pos_x':19.93030694751929,'w_pos_y':100.,'w_pos_z':.01,
              'w_velo':0.,'w_contact':80.,'w_joint':.8575242876710942,'w_joint_velo':0.,
              'w_fallen':10.,'w_quat_term':277.07876775743,'w_pos_term':1000.,'w_fallen_term':0.}
    assert args.cost_weights==expected
    ref=json.loads((HERE/'joint_reference.json').read_text())
    current=json.loads((ROOT/'configs/leap_joint_references/cube.json').read_text())
    assert ref['joint_reference']==current['joint_reference']


def test_direct_camera_mapping_matches_source_method5_adapter_chain():
    source=load_camera_config(ROOT/'configs/leap_pose_transform.json')
    deployed=load_camera_config(HERE/'camera.reference.json')
    fit=json.loads((ROOT/'camera_hand_calibration/method5/results/session_20261119T065119_646514Z/fit_001/candidate.json').read_text())
    shift=np.eye(4);shift[:3,3]=-source['object_translation_adjust_camera_m']
    adapter=np.array(fit['T_world_from_camera'])@shift@np.linalg.inv(source['camera_to_published'])
    points=np.vstack((np.random.default_rng(7).normal(size=(3,20)),np.ones(20)))
    source_points=points.copy();source_points[:3]+=source['object_translation_adjust_camera_m'][:,None]
    expected=adapter@source['camera_to_published']@source_points
    actual=deployed['camera_to_published']@points
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    np.testing.assert_array_equal(deployed['object_translation_adjust_camera_m'],0)


def test_command_builder_uses_portable_api_and_one_shared_world_frame(tmp_path):
    config=json.loads((HERE/'experiment.json').read_text())
    config['init_mask_path']=str(tmp_path/'mask.png')
    commands=launcher.build_commands(config,tmp_path)
    bridge=commands['bridge']
    assert bridge[bridge.index('--hardware-api')+1]=='third_party/LEAP_Hand_Hardware'
    for service in ['tracker','controller','mujoco','viewer']:
        command=commands[service]
        assert command[command.index('--camera-config')+1]==str(HERE/'camera.reference.json')
    assert '--cost-weight' in commands['controller']


@pytest.mark.parametrize('path',['/home/somebody/checkout/model.xml','../outside/model.xml'])
def test_manifest_runtime_paths_must_be_checkout_relative(path,tmp_path):
    with pytest.raises(ValueError):verify.local(path,tmp_path)


def test_missing_or_changed_reference_asset_is_reported(tmp_path):
    bundle=tmp_path/'deployment/leap_cube';bundle.mkdir(parents=True)
    (bundle/'bundle_manifest.json').write_text(json.dumps({'sha256':{'missing.xml':'0'*64,'changed.xml':'0'*64}}))
    (tmp_path/'changed.xml').write_text('different')
    (bundle/'experiment.json').write_text(json.dumps({'hardware_config':'missing-config.json','camera_config':'missing-camera.json'}))
    errors,_=verify.inspect(bundle/'experiment.json',root=tmp_path)
    assert any('Missing versioned reference asset' in e for e in errors)
    assert any('Reference asset changed' in e for e in errors)
