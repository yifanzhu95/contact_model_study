"""Run production pose_track on saved frames, replacing device/network adapters."""
import sys
import os
import time
import json
import importlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/chetan/contact_model_study')
sys.path[:0]=[str(ROOT/'iris-foundationpose-plusplus/src'),str(ROOT)]
os.environ['FOUNDATIONPOSE_ROOT']='/FoundationPose-plus-plus/FoundationPose'
start=time.monotonic()
import online_tracking2ctrl as tracker
from ContactModelStudy.Hardware.Camera import load_camera_config, load_registration_frame
import numpy as np
config=load_camera_config(ROOT/'configs/leap_pose_transform.json',require_extrinsics=True)
class ReplayFinished(Exception): pass
class Camera:
    def __init__(self, **kwargs):
        self.count=0
        self.pipeline=SimpleNamespace(stop=lambda: None)
    def getFrame(self):
        if self.count>=3: raise ReplayFinished()
        self.count+=1
        return color,depth,time.time()
class Publisher:
    def __init__(self, **kwargs): pass
    def publish(self,pose,image,stamp):
        assert np.isfinite(pose).all()
        frames.append(time.monotonic())
    def close(self): pass
class LCM:
    def __init__(self, **kwargs): pass
    def publish_pose(self,pose,timestamp):
        assert np.isfinite(pose).all()
        poses.append(pose.copy())
    def close(self): pass
tracker.discover_camera_devices=lambda:[dict(serial=config['camera_serial'])]
tracker.Camera=Camera
tracker.Publisher=Publisher
tracker.LCMPublisher=LCM
mesh=ROOT/'iris-foundationpose-plusplus/meshes/cube_70_mm_2_adjusted/cube_70_mm_2.obj'
for run in ['leap_experiment_20261113_211750_152086','leap_experiment_20261113_211949_369635']:
    path=ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2'/run
    color,depth,_=load_registration_frame(path/'registration_frame.npz',config['image_size'],config['camera_serial'])
    tracker.args=SimpleNamespace(camera_config_data=config,viewer_bind='unused',pose_channel='unused',lcm_url='unused',
        apply_scale=1.0,force_apply_color=False,verbose=False,registration_frame_path=str(path/'registration_frame.npz'))
    frames=[];poses=[]
    run_start=time.monotonic()
    try:
        tracker.pose_track(str(mesh),str(path/'mask.png'),config['camera_intrinsics'],None,None,None,10,5)
    except ReplayFinished:
        pass
    assert len(frames)==4 and len(poses)==4,(len(frames),len(poses))
    print('REPLAY_RESULT',json.dumps(dict(run=run,first_pose_s=frames[0]-run_start,total_s=time.monotonic()-run_start,
        frames=len(frames),poses=len(poses),position_model=poses[-1][:3,3].tolist())),flush=True)
print('PRODUCTION REPLAY PASSED',round(time.monotonic()-start,3),'s',flush=True)
