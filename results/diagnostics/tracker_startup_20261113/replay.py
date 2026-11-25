"""Replay a saved registration frame; never open camera, LCM, or hardware."""
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
ROOT = Path('/home/chetan/contact_model_study')
OUT = ROOT / 'results/diagnostics/tracker_startup_20261113'
@contextmanager
def stage(name):
    start = time.monotonic()
    print('[replay] START', name, flush=True)
    yield
    print('[replay] DONE', name, round(time.monotonic()-start, 3), 's', flush=True)
sys.path[:0] = [str(ROOT/'iris-foundationpose-plusplus/src'), '/FoundationPose-plus-plus', '/FoundationPose-plus-plus/FoundationPose', str(ROOT)]
with stage('torch import'):
    import torch
print('[replay] torch threads',torch.get_num_threads(),'interop',torch.get_num_interop_threads(), 'affinity',len(os.sched_getaffinity(0)),flush=True)
if os.environ.get('REPLAY_THREADS'):
    torch.set_num_threads(int(os.environ['REPLAY_THREADS']))
    torch.set_num_interop_threads(1)
    print('[replay] constrained torch threads',torch.get_num_threads(),flush=True)
with stage('VOT import'):
    import VOT
with stage('FoundationPose import'):
    import FoundationPose.estimater as fp
if os.environ.get('REPLAY_BOUNDED'):
    from ContactModelStudy.Hardware.VisionRuntime import install_bounded_mesh_diameter
    install_bounded_mesh_diameter(fp)
with stage('mesh and saved RGB-D input'):
    import cv2
    import numpy as np
    import trimesh
    import json
    from ContactModelStudy.Hardware.Camera import load_registration_frame
    cfg=json.loads((ROOT/'configs/leap_pose_transform.json').read_text())
    run=ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2/leap_experiment_20261113_211949_369635'
    rgb,depth,_=load_registration_frame(run/'registration_frame.npz',cfg['image_size'],cfg['camera_serial'])
    mask=cv2.imread(str(run/'mask.png'),0)>127
    mesh=trimesh.load(str(ROOT/'iris-foundationpose-plusplus/meshes/cube_70_mm_2_adjusted/cube_70_mm_2.obj'))
    if isinstance(mesh,trimesh.Scene): mesh=mesh.dump(concatenate=True)
with stage('ScorePredictor'):
    scorer=fp.ScorePredictor()
with stage('PoseRefinePredictor'):
    refiner=fp.PoseRefinePredictor()
with stage('RasterizeCudaContext'):
    glctx=fp.dr.RasterizeCudaContext()
with stage('FoundationPose constructor'):
    est=fp.FoundationPose(model_pts=mesh.vertices,model_normals=mesh.vertex_normals,mesh=mesh,scorer=scorer,refiner=refiner,glctx=glctx,debug_dir='/tmp/leap_replay')
with stage('register, 10 refinement iterations'):
    pose=est.register(K=np.asarray(cfg['camera_intrinsics'], dtype=np.float32),rgb=rgb,depth=depth,ob_mask=mask.astype(np.uint8)*255,iteration=10)
    torch.cuda.synchronize()
print('[replay] POSE',pose,flush=True)
with stage('track_one, 5 refinement iterations'):
    pose=est.track_one(rgb=rgb,depth=depth,K=np.asarray(cfg['camera_intrinsics'], dtype=np.float32),iteration=5)
    torch.cuda.synchronize()
print('[replay] PASSED',flush=True)
faulthandler.cancel_dump_traceback_later()
