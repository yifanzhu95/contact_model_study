#!/usr/bin/env python3
"""Pure manual-reference editing and rigid camera calibration fitting."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent))
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from geometry import matrix,parameters
from ContactModelStudy.Hardware.LCM import rigid_transform


class ReferenceEditor:
    """Numpad changes a display pose only. There is no hardware interface."""
    AXES={324:(0,-1),326:(0,1),322:(1,-1),328:(1,1),321:(2,-1),327:(2,1)}
    def __init__(self,config):
        self.config=config;self.pose=None;self.mode='translate';self.step_index=1;self.undo=[]
    def seed(self,pose):
        self.pose=rigid_transform(pose).copy();self.undo=[]
    def key(self,key):
        if key==325:self.mode='rotate' if self.mode=='translate' else 'translate';return 'mode'
        if key==320:self.step_index=(self.step_index+1)%len(self.config['translation_steps_m']);return 'step'
        if key==329:
            if self.undo:self.pose=self.undo.pop()
            return 'undo'
        if key not in self.AXES or self.pose is None:return None
        axis,sign=self.AXES[key];self.undo.append(self.pose.copy());self.undo=self.undo[-1000:]
        if self.mode=='translate':self.pose[axis,3]+=sign*self.config['translation_steps_m'][self.step_index]
        else:
            vector=np.zeros(3);vector[axis]=np.radians(sign*self.config['rotation_steps_deg'][self.step_index])
            self.pose[:3,:3]=Rotation.from_rotvec(vector).as_matrix()@self.pose[:3,:3]
        return 'nudge'


def pose_stability(poses):
    poses=np.asarray(poses);rot=Rotation.from_matrix(poses[:,:3,:3]);mean=rot.mean()
    return dict(position_std_m=float(np.linalg.norm(np.std(poses[:,:3,3],axis=0))),
        orientation_rms_deg=float(np.degrees(np.sqrt(np.mean((mean.inv()*rot).magnitude()**2)))))


def fit_calibration(initial,camera_object_poses,reference,config):
    """Fit all six extrinsic parameters to the operator's fixed reference.

    No depth/axis-only correction and no hardware actions are applied. The
    manually chosen reference is a fitting target, not independent metrology.
    """
    initial=rigid_transform(initial);reference=rigid_transform(reference)
    poses=np.array([rigid_transform(p) for p in camera_object_poses]);N=len(poses)
    if N<config['min_samples']:raise ValueError('Wait for more fresh tracked frames before fitting')
    stability=pose_stability(poses)
    if stability['position_std_m']>config['max_position_std_m'] or stability['orientation_rms_deg']>config['max_orientation_rms_deg']:
        raise ValueError('Object tracking is moving or unstable; keep the cube still and retry')
    def residual(x):
        T=matrix(x);position=poses[:,:3,3]@T[:3,:3].T+T[:3,3]
        rotation=np.einsum('ij,njk->nik',T[:3,:3],poses[:,:3,:3])
        error=Rotation.from_matrix(np.einsum('ij,njk->nik',reference[:3,:3].T,rotation)).as_rotvec()
        return np.c_[(position-reference[:3,3])/.001,error/np.radians(1.)].ravel()
    before=residual(parameters(initial)).reshape(N,6)
    result=least_squares(residual,parameters(initial),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=200)
    if not result.success:raise ValueError('Calibration fit did not converge: '+result.message)
    T=matrix(result.x);translation=float(np.linalg.norm(T[:3,3]-initial[:3,3]));angle=float(np.degrees(Rotation.from_matrix(T[:3,:3]@initial[:3,:3].T).magnitude()))
    old_positions=poses[:,:3,3]@initial[:3,:3].T+initial[:3,3]
    new_positions=poses[:,:3,3]@T[:3,:3].T+T[:3,3]
    anchor_displacement=float(np.sqrt(np.mean(np.sum((new_positions-old_positions)**2,axis=1))))
    anchor_limit=config.get('max_anchor_displacement_m',config['max_translation_change_m'])
    if (translation>config['max_translation_change_m'] or angle>config['max_rotation_change_deg']
            or anchor_displacement>anchor_limit):
        raise ValueError(f'Alignment change is too large: cube {anchor_displacement*1000:.1f}/{anchor_limit*1000:g} mm, '
            f'rotation {angle:.2f}/{config["max_rotation_change_deg"]:g} deg, '
            f'camera origin {translation*1000:.1f}/{config["max_translation_change_m"]*1000:g} mm. '
            'Manual reference saved; verify its position and tracked face orientation.')
    after=residual(result.x).reshape(N,6)
    def metrics(r):return dict(position_rms_mm=float(np.sqrt(np.mean(np.sum(r[:,:3]**2,axis=1)))),orientation_rms_deg=float(np.sqrt(np.mean(np.sum(r[:,3:]**2,axis=1)))))
    return T,dict(samples=N,stability=stability,before=metrics(before),after=metrics(after),
        translation_change_world_m=(T[:3,3]-initial[:3,3]).tolist(),rotation_change_deg=angle,
        camera_origin_displacement_m=translation,anchor_displacement_m=anchor_displacement,
        optimizer='scipy least_squares, soft_l1, full 6-DOF extrinsic',
        reference_source='operator visual alignment in MuJoCo world',independent_accuracy_established=False,
        object_only_correction_applied=False,hardware_commands_generated=False)
