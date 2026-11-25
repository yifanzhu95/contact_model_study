#!/usr/bin/env python3
"""Isolated CPU MuJoCo screen for the loaded-hand sine-response experiment.

No serial devices, LCM instances or hardware controller are created. Production
XML files are read only. Historical object poses approximate operator placement;
this screen cannot certify retention at a new, unmeasured hardware placement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from measure_leap_target_response import DEFAULT_CONFIG,DEFAULT_RUN,MOVING_JOINTS,measured_period,sequence
from ContactModelStudy.Hardware.Hand import control_limits

ROOT=Path(__file__).resolve().parent


def excitation_against_control(control,candidate,min_translation_mm=2.,min_rotation_deg=3.):
    """Require induced motion, rather than counting shared settling/creep."""
    a,b=np.asarray(control['trace']),np.asarray(candidate['trace'])
    if a.shape!=b.shape or not len(a) or not np.allclose(a[:,0],b[:,0]):
        return dict(pass_excitation=False,reason='No complete time-matched fixed-target control')
    translation=np.linalg.norm(a[:,1:4]-b[:,1:4],axis=1)*1000
    rotation=2*np.arccos(np.clip(abs(np.sum(a[:,4:8]*b[:,4:8],axis=1)),0,1))*180/np.pi
    return dict(pass_excitation=bool(translation.max()>=min_translation_mm or rotation.max()>=min_rotation_deg),
        max_induced_translation_mm=float(translation.max()),max_induced_rotation_deg=float(rotation.max()))


def scene():
    model=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/leap/hardware/cube.xml'))
    obj=model.body('obj').id
    # Current calibrated high-accuracy hand and 70 g, 70 mm cube, with the
    # same grouped friction overrides as the manipulation baseline.
    for index in range(model.ngeom):
        if not(model.geom_contype[index] or model.geom_conaffinity[index]):continue
        body=model.body(int(model.geom_bodyid[index])).name;name=model.geom(index).name
        sliding=.3 if body=='obj' else .5 if name in ('if_tip','mf_tip','rf_tip','th_tip') else .4
        model.geom_friction[index]=[sliding,.005,.0001]
    model.opt.gravity[:]=[0,0,-9.81]
    assert abs(model.body_mass[obj]-.07)<1e-10
    assert (model.nq,model.nv,model.nu)==(23,22,16)
    return model


def object_penetration(model,data):
    obj=model.body('obj').id
    distances=[data.contact[i].dist for i in range(data.ncon)
        if obj in (model.geom_bodyid[data.contact[i].geom1],model.geom_bodyid[data.contact[i].geom2])]
    return max(0.,-min(distances,default=0.))


def seeds():
    values=[]
    for name in ['leap_experiment_20261116_215909_720053','leap_experiment_20261116_215155_209099']:
        source=ROOT/'results/diagnostics/m2_tracking_baseline'/name/'controller.json'
        report=json.loads(source.read_text())
        for e in report['episodes']:
            if e['steps']:
                q=e.get('q_start',e['steps'][0]['q'])
                values.append(dict(name=f'{name}_e{e["episode"]+1}',object_qpos=q[16:23],source=str(source)))
    values.append(dict(name='canonical',object_qpos=[.02,.035,.085,1,0,0,0],source='canonical scene placement'))
    return values


def settle(model,reference,pose,max_lift=.025):
    data=mujoco.MjData(model)
    data.qpos[:16]=reference;data.qpos[16:]=pose;data.ctrl[:]=reference;data.act[:]=reference
    mujoco.mj_forward(model,data)
    original_penetration=object_penetration(model,data);lift=0.
    # Start a free gravity placement above colliders rather than injecting the
    # several-mm initial overlaps present in some historical tracker poses.
    while object_penetration(model,data)>.0005 and lift<max_lift:
        data.qpos[18]+=.0005;lift+=.0005;mujoco.mj_forward(model,data)
    release_penetration=object_penetration(model,data)
    positions=[]
    for i in range(round(1.5/model.opt.timestep)):
        mujoco.mj_step(model,data)
        if i>round(1.2/model.opt.timestep):positions.append(data.qpos[16:19].copy())
        if not np.isfinite(data.qpos).all() or data.qpos[18]<.04:
            return None,dict(retained=False,reason='dropped during baseline placement',initial_penetration_m=original_penetration,
                lift_m=lift,release_penetration_m=release_penetration)
    movement=float(np.max(np.linalg.norm(np.array(positions)-positions[-1],axis=1)))
    stable=data.qpos[18]>.055 and np.linalg.norm(data.qvel[16:19])<.025 and movement<.002
    return (data if stable else None),dict(retained=bool(stable),initial_penetration_m=original_penetration,
        lift_m=lift,release_penetration_m=release_penetration,settled_penetration_m=object_penetration(model,data),
        settled_object_qpos=data.qpos[16:].tolist(),settled_translation_speed_m_s=float(np.linalg.norm(data.qvel[16:19])),last_300ms_motion_m=movement)


def trial(model,baseline,reference,amplitude,samples,cycles,period):
    data=mujoco.MjData(model)
    data.qpos[:]=baseline.qpos;data.qvel[:]=baseline.qvel;data.act[:]=baseline.act;data.ctrl[:]=reference
    mujoco.mj_forward(model,data)
    origin=data.qpos[16:19].copy();rotation=data.qpos[19:23].copy()
    elapsed=0.;max_displacement=0.;min_z=float(origin[2]);max_rotation=0.;dropped=False;trace=[]
    for joint in MOVING_JOINTS:
        # Same slow sampled sine cycle and repeated fast sine cycles that will
        # be sent on hardware. All other finger targets remain at reference.
        for mode,count,hold in [('settled',1,1.),('fast',cycles,period)]:
            targets=sequence(reference,joint,amplitude,count,closing_only=True,waveform='sine',sine_samples=samples)
            for target in targets:
                data.ctrl[:]=target
                end=data.time+hold
                while data.time<end-model.opt.timestep/2:
                    mujoco.mj_step(model,data)
                    displacement=float(np.linalg.norm(data.qpos[16:19]-origin))
                    angle=float(2*np.arccos(np.clip(abs(np.dot(data.qpos[19:23],rotation)),0,1)))
                    max_displacement=max(max_displacement,displacement);min_z=min(min_z,float(data.qpos[18]));max_rotation=max(max_rotation,angle)
                    if data.qpos[18]<.04 or not np.isfinite(data.qpos).all():dropped=True;break
                trace.append([float(data.time),*data.qpos[16:23]])
                if dropped:break
            if dropped:break
            data.ctrl[:]=reference
            for _ in range(round(1./model.opt.timestep)):
                mujoco.mj_step(model,data)
                max_displacement=max(max_displacement,float(np.linalg.norm(data.qpos[16:19]-origin)))
                min_z=min(min_z,float(data.qpos[18]))
                if data.qpos[18]<.04:dropped=True;break
            if dropped:break
        if dropped:break
    final_displacement=float(np.linalg.norm(data.qpos[16:19]-origin))
    # Displacement limits are conservative screening criteria, not changes to
    # the manipulation task's cost or success thresholds.
    acceptable=not dropped and max_displacement<.012 and final_displacement<.008 and min_z>origin[2]-.012
    return dict(dropped=dropped,conservative_pass=bool(acceptable),
        max_translation_m=max_displacement,final_translation_m=final_displacement,
        min_center_z_m=min_z,max_rotation_deg=float(np.degrees(max_rotation)),
        simulated_s=float(data.time),trace=trace)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--amplitudes',type=float,nargs='+',default=[.04,.06,.08,.1])
    parser.add_argument('--samples',type=int,default=16)
    parser.add_argument('--cycles',type=int,default=3)
    parser.add_argument('--max-placement-lift',type=float,default=.025,
        help='optional vertical release before baseline settling; 0 keeps the recorded placement')
    parser.add_argument('--min-induced-translation-mm',type=float,default=2.)
    parser.add_argument('--min-induced-rotation-deg',type=float,default=3.)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    model=scene();period=measured_period(DEFAULT_RUN)
    config=json.loads(DEFAULT_CONFIG.read_text());reference=np.clip(config['robot_qref'],*control_limits(model))
    baseline=[];start=time.monotonic()
    for seed in seeds():
        data,status=settle(model,reference,seed['object_qpos'],args.max_placement_lift)
        baseline.append((seed,data,status));print('BASELINE',seed['name'],json.dumps(status),flush=True)
    accepted=[b for b in baseline if b[1] is not None]
    results=[]
    for amplitude in [0.]+args.amplitudes:
        for seed,data,status in accepted:
            # 0-amplitude control uses tiny epsilon for the shared waveform
            # validator, numerically indistinguishable from a fixed target.
            result=trial(model,data,reference,max(amplitude,1e-12),args.samples,args.cycles,period)
            result.update(amplitude_rad=amplitude,seed=seed['name'])
            results.append(result)
            print('TRIAL',json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
        (args.output/'progress.json').write_text(json.dumps(dict(completed=len(results),elapsed_s=time.monotonic()-start))+'\n')
    summaries=[]
    controls={r['seed']:r for r in results if r['amplitude_rad']==0}
    for amplitude in [0.]+args.amplitudes:
        rows=[r for r in results if r['amplitude_rad']==amplitude]
        for row in rows:
            row['excitation']=excitation_against_control(controls[row['seed']],row,
                args.min_induced_translation_mm,args.min_induced_rotation_deg)
        summaries.append(dict(amplitude_rad=amplitude,scenarios=len(rows),drops=sum(r['dropped'] for r in rows),
            conservative_passes=sum(r['conservative_pass'] for r in rows),
            excitation_passes=sum(r['excitation']['pass_excitation'] for r in rows),
            max_translation_mm=max((r['max_translation_m']*1000 for r in rows),default=None),
            max_rotation_deg=max((r['max_rotation_deg'] for r in rows),default=None)))
    passing=[r['amplitude_rad'] for r in summaries if r['amplitude_rad']>0 and r['scenarios']==len(baseline)
        and r['conservative_passes']==r['scenarios'] and r['drops']==0 and r['excitation_passes']==r['scenarios']]
    # Prefer 0.08 rad or less even if 0.10 rad survives; leave amplitude margin.
    choice=max([a for a in passing if a<=.08],default=None)
    report=dict(scope=__doc__,physics=dict(scene='scenes/leap/hardware/cube.xml',calibrated_hand=True,
        cube_mass_kg=.07,sliding_friction=dict(cube=.3,fingertip=.5,palm=.4,finger=.4),
        timestep_s=float(model.opt.timestep),gravity=model.opt.gravity.tolist(),
        geometry='calibrated hand high-accuracy collisions; 70 mm box cube'),
        command_period_s=period,update_rate_hz=1/period,sine_samples=args.samples,
        selection_criteria_version='retention_and_induced_excitation_v2',
        minimum_induced_translation_mm=args.min_induced_translation_mm,
        minimum_induced_rotation_deg=args.min_induced_rotation_deg,
        sine_frequency_hz=1/(period*args.samples),cycles=args.cycles,closing_only=True,
        reference=reference.tolist(),max_placement_lift_m=args.max_placement_lift,
        baselines=[dict(**s,status=t) for s,d,t in baseline],
        amplitudes=summaries,elapsed_wall_s=time.monotonic()-start,
        hardware_candidate=dict(approved=choice is not None,amplitude_rad=choice,
            waveform='sine',sine_samples=args.samples,cycles=args.cycles,closing_only=True),
        caveats=['Historical object poses approximate a future placement; the present cube pose was not tracked.',
            'Optional vertical release is followed by 1.5 s free gravity placement; clearance is not guaranteed by a vertical lift. Initial and settled penetration are recorded.',
            'Passing a finite simulation screen cannot guarantee no drop on hardware.'],results=results)
    (args.output/'screen.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SUMMARY',json.dumps(summaries),flush=True);print('CANDIDATE',choice,flush=True)


if __name__=='__main__':main()
