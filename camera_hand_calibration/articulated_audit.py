#!/usr/bin/env python3
"""Offline acquisition/torque/timing audit; never opens a hardware device."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import argparse
import json
import numpy as np
from articulated_fit import load_metadata
from geometry import residual_stats
from workspace import local,read_json,write_json,digest


def audit(session,inspection,result=None):
    folder=local(session);s=read_json(folder/'session.json');post=read_json(inspection)
    records=[json.loads(line) for line in (folder/'joints.jsonl').read_text().splitlines()]
    t=np.array([r['timestamp_epoch_s'] for r in records]);q=np.array([r['q'] for r in records]);target=np.array([r['command'] for r in records])
    dt=np.diff(t);speed=np.diff(target,axis=0)/dt[:,None]
    rows=load_metadata(folder);static=[r for r in rows if r['meta']['phase']=='capture'];good=[r for r in static if r['meta']['joints']['usable']]
    skew=[r['meta']['joints']['skew_ms'] for r in good]
    manifest=read_json('model/manifest.json')
    source_checks={}
    for path,key in [(ROOT.parent/'configs/leap_pose_transform.json','camera_source_sha256'),(ROOT.parent/'configs/leap_hardware.json','hardware_source_sha256'),(Path(manifest['source_xml']),'source_xml_sha256')]:
        source_checks[str(path)]=dict(current_sha256=digest(path),snapshot_sha256=manifest[key],matches_snapshot=digest(path)==manifest[key])
    result_data=dict(session=session,status=s['status'],duration_s=float(t[-1]-t[0]),
        planned_waypoints=len(s['plan']['waypoints']),completed_waypoints=len(s['waypoints']),
        rgbd_frames=len(rows),writer_frames=s['writer_frames'],writer_loss=s['frames_saved']-s['writer_frames'],
        source_frames_skipped=s['source_frames_skipped'],settled_capture_frames=len(static),usable_settled_frames=len(good),
        joint_samples=len(records),joint_interval_ms=residual_stats(dt*1000),joint_camera_skew_ms=residual_stats(skew),
        planned_target_speed_limit_rad_s=s['max_speed_rad_s'],planned_target_acceleration_limit_rad_s2=s['max_acceleration_rad_s2'],
        observed_target_finite_difference_peak_rad_s=float(np.max(abs(speed))),
        observed_joint_peak_velocity_rad_s=float(np.max(np.abs([r['v'] for r in records]))),
        maximum_abs_current_raw=float(np.max(np.abs([r['current_raw'] for r in records]))),
        observed_minimum_modeled_clearance_mm=float(min(r['minimum_clearance_m'] for r in records)*1000),
        torque_off_verified_at_sweep_end=s['torque_disabled_verified'],postcheck=inspection,
        postcheck_torque_by_motor=post['torque_by_native_motor'],postcheck_fault_by_motor=post['hardware_error_by_native_motor'],
        postcheck_temperature_c=post['temperature_by_native_motor'],single_register_read_retries=s['single_register_read_retries'],
        source_checks=source_checks,
        note='Minimum clearance is from collision geometry, not a physical gap measurement. Encoder peak velocity differs from the smooth commanded target velocity. Source frame gaps are recorded, not writer losses. No external camera accuracy claim.')
    out=local(result) if result else local('results')/('audit_'+folder.name);out.mkdir(exist_ok=True)
    write_json(out/'acquisition_audit.json',result_data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained');time=t-t[0]
    for j,name,color in [(0,'index MCP','tab:blue'),(4,'middle MCP','tab:orange'),(8,'ring MCP','tab:green')]:
        axes[0].plot(time,target[:,j],ls='--',color=color,lw=1,label=name+' target');axes[0].plot(time,q[:,j],color=color,lw=.8,label=name+' measured')
    axes[0].set_ylabel('Joint angle (rad)');axes[0].legend(ncol=3,fontsize=8)
    axes[1].plot(time,[r['minimum_clearance_m']*1000 for r in records]);axes[1].axhline(s['plan']['runtime_abort_margin_m']*1000,color='red',ls='--');axes[1].set_ylabel('Modeled clearance (mm)')
    axes[2].plot(time[1:],dt*1000,'.',ms=2);axes[2].set(xlabel='Elapsed acquisition time (s)',ylabel='Joint sample interval (ms)')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'acquisition_audit.png',dpi=170);fig.savefig(out/'acquisition_audit.pdf');plt.close(fig)
    print(json.dumps(result_data,indent=2))
    return result_data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--inspection',required=True);p.add_argument('--result');a=p.parse_args();audit(a.session,a.inspection,a.result)
