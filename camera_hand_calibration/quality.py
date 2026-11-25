"""Camera-session timing/depth QC; independent of robot hardware."""
import json
from pathlib import Path
import numpy as np
import cv2

from workspace import ROOT,local,read_json,write_json,identifier
from capture import load_frame


def inspect_session(session):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    session=local(session);meta=read_json(session/'session.json')
    timings=[json.loads(s) for s in (session/'timing.jsonl').read_text().splitlines()]
    profile=read_json(session/'camera_profile.json')
    out=local('results')/('camera_quality_'+session.name);out.mkdir(exist_ok=True)
    t=np.array([r['color_timestamp_ms'] for r in timings]);dt=np.diff(t)
    skew=np.array([r['color_depth_skew_ms'] for r in timings])
    source=read_json('configs/source_camera.json');actual=np.array(profile['color']['K'])
    result=dict(camera_serial=profile['serial'],received=meta['frames_received'],saved=meta['writer_frames'],
        requested_fps=meta['config']['camera']['fps'],median_fps=float(1000/np.median(dt)),
        interval_ms=dict(median=float(np.median(dt)),p95=float(np.percentile(dt,95)),max=float(dt.max())),
        color_depth_skew_ms=dict(median_abs=float(np.median(abs(skew))),p95_abs=float(np.percentile(abs(skew),95))),
        skipped_source_frames=meta['source_frames_skipped'],writer_loss=meta['frames_saved']-meta['writer_frames'],
        clock_bases=meta['clock_bases'],joint_subscription_enabled=meta['joints_enabled'],
        camera_intrinsics=actual.tolist(),intrinsics_change= (actual-np.asarray(source['camera_intrinsics'])).tolist(),
        pinhole_supported=profile['pinhole_supported'],extrinsics_verified=False,
        note='Camera-only stream/profile test; no joint/hardware/calibration accuracy claim.')
    frames=sorted(session.glob('frame_*.npz'))
    if frames:
        rgb,depth,_=load_frame(frames[0]);valid=depth>0
        lo,hi=np.percentile(depth[valid],[2,98]);view=cv2.applyColorMap(np.uint8(np.clip((depth-lo)/(hi-lo),0,1)*255),cv2.COLORMAP_TURBO)
        view[~valid]=0;cv2.imwrite(str(out/'depth_preview.png'),view)
        cv2.imwrite(str(out/'rgb_preview.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot((t[1:]-t[0])*.001,dt,'.',ms=3);axes[0].axhline(1000/result['requested_fps'],ls='--',c='black')
    axes[0].set(xlabel='Camera time (s)',ylabel='RGB interval (ms)')
    axes[1].plot((t-t[0])*.001,skew,'.',ms=3);axes[1].set(xlabel='Camera time (s)',ylabel='RGB minus depth timestamp (ms)')
    for ax in axes:ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(out/f'timing.{ext}',dpi=170)
    plt.close(fig)
    write_json(out/'report.json',result)
    (out/'index.html').write_text('<meta charset="utf-8"><pre>'+__import__('html').escape(json.dumps(result,indent=2))+'</pre><img src="rgb_preview.png"><img src="depth_preview.png"><img width="100%" src="timing.png">')
    return dict(result_directory=str(out.relative_to(ROOT)),**result)


def workspace_status():
    import html
    sessions=[]
    for path in sorted(local('datasets').glob('*/session.json')):
        session=read_json(path)
        sessions.append(dict(directory=str(path.parent.relative_to(ROOT)),status=session['status'],
            frames_received=session.get('frames_received'),frames_saved=session.get('writer_frames'),
            joint_subscription=session.get('joints_enabled',False),
            direct_motor_motion=session.get('motor_motion_authorized',False),
            torque_disabled_verified=session.get('torque_disabled_verified')))
    quality=[]
    for path in sorted(local('results').glob('camera_quality_*/report.json')):
        quality.append(dict(path=str(path.relative_to(ROOT)),**read_json(path)))
    fits=[]
    for path in sorted(local('results').glob('*/report.json')):
        item=read_json(path)
        if 'absolute_accuracy_established' in item:
            fits.append(dict(directory=str(path.parent.relative_to(ROOT)),status=item['status'],
                             synthetic_test_only=item.get('synthetic_test_only',False)))
    subscribed=any(s['joint_subscription'] for s in sessions)
    stationary=[]
    for path in sorted(local('results').glob('stationary_rgbd_*/comparison.json')):
        if not (path.parent/'index.html').exists():continue
        item=read_json(path)
        stationary.append(dict(directory=str(path.parent.relative_to(ROOT)),status=item['status'],
            relative_rotation_deg=item['relative_rotation_deg'],
            camera_center_difference_mm=item['camera_center_distance_mm'],absolute_accuracy_established=False))
    status=dict(camera_sessions=sessions,camera_quality=quality,fit_candidates=fits,
        stationary_rgbd_candidates=stationary,
        real_fit_candidates_present=bool(stationary) or any(not r['synthetic_test_only'] for r in fits),
        passive_joint_sessions_present=subscribed,direct_motor_access_implemented=(ROOT/'articulated_motion.py').exists(),
        hardware_phase=('Operator-authorized articulated sweeps recorded. See per-session torque verification.' if any(s['direct_motor_motion'] for s in sessions) else
                        'Hardware motion requires explicit operator authorization; camera-only capture is independent.'),
        production_configuration_modified=False)
    write_json('results/workspace_status.json',status)
    body='<meta charset="utf-8"><title>Hand camera calibration workspace</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Marker-free camera / hand calibration</h1><p><a href="README.md">Workflow and reproducibility instructions</a></p><p>Camera-only and operator-authorized articulated data are available. Candidate transforms require independent geometric cross-checks. Synthetic fits are tests, not real calibration.</p>'
    if local('results/rigid_surface_latest.json').exists():
        latest=read_json('results/rigid_surface_latest.json')['directory'];item=read_json(latest+'/comparison.json')
        body+=f'<h2>Latest articulated cross-check</h2><p><a href="{latest}/index.html">Report</a>: {html.escape(item["status"])}</p><img src="{latest}/cross_checks.png">'
    for fit in stationary:
        directory=fit['directory']
        body+=f'<h2>Stationary RGB-D candidate</h2><p><a href="{directory}/index.html">Comparison report</a> — {html.escape(fit["status"])}</p><img src="{directory}/residuals.png">'
    for q in quality:
        directory=str(Path(q['path']).parent)
        body+=f'<h2>{html.escape(directory)}</h2><p><a href="{directory}/index.html">Camera report</a></p><img src="{directory}/rgb_preview.png"><img src="{directory}/timing.png">'
    body+='<h2>Landmark catalog</h2><img src="results/landmarks_palm.png"><h2>Machine-readable status</h2><pre>'+html.escape(json.dumps(status,indent=2))+'</pre>'
    local('index.html').write_text(body)
    return status
