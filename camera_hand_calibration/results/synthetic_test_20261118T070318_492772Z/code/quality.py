"""Camera-session timing/depth QC; independent of robot hardware."""
import json
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
