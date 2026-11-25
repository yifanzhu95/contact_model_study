"""Combine the latest 4+7+1 completed episodes and compare with the prior 12.

Recorded-data analysis only. Videos are linked/copied without reencoding.
No files are deleted and no hardware/simulation processes are started.
"""
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

os.environ.setdefault('MPLCONFIGDIR','/tmp/leap_analysis_matplotlib')
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
PREVIOUS=BASE/'combined_12_completed_20261121_030146_030714'
spec=importlib.util.spec_from_file_location('prior_completed_metrics',PREVIOUS/'analyze_and_group.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
np,plt=m.np,m.plt
ROOT=m.ROOT
NAMES=['leap_experiment_20261121_031802_821943','leap_experiment_20261121_032216_618091',
       'leap_experiment_20261121_032712_085446']
ANALYSIS=OUT/'analysis'
KEYS=['weights','hardware_config','joint_reference','planner_config','rollout_config','pose_transform',
      'success_thresholds','effective_ctrl_range','joint_limits','temporal_plan_blending','failure_z','fixed_goal_quat','execution']


def config_checks(a,b):
    result={k:a[k]==b[k] for k in KEYS}
    result.update({role+'_scene_equal_except_mesh_mime':m.normalized_xml(a['scenes'][role]['scene_xml'])==
        m.normalized_xml(b['scenes'][role]['scene_xml']) for role in ['rollout','eval']})
    return result


def main():
    ANALYSIS.mkdir(exist_ok=True);(OUT/'video').mkdir(exist_ok=True);(OUT/'source_runs').mkdir(exist_ok=True)
    docs=[json.loads((BASE/n/'controller.json').read_text()) for n in NAMES]
    equivalence={n:config_checks(docs[0],d) for n,d in zip(NAMES,docs)}
    assert all(all(checks.values()) for checks in equivalence.values()),equivalence
    previous=json.loads((PREVIOUS/'controller.json').read_text())
    comparison_checks=config_checks(docs[0],previous)
    assert all(comparison_checks.values()),comparison_checks
    selected=[];sources=[];excluded=[];videos=[]
    for i,(name,d) in enumerate(zip(NAMES,docs),1):
        source=BASE/name;target=OUT/'source_runs'/f'source_{i:02d}';target.mkdir(exist_ok=True)
        hashes={}
        for filename in ['controller.json','launch.json','summary.json','steps.csv','episodes.csv',
                         'bridge.log','controller.log','video.log','mujoco.log','preparation_trace.jsonl',
                         'episode_status.json','bridge_status.json','goal.json']:
            p=source/filename
            if p.is_file():shutil.copy2(p,target/filename);hashes[filename]=m.sha(p)
        recording=json.loads((source/'video/recording.json').read_text())
        m.write_json(target/'recording.original.json',recording)
        mapping={}
        for e in d['episodes']:
            if not e.get('completed'):
                excluded.append(dict(source_run=name,source_episode=e['episode']+1,outcome=e['finish_reason']));continue
            number=len(selected)+1
            assert e['finish_reason'] in ['success','failure','timeout']
            assert len(e['steps'])==e['n_steps'] and not e.get('failed_step')
            new=copy.deepcopy(e);new.update(episode=number-1,source_run=name,source_episode=e['episode']+1)
            selected.append(new);mapping[new['source_episode']]=number
            records=[r for r in recording['episodes'] if r['episode']==new['source_episode']]
            assert len(records)==1
            record=copy.deepcopy(records[0])
            assert record['outcome']==new['finish_reason'] and record['error'] is None
            assert record['started_at_utc']==new['started_at_utc'] and record['finished_at_utc']==new['finished_at_utc']
            files=[]
            for suffix in ['mkv','frames.csv','ffmpeg.log']:
                src=source/'video'/f"episode_{new['source_episode']:02d}.{suffix}"
                dst=OUT/'video'/f'episode_{number:02d}.{suffix}'
                if not dst.exists():
                    try:os.link(src,dst)
                    except OSError:shutil.copy2(src,dst)
                digest=m.sha(src);assert m.sha(dst)==digest
                files.append(dict(source=str(src.relative_to(ROOT)),path=str(dst.relative_to(OUT)),bytes=dst.stat().st_size,
                    sha256=digest,hardlinked=os.path.samefile(src,dst)))
            record.update(episode=number,source_episode=new['source_episode'],source_run=name,
                          video=f'episode_{number:02d}.mkv',timestamps=f'episode_{number:02d}.frames.csv')
            probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height',
                '-of','json',str(OUT/'video'/record['video'])],capture_output=True,text=True,check=True)
            probe=json.loads(probe.stdout);assert probe['streams']
            assert abs(float(probe['format']['duration'])-record['video_duration_s'])<.15
            videos.append(dict(episode=number,source_run=name,source_episode=new['source_episode'],recording=record,files=files,probe=probe))
        for record in recording['episodes']:
            if record['episode'] not in mapping:
                excluded.append(dict(source_run=name,source_episode=record['episode'],outcome=record['outcome'],
                    source='video recording metadata; absent from last completed controller checkpoint'))
        launch=json.loads((source/'launch.json').read_text())
        controller_log=(source/'controller.log').read_text();bridge_log=(source/'bridge.log').read_text()
        error=re.findall(r'Software failure: (.+?)\. MPPI stopped;',controller_log)
        sources.append(dict(run=name,archived_logs=str(target.relative_to(OUT)),selected_episodes=mapping,
            source_hashes=hashes,source_status=d['status'],launch_reason=launch['reason'],exit_codes=launch['exit_codes'],
            software_failure_reason=error[-1] if error else None,
            torque_off_verified='Torque disabled and verified on all 16 motors.' in bridge_log))
    assert len(selected)==12 and [len(s['selected_episodes']) for s in sources]==[4,7,1]
    assert len({(e['source_run'],e['source_episode']) for e in selected})==12
    aggregate=copy.deepcopy(docs[0]);aggregate.pop('summary',None)
    aggregate.update(record_type='completed_episode_aggregate',status='complete',episodes=selected,
        source_runs=NAMES,started_at_utc=selected[0]['started_at_utc'],finished_at_utc=selected[-1]['finished_at_utc'],
        wall_time_s=sum(e['manipulation_wall_time_s'] for e in selected),
        aggregation_note='Completed successes, failures and timeouts only. Do not add this view to source counts. Wall time excludes preparation, gates, resets, and aborted attempts.')
    aggregate['args'].update(n_episodes=12,results=str(OUT/'controller.json'))
    summary=m.metrics(selected,aggregate)
    prior_metrics=m.metrics(previous['episodes'],previous)
    rows=[];curves=[]
    for e in selected:
        row,curve=m.episode_analysis(e,aggregate);rows.append(row);curves.append(curve)
    aggregate['summary']=dict(requested_episodes=12,completed_episodes=12,successful_episodes=summary['successes'],
        unsuccessful_episodes=12-summary['successes'],success_rate_completed=summary['success_rate'],
        aborted_episodes=0,completed_steps=summary['actions'],wall_time_s=aggregate['wall_time_s'])
    m.write_json(OUT/'controller.json',aggregate);m.write_json(OUT/'summary.json',aggregate['summary'])
    m.write_csv(OUT/'episodes.csv',rows)
    m.write_csv(OUT/'steps.csv',[dict(episode=e['episode']+1,source_run=e['source_run'],source_episode=e['source_episode'],
        step=s['step'],action_timestamp_us=s['action_timestamp_us'],planning_time_s=s['planning_time'],
        pacing_wait_s=s.get('pacing_wait_s',0),position_error_m=s['goal_errors']['pos'],
        orientation_error_deg=float(np.degrees(s['quaternion_error_rad'])),
        estimated_joint_limit_clipped_joints=s['joint_limit_clipping']['estimated_clipped_joints']) for e in selected for s in e['steps']])
    m.write_json(OUT/'video/recording.json',dict(status='complete',mode='completed_episode_aggregate',
        episodes=[v['recording'] for v in videos],codec='mjpeg',container='matroska',fps=30,reencoding=False,
        total_video_duration_s=sum(v['recording']['video_duration_s'] for v in videos),
        queue_dropped_frames=sum(v['recording']['queue_dropped_frames'] for v in videos)))
    m.write_json(OUT/'sources.json',dict(sources=sources,configuration_equivalence=equivalence,
        excluded_episodes=excluded,videos=videos,created_at_utc=datetime.now(timezone.utc).isoformat(),
        note='Raw files and all existing videos preserved. Only selected videos are linked/copied here.'))
    orientation=[]
    prior_rows=[m.episode_analysis(e,previous)[0] for e in previous['episodes']]
    for rotation in ['-X','+X','-Y','+Y','-Z','+Z']:
        subset=[r for r in rows if r['rotation']==rotation]
        old=[r for r in prior_rows if r['rotation']==rotation]
        orientation.append(dict(rotation=rotation,episodes=len(subset),successes=sum(r['outcome']=='success' for r in subset),
            drops=sum(r['outcome']=='failure' for r in subset),timeouts=sum(r['outcome']=='timeout' for r in subset),
            previous_episodes=len(old),previous_successes=sum(r['outcome']=='success' for r in old),
            combined_episodes=len(old)+len(subset),combined_successes=sum(r['outcome']=='success' for r in old+subset)))
    current_usb=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),latency_timers_ms={
        str(p):int(p.read_text().strip()) for p in Path('/sys/bus/usb-serial/devices').glob('*/latency_timer')},
        note='Read-only current inspection; setting during the episodes was not explicitly snapshotted.')
    report=dict(summary=summary,previous12=prior_metrics,comparison_configuration_checks=comparison_checks,
        previous_aggregate=str(PREVIOUS.relative_to(ROOT)),configuration=dict(weights=aggregate['weights'],
            planner=aggregate['planner_config'],hardware_config=aggregate['hardware_config'],joint_limits=aggregate['joint_limits'],
            pose_transform=aggregate['pose_transform'],goal_pos_m=selected[0]['target_pos']),
        episodes=rows,starting_rotation_groups=orientation,sources=sources,excluded_episodes=excluded,current_usb_observation=current_usb,
        initially_off_center_episodes=[r['episode'] for r in rows if r['initial_position_error_mm']>20],
        successes=m.metrics([e for e in selected if e['success']],aggregate),
        failures=m.metrics([e for e in selected if not e['success']],aggregate),
        successful_final_angle_deg=m.stats([r['final_angle_deg'] for r in rows if r['outcome']=='success']),
        successful_final_position_mm=m.stats([r['final_position_error_mm'] for r in rows if r['outcome']=='success']),
        limitations=['Small unpaired samples, even when starting rotations match. A one-episode difference is inconclusive.',
            'Two later software-fault attempts appear in video/controller logs but not the last completed controller.json checkpoint; excluded.',
            'Clipping and target gaps are controller estimates. No bridge telemetry was recorded, so exact servo targets/current are unavailable.',
            'Clipping is step weighted; the 1000-step timeout contributes substantially. Episode-level and first-100-step metrics are also supplied.',
            'Episode 11 starts 38.3mm from the target; retained as completed and flagged. Other starts are about 6–9mm away.',
            'No hardware, simulator, planner, tracking restart, calibration edit, or parameter change was performed.'])
    first100=[dict(e,steps=e['steps'][:100]) for e in selected]
    report['first100_actions_clipping']=m.clip_stats([s for e in first100 for s in e['steps']],aggregate)
    report['median_episode_clipping_percent']={key:float(np.median([r[key] for r in rows])) for key in
        ['delta_clipped_percent','absolute_clipped_percent','either_clipped_percent']}
    m.write_json(ANALYSIS/'report.json',report);m.write_csv(ANALYSIS/'joint_clipping.csv',summary['clipping']['joints'])
    comparison_rows=[]
    getters=[('Success rate (%)',lambda s:s['success_rate']*100),('Successes',lambda s:s['successes']),
        ('Drops',lambda s:s['drops']),('Timeouts',lambda s:s['timeouts']),('Actions',lambda s:s['actions']),
        ('Successful duration mean (s)',lambda s:s['successful_episode_duration_s']['mean']),
        ('Successful duration median (s)',lambda s:s['successful_episode_duration_s']['median']),
        ('Planning mean (ms)',lambda s:s['planning_ms']['mean']),('Planning p95 (ms)',lambda s:s['planning_ms']['p95']),
        ('Planning max (ms)',lambda s:s['planning_ms']['max']),('Control frequency (Hz)',lambda s:s['publication_rate_hz']),
        ('Control frequency min (Hz)',lambda s:s['instantaneous_frequency_hz']['min']),
        ('Control frequency max (Hz)',lambda s:s['instantaneous_frequency_hz']['max']),
        ('Planning exceeded 64ms (%)',lambda s:s['planning_exceeded_64ms_percent']),
        ('Pacing waits (%)',lambda s:100*s['pacing_wait_steps']/s['actions']),
        ('Command joint age (ms)',lambda s:s['command_joint_age_ms']['mean']),
        ('Command object age (ms)',lambda s:s['command_object_age_ms']['mean']),
        ('Delta clipping (%)',lambda s:s['clipping']['delta_clipped_percent']),
        ('Absolute limit clipping (%)',lambda s:s['clipping']['absolute_clipped_percent']),
        ('Either clipping (%)',lambda s:s['clipping']['either_clipped_percent']),
        ('Estimated max joint target change p95 (deg)',lambda s:s['estimated_target_change_max_joint_deg']['p95']),
        ('Estimated target tracking gap mean (deg)',lambda s:s['estimated_previous_target_tracking_gap_deg']['mean'])]
    for label,get in getters:comparison_rows.append(dict(metric=label,previous12=get(prior_metrics),latest12=get(summary)))
    m.write_csv(ANALYSIS/'comparison.csv',comparison_rows)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,constrained_layout=True)
    colors=plt.get_cmap('tab20').colors
    for row,c in zip(rows,curves):
        axes[0].plot(c['time_s'],c['angle_deg'],color=colors[row['episode']-1],label=f"{row['episode']}: {row['outcome']}")
        axes[1].plot(c['time_s'],c['position_mm'],color=colors[row['episode']-1])
    angle_threshold=np.degrees(2*np.arcsin(np.sqrt(aggregate['success_thresholds']['quat'])))
    axes[0].axhline(angle_threshold,color='black',ls='--');axes[1].axhline(20,color='black',ls='--')
    axes[0].set(ylabel='Orientation error (deg)',title='Latest 12 completed M2 episodes — 4 + 7 + 1')
    axes[1].set(ylabel='Position error (mm)',xlabel='Manipulation time (s)',ylim=(0,100))
    axes[0].legend(ncol=4,fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(ANALYSIS/f'episode_errors.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    groups=[prior_metrics,summary];labels=['Previous 12','Latest 12']
    axes[0].bar(labels,[s['success_rate']*100 for s in groups],color=['#888888','#367ca0']);axes[0].set(ylabel='Success (%)',ylim=(0,100))
    axes[1].bar(labels,[s['command_joint_age_ms']['mean'] for s in groups],color=['#888888','#367ca0']);axes[1].set_ylabel('Joint age at command (ms)')
    axes[2].bar(labels,[s['publication_rate_hz'] for s in groups],color=['#888888','#367ca0']);axes[2].set(ylabel='Effective control rate (Hz)',ylim=(0,18))
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(ANALYSIS/'comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    x=np.arange(6);ss=np.array([r['successes'] for r in orientation]);dd=np.array([r['drops'] for r in orientation]);tt=np.array([r['timeouts'] for r in orientation])
    ax.bar(x,ss,color='#418a66',label='Success');ax.bar(x,dd,bottom=ss,color='#be6958',label='Drop');ax.bar(x,tt,bottom=ss+dd,color='#c9a640',label='Timeout')
    ax.set(xticks=x,xticklabels=[r['rotation'] for r in orientation],ylabel='Episodes',title='Starting rotation distribution',yticks=range(4));ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(ANALYSIS/'starting_rotation.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
    for row,c in zip(rows,curves):
        for j,ax in enumerate(axes):
            ax.plot(c['time_s'],np.array(c['position_axes_mm'])[:,j],color=colors[row['episode']-1],
                label=f"{row['episode']}: {row['outcome']}")
    for j,ax in enumerate(axes):
        ax.axhline(20,color='black',ls='--',alpha=.5);ax.axhline(-20,color='black',ls='--',alpha=.5)
        ax.axhline(0,color='grey',lw=.7);ax.set(ylabel=f"{'XYZ'[j]} error (mm)",ylim=(-100,100));ax.grid(alpha=.2)
    axes[0].set_title('Object displacement by axis — +/-20mm on each axis is necessary, not sufficient, for position success')
    axes[0].legend(ncol=4,fontsize=8);axes[-1].set_xlabel('Manipulation time (s)')
    fig.savefig(ANALYSIS/'position_axes.png',dpi=170);plt.close(fig)
    body=['<!doctype html><meta charset="utf-8"><title>Latest M2 twelve completed episodes</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}img{max-width:100%}</style>',
        '<h1>M2 — twelve completed episodes (4 + 7 + 1)</h1>',
        f"<p>{summary['successes']}/12 successes; {summary['drops']} drops; {summary['timeouts']} timeout. Two later software-fault attempts excluded. Raw logs and all videos preserved.</p>",
        '<p>Saved configuration matches the prior twelve episodes: paced M2, 512 samples, 0.256s horizon, 0.064s control step, temperature 0.1, noise 0.05, shifted warm start; action-rate weight 10, blending alpha 0.75, measured-limit cost 100, +/-0.2 hardware clipping followed by absolute limits; w_pos_y=1 and w_contact=80. Current USB latency timer is 1ms; joint-feedback age is measured separately in the logs.</p>',
        '<p><a href="controller.json">Combined controller log</a> · <a href="episodes.csv">Episode metrics</a> · <a href="sources.json">Sources/video checksums</a> · <a href="analysis/report.json">Full analysis</a> · <a href="analysis/comparison.csv">Comparison CSV</a></p>',
        '<table><tr><th>Metric</th><th>Previous 12</th><th>Latest 12</th></tr>']
    for r in comparison_rows:body.append(f"<tr><td>{r['metric']}</td><td>{r['previous12']:.2f}</td><td>{r['latest12']:.2f}</td></tr>")
    body+=['</table><h2>Episodes</h2><table><tr><th>Episode</th><th>Rotation</th><th>Outcome</th><th>Steps</th><th>Duration</th><th>Final angle</th><th>Final position</th><th>Video</th></tr>']
    for r in rows:body.append(f"<tr><td>{r['episode']}</td><td>{r['rotation']}</td><td>{r['outcome']}</td><td>{r['steps']}</td><td>{r['duration_s']:.2f}s</td><td>{r['final_angle_deg']:.1f}deg</td><td>{r['final_position_error_mm']:.1f}mm</td><td><a href=\"video/episode_{r['episode']:02d}.mkv\">Video {r['episode']:02d}</a></td></tr>")
    body+=['</table><p>The latest result does not establish a success improvement: 6/12 versus 7/12 previously. Joint-feedback age improved markedly. Across both batches, +Z has 0/4 successes while -Z has 4/4. Latest episodes 3, 7 and 10 reached the orientation threshold but drifted in +Y before falling; episode 1 timed out with position near goal but orientation still about 59 degrees. Episode 11 started 38.3mm off-center and is retained.</p>',
        '<p>Two software-fault attempts after the completed episodes were excluded: stale planned state in source 1 episode 5, and planning-feedback synchronization deadline in source 2 episode 8. All source runs verified torque-off. No motor-current fault was logged.</p>',
        '<img src="analysis/comparison.png"><img src="analysis/starting_rotation.png"><img src="analysis/episode_errors.png"><img src="analysis/position_axes.png">']
    (OUT/'index.html').write_text('\n'.join(body))
    print(json.dumps(dict(out=str(OUT),comparison=comparison_rows,episodes=rows,rotation=orientation),indent=2))


if __name__=='__main__':main()
