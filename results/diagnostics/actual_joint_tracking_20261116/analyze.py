"""Offline bridge-held-target tracking analysis. No hardware/LCM/physics calls."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
RUNS=['leap_experiment_20261116_215155_209099','leap_experiment_20261116_215909_720053']
THRESHOLDS=[.02,.05,.1,.2]


def stats(values):
    a=np.asarray(values,dtype=float).ravel()
    if not a.size:
        return None
    return dict(count=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
        p95=float(np.percentile(a,95)),max=float(a.max()),rms=float(np.sqrt(np.mean(a*a))))


def errors_summary(a):
    a=np.abs(np.asarray(a,dtype=float)).reshape(-1,16)
    return dict(frames=len(a),joint_samples=int(a.size),absolute_error_rad=stats(a),
        absolute_error_deg=stats(np.degrees(a)),maximum_joint_error_per_frame_deg=stats(np.degrees(a.max(axis=1))) if len(a) else None,
        thresholds={str(t):dict(joint_fraction=float(np.mean(a>t)) if a.size else None,
            frame_any_joint_fraction=float(np.mean(a.max(axis=1)>t)) if a.size else None,
            joint_count=int(np.sum(a>t)),frame_count=int(np.sum(np.any(a>t,axis=1)))) for t in THRESHOLDS})


def write_csv(name,rows):
    if rows:
        with (OUT/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    reports=[];joint_rows=[];episode_rows=[];sample_rows=[];plot_data={}
    for name in RUNS:
        directory=ROOT/'results/diagnostics/m2_tracking_baseline'/name
        source=directory/'controller.json';controller=json.loads(source.read_text())
        events=[json.loads(line) for line in (directory/'bridge_telemetry.jsonl').read_text().splitlines()]
        cycles=[r for r in events if r.get('event')=='cycle']
        published={s['action_timestamp_us']:(ep,s) for ep in controller['episodes'] for s in ep['steps'] if s.get('action_timestamp_us')}
        written={r['written_action_timestamp_us']:r for r in cycles if r.get('write_succeeded') and r.get('new_action') and r.get('command_type')=='delta'}
        # Verify that the target logged BEFORE every read equals the previous
        # SDK write. Never compare a joint read to a target first written AFTER it.
        previous=None;verified=0
        for r in cycles:
            if previous is not None:
                np.testing.assert_array_equal(r['held_target_before_read'],previous['written_target'])
                assert r['held_action_timestamp_us']==previous['written_action_timestamp_us']
                verified+=1
            if r.get('write_succeeded'):
                previous=r
        records=[]
        for r in cycles:
            identity=r.get('held_action_timestamp_us')
            if (r.get('held_command_type')!='delta' or identity not in published
                    or r.get('held_target_since_us') is None or 'q' not in r):
                continue
            ep,step=published[identity]
            if r['read_started_us']>datetime.fromisoformat(ep['finished_at_utc']).timestamp()*1e6:
                continue
            age=(r['read_started_us']-r['held_target_since_us'])*1e-6
            assert age>=0
            error=np.asarray(r['q'])-r['held_target_before_read']
            records.append(dict(trace=r,episode=ep['episode']+1,completed=ep['completed'],age=age,error=error))
            sample_rows.append(dict(run=name,episode=ep['episode']+1,completed=ep['completed'],
                cycle=r['cycle'],read_started_us=r['read_started_us'],held_action_timestamp_us=identity,
                unchanged_target_age_at_read_start_ms=age*1000,mean_abs_error_rad=float(np.abs(error).mean()),
                max_abs_error_rad=float(np.abs(error).max()),joint_count_over_005=int(np.sum(abs(error)>.05)),
                joint_count_over_010=int(np.sum(abs(error)>.1))))
        a=np.array([r['error'] for r in records]);age=np.array([r['age'] for r in records])
        completed=np.array([r['completed'] for r in records])
        groups=dict(all_manipulation=errors_summary(a),completed_episodes_only=errors_summary(a[completed]),
            aborted_episode_only=errors_summary(a[~completed]),
            target_unchanged_at_least_50ms=errors_summary(a[age>=.05]))
        for i,joint in enumerate(controller['joint_order']):
            q=np.abs(a[:,i]);joint_rows.append(dict(run=name,joint=joint,frames=len(q),
                mean_abs_error_deg=float(np.degrees(q.mean())),p95_abs_error_deg=float(np.degrees(np.percentile(q,95))),
                max_abs_error_deg=float(np.degrees(q.max())),fraction_over_005=float(np.mean(q>.05)),fraction_over_010=float(np.mean(q>.1))))
        for ep in controller['episodes']:
            subset=[r for r in records if r['episode']==ep['episode']+1]
            data=errors_summary([r['error'] for r in subset])
            episode_rows.append(dict(run=name,episode=ep['episode']+1,completed=ep['completed'],outcome=ep['finish_reason'],
                actions=ep['n_steps'],tracking_frames=data['frames'],mean_abs_error_deg=data['absolute_error_deg']['mean'] if data['frames'] else None,
                p95_abs_error_deg=data['absolute_error_deg']['p95'] if data['frames'] else None,
                joint_fraction_over_010=data['thresholds']['0.1']['joint_fraction'],frame_any_joint_fraction_over_010=data['thresholds']['0.1']['frame_any_joint_fraction']))
        fault=next(r for r in cycles if r.get('error'))
        current=np.asarray(fault['current_raw_native']);motor=int(np.argmax(np.abs(current)))
        # Native LEAP IDs swap the first two joints in each of the three fingers.
        native_to_mujoco=[1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15]
        j=native_to_mujoco[motor]
        ep=next(ep for ep in controller['episodes'] if ep.get('failed_step'));failure=ep['failed_step']
        failed_us=failure['feedback']['robot_timestamp']+failure['rejection']['robot_age_s']*1e6
        target_age_bins=[]
        for low,high in [(0,.025),(.025,.05),(.05,.075),(.075,.1),(.1,float('inf'))]:
            target_age_bins.append(dict(lower_ms=low*1000,upper_ms=high*1000 if np.isfinite(high) else None,
                **errors_summary(a[(age>=low)&(age<high)])))
        time=[r['state_timestamp_us'] for r in cycles if 'state_timestamp_us' in r]
        report=dict(run=name,controller_source=str(source.relative_to(ROOT)),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),summary=controller['summary'],
            comparison_verified_frames=verified,groups=groups,target_age_ms=stats(age*1000),target_age_bins=target_age_bins,
            published_actions=len(published),written_actions_matched=len(published.keys()&written.keys()),
            unapplied_published_actions=[dict(action_timestamp_us=k,episode=published[k][0]['episode']+1,
                step=published[k][1]['step'],published_after_fault=k>=fault['read_finished_us']) for k in sorted(published.keys()-written.keys())],
            publication_to_first_write_ms=stats([(written[k]['write_started_us']-k)*.001 for k in published.keys()&written.keys()]),
            bridge_state_interval_ms=stats(np.diff(time)*.001),
            serial_read_ms=stats([r['read_duration_s']*1000 for r in cycles if 'read_duration_s' in r]),
            health_transaction_ms=stats([r['health_duration_s']*1000 for r in cycles if 'health_duration_s' in r]),
            planning_ms=stats([s['planning_time']*1000 for ep in controller['episodes'] for s in ep['steps']]),
            recorder_status=json.loads((directory/'bridge_telemetry.status.json').read_text()),
            fault=dict(primary=fault['error'],motor_id=motor,joint=controller['joint_order'][j],
                current_raw=float(current[motor]),limit_raw=controller['hardware_config']['max_current_raw'],
                read_finished_us=fault['read_finished_us'],controller_rejection_us=failed_us,
                controller_rejection_ms_after_current_fault=(failed_us-fault['read_finished_us'])*.001,
                last_written_target_rad=fault['held_target_before_read'][j],measured_q_rad=fault['q'][j],
                target_error_deg=float(np.degrees(fault['held_target_before_read'][j]-fault['q'][j])),
                measured_velocity_rad_s=fault['qvel'][j],
                torque_off_verified='Torque disabled and verified on all 16 motors.' in (directory/'bridge.log').read_text()))
        reports.append(report);plot_data[name]=(a,age,cycles,report)
    report=dict(runs=reports,method=dict(
        measured_error='Measured q minus held_target_before_read from the same bridge cycle. Target verified against the preceding successful SDK write.',
        sample_selection='Only reads with a previously applied MPPI delta action ID joined to a controller step. Preparation, operator holds and zero return excluded; faulting read retained. Aborted manipulation reported separately.',
        target_age='Read-start timestamp minus the most recent write that changed target values. >=50 ms ensures the same target was present for the whole read after at least 50 ms.',
        thresholds='Descriptive diagnostic thresholds, not controller pass/fail settings.',
        caveats=['These are dynamic position errors while targets change and contacts apply load, not settled servo accuracy or a diagnosis of motor incapability.',
            'The SDK returning from a write is not per-motor target-register acknowledgement. There is no added serial readback.',
            'Joint data are acquired during each approximately 16 ms serial read; exact per-motor sample instants are not logged.',
            'Fractions count observed joint samples/frames; longer episodes contribute more samples. They are not episode success probabilities.',
            '50 ms is not a settling guarantee. Holding force may require nonzero position error.',
            'Recorder health and error chronology identify the immediate shutdown trigger. No matched hardware diagnostics-off trial exists here, so an indirect effect on trajectories cannot be causally excluded.']))
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_csv('joints.csv',joint_rows);write_csv('episodes.csv',episode_rows);write_csv('samples.csv',sample_rows)
    plots(plot_data,joint_rows)
    page(report,episode_rows,joint_rows)
    for r in reports:
        print(r['run'],json.dumps(dict(groups=r['groups'],fault=r['fault'],transport=r['publication_to_first_write_ms'],matched=r['written_actions_matched'],published=r['published_actions'])))


def plots(data,joints):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    latest=RUNS[-1];a,age,cycles,report=data[latest]
    fig,ax=plt.subplots(figsize=(7.5,4.5),constrained_layout=True)
    for label,mask in [('All manipulation reads',np.ones(len(a),dtype=bool)),('Same target for at least 50 ms',age>=.05)]:
        x=np.sort(np.degrees(np.abs(a[mask]).ravel()))
        ax.plot(x,np.arange(1,len(x)+1)/len(x),label=f'{label} (n={len(x):,} joints)')
    ax.set(xlim=(0,20),ylim=(0,1),xlabel='Measured joint / previously written target gap (degrees)',ylabel='Fraction of joint samples',title='Actual held-target tracking: latest M2 diagnostic run')
    ax.grid(alpha=.2);ax.legend(loc='lower right')
    fig.savefig(OUT/'tracking_distribution.png',dpi=180);fig.savefig(OUT/'tracking_distribution.pdf');plt.close(fig)
    rows=[r for r in joints if r['run']==latest];fig,ax=plt.subplots(figsize=(10,4.6),constrained_layout=True)
    x=np.arange(16);ax.bar(x-.2,[r['mean_abs_error_deg'] for r in rows],.4,label='Mean');ax.bar(x+.2,[r['p95_abs_error_deg'] for r in rows],.4,label='95th percentile')
    ax.set(xticks=x,xticklabels=[r['joint'] for r in rows],ylabel='Absolute joint error (degrees)',title='Latest run: dynamic error by joint');ax.tick_params(axis='x',rotation=60);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'per_joint_error.png',dpi=180);plt.close(fig)
    fault=report['fault'];end=fault['read_finished_us'];j=[r['joint'] for r in rows].index(fault['joint']);motor=fault['motor_id']
    selected=[r for r in cycles if end-1.5e6<=r.get('read_finished_us',0)<=end]
    t=np.array([r['read_finished_us']-end for r in selected])*1e-6
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
    writes=[r for r in cycles if r.get('write_succeeded') and r['write_finished_us']<=end]
    before=[r for r in writes if r['write_finished_us']<end-1.5e6]
    writes=before[-1:]+[r for r in writes if r['write_finished_us']>=end-1.5e6]
    target_times=[(r['write_finished_us']-end)*1e-6 for r in writes]+[0.]
    targets=[r['written_target'][j] for r in writes]+[fault['last_written_target_rad']]
    axs[0].plot(t,[r['q'][j] for r in selected],label='Measured (read completion time)')
    axs[0].step(target_times,targets,where='post',label='Target (SDK write completion time)')
    axs[0].set_xlim(-1.5,.02)
    axs[0].set(ylabel='Joint position (rad)',title=f"Final 1.5 seconds: motor {motor}, {fault['joint']}");axs[0].legend()
    axs[1].plot(t,[r['current_raw_native'][motor] for r in selected],label='Measured current');limit=fault['limit_raw']
    for sign in [-1,1]:axs[1].axhline(sign*limit,color='red',linestyle='--',label='Software cutoff' if sign==1 else None)
    axs[1].set(xlabel='Seconds relative to current cutoff',ylabel='Current (raw)');axs[1].legend()
    for ax in axs:ax.axvline(0,color='black',linestyle=':');ax.grid(alpha=.2)
    fig.savefig(OUT/'shutdown_timeline.png',dpi=180);plt.close(fig)


def page(report,episodes,joints):
    r=report['runs'][-1];g=r['groups'];a=g['all_manipulation'];b=g['target_unchanged_at_least_50ms']
    content=['<!doctype html><meta charset="utf-8"><title>Actual LEAP joint tracking</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}img{max-width:100%}</style>',
        '<h1>Measured joints versus actual bridge-held targets</h1>',
        '<p>Offline analysis; no hardware, simulation or controller changes. Latest run: '+html.escape(r['run'])+'.</p>',
        '<p>Each measured pose is compared with the target held <strong>before that read began</strong>, verified against the previous successful SDK write. This avoids comparing a new command against a pose sampled before that command existed.</p>',
        '<table><tr><th>Metric</th><th>All manipulation reads</th><th>Same target at least 50 ms</th></tr>']
    for label,key in [('Frames','frames'),('Joint samples','joint_samples')]:content.append(f'<tr><td>{label}</td><td>{a[key]:,}</td><td>{b[key]:,}</td></tr>')
    for key in ['mean','median','p95','max']:
        content.append(f'<tr><td>{key} absolute joint error (deg)</td><td>{a["absolute_error_deg"][key]:.3f}</td><td>{b["absolute_error_deg"][key]:.3f}</td></tr>')
    for t in [.05,.1,.2]:
        for key,label in [('joint_fraction','Joint samples'),('frame_any_joint_fraction','Frames with any joint')]:
            content.append(f'<tr><td>{label} above {t} rad</td><td>{a["thresholds"][str(t)][key]*100:.2f}%</td><td>{b["thresholds"][str(t)][key]*100:.2f}%</td></tr>')
    content+=['</table><img src="tracking_distribution.png"><img src="per_joint_error.png">',
        '<h2>Latest shutdown</h2><p>'+html.escape(r['fault']['primary'])+'. The later controller freshness rejection followed '+f'{r["fault"]["controller_rejection_ms_after_current_fault"]:.1f} ms'+' after the current fault. Torque-off was verified.</p>',
        '<img src="shutdown_timeline.png"><h2>Limits of this evidence</h2><ul>']
    content+=['<li>'+html.escape(x)+'</li>' for x in report['method']['caveats']]
    content+=['</ul><p><a href="report.json">Full report including previous run</a> · <a href="joints.csv">Per-joint statistics</a> · <a href="episodes.csv">Per-episode statistics</a> · <a href="samples.csv">Sample metrics</a> · <a href="analyze.py">Reproduce analysis</a></p>']
    (OUT/'index.html').write_text('\n'.join(content))


if __name__=='__main__':
    main()
