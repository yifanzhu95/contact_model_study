"""Offline evidence about tracker age/reuse and camera-depth discontinuities.

Only original controller/launcher/video logs are read. No camera, GPU, simulator
or hardware calls. A discontinuity is a candidate, not a diagnosed tracker error.
"""
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p/'launch_leap_experiment.py').exists())
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'mpl_cache'))
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ROOT/'results/horizon_trials/hardware/cube_reorient'
M2_HIGH = ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial358_shift512_h0256_paced_measured_limits/leap_experiment_20261123_124445_598769'
THRESHOLDS = [20, 30]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csv_save(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(rows)


def epoch(value):
    return datetime.fromisoformat(value).timestamp() if value else None


def bin_label(age):
    return '0–2 min' if age<2 else '2–5 min' if age<5 else '5–10 min' if age<10 else '10–20 min' if age<20 else '20+ min'


def stats(values):
    return dict(n=len(values), median=float(np.median(values)), p95=float(np.percentile(values,95)), max=float(max(values))) if len(values) else dict(n=0)


def clusters(rows, threshold):
    result=[]
    for row in sorted((r for r in rows if abs(r['depth_jump_mm'])>=threshold), key=lambda r:(r['run'],r['episode'],r['timestamp_s'])):
        if result and (result[-1]['run'],result[-1]['episode'])==(row['run'],row['episode']) and row['timestamp_s']-result[-1]['last_timestamp_s']<=.3:
            result[-1]['transitions']+=1
            result[-1]['last_timestamp_s']=row['timestamp_s']
            if abs(row['depth_jump_mm'])>abs(result[-1]['max_depth_jump_mm']):
                result[-1]['max_depth_jump_mm']=row['depth_jump_mm']
                result[-1]['peak_step']=row['step']
                result[-1]['peak_elapsed_s']=row['elapsed_s']
        else:
            result.append(dict(run=row['run'],episode=row['episode'],session=row['session'],model=row['model'],hand=row['hand'],
                first_timestamp_s=row['timestamp_s'],last_timestamp_s=row['timestamp_s'],session_age_min=row['session_age_min'],
                age_bin=row['age_bin'],run_reused=row['run_reused'],after_recorded_drop=row['after_recorded_drop'],
                first_episode_in_session=row['first_episode_in_session'],max_depth_jump_mm=row['depth_jump_mm'],
                step=row['step'],elapsed_s=row['elapsed_s'],peak_step=row['step'],peak_elapsed_s=row['elapsed_s'],transitions=1))
    return result


def summary(rows):
    duration=sum(r['dt_ms']/1000 for r in rows)
    result=dict(transitions=len(rows),eligible_observed_minutes=duration/60,episodes=len({(r['run'],r['episode']) for r in rows}),sessions=len({r['session'] for r in rows}))
    for t in THRESHOLDS:
        hits=[r for r in rows if abs(r['depth_jump_mm'])>=t]
        event=clusters(rows,t)
        result[f'jump{t}_transitions']=len(hits)
        result[f'jump{t}_percent']=len(hits)/len(rows)*100 if rows else None
        result[f'jump{t}_clusters']=len(event)
        result[f'jump{t}_clusters_per_min']=len(event)/(duration/60) if duration else None
        result[f'jump{t}_episodes']=len({(r['run'],r['episode']) for r in hits})
    return result


def figure(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(OUT/'plots'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    (OUT/'plots').mkdir(exist_ok=True)
    launch_paths=sorted(BASE.glob('hand-*_obj-high/M[12]/*/leap_experiment_*/launch.json'))+[M2_HIGH/'launch.json']
    runs, sessions={},{}
    for path in launch_paths:
        d=json.loads(path.read_text()); tracking=d.get('tracking') or {};s=tracking.get('session')
        if not s: continue
        directory=Path(s['directory']);sid=directory.name
        assert s.get('started_at')
        commands=s.get('commands',{}).get('tracker',[])
        record=dict(session=sid,directory=str(directory.relative_to(ROOT)),started_at_s=s['started_at'],
            started_at_utc=datetime.fromtimestamp(s['started_at'],timezone.utc).isoformat(),
            started_at_phoenix=datetime.fromtimestamp(s['started_at'],ZoneInfo('America/Phoenix')).isoformat(),
            tracker_identity=s['processes'].get('tracker'),kalman='--activate_kalman_filter' in commands,
            tracker_2d='--activate_2d_tracker' in commands)
        if sid in sessions: assert record['started_at_s']==sessions[sid]['started_at_s']
        else: sessions[sid]=record
        run=path.parent
        controller=run/'controller.json'
        doc=json.loads(controller.read_text()) if controller.exists() else None
        videos=run/'video/recording.json'
        video=json.loads(videos.read_text()) if videos.exists() else {}
        episodes={e['episode']+1:e for e in doc.get('episodes',[])} if doc else {}
        starts=[]
        for v in video.get('episodes',[]):
            e=episodes.get(v['episode'],{})
            start=epoch(e.get('started_at_utc') or v.get('started_at_utc'))
            end=epoch(e.get('finished_at_utc') or v.get('finished_at_utc'))
            if start:
                starts.append(dict(run=run.name,episode=v['episode'],start=start,end=end,
                    outcome=e.get('finish_reason') if e.get('completed') else v.get('outcome')))
        runs[run.name]=dict(path=run,launch=d,doc=doc,session=sid,reused=tracking['reused'],starts=starts)
    by_session={sid:sorted([e for run in runs.values() if run['session']==sid for e in run['starts']],key=lambda e:e['start']) for sid in sessions}
    observations=[];episode_rows=[]
    for name,run in runs.items():
        doc=run['doc']
        if not doc:continue
        sid=run['session'];session=sessions[sid]
        T=doc['pose_transform'].get('selected_calibration',{}).get('T_world_from_camera')
        if T is None:raise ValueError(f'Missing selected camera transform: {name}')
        R=np.linalg.inv(np.array(T))[:3,:3]
        starts=by_session[sid]
        for e in doc.get('episodes',[]):
            if not e.get('steps') or not e.get('started_at_utc'):continue
            number=e['episode']+1;start=epoch(e['started_at_utc'])
            prior=[p for p in starts if p['start']<start-.001]
            drops=[p for p in prior if p['outcome']=='failure' and p['end'] and p['end']<=start]
            first=not prior
            previous=None;rows=[]
            for step in e['steps']:
                if 'q' not in step or len(step['q'])<23:continue
                ts=step.get('feedback',{}).get('pose_timestamp')
                if not ts:continue
                ts=ts/1e6
                pos=np.array(step['q'][16:19]);quat=np.array(step['q'][19:23])
                current=(ts,pos,quat)
                if previous is None:previous=current;continue
                if ts==previous[0]:continue
                dt=ts-previous[0]
                if dt>0 and dt<=.120:
                    change=R@(pos-previous[1]);angle=np.degrees(2*np.arccos(np.clip(abs(np.dot(quat,previous[2]))/(np.linalg.norm(quat)*np.linalg.norm(previous[2])),0,1)))
                    remaining=max(s.get('elapsed_s',0) for s in e['steps'])-step['elapsed_s']
                    is_failure=e.get('finish_reason')=='failure'
                    age=(ts-session['started_at_s'])/60
                    row=dict(run=name,path=str(run['path'].relative_to(ROOT)),session=sid,
                        model=doc['args']['rollout_model'],hand=doc['args']['hand_acc'],horizon_s=doc['args']['time_horizon'],
                        episode=number,outcome=e.get('finish_reason'),step=step['step'],elapsed_s=step['elapsed_s'],timestamp_s=ts,
                        dt_ms=dt*1000,depth_jump_mm=float(change[2]*1000),lateral_jump_mm=float(np.linalg.norm(change[:2])*1000),
                        angle_jump_deg=float(angle),session_age_min=age,age_bin=bin_label(age),
                        run_reused=run['reused'],first_episode_in_session=first,prior_episodes_in_session=len(prior),
                        after_recorded_drop=bool(drops),prior_recorded_drops=len(drops),
                        seconds_after_last_recorded_drop=ts-drops[-1]['end'] if drops else None,
                        failure_last1s=is_failure and remaining<=1,failure_last2s=is_failure and remaining<=2,
                        pose_rejections=step['feedback'].get('rejected_poses'))
                    rows.append(row)
                previous=current
            observations.extend(rows)
            if rows:
                eligible=[r for r in rows if not r['failure_last1s']]
                episode_rows.append(dict(run=name,episode=number,model=doc['args']['rollout_model'],hand=doc['args']['hand_acc'],
                    session=sid,session_age_at_start_min=(start-session['started_at_s'])/60,run_reused=run['reused'],
                    first_episode_in_session=first,after_recorded_drop=bool(drops),outcome=e.get('finish_reason'),
                    **summary(eligible)))
    clean=[r for r in observations if not r['failure_last1s']]
    models=['M1','M2'];bins=['0–2 min','2–5 min','5–10 min','10–20 min','20+ min']
    tables=[]
    for model in models:
        rows=[r for r in clean if r['model']==model]
        for grouping,fn,values in [('age_bin',lambda r:r['age_bin'],bins),
            ('run_reused',lambda r:r['run_reused'],[False,True]),
            ('after_recorded_drop',lambda r:r['after_recorded_drop'],[False,True]),
            ('first_episode_in_session',lambda r:r['first_episode_in_session'],[True,False])]:
            for value in values:
                tables.append(dict(model=model,grouping=grouping,group=value,**summary([r for r in rows if fn(r)==value])))
    session_rows=[]
    for sid,s in sorted(sessions.items(),key=lambda kv:kv[1]['started_at_s']):
        directory=ROOT/s['directory'];log=directory/'tracker.log'
        logtext=log.read_text(errors='replace') if log.exists() else ''
        fps=[float(x) for x in re.findall(r'^FPS: ([0-9.]+)',logtext,re.M)]
        mrows=[r for r in clean if r['session']==sid]
        row=dict(**s,runs=[name for name,run in runs.items() if run['session']==sid],
            recorded_episode_attempts=len(by_session[sid]),
            initial_registration_start_messages=logtext.count('[tracking startup] initial object registration...'),
            lost_reacquired_messages=[line for line in logtext.splitlines() if re.search(r'\b(lost|reacquir\w*|reinitializ\w*)\b',line,re.I)],
            tracker_log_bytes=len(logtext.encode()),fps=stats(fps),
            fps_first_third=stats(fps[:max(1,len(fps)//3)]),fps_last_third=stats(fps[-max(1,len(fps)//3):]),
            observed_max_session_age_min=max([r['session_age_min'] for r in mrows],default=None),**summary(mrows))
        session_rows.append(row)
    sensitivity=[]
    for model in models:
        for name,filter_fn in [('all',lambda r:True),('exclude_last1s_drop',lambda r:not r['failure_last1s']),
            ('exclude_last2s_drop',lambda r:not r['failure_last2s']),
            ('exclude_last1s_dt_le80ms',lambda r:not r['failure_last1s'] and r['dt_ms']<=80),
            ('exclude_last1s_depth_dominant',lambda r:not r['failure_last1s'] and r['lateral_jump_mm']<=10)]:
            subset=[r for r in observations if r['model']==model and filter_fn(r)]
            sensitivity.append(dict(model=model,selection=name,**summary(subset)))
    corr={}
    for model in models:
        rows=[e for e in episode_rows if e['model']==model and e['eligible_observed_minutes']>0]
        age=[r['session_age_at_start_min'] for r in rows]
        vals=[r['jump20_clusters_per_min'] for r in rows]
        c=spearmanr(age,vals)
        corr[model]=dict(episodes=len(rows),rho=float(c.statistic),p=float(c.pvalue),note='Exploratory episode rates, many zeros; sessions/episodes not independent; not a causal or well-powered age test.')
    candidates=clusters(clean,20)
    manifest=[]
    for name,run in sorted(runs.items()):
        paths=[run['path']/'launch.json',run['path']/'controller.json',run['path']/'video/recording.json']
        manifest.append(dict(run=name,session=run['session'],inputs=[dict(path=str(p.relative_to(ROOT)),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths if p.exists()]))
    report=dict(scope='Raw M1/M2 horizon-trial runs plus the designated high-M2 Trial358 baseline. No derived aggregates counted.',
        age_definition='Seconds since saved persistent-session launch registration; includes tracker startup and idle time. Not time since last successful visual lock.',
        summary={model:summary([r for r in clean if r['model']==model]) for model in models},
        tables=tables,sessions=session_rows,episodes=episode_rows,candidate_events20=candidates,
        candidate_events30=clusters(clean,30),sensitivity=sensitivity,exploratory_age_rate_correlation=corr,
        limitations=['Controller-subsampled pose transitions, not every camera frame; observations between episodes and most aborted episodes are absent.',
            'At least 20/30 mm camera-depth changes within 120 ms are screening criteria, not verified tracking errors. Real axial motion and falls can qualify.',
            'Last one second of recorded drops excluded; two-second and sampling-interval sensitivity provided. This cannot remove all legitimate physical motion.',
            'Recorded prior drop is an exposure proxy, not a confirmed tracking-loss/reacquisition event. Hand occlusion/object removal can also occur after successful episodes.',
            'Tracker reuse describes process continuity across launches; later episodes within a fresh launch also reuse its pose state.',
            'Only a few independent tracker sessions; age correlates with episode order, rotation, model, horizon and handling. Longest ages have limited exposure.',
            'No confidence/visibility state or lost/reacquired event markers were recorded; persistent bias without a jump is not detectable with this scan.',
            'FPS logs have ordering but no per-line wall-clock timestamps; early/late thirds do not align precisely with episode times.'])
    save(OUT/'report.json',report)
    save(OUT/'sources.json',manifest)
    save(OUT/'transitions.json',observations)
    csv_save(OUT/'grouped_rates.csv',tables)
    csv_save(OUT/'episodes.csv',episode_rows)
    csv_save(OUT/'candidate_events20.csv',candidates)
    csv_save(OUT/'sensitivity.csv',sensitivity)
    csv_save(OUT/'session_summary.csv',[{k:v for k,v in s.items() if k not in ['tracker_identity','fps_first_third','fps_last_third','lost_reacquired_messages']} for s in session_rows])
    # Age-normalized rates, with exposure printed to make sparse bins visible.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,model in zip(axes,models):
        selected=[r for r in tables if r['model']==model and r['grouping']=='age_bin']
        vals=[r['jump20_clusters_per_min'] or 0 for r in selected]
        ax.bar(range(5),vals,color='#3977ad' if model=='M1' else '#de8836')
        for i,r in enumerate(selected):ax.text(i,vals[i]+.03,
            f'{r["jump20_clusters"]} / {r["eligible_observed_minutes"]:.1f}m' if r['transitions'] else 'No data',ha='center',fontsize=8)
        ax.set(xticks=range(5),xticklabels=['0–2','2–5','5–10','10–20','20+'],xlabel='Tracker-session age (minutes)',ylabel='Depth-jump candidate clusters / observed minute',title=model)
        ax.set_ylim(0,max(vals+[1]) * 1.25)
    figure(fig,'candidate_rate_by_tracker_age')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,group,title,labels in [(axes[0],'after_recorded_drop','Prior recorded drop in same session',['No prior drop','After prior drop']),
                                 (axes[1],'first_episode_in_session','Repeated episode handling',['Later episodes','First episode'])]:
        for j,model in enumerate(models):
            selected=[r for r in tables if r['model']==model and r['grouping']==group]
            selected=sorted(selected,key=lambda r:r['group'])
            vals=[r['jump20_clusters_per_min'] or 0 for r in selected]
            x=np.arange(2)+(j-.5)*.34
            ax.bar(x,vals,.32,color='#3977ad' if model=='M1' else '#de8836',label=model)
            for xx,r,v in zip(x,selected,vals):ax.text(xx,v+.03,f'{r["jump20_clusters"]}/{r["eligible_observed_minutes"]:.1f}m',ha='center',fontsize=8)
        ax.set(xticks=[0,1],xticklabels=labels,title=title,ylabel='Candidate clusters / observed minute');ax.legend()
        ax.margins(y=.3)
    figure(fig,'candidate_rate_after_handling')
    fig,ax=plt.subplots(figsize=(12,6))
    used=[s for s in session_rows if s['transitions']]
    for i,s in enumerate(used):
        sr=[r for r in clean if r['session']==s['session']]
        # Actual observed episode windows, not a continuous-exposure line.
        for name,number in sorted({(r['run'],r['episode']) for r in sr}):
            age=[r['session_age_min'] for r in sr if (r['run'],r['episode'])==(name,number)]
            ax.plot([min(age),max(age)],[i,i],color='#bbbbbb',lw=5,solid_capstyle='butt')
        cr=[r for r in candidates if r['session']==s['session']]
        ax.scatter([r['session_age_min'] for r in cr],[i]*len(cr),color='#c7473a',s=20,zorder=3)
    ax.set(yticks=range(len(used)),yticklabels=[s['session'].replace('leap_experiment_','') for s in used],xlabel='Age since tracker-session launch (minutes)',title='Gray: analyzed episode exposure; red: ≥20 mm depth-jump candidate cluster')
    figure(fig,'session_timeline')
    body=['<!doctype html><meta charset="utf-8"><title>Tracker persistence investigation</title>',
        '<style>body{font:16px sans-serif;line-height:1.5;max-width:1300px;margin:30px auto}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}img{max-width:100%}</style>',
        '<h1>Tracker age, reuse and apparent recovery</h1>',
        '<p>No increasing depth-jump candidate rate with age is evident in these logs. Actual visual loss/recovery is unmarked, so that association remains unresolved.</p>',
        '<p>118 numeric episode traces across 16 sessions. Large depth changes are candidates, not confirmed tracking errors. Fall-window exclusions and sampling sensitivity are retained.</p>',
        '<p><a href="FINDINGS.txt">Full findings and code explanation</a> · <a href="grouped_rates.csv">Grouped rates</a> · <a href="candidate_events20.csv">Candidate events</a> · <a href="session_summary.csv">Session summary</a> · <a href="report.json">Full analysis</a> · <a href="sources.json">Hashed inputs</a></p>',
        '<table><tr><th>Model</th><th>Grouping</th><th>Group</th><th>Observed minutes</th><th>≥20 mm clusters</th><th>Clusters/min</th><th>≥30 mm clusters</th></tr>']
    for r in tables:
        rate='No data' if r['jump20_clusters_per_min'] is None else f'{r["jump20_clusters_per_min"]:.3f}'
        body.append(f'<tr><td>{r["model"]}</td><td>{r["grouping"]}</td><td>{r["group"]}</td><td>{r["eligible_observed_minutes"]:.3f}</td><td>{r["jump20_clusters"]}</td><td>{rate}</td><td>{r["jump30_clusters"]}</td></tr>')
    body.append('</table>')
    for name in ['candidate_rate_by_tracker_age','candidate_rate_after_handling','session_timeline']:
        body.append(f'<p><a href="plots/{name}.pdf">PDF</a></p><img src="plots/{name}.png">')
    body.append('<h2>Limitations</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in report['limitations'])+'</ul>')
    (OUT/'index.html').write_text('\n'.join(body))
    print(json.dumps(dict(summary=report['summary'],rates=tables,age_correlation=corr,candidates20=candidates,
        sessions=[dict(session=s['session'],runs=len(s['runs']),observed_age=s['observed_max_session_age_min'],registrations=s['initial_registration_start_messages'],events=s['jump20_clusters']) for s in session_rows]),indent=2))


if __name__=='__main__':main()
