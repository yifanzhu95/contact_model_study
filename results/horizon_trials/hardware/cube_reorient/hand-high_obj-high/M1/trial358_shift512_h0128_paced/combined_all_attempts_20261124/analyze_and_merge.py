"""Combine all original two-step M1 runs, distinguishing incomplete attempts."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=next(p for p in OUT.parents if (p/'run_leap_hardware.py').exists())
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'analysis/mpl_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REFERENCE=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial358_shift512_h0256_paced_measured_limits/leap_experiment_20261123_124445_598769/analysis/analyze.py'
spec=importlib.util.spec_from_file_location('analysis_helper',REFERENCE)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_save(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def link(a,b):
    if b.exists():assert a.stat().st_size==b.stat().st_size;return
    try:os.link(a,b)
    except OSError:shutil.copy2(a,b)


def main():
    for name in ['analysis','video','source_runs']:(OUT/name).mkdir(exist_ok=True)
    paths=sorted(BASE.glob('leap_experiment_*/controller.json'))
    docs=[json.loads(p.read_text()) for p in paths]
    common=['weights','hardware_config','planner_config','rollout_config','joint_reference','pose_transform','joint_limits','success_thresholds']
    checks={p.parent.name:{k:d[k]==docs[0][k] for k in common} for p,d in zip(paths,docs)}
    assert all(all(v.values()) for v in checks.values()),checks
    assert all(d['args']['time_horizon']==.128 and d['args']['rollout_model']=='M1' for d in docs)
    attempts=[];completed=[];incomplete=[];sources=[];videos=[]
    for i,(path,doc) in enumerate(zip(paths,docs),1):
        raw=path.parent;folder=OUT/'source_runs'/f'source_{i:02d}';folder.mkdir(exist_ok=True)
        for file in raw.iterdir():
            if file.is_file():link(file,folder/file.name)
        by_episode={e['episode']+1:e for e in doc['episodes']}
        recording=json.loads((raw/'video/recording.json').read_text())
        for v in recording['episodes']:
            number=len(attempts)+1;original=v['episode'];e=copy.deepcopy(by_episode.get(original))
            done=e is not None and e.get('completed',False)
            assert v['outcome'] in ['error','success','failure','timeout']
            assert not done or v['outcome']==e['finish_reason']
            record=dict(attempt=number,source_run=raw.name,source_episode=original,
                completed=done,outcome=e['finish_reason'] if done else 'aborted',
                tracking_steps_available=e is not None,steps=e.get('n_steps') if e else None,
                rotation=e.get('starting_orientation',{}).get('nearest_quarter_turn') if e else None,
                manipulation_wall_time_s=e.get('manipulation_wall_time_s') if e else None,
                video=f'video/attempt_{number:02d}.mkv')
            for field,extension in [('video','.mkv'),('timestamps','.frames.csv')]:
                link(raw/'video'/v[field],OUT/'video'/f'attempt_{number:02d}{extension}')
            videos.append(dict(v,attempt=number,source_run=raw.name,source_episode=original,video=record['video']))
            if done:
                e.update(source_run=raw.name,source_episode=original,episode=len(completed),aggregate_attempt=number)
                completed.append(e);record['completed_episode']=len(completed)
            else:
                incomplete.append(dict(record,episode_record=e,
                    data_limitation=None if e is not None else 'Controller was terminated before final checkpoint; original logs and video document attempt. No state samples fabricated.'))
            attempts.append(record)
        sources.append(dict(run=raw.name,source_path=str(raw.relative_to(ROOT)),
            controller_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            controller_episodes=len(doc['episodes']),video_attempts=len(recording['episodes'])))
    aggregate=copy.deepcopy(docs[0]);aggregate.update(status='completed_only_aggregate',episodes=completed,
        wall_time_s=sum(e['manipulation_wall_time_s'] for e in completed),
        aggregation_note='Completed-only metrics; all aborted attempts are separately preserved in aborted_attempts.json and attempts.csv. Do not double-count source runs.')
    aggregate['args'].update(n_episodes=len(completed),results=str(OUT/'controller_completed.json'))
    metrics,rows,curves,timing=helper.collect(aggregate)
    for row,e in zip(rows,completed):row['attempt']=e['aggregate_attempt'];row['video']=f'video/attempt_{e["aggregate_attempt"]:02d}.mkv'
    total=dict(attempts=len(attempts),completed=len(completed),successes=metrics['successes'],drops=metrics['drops'],timeouts=metrics['timeouts'],
        aborted=len(incomplete),aborted_with_state_data=sum(e['tracking_steps_available'] for e in incomplete),
        aborted_missing_final_checkpoint=sum(not e['tracking_steps_available'] for e in incomplete),
        success_rate_completed=metrics['success_rate'],success_rate_all_attempts=metrics['successes']/len(attempts))
    aggregate['summary']=dict(total,completed_steps=metrics['actions'],completed_manipulation_time_s=metrics['total_manipulation_s'])
    save(OUT/'controller_completed.json',aggregate);save(OUT/'summary.json',total)
    save(OUT/'aborted_attempts.json',incomplete);csv_save(OUT/'attempts.csv',attempts);csv_save(OUT/'completed_episodes.csv',rows)
    save(OUT/'sources.json',dict(sources=sources,configuration_checks=checks,note='All videos linked without reencoding; original source files retained.'))
    save(OUT/'video/recording.json',dict(mode='all_recorded_attempts',episodes=videos))
    save(OUT/'analysis/report.json',dict(summary=total,completed_metrics=metrics,completed_episodes=rows,configuration_checks=checks))
    save(OUT/'analysis/curves.json',curves)
    csv_save(OUT/'analysis/joint_clipping.csv',metrics['clipping']['joints'])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.sort(timing['planning_ms']);axes[0].plot(x,np.arange(1,len(x)+1)/len(x));axes[0].axvline(64,color='black',ls='--')
    axes[0].set(xlabel='Planning time (ms)',ylabel='Cumulative fraction',title='Completed episodes only; aborted spikes separate')
    for row,curve in zip(rows,curves):axes[1].plot(curve['time_s'],curve['angle_deg'],label=f'{row["rotation"]}: {row["outcome"]}')
    axes[1].axhline(np.degrees(2*np.arcsin(.2)),color='black',ls='--');axes[1].set(xlabel='Manipulation time (s)',ylabel='Angular error (degrees)');axes[1].legend(fontsize=8);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'analysis/timing_and_errors.{ext}',dpi=170)
    body=['<!doctype html><meta charset="utf-8"><title>M1 two-step all attempts</title>',
        '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}td,th{border:1px solid #ccc;padding:7px}table{border-collapse:collapse}img{max-width:100%}</style>',
        '<h1>M1 high/high: all two-step attempts</h1>',f'<p>{len(attempts)} attempts: {len(completed)} completed, {len(incomplete)} aborted. Completed: {metrics["successes"]} success, {metrics["drops"]} drop, {metrics["timeouts"]} timeouts.</p>',
        '<p><a href="analysis/FINDINGS.txt">Findings</a> · <a href="attempts.csv">All attempts</a> · <a href="completed_episodes.csv">Completed metrics</a> · <a href="controller_completed.json">Completed trajectories</a> · <a href="aborted_attempts.json">Aborted data</a> · <a href="sources.json">Sources</a></p>',
        '<img src="analysis/timing_and_errors.png">','<table><tr><th>Attempt</th><th>Source episode</th><th>Rotation</th><th>Outcome</th><th>State data</th><th>Video</th></tr>']
    for a in attempts:body.append(f'<tr><td>{a["attempt"]}</td><td>{a["source_run"]}: {a["source_episode"]}</td><td>{a["rotation"] or "unavailable"}</td><td>{a["outcome"]}</td><td>{a["tracking_steps_available"]}</td><td><a href="{a["video"]}">Video</a></td></tr>')
    body.append('</table>');(OUT/'index.html').write_text('\n'.join(body))
    assert len(list((OUT/'video').glob('*.mkv')))==len(attempts)
    print(json.dumps(dict(out=str(OUT),summary=total,frequency_hz=metrics['publication_frequency_hz'],planning_ms=metrics['planning_ms']),indent=2))


if __name__=='__main__':main()
