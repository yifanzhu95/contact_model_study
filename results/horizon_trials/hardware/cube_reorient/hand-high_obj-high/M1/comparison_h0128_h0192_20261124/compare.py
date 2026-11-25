"""Compare completed, recorded M1 hardware episodes. No hardware or simulation."""
import csv
import html
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
SOURCES={'two':BASE/'trial358_shift512_h0128_paced/combined_12_completed_20261123_235834_20261124_003707',
         'three':BASE/'trial358_shift512_h0192_paced/combined_12_completed_20261123_232724_235225'}
LABELS={'two':'2 steps / 0.128 s','three':'3 steps / 0.192 s'}
COLORS={'two':'#dd8736','three':'#377eb8'}


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    (OUT/'plots').mkdir(exist_ok=True)
    reports={k:json.loads((p/'analysis/report.json').read_text()) for k,p in SOURCES.items()}
    docs={k:json.loads((p/'controller.json').read_text()) for k,p in SOURCES.items()}
    keys=['weights','hardware_config','planner_config','joint_reference','pose_transform','joint_limits','success_thresholds','failure_z','control_policy','fixed_goal_quat']
    checks={k:docs['two'][k]==docs['three'][k] for k in keys};assert all(checks.values()),checks
    simkeys=set(docs['two']['rollout_config'])|set(docs['three']['rollout_config'])
    simdiff={k:[docs['two']['rollout_config'].get(k),docs['three']['rollout_config'].get(k)] for k in simkeys if docs['two']['rollout_config'].get(k)!=docs['three']['rollout_config'].get(k)}
    assert set(simdiff)=={'time_horizon'},simdiff
    metrics={k:r['metrics'] for k,r in reports.items()}
    aborted={}
    for key,folder in SOURCES.items():
        entries=[]
        for path in sorted(folder.parent.glob('leap_experiment_*/video/recording.json')):
            for e in json.loads(path.read_text())['episodes']:
                if e['outcome']=='error':entries.append(dict(source_run=path.parents[1].name,source_episode=e['episode'],video=str(path.parent/e['video']),outcome='aborted',note='Excluded from completed-only metrics; original retained.'))
        aborted[key]=entries
    save(SOURCES['two']/'excluded_aborted_attempts.json',aborted['two'])
    p=float(fisher_exact([[metrics['two']['successes'],12-metrics['two']['successes']],[metrics['three']['successes'],12-metrics['three']['successes']]]).pvalue)
    save(OUT/'report.json',dict(sources={k:str(p) for k,p in SOURCES.items()},summaries=metrics,configuration_checks=checks,rollout_differences=simdiff,
        excluded_aborted_attempts=aborted,fisher_two_sided_p=p,
        limitations=['Twelve completed episodes per horizon; exact physical placements are not paired.',
            'Timing metrics here exclude aborted attempts, including rejected plans of 161-230ms.',
            'Clipping uses controller estimates, not exact bridge-written telemetry.',
            'Changing horizon at fixed weights changes lookahead and cumulative running/terminal balance.',
            'No production settings or baseline designation changed.']))
    fields={'Successes':lambda m:m['successes'],'Drops':lambda m:m['drops'],'Timeouts':lambda m:m['timeouts'],
        'Control average Hz':lambda m:m['publication_frequency_hz'],'Control min Hz':lambda m:m['instantaneous_frequency_hz']['min'],
        'Control max Hz':lambda m:m['instantaneous_frequency_hz']['max'],'Planning mean ms':lambda m:m['planning_ms']['mean'],
        'Planning p95 ms':lambda m:m['planning_ms']['p95'],'Planning p99 ms':lambda m:m['planning_ms']['p99'],
        'Planning max ms':lambda m:m['planning_ms']['max'],'Plans over 64ms %':lambda m:m['planning_over_64ms_percent'],
        'Pacing wait %':lambda m:m['paced_wait_percent'],'Delta clipping %':lambda m:m['clipping']['delta_clipped_percent'],
        'Absolute clipping %':lambda m:m['clipping']['absolute_clipped_percent'],'Either clipping %':lambda m:m['clipping']['either_clipped_percent'],
        'Manipulation total s':lambda m:m['total_manipulation_s'],'Successful duration mean s':lambda m:m['successful_duration_s']['mean'],
        'Successful endpoint position mean mm':lambda m:m['successful_final_pos_mm']['mean'],
        'Successful endpoint angle mean deg':lambda m:m['successful_final_angle_deg']['mean']}
    table=[dict(metric=name,**{k:f(m) for k,m in metrics.items()}) for name,f in fields.items()]
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['metric','two','three']);w.writeheader();w.writerows(table)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    bottom=np.zeros(2)
    for field,color in [('successes','#419c64'),('drops','#c36550'),('timeouts','#c7a443')]:
        vals=[metrics[k][field] for k in SOURCES];axes[0].bar([LABELS[k] for k in SOURCES],vals,bottom=bottom,color=color,label=field);bottom+=vals
    axes[0].set(ylabel='Completed episodes',yticks=range(0,13,2));axes[0].legend()
    for k,doc in docs.items():
        x=np.sort([s['planning_time']*1000 for e in doc['episodes'] for s in e['steps'] if s.get('action_published')]);axes[1].plot(x,np.arange(1,len(x)+1)/len(x),color=COLORS[k],label=LABELS[k])
    axes[1].axvline(64,color='black',ls='--');axes[1].set(xlabel='Planning time (ms)',ylabel='Fraction of published plans',title='Completed episodes only; aborted spikes excluded');axes[1].legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'plots/outcomes_and_timing.{ext}',dpi=170)
    fig,ax=plt.subplots(figsize=(9,3));rotations=['+X','-X','+Y','-Y','+Z','-Z'];a=np.array([[metrics[k]['rotations'][r]['successes'] for r in rotations] for k in SOURCES]);ax.imshow(a,vmin=0,vmax=2,cmap='RdYlGn',aspect='auto')
    for i in range(2):
        for j in range(6):ax.text(j,i,f'{a[i,j]}/2',ha='center',va='center')
    ax.set(xticks=range(6),xticklabels=rotations,yticks=[0,1],yticklabels=list(LABELS.values()),title='Successes by starting rotation');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'plots/rotation_success.{ext}',dpi=170)
    style='<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;line-height:1.5}td,th{border:1px solid #ccc;padding:7px}table{border-collapse:collapse}img{max-width:100%}</style>'
    for key,folder in SOURCES.items():
        if key=='three':continue
        body=['<!doctype html><meta charset="utf-8">'+style,'<h1>M1 two-step horizon: 12 completed episodes</h1>',
            '<p>3 successes, 5 drops, 4 timeouts. Fourteen additional aborted attempts are excluded from these metrics and remain in original logs/videos.</p>',
            f'<p><a href="{os.path.relpath(OUT/"index.html",folder)}">Compare with three steps</a> · <a href="controller.json">Full trajectories</a> · <a href="episodes.csv">Episodes</a> · <a href="sources.json">Sources</a> · <a href="excluded_aborted_attempts.json">Aborted attempt manifest</a></p>',
            '<table><tr><th>Episode</th><th>Rotation</th><th>Outcome</th><th>Seconds</th><th>Video</th></tr>']
        for e in reports[key]['episodes']:
            n=e['episode'];body.append(f'<tr><td>{n}</td><td>{e["rotation"]}</td><td>{e["outcome"]}</td><td>{e["duration_s"]:.2f}</td><td><a href="video/episode_{n:02d}.mkv">Video</a></td></tr>')
        body.append('</table>');(folder/'index.html').write_text('\n'.join(body))
    body=['<!doctype html><meta charset="utf-8">'+style,'<h1>M1: two versus three control steps</h1>','<p>12 completed episodes each, balanced rotations. Aborted attempts remain separate; do not double-count the source runs.</p>',
        '<p><a href="FINDINGS.txt">Interpretation</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="report.json">Full report</a></p>']
    for k,folder in SOURCES.items():body.append(f'<p><a href="{os.path.relpath(folder/"index.html",OUT)}">{LABELS[k]}: completed episodes and videos</a></p>')
    body.append('<table><tr><th>Metric</th><th>2 steps</th><th>3 steps</th></tr>')
    for row in table:body.append('<tr><td>'+html.escape(row['metric'])+'</td>'+''.join(f'<td>{row[k]:.3f}</td>' for k in SOURCES)+'</tr>')
    body.append('</table><img src="plots/outcomes_and_timing.png"><img src="plots/rotation_success.png">');(OUT/'index.html').write_text('\n'.join(body))
    for folder in SOURCES.values():assert len(list((folder/'video').glob('*.mkv')))==12
    print(json.dumps(dict(two_step_successes=metrics['two']['successes'],three_step_successes=metrics['three']['successes'],aborted_counts={k:len(v) for k,v in aborted.items()},fisher_two_sided_p=p),indent=2))


if __name__=='__main__':main()
