"""Empirical MPPI output deltas, before hardware clipping. Offline only."""
from pathlib import Path
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
SOURCE=ROOT/'results/diagnostics/m2_tracking_baseline/leap_experiment_20261116_215909_720053/controller.json'
THRESHOLDS=[.01,.02,.05,.1,.15,.2,.3]


def summarize(a):
    a=np.asarray(a,dtype=float);b=abs(a);maximum=b.max(axis=1)
    return dict(actions=len(a),joint_values=int(a.size),
        signed_rad=dict(min=float(a.min()),max=float(a.max()),mean=float(a.mean()),std=float(a.std()),
            percentiles={str(q):float(np.percentile(a,q)) for q in [1,5,25,50,75,95,99]}),
        absolute_rad=dict(mean=float(b.mean()),percentiles={str(q):float(np.percentile(b,q)) for q in [50,75,90,95,99]},max=float(b.max())),
        hypothetical_clipping={str(t):dict(joint_count=int(np.sum(b>t+1e-7)),
            joint_fraction=float(np.mean(b>t+1e-7)),action_count=int(np.sum(maximum>t+1e-7)),
            action_fraction=float(np.mean(maximum>t+1e-7))) for t in THRESHOLDS})


def main():
    c=json.loads(SOURCE.read_text());all_steps=[];episodes=[]
    for ep in c['episodes']:
        steps=[s for s in ep['steps'] if s.get('action_published')]
        if not steps:continue
        a=np.array([s['action_delta'] for s in steps])
        np.testing.assert_allclose(a,[np.array(s['planned_u'])-s['plan_base'] for s in steps],rtol=0,atol=1e-10)
        np.testing.assert_allclose([s['plan_base'] for s in steps],[s['q'][:16] for s in steps],rtol=0,atol=1e-10)
        all_steps.extend(steps)
        episodes.append(dict(episode=ep['episode']+1,outcome=ep['finish_reason'],completed=ep['completed'],**summarize(a)))
    a=np.array([s['action_delta'] for s in all_steps])
    completed=np.array([s['action_delta'] for ep in c['episodes'] if ep['completed'] for s in ep['steps'] if s.get('action_published')])
    moving=np.array([s['action_delta'] for ep in c['episodes'] if ep['completed'] and ep['finish_reason']!='timeout'
                     for s in ep['steps'] if s.get('action_published')])
    joint_rows=[]
    for j,joint in enumerate(c['joint_order']):
        b=a[:,j];joint_rows.append(dict(joint=joint,mean_signed_rad=float(b.mean()),std_rad=float(b.std()),
            signed_min_rad=float(b.min()),signed_max_rad=float(b.max()),median_abs_rad=float(np.median(abs(b))),
            p95_abs_rad=float(np.percentile(abs(b),95)),fraction_over_001=float(np.mean(abs(b)>.0100001)),
            fraction_over_010=float(np.mean(abs(b)>.1000001)),fraction_over_020=float(np.mean(abs(b)>.2000001))))
    # Join by action ID to distinguish published requests from bridge writes.
    telemetry=SOURCE.parent/'bridge_telemetry.jsonl'
    rows=[json.loads(line) for line in telemetry.read_text().splitlines()]
    writes={r['written_action_timestamp_us']:r for r in rows if r.get('write_succeeded') and r.get('new_action') and r.get('command_type')=='delta'}
    matched=[s for s in all_steps if s['action_timestamp_us'] in writes]
    applied=np.array([np.array(writes[s['action_timestamp_us']]['written_target'])-writes[s['action_timestamp_us']]['application_base'] for s in matched])
    for s in matched:
        r=writes[s['action_timestamp_us']]
        np.testing.assert_allclose(r['delta'],s['action_delta'],rtol=0,atol=1e-7)
        np.testing.assert_allclose(np.array(r['written_target'])-r['application_base'],np.clip(r['delta'],-.2,.2),rtol=0,atol=1e-10)
    bounds=np.array([0,.01,.02,.05,.1,.2,.3,.5,np.inf]);counts=np.histogram(abs(a).ravel(),bins=bounds)[0]
    bins=[dict(lower_rad=float(lo),upper_rad=float(hi) if np.isfinite(hi) else None,
        joint_values=int(count),fraction=float(count/a.size)) for lo,hi,count in zip(bounds[:-1],bounds[1:],counts)]
    report=dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        definition='Optimized first-action delta = planned_u - plan_base; plan_base equals the measured hand used to initialize planning. The bridge later adds this delta to its fresher measured joints and clips the resulting target around those joints.',
        scope='Latest actual MPPI run, not manual step/sine tests or the distribution of rollout perturbations.',
        settings={k:c['args'].get(k) for k in ['task','rollout_model','n_samples','temperature','noise_sigma','time_horizon','warm_start','control_mode']},
        hardware_bounds=c['hardware_config']['ctrl_range'],cohorts=dict(all_published=summarize(a),completed_only=summarize(completed),
            completed_without_timeout=summarize(moving)),episodes=episodes,joints=joint_rows,magnitude_bins=bins,
        actual_bridge=dict(published=len(all_steps),matched_applied=len(matched),applied_offsets=summarize(applied)),
        caveats=['Published output deltas only; failed/unpublished plans are excluded.',
            'One 1000-step timeout contributes 84.1% of latest-run actions. Per-episode and non-timeout subsets are supplied.',
            'All-published includes 19 valid published actions from the later aborted episode; completed-only is supplied separately.',
            'Hypothetical clipping at tighter bounds re-evaluates these stored requests; it does not predict the trajectories a new closed-loop run would generate.',
            'Per-joint percentiles pool different joints, states and episodes; no Gaussian output distribution is assumed.',
            'Current step rounding tolerance for clipping frequency: 1e-7 rad.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (OUT/'joints.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(joint_rows[0]));writer.writeheader();writer.writerows(joint_rows)
    fig,axs=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    axs[0].hist(a.ravel(),bins=np.linspace(-.45,.45,91),density=True,color='#3366aa',alpha=.85)
    axs[0].axvline(0,color='black',linewidth=.8)
    for sign in [-1,1]:axs[0].axvline(sign*.2,color='red',linestyle='--',label='Hardware ±0.2 rad' if sign==1 else None)
    axs[0].set(xlabel='Raw planner delta (rad)',ylabel='Probability density',title='Signed optimized first-action deltas');axs[0].legend()
    for values,label in [(a,'All 1,189 actions'),(moving,'170 completed non-timeout actions')]:
        values=np.sort(abs(values).ravel());axs[1].plot(values,np.arange(1,len(values)+1)/len(values),label=label)
    axs[1].set(xlim=(0,.3),ylim=(0,1),xlabel='Absolute planner delta (rad)',ylabel='Fraction of joint values',title='Magnitude distribution; separate the long timeout')
    axs[1].legend();axs[1].grid(alpha=.2)
    fig.savefig(OUT/'delta_distribution.png',dpi=180);fig.savefig(OUT/'delta_distribution.pdf');plt.close(fig)
    body=['<!doctype html><meta charset="utf-8"><title>Planner delta distribution</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}img{max-width:100%}</style>',
        '<h1>MPPI output deltas before hardware clipping</h1><p>'+html.escape(report['definition'])+'</p>',
        '<p>Source: '+html.escape(str(SOURCE.relative_to(ROOT)))+'. M2 / 512 samples / noise 0.05 / temperature 0.1 / shift warm start / horizon 0.32 s.</p>',
        '<img src="delta_distribution.png"><table><tr><th>Absolute delta metric (rad)</th><th>All published</th><th>Completed non-timeout subset</th></tr>']
    for label,key in [('Mean','mean'),('Median','50'),('P90','90'),('P95','95'),('P99','99'),('Maximum','max')]:
        vals=[]
        for cohort in ['all_published','completed_without_timeout']:
            v=report['cohorts'][cohort]['absolute_rad'];vals.append(v[key] if key in v else v['percentiles'][key])
        body.append(f'<tr><td>{label}</td><td>{vals[0]:.5f}</td><td>{vals[1]:.5f}</td></tr>')
    body+=['</table><h2>Stored requests affected by different bounds</h2><table><tr><th>Symmetric bound (rad)</th><th>Joint requests clipped</th><th>Actions with any joint clipped</th></tr>']
    for t,row in report['cohorts']['all_published']['hypothetical_clipping'].items():
        body.append(f'<tr><td>±{t}</td><td>{row["joint_fraction"]*100:.2f}%</td><td>{row["action_fraction"]*100:.2f}%</td></tr>')
    body+=['</table><ul>']+['<li>'+html.escape(x)+'</li>' for x in report['caveats']]
    body+=['</ul><p><a href="report.json">Full report</a> · <a href="joints.csv">Per-joint statistics</a> · <a href="delta_distribution.pdf">PDF plot</a> · <a href="analyze.py">Reproduce</a></p>']
    (OUT/'index.html').write_text('\n'.join(body))
    print(json.dumps({k:report['cohorts'][k] for k in ['all_published','completed_without_timeout']},indent=2))


if __name__=='__main__':main()
