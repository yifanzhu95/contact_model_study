"""Compare optional blending on regenerated plans along fixed recorded states."""
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0,str(ROOT))
from ContactModelStudy.Hardware.PlanBlending import TemporalPlanBlend

ALPHAS = [1., .9, .8, .6, .4, .2]


def stat(x):
    x=np.asarray(x).ravel()
    return dict(n=int(x.size),mean=float(x.mean()),median=float(np.median(x)),
                p95=float(np.percentile(x,95)),max=float(x.max()))


def summarize(items):
    cat=lambda name:np.concatenate([r[name] for r in items])
    raw=cat('delta');limited=cat('limited');jumps=cat('jumps')
    return dict(steps=len(raw),episodes=len(items),
        estimated_target_jump_deg=stat(np.degrees(abs(jumps))),
        estimated_target_jump_l2_rad=stat(np.linalg.norm(jumps,axis=1)),
        delta_jump_deg=stat(np.degrees(abs(cat('delta_jumps')))),
        delta_abs_deg=stat(np.degrees(abs(raw))),
        clipped_actions_percent=float(np.mean(np.any(abs(raw-limited)>1e-7,axis=1))*100),
        clipped_joint_values_percent=float(np.mean(abs(raw-limited)>1e-7)*100),
        clip_removed_mean_rad=float(np.mean(abs(raw-limited))),
        blend_adjustment_deg=stat(np.degrees(abs(cat('adjustment')))))


def main():
    data_dir=OUT/'recorded_state_replans'
    manifest=json.loads((data_dir/'manifest.json').read_text())
    assert len(manifest['episodes']) == 11
    assert all(e['full_recorded_episode'] for e in manifest['episodes'])
    bag=defaultdict(list);by_cohort=defaultdict(list);rows=[];slots=Counter()
    within=[];between=[];regen_errors=[];example=None
    for meta in manifest['episodes']:
        path=ROOT/meta['source']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['source_sha256']
        assert not(path.parent.name.endswith('022352_015441') and meta['source_episode']==6)
        z=np.load(data_dir/meta['file'])
        mean,delta=z['mean'],z['regenerated_delta']
        base=z['command_base'];bound=meta['bounds'];period=meta['ctrl_period']
        # Restore integer microseconds before subtraction to avoid epoch-float
        # rounding moving a publication across an exact 64 ms slot boundary.
        ticks=np.rint(z['published_s']*1e6).astype(np.int64)
        times=(ticks-ticks[0])*1e-6
        regen_errors.extend(np.degrees(abs(delta-z['recorded_delta'])).ravel())
        for alpha in ALPHAS:
            blend=TemporalPlanBlend(alpha,period);sent=[];adjustments=[]
            for i,t in enumerate(times):
                action,context=blend.prepare(mean[i],delta[i],t)
                blend.commit(context,t)
                sent.append(action);adjustments.append(action-delta[i])
                if alpha==1:
                    record=blend.metrics(context)
                    slots[record['status'] if record['status']!='overlap' else str(record['old_slot'])]+=1
                    within.extend(record['within_plan_diff_l2'])
                    between.extend(record['aligned_plan_diff_l2'] or [])
            sent=np.asarray(sent)
            if alpha==1:np.testing.assert_array_equal(sent,delta)
            limited=np.clip(sent,-np.asarray(bound['lower']),np.asarray(bound['upper']))
            target=base+limited
            item=dict(delta=sent,limited=limited,jumps=np.diff(target,axis=0),
                delta_jumps=np.diff(sent,axis=0),adjustment=np.asarray(adjustments))
            bag[alpha].append(item);by_cohort[(str(bound['upper']),alpha)].append(item)
            summary=summarize([item])
            rows.append(dict(source=meta['source'],episode=meta['source_episode'],
                original_reference_outcome=meta['outcome'],bound=bound['upper'],alpha=alpha,
                steps=len(sent),target_jump_mean_deg=summary['estimated_target_jump_deg']['mean'],
                target_jump_p95_deg=summary['estimated_target_jump_deg']['p95'],
                clipped_actions_percent=summary['clipped_actions_percent'],
                clipped_joint_values_percent=summary['clipped_joint_values_percent'],
                clip_removed_mean_rad=summary['clip_removed_mean_rad'],
                delta_abs_mean_deg=summary['delta_abs_deg']['mean']))
            if meta['source'].endswith('022352_015441/controller.json') and meta['source_episode']==2:
                if example is None:example=dict(times=times,delta=delta,targets={})
                example['targets'][alpha]=target
    summary={str(a):summarize(bag[a]) for a in ALPHAS}
    baseline=summary['1.0']
    for alpha in ALPHAS:
        g=summary[str(alpha)]
        g['mean_target_jump_change_percent']=100*(g['estimated_target_jump_deg']['mean']/baseline['estimated_target_jump_deg']['mean']-1)
        g['p95_target_jump_change_percent']=100*(g['estimated_target_jump_deg']['p95']/baseline['estimated_target_jump_deg']['p95']-1)
        comparisons=[]
        for b,a in zip(bag[1.],bag[alpha]):
            comparisons.append(float(100*(np.mean(abs(a['jumps']))/np.mean(abs(b['jumps']))-1)))
        g['paired_episode_jump_change_percent']=comparisons
        g['episodes_with_smaller_mean_jump']=sum(x<0 for x in comparisons)
    assert baseline['steps'] == 1969
    report=dict(method=manifest['method'],episodes=len(manifest['episodes']),steps=baseline['steps'],
        original_reference_outcomes=dict(Counter(e['outcome'] for e in manifest['episodes'])),
        regenerated_vs_recorded_delta_error_deg=stat(regen_errors),
        within_plan_delta_difference_l2_rad=stat(within),
        aligned_plan_delta_difference_l2_rad=stat(between),alignment_slots=dict(slots),
        alpha_results=summary,
        cohorts={f'bound_{b}_alpha_{a}':summarize(items) for (b,a),items in by_cohort.items()},
        regeneration_wall_time_s=manifest['elapsed_s'],
        exclusions=['Episode 6 of the latest recorded run, per user instruction.', 'All aborted/incomplete episodes.'],
        limitations=manifest['limitations']+[
            'Regenerated first actions diverge materially from the historical ones after several replans; these are reconstructed forecasts, not recovered original forecasts.',
            'The full alpha sweep shares identical recorded states, forecasts and publication times. It measures signal transformations, not the resulting hand/object dynamics.',
            'Hardware clipping and target differences are controller-side estimates using recorded post-planning joint feedback, not counterfactual bridge-written targets.',
            'Original success/drop labels describe only the recordings. No blended success rate is inferred.',
            'Mean/std or percentile reductions do not prove improved grip, tracking, safety or task performance.'])
    (OUT/'offline_report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'offline_episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),constrained_layout=True)
    xx=np.arange(len(ALPHAS))
    for ax,field,title in [(axes[0],'mean_target_jump_change_percent','Mean absolute target change'),
                            (axes[1],'p95_target_jump_change_percent','95th-percentile target change')]:
        ax.bar(xx,[summary[str(a)][field] for a in ALPHAS]);ax.axhline(0,color='black',lw=1)
        ax.set(ylabel='Change from alpha=1 (%)',title=title)
    for bound in sorted({k[0] for k in by_cohort}):
        axes[2].plot(xx,[report['cohorts'][f'bound_{bound}_alpha_{a}']['clipped_actions_percent'] for a in ALPHAS],'-o',label=f'Bound ±{bound}')
    axes[2].set(ylabel='Estimated clipped actions (%)',title='Clipping after blending');axes[2].legend()
    for ax in axes:ax.set_xticks(xx,ALPHAS);ax.set_xlabel('New-plan weight alpha');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Offline regenerated-plan signal test; recorded states held fixed')
    for ext in ['png','pdf']:fig.savefig(OUT/f'offline_blend_effect.{ext}',dpi=170)
    plt.close(fig)
    if example:
        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,constrained_layout=True)
        selected=example['times']<8
        for j,ax in zip([2,6],axes):
            for a in [1.,.8,.6,.4]:ax.plot(example['times'][selected],np.degrees(example['targets'][a][selected,j]),label=f'alpha={a}')
            ax.set_ylabel(f'Joint {j} target (deg)');ax.grid(alpha=.2);ax.legend(ncol=4)
        axes[-1].set_xlabel('Recorded time since first publication (s)')
        fig.suptitle('Example: regenerated targets on one recorded successful episode\nSuccess label belongs to the recording, not these alternative commands')
        for ext in ['png','pdf']:fig.savefig(OUT/f'offline_target_example.{ext}',dpi=170)
        plt.close(fig)
    lines=['# Offline temporal-plan blending test','',
        f'Regenerated {report["steps"]} plans across {report["episodes"]} completed recorded episodes. No hardware or LCM clients were created. The final episode of the latest run remains excluded. The two source runs use ±.2 unpaced and ±.15 paced clipping configurations, trial-314 weights, P011, calibrated high/high model, 70 g cube, friction .3/.5/.4/.4, noise .05, temperature .1, N512 and H=.32 s.',
        '', '**Important:** old logs did not store full trajectories or exact observer activations. Recreated first actions match initially but diverge during warm-started replay. This is a regenerated-plan command-signal test on fixed recorded states, not an exact historical replay or a closed-loop success test.',
        '',f'Mean regenerated-versus-recorded delta difference: {report["regenerated_vs_recorded_delta_error_deg"]["mean"]:.3f}° per joint value; P95 {report["regenerated_vs_recorded_delta_error_deg"]["p95"]:.3f}°. This limits direct inference about the original actions.',
        '', '| Alpha | Mean target jump (deg/joint) | P95 jump (deg/joint) | Mean jump change | Clipped actions | Clipped joint values | Delta magnitude (deg) |',
        '|---|---|---|---|---|---|---|']
    for a in ALPHAS:
        g=summary[str(a)];lines.append(f'| {a} | {g["estimated_target_jump_deg"]["mean"]:.3f} | {g["estimated_target_jump_deg"]["p95"]:.3f} | {g["mean_target_jump_change_percent"]:+.1f}% | {g["clipped_actions_percent"]:.1f}% | {g["clipped_joint_values_percent"]:.1f}% | {g["delta_abs_deg"]["mean"]:.3f} |')
    lines+=['','Target jumps are differences between consecutive clipped absolute target estimates; samples across episode boundaries are never differenced. The mean column is joint/step-weighted. Per-episode paired changes are in offline_report.json and offline_episodes.csv. All alphas use identical raw plans; warm-start updates and noise are never altered by the blending calculation.',
        '',f'Alignment slots: {report["alignment_slots"]}. History resets every episode. Alpha=1 was checked for exact equality with the regenerated unblended deltas.',
        '', 'With shifted warm start and old slot 1, blending effectively damps the first-row optimizer update: old_mean[1] + alpha*(new_mean[0]-old_mean[1]). It leaves future raw means unchanged. This explains why it can suppress both sampling variation and useful corrections.',
        '', '## Interpretation', '',
        'The signal-smoothing benefit is modest. At alpha=.6 the mean target jump falls 4.9% and the P95 falls 4.5%, while clipping falls from 59.1% to 49.2% of actions. All 11 regenerated episode sequences have a smaller mean target jump at this alpha. Mean delta magnitude falls from 4.145 to 3.801 degrees (8.3%), so this is not a pure continuity improvement at unchanged command strength.',
        '', 'There are diminishing returns: alpha=.2 improves mean target jumps by only 6.5%, versus 4.9% at .6, despite discarding much more of the newest correction. The maximum individual joint-target jump actually rises from 28.87 to 30.30 degrees. Temporal blending is not a hard target-slew bound.',
        '', 'Mean within-plan delta differences are .359 rad in 16-joint L2 norm, compared with .163 rad for time-aligned between-plan differences. In these regenerated plans, substantial variation is already present within the horizon; damping replan disagreement addresses only part of the command variation. These numbers do not demonstrate that the within-plan variation is unnecessary or physically harmful.',
        '', 'The conservative next step is an alpha=1 --plan-blend-metrics baseline on a future normal hardware experiment, whenever the operator chooses to run one. It records actual full plans, avoiding this reconstruction limitation. Alpha=.8 would be a mild subsequent candidate; the offline test does not justify claiming a manipulation improvement or selecting an aggressive alpha. Production alpha remains 1.',
        '', 'Limitations:']+['- '+v for v in report['limitations']]
    (OUT/'offline_review.md').write_text('\n'.join(lines)+'\n')
    body='<meta charset="utf-8"><title>Offline temporal blending</title><style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:20px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
    body+='<pre>'+html.escape('\n'.join(lines))+'</pre><img src="offline_blend_effect.png"><img src="offline_target_example.png">'
    (OUT/'offline_index.html').write_text(body)
    print(json.dumps({k:report[k] for k in ['episodes','steps','regenerated_vs_recorded_delta_error_deg','alignment_slots']},indent=2))
    for a,g in summary.items():print(a,{k:g[k] for k in ['mean_target_jump_change_percent','p95_target_jump_change_percent','clipped_actions_percent','clipped_joint_values_percent','episodes_with_smaller_mean_jump']})


if __name__=='__main__':main()
