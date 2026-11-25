"""Signal-trajectory comparison for the offline action-rate ablation."""
from collections import defaultdict
import argparse
import csv
import html
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2];sys.path.insert(0,str(ROOT))
from ContactModelStudy.Hardware.PlanBlending import TemporalPlanBlend

CACHE=OUT/'recorded_state_replans';DEST=OUT/'rate_study'
LABELS={'baseline':'Baseline','blend':'Blending only','rate':'Rate penalty only','rate_blend':'Rate + blending'}
COLORS=dict(baseline='#777777',blend='#377eb8',rate='#ff7f00',rate_blend='#228833')


def stats(a):
    a=np.asarray(a,float).ravel();a=a[np.isfinite(a)]
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
        p95=float(np.percentile(a,95)),max=float(a.max()),rms=float(np.sqrt(np.mean(a*a))))


def metrics(items):
    cat=lambda field:np.concatenate([r[field] for r in items])
    sent,limited=cat('sent'),cat('limited');jump=cat('jump')
    return dict(episodes=len(items),steps=len(sent),
        target_jump_deg=stats(np.degrees(abs(jump))),
        target_rate_deg_s=stats(np.degrees(abs(cat('target_rate')))),
        target_acceleration_deg_s2=stats(np.degrees(abs(cat('target_acceleration')))),
        raw_planned_target_jump_deg=stats(np.degrees(abs(cat('raw_planned_jump')))),
        blended_planned_target_jump_deg=stats(np.degrees(abs(cat('blended_planned_jump')))),
        requested_target_jump_deg=stats(np.degrees(abs(cat('requested_jump')))),
        sent_delta_abs_deg=stats(np.degrees(abs(sent))),
        servo_offset_abs_deg=stats(np.degrees(abs(limited))),
        within_plan_delta_l2=stats(cat('within')),
        between_plan_delta_l2=stats(cat('between')),
        candidate_ess=stats(cat('ess')),
        fraction_steps_ess_below2=float(np.mean(cat('ess')<2)),
        clipped_actions_percent=float(100*np.mean(np.any(abs(sent-limited)>1e-7,axis=1))),
        clipped_joint_percent=float(100*np.mean(abs(sent-limited)>1e-7)),
        clip_removed_mean_rad=float(np.mean(abs(sent-limited))),
        expected_candidate_task_cost=stats(cat('expected_task_cost')) if 'expected_task_cost' in items[0] else None,
        expected_candidate_rate=stats(cat('expected_rate')) if 'expected_rate' in items[0] else None)


def main():
    global DEST
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired',action='store_true')
    args=parser.parse_args()
    cached=json.loads((CACHE/'manifest.json').read_text())
    calibration=json.loads((DEST/'calibration.json').read_text())
    if args.paired:
        DEST=DEST/'paired'
        common_manifest=json.loads((DEST/'manifest.json').read_text())
        manifests={case:common_manifest for case in LABELS}
    else:
        manifests={case:json.loads((DEST/case/'manifest.json').read_text()) for case in ['baseline_check','rate','rate_blend']}
    assert len(cached['episodes'])==11
    for m in manifests.values():assert len(m['episodes'])==11 and sum(e['steps'] for e in m['episodes'])==1969
    bag=defaultdict(list);cohorts=defaultdict(list);rows=[];repro=[];example=None
    for i,meta in enumerate(cached['episodes']):
        saved=np.load(CACHE/meta['file'])
        q=saved['q'][:,:16];base=saved['command_base'];n=len(q)
        ticks=np.rint(saved['published_s']*1e6).astype(np.int64);times=(ticks-ticks[0])*1e-6
        dt=np.diff(times);assert np.all(dt>0)
        low,high=meta['bounds']['lower'],meta['bounds']['upper']
        packets={case:np.load(DEST/case/meta['file']) for case in manifests}
        for case in packets:
            assert manifests[case]['episodes'][i]['file']==meta['file']
            assert packets[case]['raw_delta'].shape==(n,16)
        repro.extend(np.degrees(abs(packets['baseline' if args.paired else 'baseline_check']['raw_delta']-saved['regenerated_delta'])).ravel())
        all_cases=(packets if args.paired else
                   {'baseline':packets['baseline_check'],'blend':packets['baseline_check'],
                    'rate':packets['rate'],'rate_blend':packets['rate_blend']})
        for case,packet in all_cases.items():
            alpha=calibration['blending_alpha'] if case in ('blend','rate_blend') else 1.
            blend=TemporalPlanBlend(alpha,meta['ctrl_period']);actions=[];within=[];between=[]
            for k,t in enumerate(times):
                action,context=blend.prepare(packet['mean'][k],packet['raw_delta'][k],t)
                blend.commit(context,t);record=blend.metrics(context)
                actions.append(action);within.extend(record['within_plan_diff_l2']);between.extend(record['aligned_plan_diff_l2'] or [])
            sent=np.asarray(actions)
            if args.paired or case!='blend':np.testing.assert_allclose(sent,packet['sent_delta'],rtol=0,atol=1e-12)
            limited=np.clip(sent,-low,high);target=base+limited;jump=np.diff(target,axis=0)
            if args.paired or case!='blend':np.testing.assert_allclose(target,packet['target'],rtol=0,atol=1e-12)
            velocity=jump/dt[:,None]
            item=dict(sent=sent,limited=limited,jump=jump,ess=packet['ess'],
                target_rate=velocity,target_acceleration=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])[:,None]/2),
                raw_planned_jump=np.diff(q+packet['raw_delta'],axis=0),
                blended_planned_jump=np.diff(q+sent,axis=0),requested_jump=np.diff(base+sent,axis=0),
                within=np.asarray(within),between=np.asarray(between))
            if args.paired or case in ('rate','rate_blend'):
                item.update(expected_task_cost=packet['expected_task_cost'],expected_rate=packet['expected_rate'])
            bag[case].append(item);cohorts[(str(high),case)].append(item)
            result=metrics([item])
            rows.append(dict(source=meta['source'],episode=meta['source_episode'],
                original_reference_outcome=meta['outcome'],bound=high,case=case,steps=n,
                mean_target_jump_deg=result['target_jump_deg']['mean'],p95_target_jump_deg=result['target_jump_deg']['p95'],
                max_target_jump_deg=result['target_jump_deg']['max'],clipped_actions_percent=result['clipped_actions_percent'],
                clipped_joint_percent=result['clipped_joint_percent'],mean_servo_offset_deg=result['servo_offset_abs_deg']['mean']))
            if meta['file']=='source1_episode02.npz':
                if example is None:example=dict(times=times,targets={},bound=high)
                example['targets'][case]=target
    aggregate={case:metrics(items) for case,items in bag.items()}
    baseline=aggregate['baseline']
    for case,summary in aggregate.items():
        summary['mean_jump_change_percent']=100*(summary['target_jump_deg']['mean']/baseline['target_jump_deg']['mean']-1)
        summary['p95_jump_change_percent']=100*(summary['target_jump_deg']['p95']/baseline['target_jump_deg']['p95']-1)
        summary['paired_episode_mean_jump_change_percent']=[float(100*(np.mean(abs(a['jump']))/np.mean(abs(b['jump']))-1)) for a,b in zip(bag[case],bag['baseline'])]
        summary['episodes_with_smaller_mean_jump']=sum(x<0 for x in summary['paired_episode_mean_jump_change_percent'])
    report=dict(episodes=11,steps=1969,weight=calibration['selected_weight'],alpha=calibration['blending_alpha'],
        method='paired common-candidate rescoring' if args.paired else 'independent warm-start controller histories',
        calibration=dict(states=calibration['states'],criterion=calibration['criterion'],candidate_stats=calibration['candidates']),
        aggregate=aggregate,by_bound={f'{bound}_{case}':metrics(items) for (bound,case),items in cohorts.items()},
        zero_penalty_reproduction_error_deg=stats(repro),
        limitations=['Fixed recorded hand/object poses, velocities, estimated activations and publication times for every variant.',
            'Counterfactual hand/object trajectories and success rates cannot be inferred from this teacher-forced replay.',
            'Old full plans and exact observer states were not saved; regenerated plans do not faithfully recover the historical plans.',
            ('Paired cases share candidate trajectories, warm-start contexts and recorded prior-target references; feedback-history adaptation is not tested.' if args.paired else
             'Each regularized case maintains its own warm start and prior estimated clipped target, while physical inputs remain fixed.'),
            'Target/clip statistics are controller-side estimates based on recorded post-planning joints, not actual alternative bridge writes.',
            'Candidate task/rate costs are diagnostic model expectations under different sampled distributions, not measured task progress.',
            'Only one noise seed and a heuristic mild weight were tested; this does not establish optimal weights or hardware performance.'])
    if args.paired:report['paired_cpu_vs_gpu_mean_error_deg']=common_manifest['cpu_vs_gpu_mean_error_deg']
    (DEST/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (DEST/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cases=list(LABELS);xx=np.arange(4)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key,title in [(axes[0,0],'target_jump_deg','Mean estimated target change (deg/joint)'),
                         (axes[0,1],'target_jump_deg','P95 estimated target change (deg/joint)'),
                         (axes[1,1],'servo_offset_abs_deg','Mean clipped delta / servo offset (deg)')]:
        metric='p95' if ax is axes[0,1] else 'mean'
        ax.bar(xx,[aggregate[c][key][metric] for c in cases],color=[COLORS[c] for c in cases]);ax.set_title(title)
    for bound in sorted({b for b,_ in cohorts}):
        axes[1,0].plot(xx,[report['by_bound'][f'{bound}_{case}']['clipped_actions_percent'] for case in cases],'-o',label=f'Bound ±{bound}')
    axes[1,0].set_title('Estimated actions clipped (%)');axes[1,0].legend()
    for ax in axes.flat:ax.set_xticks(xx,[LABELS[c].replace(' ','\n') for c in cases]);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Offline recorded-state replanning: action-rate penalty and temporal blending')
    save(fig,'metrics')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    phases=['raw_planned_target_jump_deg','blended_planned_target_jump_deg','requested_target_jump_deg','target_jump_deg']
    for k,case in enumerate(cases):ax.bar(np.arange(4)+(k-1.5)*.19,[aggregate[case][key]['mean'] for key in phases],width=.19,label=LABELS[case],color=COLORS[case])
    ax.set_xticks(range(4),['Raw planner target','After delta blending','After measured-state\nrebasing','After hardware\nclipping estimate'])
    ax.set(ylabel='Mean absolute target change (deg/joint)',title='Where command variation changes in the execution pipeline');ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'pipeline')
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for case in cases:
        x=np.sort(np.degrees(abs(np.concatenate([r['jump'] for r in bag[case]]))).ravel());ax.plot(x,np.arange(1,len(x)+1)/len(x),label=LABELS[case],color=COLORS[case])
    ax.set(xlabel='Estimated clipped target change (deg/joint)',ylabel='Cumulative fraction',title='Command-change distribution on identical recorded state inputs');ax.legend();ax.grid(alpha=.2);save(fig,'jump_distribution')
    if example:
        from ContactModelStudy.Hardware.Scene import JOINTS
        fig,axes=plt.subplots(4,4,figsize=(17,11),sharex=True,constrained_layout=True)
        mask=example['times']<8
        for j,ax in enumerate(axes.flat):
            for case in cases:ax.plot(example['times'][mask],np.degrees(example['targets'][case][mask,j]),lw=1,label=LABELS[case],color=COLORS[case],alpha=.85)
            ax.set_title(JOINTS[j]);ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
        fig.supxlabel('Recorded time since first publication (s)');fig.supylabel('Estimated target (degrees)')
        fig.suptitle('Example commanded trajectories, all 16 joints — recorded states held fixed\nThese curves do not predict the resulting physical hand/object motion')
        save(fig,'joint_trajectories')
    lines=['# Offline action-rate penalty + temporal blending comparison','',
        'Compared 1,969 steps from the same 11 completed recorded episodes as the prior blending study. No hardware/LCM clients were created. The last episode of the latest run remains excluded. No production parameters were changed.',
        '',f'Action-rate weight **{report["weight"]}**, blending alpha **{report["alpha"]}**. All other MPPI/task/physical settings are unchanged: trial-314 weights, P011, N512, noise .05, temperature .1, shifted warm start, .32 s horizon, .064 s control period, calibrated high/high hand, 70 g mass, sliding friction .3/.5/.4/.4. Both ±.2 and ±.15 recorded clipping cohorts are retained and also reported separately.',
        '',f'The weight was chosen on {calibration["states"]} distinct recorded states from a grid {.005,.01,.02,.05,.1,.2}. It is a mild scale-based choice, not an optimized or hardware-validated setting. Selection targeted .25 temperature units of median rate-cost IQR among competitive candidates, subject to mean distribution KL≤.05. At the selected weight the median scaled IQR is {calibration["candidates"][str(report["weight"])]["elite_rate_iqr_in_temperatures"]["median"]:.3f}, mean KL {calibration["candidates"][str(report["weight"])]["kl"]["mean"]:.5f}, and mean first-command change from matched one-step rescoring is {calibration["candidates"][str(report["weight"])]["first_delta_change_deg"]["mean"]:.4f} degrees/joint.',
        '', '## Method', '',
        ('This is a controlled one-step objective ablation. At every recorded state all four cases share the EXACT same candidate trajectories, base task costs, noise and warm-start seed. The regularized case changes only candidate weights using J_task + .2*J_rate; blending is then applied to the mean sequences. The rate reference is the recorded previous estimated target (robot_qref for the first step). This isolates immediate cost effects without numerical replay divergence; it does NOT test a controller adapting its own warm-start and command history. Physical hand/object states, observer activations and publication times remain fixed.' if args.paired else
         'This is a teacher-forced policy replay, not a closed-loop manipulation experiment. All variants receive identical recorded q/v, reconstructed actuator-observer states and publication times. Each regularized variant maintains its own optimized warm-start sequence and previous estimated clipped command. Consequently, penalty+blending is separately replanned; it is not a post-hoc blend of penalty-only plans. The initial rate reference is robot_qref, matching the existing positive-rate-weight hardware path. Blending-only is computed from the zero-penalty plans because previous commands do not affect the position-relative planner when the rate weight is zero and activation is fixed.'),
        '', 'The existing rate penalty is the sum of squared consecutive absolute requested targets, including the first relative to the prior command and the final horizon step. It has no timestep division. Costs and sampling noise are unchanged apart from this added term. A diagnostic copy of the same rate kernel records the rate component without altering task costs.',
        '',f'Baseline delta difference versus the previous offline cache: mean {report["zero_penalty_reproduction_error_deg"]["mean"]:.6f} degrees, P95 {report["zero_penalty_reproduction_error_deg"]["p95"]:.6f}, maximum {report["zero_penalty_reproduction_error_deg"]["max"]:.6f}. '+('For paired cases the candidate means are pinned to common cached contexts, so differences cannot accumulate through warm starts. Cross-case comparisons use the same newly computed costs exactly.' if args.paired else 'The independent zero-penalty repeat differs materially pointwise: long-history improvements cannot be confidently attributed to a small cost change. Use the paired companion report for the isolated immediate effect.'),
        '', '## Estimated command-trajectory effects', '',
        '| Variant | Mean jump (deg/joint) | P95 jump (deg/joint) | Mean jump change | Actions clipped | Clipped joint values | Mean servo offset (deg) |',
        '|---|---|---|---|---|---|---|']
    for case in cases:
        g=aggregate[case];lines.append(f'| {LABELS[case]} | {g["target_jump_deg"]["mean"]:.3f} | {g["target_jump_deg"]["p95"]:.3f} | {g["mean_jump_change_percent"]:+.1f}% | {g["clipped_actions_percent"]:.1f}% | {g["clipped_joint_percent"]:.1f}% | {g["servo_offset_abs_deg"]["mean"]:.3f} |')
    lines+=['','A jump is a difference between consecutive estimated absolute targets after bridge-equivalent clipping around the recorded post-planning measured joints. Episode boundaries are never differenced. Servo offset is the clipped delta magnitude relative to that measurement; it is not actual tracking error or measured contact force. Command rates/accelerations in comparison.json are target-signal derivatives, not physical joint velocity/acceleration.',
        '', '## Per-bound results', '', '| Bound | Variant | Mean jump | Actions clipped |','|---|---|---|---|']
    for bound in sorted({b for b,_ in cohorts}):
        for case in cases:
            g=report['by_bound'][f'{bound}_{case}'];lines.append(f'| ±{bound} | {LABELS[case]} | {g["target_jump_deg"]["mean"]:.3f}° | {g["clipped_actions_percent"]:.1f}% |')
    lines+=['','## Interpretation','']
    if args.paired:
        lines += [
            f'The mild action-rate penalty alone changes mean command jumps by {aggregate["rate"]["mean_jump_change_percent"]:.2f}%, versus {aggregate["blend"]["mean_jump_change_percent"]:.2f}% for blending and {aggregate["rate_blend"]["mean_jump_change_percent"]:.2f}% for their combination. The combination improves mean jumps by only {aggregate["blend"]["target_jump_deg"]["mean"]-aggregate["rate_blend"]["target_jump_deg"]["mean"]:.4f} degrees/joint beyond blending alone. This is little evidence for adding the extra cost at this weight.',
            '',f'The median baseline effective sample size is {baseline["candidate_ess"]["median"]:.2f} out of 512; {100*baseline["fraction_steps_ess_below2"]:.1f}% of steps have ESS below 2. The distribution is often dominated by one candidate, which helps explain why this small cost perturbation leaves the action largely unchanged. This is not a proof that increasing the penalty or temperature would improve manipulation.',
            '',f'CPU common-candidate weighting matches the existing GPU mean calculation to maximum {report["paired_cpu_vs_gpu_mean_error_deg"]["max"]:.8f} degrees. The comparison isolates actual candidate reweighting, rather than applying a target filter in place of the cost.',
            '', 'Blending and the combination lower average clipped servo offsets by about 7%. This may reduce useful holding effort as well as oscillation. The maximum individual target jump is not improved: approximately 28.87 degrees for baseline versus 29.58 for the combination. No setting here is demonstrated to retain the cube better.',
            '', 'Recommendation: keep production defaults unchanged. A genuine saved-plan baseline or controlled closed-loop simulation/hardware validation is required before promoting a penalty/blending setting. Do not increase the penalty merely to minimize command variation: a controller that holds its target or stops manipulating could score well on that signal metric.']
    else:
        lines += ['The independent-history replay is retained as exploratory evidence. Repeating zero penalty changes the command sequence materially on the same physical inputs, so its small differences cannot be confidently attributed to the regularizer. Use paired/report.md for the matched-candidate immediate-effect comparison.']
    lines+=['','## Interpretation limits','']+['- '+v for v in report['limitations']]
    (DEST/'report.md').write_text('\n'.join(lines)+'\n')
    body='<meta charset="utf-8"><title>Offline rate penalty study</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px}img{max-width:100%}pre{white-space:pre-wrap}</style><pre>'+html.escape('\n'.join(lines))+'</pre>'
    for name in ['metrics','pipeline','jump_distribution','joint_trajectories']:body+=f'<img src="{name}.png">'
    (DEST/'index.html').write_text(body)
    print('Reproducibility:',report['zero_penalty_reproduction_error_deg'])
    for case,g in aggregate.items():print(case,{k:g[k] for k in ['target_jump_deg','mean_jump_change_percent','p95_jump_change_percent','clipped_actions_percent','clipped_joint_percent','episodes_with_smaller_mean_jump']})


def save(fig,name):
    for extension in ['png','pdf']:fig.savefig(DEST/f'{name}.{extension}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
