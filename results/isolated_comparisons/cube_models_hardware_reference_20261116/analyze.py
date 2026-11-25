"""Summarize matched rollout-model trials against the common MuJoCo evaluator."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/'manifest.json').read_text())
MODELS=M['models'];assert MODELS==['M1','M2','M3','M4','M6']
LABELS=M['weight_sets']
summary={};episodes={};episode_rows=[];presets={}
for label in LABELS:
    old=Path(M['reused_results'][label]['M2'])
    base_config=json.loads((old.parent/'config.json').read_text())
    baseline=[json.loads((old/f'episode_{case+1:02d}.json').read_text()) for case in range(12)]
    summary[label]={};presets[label]={}
    for model in MODELS:
        folder=old if model=='M2' else ROOT/label/model/'hardware'
        config=json.loads((folder.parent/'config.json').read_text())
        for key in ['weights','references','initial_hand','target_pos','target_quat','fallen_z','failure_z',
                    'mass_kg','friction','hand','hand_acc','obj_acc','eval_sim','n_samples','n_iterations',
                    'temperature','noise_sigma','warm_start','control_mode','delta_range','hardware_target_clip',
                    'action_rate_weight','timestep','ctrl_time_step','time_horizon','max_steps','settle_s','seeds']:
            assert config[key]==base_config[key],(label,model,key)
        es=[json.loads((folder/f'episode_{case+1:02d}.json').read_text()) for case in range(12)]
        assert all(e['status']=='complete' for e in es),(label,model)
        episodes[label,model]=es
        arrays=[]
        for case,e in enumerate(es):
            b=baseline[case]
            for key in ['initial_q','initial_v','initial_u','initial_state_sha256','seed','noise_seed',
                        'initial_resample_count','planner_config','joint_reference']:
                assert e[key]==b[key],(label,model,case,key)
            assert e['initial_resample_count']==0
            episode_rows.append(dict(weight_set=label,model=model,case=case+1,rotation=e['rotation'],
                repetition=e['repetition'],seed=e['seed'],outcome=e['outcome'],success=e['success'],
                steps=e['n_steps'],simulation_s=e['simulation_s'],pos_mm=e['final_errors']['pos']*1000,
                angle_deg=e['final_errors']['angle_deg'],failed_plan_ms=e.get('failed_plan_ms'),
                finite_rollout_costs_at_failure=e.get('finite_rollout_costs')))
            arrays.append(np.load(folder/f'episode_{case+1:02d}.npz'))
        presets[label][model]=es[0]['rollout_config']
        valid=[e for e in es if e['n_steps']>0]
        good=[e for e in es if e['success']]
        plan_arrays=[a['planning_s'] for a in arrays if len(a['planning_s'])]
        plans=np.concatenate(plan_arrays) if plan_arrays else np.array([])
        outcomes=dict(Counter(e['outcome'] for e in es))
        stats=dict(attempts=12,episodes_with_applied_actions=len(valid),successes=len(good),success_rate=len(good)/12,
            outcomes=outcomes,success_time_median_sim_s=float(np.median([e['simulation_s'] for e in good])) if good else None,
            final_position_mean_mm=float(np.mean([e['final_errors']['pos']*1000 for e in valid])) if valid else None,
            final_position_median_mm=float(np.median([e['final_errors']['pos']*1000 for e in valid])) if valid else None,
            final_angle_mean_deg=float(np.mean([e['final_errors']['angle_deg'] for e in valid])) if valid else None,
            final_angle_median_deg=float(np.median([e['final_errors']['angle_deg'] for e in valid])) if valid else None,
            total_steps=sum(e['n_steps'] for e in es),
            planning_mean_ms=float(plans.mean()*1000) if len(plans) else None,
            planning_p95_ms=float(np.percentile(plans,95)*1000) if len(plans) else None,
            software_clipped_steps=sum(e['hardware_clipped_steps'] for e in es),
            native_actuator_range_exceeded_steps=sum(e['targets_outside_native_actuator_range_steps'] for e in es),
            reused=model=='M2')
        summary[label][model]=stats
        for a in arrays:a.close()

aggregate={}
for label in LABELS:
    successes=sum(summary[label][model]['successes'] for model in MODELS)
    outcomes=Counter()
    for model in MODELS:outcomes.update(summary[label][model]['outcomes'])
    aggregate[label]=dict(successes=successes,attempts=60,success_rate=successes/60,
                          worst_model_successes=min(summary[label][model]['successes'] for model in MODELS),
                          outcomes=dict(outcomes))
report=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),summary=summary,aggregate=aggregate,
            matched_inputs_verified=True,models=MODELS,weight_sets=LABELS,backend_configs=presets,
            notes=['Rollout model is varied; the evaluation simulator is MuJoCo for every model.',
                   'M2 results are reused from the completed matching hardware-reference simulation runs.',
                   '12 matched attempts/model: the same six signed 90-degree initial rotations, twice; seeds 64-75.',
                   'Uncalibrated high/high, 70g cube, sliding friction object/tips/palm/fingers=.3/.5/.4/.4.',
                   'N=256, one MPPI iteration, warm-start shift, H=.32s, control=.064s, physics=.004s, T=.1, sigma=.05.',
                   'Fixed target z=.08m, fallen-penalty z=.065m, failure below z=0, maximum 1000 control steps.',
                   'No software target/delta clipping. Native XML actuator/joint limits remain.',
                   'Model-specific preset/task contact parameters are preserved without tuning.',
                   'All start from the same hardware robot_qref at zero velocity with no settling. The shared initial geometry has up to 7.74mm cube/finger penetration.',
                   'Planner failures are unsuccessful attempts, reported separately from physical drops/timeouts. Error summaries exclude attempts with no applied actions.',
                   'Private MPS and changing concurrent workload mean planning times are observational, not isolated GPU speed benchmarks.',
                   'Four resource-interrupted attempts were archived and restarted after unrelated GPU jobs reduced memory headroom; interrupted attempts are excluded, and completed episodes were retained.',
                   'Simulation evidence for this initialization does not establish hardware success rates.'])
(ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (ROOT/'episodes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(episode_rows[0]));w.writeheader();w.writerows(episode_rows)
metrics=[]
for label in LABELS:
    for model in MODELS:metrics.append(dict(weight_set=label,model=model,**summary[label][model]))
with (ROOT/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def finish(fig,name):
    fig.tight_layout();fig.savefig(ROOT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(len(LABELS),1,figsize=(8,4*len(LABELS)),squeeze=False)
for ax,label in zip(axs[:,0],LABELS):
    vals=[summary[label][model]['success_rate'] for model in MODELS]
    bars=ax.bar(MODELS,vals,color=['#2878b5','#439a68','#d09535','#9b6bb0','#d46d5a'])
    for b,model in zip(bars,MODELS):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.025,f"{summary[label][model]['successes']}/12",ha='center')
    ax.set(ylim=(0,1.15),ylabel='Success rate',title=f'{label} weights — hardware joint-cost reference')
finish(fig,'success_by_model')

fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(5);width=.75/len(LABELS)
for i,label in enumerate(LABELS):
    values=[summary[label][model]['success_rate'] for model in MODELS]
    bars=ax.bar(x+(i-(len(LABELS)-1)/2)*width,values,width=width,label=label)
    for b,model in zip(bars,MODELS):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,f"{summary[label][model]['successes']}/12",ha='center',fontsize=8)
ax.set(xticks=x,xticklabels=MODELS,ylim=(0,1.15),ylabel='Success rate',title='Hardware joint-cost reference: weight sets across rollout models')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False);finish(fig,'weights_by_model')

fig,axs=plt.subplots(len(LABELS),1,figsize=(9,4*len(LABELS)),squeeze=False)
rotations=M['settings']['rotations']
for ax,label in zip(axs[:,0],LABELS):
    data=np.array([[sum(e['success'] for e in episodes[label,model] if e['rotation']==d) for d in rotations] for model in MODELS])
    ax.imshow(data,vmin=0,vmax=2,cmap='YlGnBu',aspect='auto')
    for i in range(5):
        for j in range(6):ax.text(j,i,f'{data[i,j]}/2',ha='center',va='center',color='white' if data[i,j]==2 else 'black')
    ax.set(yticks=np.arange(5),yticklabels=MODELS,xticks=np.arange(6),xticklabels=rotations,
           xlabel='Initial signed 90-degree rotation',title=f'{label}: success by initial rotation')
finish(fig,'success_by_rotation')

fig,axs=plt.subplots(len(LABELS),1,figsize=(9,4*len(LABELS)),squeeze=False)
outcome_colors={'success':'#439a68','timeout':'#d09535','drop':'#d46d5a','planner_failure':'#9b6bb0','numerical_failure':'#535353'}
for ax,label in zip(axs[:,0],LABELS):
    bottom=np.zeros(5)
    for outcome,color in outcome_colors.items():
        values=np.array([summary[label][model]['outcomes'].get(outcome,0) for model in MODELS])
        ax.bar(MODELS,values,bottom=bottom,color=color,label=outcome.replace('_',' '));bottom+=values
    ax.set(ylim=(0,12.5),ylabel='Attempts',title=f'{label}: outcomes')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=8,frameon=False)
finish(fig,'outcomes')

fig,axs=plt.subplots(len(LABELS),2,figsize=(12,4*len(LABELS)),squeeze=False)
for i,label in enumerate(LABELS):
    for ax,key,scale,title,threshold in [(axs[i,0],'pos',1000,'Final position error (mm)',20),
                                       (axs[i,1],'angle_deg',1,'Final orientation error (degrees)',float(np.rad2deg(2*np.arcsin(np.sqrt(.04)))) )]:
        rng=np.random.default_rng(0)
        for x,model in enumerate(MODELS):
            vals=[e['final_errors'][key]*scale for e in episodes[label,model] if e['n_steps']>0]
            if not vals:
                ax.annotate('No valid\nactions',(x,threshold),ha='center',fontsize=8)
                continue
            ax.scatter(x+rng.uniform(-.15,.15,len(vals)),vals,s=22,alpha=.65,color='#2878b5')
            ax.plot([x-.2,x+.2],[np.median(vals)]*2,color='#d46d5a',lw=2)
        ax.axhline(threshold,color='.3',ls=':');ax.set(xticks=np.arange(5),xticklabels=MODELS,xlim=(-.5,4.5),
            ylabel=title,title=f'{label}: {title}',ylim=(0,None))
finish(fig,'final_errors')

def fmt(v):return '—' if v is None else f'{v:.2f}'
html=['<html><head><meta charset="utf-8"><title>Cube contact-model comparison</title></head><body>',
      '<h1>Cube rollout models — hardware joint-cost reference</h1>',
      '<p>MuJoCo evaluation; M1/M2/M3/M4/M6; M5 excluded. Production settings unchanged.</p>']
html.append('<table border="1" cellpadding="6"><tr><th>Weight set</th><th>Total successes</th><th>Worst-model successes</th><th>Outcomes</th></tr>')
for label,a in aggregate.items():
    html.append(f"<tr><td>{label}</td><td>{a['successes']}/60</td><td>{a['worst_model_successes']}/12</td><td>{a['outcomes']}</td></tr>")
html.append('</table>')
for label in LABELS:
    html.append(f'<h2>{label}</h2><table border="1" cellpadding="6"><tr><th>Model</th><th>Success</th><th>Timeout</th><th>Drop</th><th>Planning failure</th><th>Mean final pos (mm)</th><th>Mean final angle (deg)</th><th>Median success time (sim s)</th></tr>')
    for model in MODELS:
        s=summary[label][model];o=s['outcomes'];vals=[model,f"{s['successes']}/12",o.get('timeout',0),o.get('drop',0),o.get('planner_failure',0)+o.get('numerical_failure',0),fmt(s['final_position_mean_mm']),fmt(s['final_angle_mean_deg']),fmt(s['success_time_median_sim_s'])]
        html.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in vals)+'</tr>')
    html.append('</table>')
for note in report['notes']:html.append('<p>'+note+'</p>')
for name in ['weights_by_model','success_by_model','success_by_rotation','outcomes','final_errors']:html.append(f'<p><img src="{name}.png" style="max-width:100%"></p>')
html.append('</body></html>');(ROOT/'index.html').write_text('\n'.join(html))
print(json.dumps(summary,indent=2))
