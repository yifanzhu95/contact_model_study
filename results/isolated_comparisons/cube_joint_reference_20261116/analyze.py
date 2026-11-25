"""Summarize the matched simulation runs without loading a GPU or hardware code."""
from pathlib import Path
from collections import Counter
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
C=json.loads((OUT/'config.json').read_text())
VARIANTS=['optimization','hardware']
LABELS=['Optimization reference','Hardware reference']
COLORS=['#2878b5','#e07828']
records={name:[json.loads((OUT/name/f'episode_{i+1:02d}.json').read_text()) for i in range(12)] for name in VARIANTS}
for name,episodes in records.items():
    assert len(episodes)==12 and all(e['status']=='complete' for e in episodes)
for a,b in zip(records['optimization'],records['hardware']):
    for key in ['case','rotation','repetition','seed','initial_q','initial_v','initial_u',
                'initial_state_sha256','noise_seed','initial_resample_count','planner_config','rollout_config']:
        assert a[key]==b[key],(key,a['case'])
    assert a['initial_resample_count']==0
    assert np.allclose(a['initial_q'][16:19],C['target_pos'])
    assert np.isclose(abs(a['initial_q'][19]),np.sqrt(.5))

metrics={};rows=[]
for name,episodes in records.items():
    arrays=[np.load(OUT/name/f'episode_{i+1:02d}.npz') for i in range(12)]
    plans=np.concatenate([a['planning_s'] for a in arrays])
    changes=np.concatenate([a['target_delta'] for a in arrays])
    successes=[e for e in episodes if e['success']]
    metrics[name]=dict(episodes=12,successes=len(successes),success_rate=len(successes)/12,
        outcomes=dict(Counter(e['outcome'] for e in episodes)),
        final_angle_mean_deg=float(np.mean([e['final_errors']['angle_deg'] for e in episodes])),
        final_angle_median_deg=float(np.median([e['final_errors']['angle_deg'] for e in episodes])),
        final_position_mean_mm=float(np.mean([e['final_errors']['pos'] for e in episodes])*1000),
        minimum_angle_mean_deg=float(np.mean([e['minimum_angle_deg'] for e in episodes])),
        success_time_median_sim_s=float(np.median([e['simulation_s'] for e in successes])) if successes else None,
        total_steps=int(sum(e['n_steps'] for e in episodes)),
        planning_mean_ms=float(plans.mean()*1000),planning_p95_ms=float(np.percentile(plans,95)*1000),
        target_change_mean_max_joint_rad=float(np.abs(changes).max(axis=1).mean()),
        hardware_clipped_steps=0,
        outside_native_actuator_range_steps=sum(e['targets_outside_native_actuator_range_steps'] for e in episodes),
        outside_native_actuator_range_joints=sum(e['targets_outside_native_actuator_range_joints'] for e in episodes))
    for e in episodes:
        rows.append(dict(variant=name,case=e['case'],rotation=e['rotation'],repetition=e['repetition'],seed=e['seed'],
             outcome=e['outcome'],steps=e['n_steps'],simulation_s=e['simulation_s'],
             final_angle_deg=e['final_errors']['angle_deg'],final_position_mm=e['final_errors']['pos']*1000,
             minimum_angle_deg=e['minimum_angle_deg'],planning_mean_ms=e['planning_mean_ms'],
             native_actuator_range_steps=e['targets_outside_native_actuator_range_steps']))
    for a in arrays:a.close()
paired=Counter()
for a,b in zip(records['optimization'],records['hardware']):
    paired['both_success' if a['success'] and b['success'] else
           'optimization_only_success' if a['success'] else
           'hardware_only_success' if b['success'] else 'neither_success']+=1
report=dict(settings=C,metrics=metrics,paired_successes=dict(paired),paired_input_validation_passed=True,
            caveats=['12 matched pairs, two MPPI seeds per rotation; simulation evidence only.',
             'Each start is exactly 90 degrees from the fixed identity goal; zero settling preserves that start.',
             'Initial hand is hardware robot_qref in both variants; only joint-cost reference changes.',
             'The common initial cube/hand geometry overlaps by up to 7.74 mm; this is a controlled comparison of that initialization, not a calibrated hardware success-rate prediction.',
             'No hardware ctrl_range clipping and no planner delta clipping. Original XML actuator and joint limits remain.',
             'M2 GPU rollouts, MuJoCo CPU evaluation, original uncalibrated high/high hand.'])
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'episodes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'metrics.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['metric',*LABELS])
    for k in metrics['optimization']:w.writerow([k,*[metrics[n][k] for n in VARIANTS]])

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(5.8,4))
counts=[metrics[n]['successes'] for n in VARIANTS]
bars=ax.bar(LABELS,np.array(counts)/12,color=COLORS,width=.55)
for b,k in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.035,f'{k}/12',ha='center')
ax.set(ylim=(0,1.12),ylabel='Success rate',title='Joint-cost reference comparison — same trial-622 weights')
finish(fig,'success')

fig,ax=plt.subplots(figsize=(8,4))
x=np.arange(6)
for j,name in enumerate(VARIANTS):
    counts=[sum(e['success'] for e in records[name] if e['rotation']==d) for d in C['rotations']]
    bars=ax.bar(x+(j-.5)*.36,np.array(counts)/2,width=.36,color=COLORS[j],label=LABELS[j])
    for b,k in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.025,f'{k}/2',ha='center',fontsize=8)
ax.set(xticks=x,xticklabels=C['rotations'],ylim=(0,1.18),ylabel='Success rate',xlabel='Initial rotation from goal (90°)',title='Success by matched initial rotation')
ax.legend(loc='upper right',fontsize=8);finish(fig,'success_by_rotation')

fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for ax,key,scale,label,threshold in [(axs[0],'angle_deg',1,'Final orientation error (degrees)',np.rad2deg(2*np.arcsin(np.sqrt(.04)))),(axs[1],'pos',1000,'Final position error (mm)',20)]:
    for j,name in enumerate(VARIANTS):
        values=np.array([e['final_errors'][key]*scale for e in records[name]])
        ax.plot(np.arange(1,13),values,marker='o' if j==0 else '^',color=COLORS[j],label=LABELS[j])
    ax.axhline(threshold,ls=':',color='.3',label='Success threshold')
    ax.set(xticks=np.arange(1,13),xticklabels=[f'{d}\n{1+i//6}' for i,d in enumerate(C['rotations']*2)],
           xlabel='Initial rotation / repeat (matched pairs)',ylabel=label,ylim=(0,None))
    ax.legend(fontsize=8)
finish(fig,'paired_final_errors')

fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
for col,name in enumerate(VARIANTS):
    for i,e in enumerate(records[name]):
        with np.load(OUT/name/f'episode_{i+1:02d}.npz') as data:
            t=np.arange(e['n_steps']+1)*.064
            angle=np.r_[data['angle_deg'],e['final_errors']['angle_deg']]
            pos=np.r_[data['pos_error']*1000,e['final_errors']['pos']*1000]
            for ax,y in [(axs[0,col],angle),(axs[1,col],pos)]:
                ax.plot(t,y,color=plt.cm.tab10(i%6),alpha=.7,lw=1,ls='-' if i<6 else '--')
    axs[0,col].axhline(np.rad2deg(2*np.arcsin(np.sqrt(.04))),color='.25',ls=':',label='Success threshold')
    axs[1,col].axhline(20,color='.25',ls=':')
    axs[0,col].set(title=LABELS[col],ylabel='Orientation error (degrees)')
    axs[1,col].set(ylabel='Position error (mm)',xlabel='Simulation time (s)')
finish(fig,'trajectories')

lines=['<html><head><meta charset="utf-8"><title>Cube joint-reference comparison</title></head><body>',
       '<h1>Cube joint-cost reference comparison</h1>',
       '<p>Trial 622; uncalibrated high/high; M2; N=256; horizon=.32 s; control=.064 s; shift; T=.1; sigma=.05.</p>',
       '<p>Mass=.07 kg; friction object/tips/palm/fingers=.3/.5/.4/.4; target z=.08 m; fallen-penalty z=.065 m.</p>',
       '<table border="1" cellpadding="6"><tr><th>Metric</th><th>Optimization reference</th><th>Hardware reference</th></tr>']
for key in metrics['optimization']:
    lines.append('<tr><td>'+key+'</td>'+''.join('<td>'+str(metrics[n][key])+'</td>' for n in VARIANTS)+'</tr>')
lines.append('</table><p>Paired successes: '+str(dict(paired))+'</p>')
for caveat in report['caveats']:lines.append('<p>'+caveat+'</p>')
for name in ['success','success_by_rotation','paired_final_errors','trajectories']:
    lines.append(f'<p><img src="{name}.png" style="max-width:100%"></p>')
lines.append('</body></html>');(OUT/'index.html').write_text('\n'.join(lines))
print(json.dumps(dict(metrics=metrics,paired_successes=dict(paired),paired_input_validation_passed=True),indent=2))
