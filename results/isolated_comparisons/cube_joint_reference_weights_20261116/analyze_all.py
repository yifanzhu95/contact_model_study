"""Compare reference effects for original, trial-314, and completed trial-622 weights."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime,timezone
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text())
folders={'original':ROOT/'original','trial314':ROOT/'trial314','trial622':Path(manifest['baseline_trial622'])}
labels={'original':'Original weights','trial314':'Trial 314','trial622':'Trial 622 (previous run)'}
refs=['optimization','hardware'];ref_labels=['Optimization reference','Hardware reference']
colors=['#2878b5','#e07828']
baseline=json.loads((folders['trial622']/'config.json').read_text())
administrative={'created_at_utc','out','code','scenes','weights','weight_label','comparison_parent','inputs_sha256'}
episodes={};summaries={};rows=[]
for label,folder in folders.items():
    config=json.loads((folder/'config.json').read_text())
    assert {k:v for k,v in config.items() if k not in administrative}=={k:v for k,v in baseline.items() if k not in administrative},label
    for role in ['rollout','eval']:
        assert hashlib.sha256(Path(config['scenes'][role]).read_bytes()).hexdigest()==hashlib.sha256(Path(baseline['scenes'][role]).read_bytes()).hexdigest()
    if label!='trial622':
        subprocess.run([sys.executable,str(folder/'analyze.py')],check=True,stdout=subprocess.DEVNULL)
    for ref in refs:
        es=[json.loads((folder/ref/f'episode_{case+1:02d}.json').read_text()) for case in range(12)]
        assert all(e['status']=='complete' for e in es)
        episodes[label,ref]=es
        for case,e in enumerate(es):
            assert e['joint_reference']==config['references'][ref],(label,ref,case,'joint_reference')
            b=json.loads((folders['trial622']/'optimization'/f'episode_{case+1:02d}.json').read_text())
            for key in ['initial_state_sha256','initial_q','initial_v','initial_u','seed','noise_seed',
                        'initial_resample_count','planner_config','rollout_config']:
                assert e[key]==b[key],(label,ref,case,key)
            rows.append(dict(weights=label,reference=ref,case=case+1,rotation=e['rotation'],repetition=e['repetition'],
                 seed=e['seed'],outcome=e['outcome'],success=e['success'],steps=e['n_steps'],
                 simulation_s=e['simulation_s'],final_pos_mm=e['final_errors']['pos']*1000,
                 final_angle_deg=e['final_errors']['angle_deg'],planning_mean_ms=e['planning_mean_ms']))
        own=json.loads((folder/'comparison.json').read_text())['metrics'][ref]
        summaries[label,ref]={**own,'final_position_median_mm':float(np.median([e['final_errors']['pos']*1000 for e in es]))}
with (ROOT/'episodes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics_rows=[]
for label in folders:
    for ref in refs:metrics_rows.append(dict(weights=label,reference=ref,**summaries[label,ref]))
with (ROOT/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics_rows[0]));w.writeheader();w.writerows(metrics_rows)
paired={}
for label in folders:
    paired[label]=dict(Counter('both_success' if a['success'] and b['success'] else
        'optimization_only_success' if a['success'] else 'hardware_only_success' if b['success'] else 'neither_success'
        for a,b in zip(episodes[label,'optimization'],episodes[label,'hardware'])))
report=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),new_episodes=48,reused_trial622_episodes=24,
    all_72_episode_inputs_and_planner_settings_verified_matched=True,
    summaries={label:{ref:summaries[label,ref] for ref in refs} for label in folders},paired_successes=paired,
    weight_sets={label:json.loads((folder/'config.json').read_text())['weights'] for label,folder in folders.items()},
    assumptions=baseline,
    notes=['Original means the pre-friction BO block at CubeReorient.py lines 118-131, with w_quat=26.342 and w_joint=.0538.',
           'Trial 314 uses full-precision archived weights, rather than the rounded commented copy.',
           'The two new weight sets ran concurrently under private MPS. Planning timings are load-dependent and are not an isolated speed comparison with trial 622.',
           'All episodes start from the same robot_qref and exact +/-90 degree orientations at target position, with zero initial velocity and no settling.',
           'No software target or delta clipping; original XML actuator/joint limits remain.',
           'Shared initial cube/hand penetration is up to 7.74 mm in the uncalibrated geometry; results are comparative simulation evidence, not a hardware success-rate prediction.'])
(ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def finish(fig,name):
    fig.tight_layout();fig.savefig(ROOT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(3)
for j,ref in enumerate(refs):
    counts=[summaries[label,ref]['successes'] for label in folders]
    bars=ax.bar(x+(j-.5)*.35,np.array(counts)/12,width=.35,color=colors[j],label=ref_labels[j])
    for b,n in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,f'{n}/12',ha='center')
ax.set(xticks=x,xticklabels=[labels[k] for k in folders],ylim=(0,1.2),ylabel='Success rate',title='Effect of joint-cost reference across weight sets')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False,fontsize=9);finish(fig,'success_by_weights')

fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True,sharey=True);x=np.arange(6)
for ax,label in zip(axs,folders):
    for j,ref in enumerate(refs):
        counts=[sum(e['success'] for e in episodes[label,ref] if e['rotation']==d) for d in baseline['rotations']]
        bars=ax.bar(x+(j-.5)*.35,np.array(counts)/2,width=.35,color=colors[j],label=ref_labels[j])
        for b,n in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,f'{n}/2',ha='center',fontsize=8)
    ax.set(title=labels[label],ylim=(0,1.2),ylabel='Success rate')
axs[0].legend(fontsize=8);axs[-1].set(xticks=x,xticklabels=baseline['rotations'],xlabel='Initial rotation from goal (90 degrees)')
finish(fig,'success_by_rotation')

for key,scale,name,ylabel,threshold in [('pos',1000,'final_position_errors','Final position error (mm)',20),
        ('angle_deg',1,'final_orientation_errors','Final orientation error (degrees)',float(np.rad2deg(2*np.arcsin(np.sqrt(.04)))))]:
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True,sharey=True)
    for ax,label in zip(axs,folders):
        for j,ref in enumerate(refs):
            vals=[e['final_errors'][key]*scale for e in episodes[label,ref]]
            ax.plot(np.arange(1,13),vals,marker='o' if j==0 else '^',color=colors[j],label=ref_labels[j])
        ax.axhline(threshold,ls=':',color='.3',label='Success threshold');ax.set(title=labels[label],ylabel=ylabel,ylim=(0,None))
    axs[0].legend(fontsize=8);axs[-1].set(xticks=np.arange(1,13),xticklabels=[f'{d}\n{1+i//6}' for i,d in enumerate(baseline['rotations']*2)],xlabel='Initial rotation / repeat (matched cases)')
    finish(fig,name)

lines=['<html><head><meta charset="utf-8"><title>Cube weights and joint reference</title></head><body>',
 '<h1>Cube weights × joint-cost reference</h1>',
 '<p>Uncalibrated high/high; M2; 256 samples; horizon .32 s; control .064 s; shift; T=.1; sigma=.05.</p>',
 '<p>Mass .07 kg; sliding friction object/tips/palm/fingers .3/.5/.4/.4; target z=.08 m; fallen-penalty z=.065 m.</p>',
 '<table border="1" cellpadding="6"><tr><th>Weights</th><th>Reference</th><th>Success</th><th>Drops</th><th>Timeouts</th><th>Mean final pos (mm)</th><th>Median final pos (mm)</th><th>Mean final angle (deg)</th><th>Median final angle (deg)</th></tr>']
for label in folders:
    for ref in refs:
        m=summaries[label,ref];o=m['outcomes']
        vals=[labels[label],ref,f"{m['successes']}/12",o.get('drop',0),o.get('timeout',0),
              f"{m['final_position_mean_mm']:.2f}",f"{m['final_position_median_mm']:.2f}",
              f"{m['final_angle_mean_deg']:.2f}",f"{m['final_angle_median_deg']:.2f}"]
        lines.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in vals)+'</tr>')
lines.append('</table>')
for note in report['notes']:lines.append('<p>'+note+'</p>')
for name in ['success_by_weights','success_by_rotation','final_position_errors','final_orientation_errors']:
    lines.append(f'<p><img src="{name}.png" style="max-width:100%"></p>')
lines.append('</body></html>');(ROOT/'index.html').write_text('\n'.join(lines))
print(json.dumps(dict(summaries=report['summaries'],paired_successes=paired),indent=2))
