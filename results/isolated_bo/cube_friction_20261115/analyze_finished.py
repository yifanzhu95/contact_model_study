"""Read-only analysis of completed BO observations; writes separate report artifacts."""
import ast
import collections
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
CELL=ROOT/'search_results/cell_0000_cube_mass070g_T0.1'
OUT=ROOT/'analysis_500'
OUT.mkdir(exist_ok=True)
MODELS=['M1','M2','M3','M4']
trials=[json.loads(p.read_text()) for p in sorted(CELL.glob('trial_*/trial.json'))]
assert len(trials)==500 and {t['trial'] for t in trials}==set(range(500))
assert all(t['status']=='done' and math.isfinite(t['objective']) for t in trials)
ranked=sorted(trials,key=lambda t:t['objective'])
best=ranked[0]
all_outcomes=collections.Counter();nonfinite=[];summaries=[]
for t in trials:
    for m in MODELS:
        d=json.loads((CELL/f'trial_{t["trial"]:04d}'/(m+'.json')).read_text())
        assert len(d['episodes'])==4
        for i,e in enumerate(d['episodes']):
            s=e['summary'];all_outcomes[e['finish_reason']]+=1
            for key in ['q_end','q_dot_end']:
                if not np.isfinite(np.asarray(s[key],float)).all():
                    nonfinite.append(dict(trial=t['trial'],model=m,episode=i,key=key))
            if not all(math.isfinite(float(v)) for v in s['goal_errors_end'].values()):
                nonfinite.append(dict(trial=t['trial'],model=m,episode=i,key='goal_errors_end'))
        if t['trial'] not in [0,308,372,best['trial']]:continue
        eps=d['episodes'];err=[e['summary']['goal_errors_end'] for e in eps]
        counts=collections.Counter(e['finish_reason'] for e in eps)
        qangles=[math.degrees(2*math.asin(math.sqrt(max(0.,min(1.,v['quat']))))) for v in err]
        summaries.append(dict(trial=t['trial'],model=m,successes=t['per_model'][m]['n_success'],
            episodes=4,failures=counts['failure'],timeouts=counts['timeout'],
            mean_final_position_mm=np.mean([v['pos'] for v in err])*1000,
            mean_final_orientation_deg=float(np.mean(qangles)),
            mean_final_quaternion_error=float(np.mean([v['quat'] for v in err])),
            mean_success_steps=t['per_model'][m]['mean_steps_to_success'],
            normalized_goal_error=t['per_model'][m]['mean_norm_goal_err'],
            model_objective=t['per_model'][m]['objective']))
rows=[];incumbent=math.inf;improvements=[]
for t in trials:
    if t['objective']<incumbent-1e-12:
        incumbent=t['objective'];improvements.append(dict(trial=t['trial'],objective=incumbent))
    row=dict(trial=t['trial'],source=t['source'],objective=t['objective'],best_so_far=incumbent,
        overall_success_rate=t['success_rate'],worst_model_success_rate=min(v['success_rate'] for v in t['per_model'].values()),
        candidate_wall_s=t['wall_s'])
    for m in MODELS:row[m+'_success_rate']=t['per_model'][m]['success_rate']
    rows.append(row)
for name,data in [('trials.csv',rows),('evaluation_comparison.csv',summaries)]:
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
state=json.loads((ROOT/'status.json').read_text())
rev=Path(state['execution_revision'])
gpu_samples=[s for line in (rev/'gpu_metrics.jsonl').read_text().splitlines()
             for s in json.loads(line)['gpu']]
applied=json.loads((ROOT/'applied_comment_weights.json').read_text())
preserved=json.loads((rev/'preserved_trial_hashes.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in preserved.items())
task=Path(applied['target']);before=(ROOT/'CubeReorient.before.py').read_text()
ast_unchanged=ast.dump(ast.parse(task.read_text()))==ast.dump(ast.parse(before))
assert ast_unchanged
start=datetime.fromisoformat(state['started_at_utc']);end=datetime.fromisoformat(state['finished_at_utc'])
report=dict(completed=True,trials=500,episodes=8000,
    sources=dict(collections.Counter(t['source'] for t in trials)),
    episode_execution_errors=sum(v['n_errors'] for t in trials for v in t['per_model'].values()),
    nonfinite_results=nonfinite,episode_outcomes=dict(all_outcomes),
    resumed_trials=25,previous_result_files_preserved=len(preserved),
    accelerated_elapsed_hours=(end-start).total_seconds()/3600,
    finished_phoenix=end.astimezone(ZoneInfo('America/Phoenix')).isoformat(),
    peak_gpu_memory_gib=max(s['memory_used_mib'] for s in gpu_samples)/1024,
    minimum_gpu_headroom_gib=min(s['headroom_mib'] for s in gpu_samples)/1024,
    peak_gpu_temperature_c=max(s['temperature_c'] for s in gpu_samples),
    fallback_used=state['attempt']!=0,exit_code=state['last_returncode'],
    baseline=trials[0],best=best,runner_up=ranked[1],improvements=improvements,
    best_trial_is_zero_based=True,trials_after_last_improvement=499-improvements[-1]['trial'],
    maximum_worst_model_success_rate=max(r['worst_model_success_rate'] for r in rows),
    highest_overall_success_trial=max(trials,key=lambda t:t['success_rate']),
    evaluation_comparison=summaries,weights=applied['weights'],
    active_code_ast_unchanged=ast_unchanged,commented_weights_updated=True,
    limitations=['Only four fixed seeded episodes per model, used during optimization.',
        'No independent held-out evaluation or hardware validation is included.',
        'Worst-model objective can trade away higher overall or M2 success.',
        'No new simulations or hardware runs performed for this report.'],
    success_thresholds=dict(position_m=.02,quaternion_error=.04,
        equivalent_orientation_deg=math.degrees(2*math.asin(math.sqrt(.04))),combined_object_velocity_norm=10.))
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(OUT/'best_weights.json').write_text(json.dumps(applied['weights'],indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
for src,color in [('defaults','#888888'),('random','#83a7c7'),('gp','#c3b1df')]:
    part=[r for r in rows if r['source']==src]
    ax.scatter([r['trial']+1 for r in part],[r['objective'] for r in part],s=11,alpha=.6,c=color,label=src)
ax.plot([r['trial']+1 for r in rows],[r['best_so_far'] for r in rows],color='#17365d',lw=2,label='Best observed')
ax.axvline(120,color='#777777',ls='--',lw=1,label='Initial budget: 120')
ax.scatter([best['trial']+1],[best['objective']],marker='*',s=180,c='#dc7f20',edgecolor='black',zorder=4)
ax.annotate(f'Winner: trial ID {best["trial"]}\nJ = {best["objective"]:.6f}',
    xy=(best['trial']+1,best['objective']),xytext=(best['trial']+20,-.15),
    arrowprops=dict(arrowstyle='->',color='#555555'),fontsize=10)
ax.set(xlabel='Candidate number (trial ID + 1)',ylabel='Worst-model objective (lower is better)',
    title='70 g cube, T = 0.1, noise = 0.05: 500-trial optimization')
ax.legend(ncol=2,fontsize=9,loc='upper right');ax.grid(axis='y',alpha=.2)
for ext in ['png','pdf']:fig.savefig(OUT/f'convergence.{ext}')
plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,4.5),constrained_layout=True)
x=np.arange(4);width=.25
for offset,index,color,label in [(-1,0,'#a5a5a5','Baseline (ID 0)'),(0,best['trial'],'#3278a5','BO winner (ID 314)'),(1,372,'#e5a24a','Highest overall (ID 372)')]:
    vals=[trials[index]['per_model'][m]['success_rate']*100 for m in MODELS]
    bars=ax.bar(x+offset*width,vals,width,color=color,label=label)
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+2,f'{int(v/25)}/4',ha='center',fontsize=9)
ax.set(xticks=x,xticklabels=MODELS,ylim=(0,105),yticks=[0,25,50,75,100],
    ylabel='Success rate (%)',title='Fixed-seed BO evaluation: four episodes per model')
ax.legend(fontsize=9,loc='upper right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
for ext in ['png','pdf']:fig.savefig(OUT/f'success_comparison.{ext}')
plt.close(fig)
print(json.dumps({k:report[k] for k in ['trials','episodes','episode_execution_errors','nonfinite_results',
    'accelerated_elapsed_hours','peak_gpu_memory_gib','minimum_gpu_headroom_gib','peak_gpu_temperature_c',
    'trials_after_last_improvement','maximum_worst_model_success_rate']},indent=2))
print('Highest overall:',report['highest_overall_success_trial']['trial'],report['highest_overall_success_trial']['success_rate'])
print('Comparison:',json.dumps([s for s in summaries if s['trial']==best['trial']],indent=2))
print('Artifacts:',OUT)
