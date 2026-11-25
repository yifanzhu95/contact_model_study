from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/leap_timing_matplotlib')
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high/M6/leap_experiment_20261113_230803_006211'
OUT=Path(__file__).resolve().parent
log=json.loads((RUN/'controller.json').read_text())
def stats(a):
 a=np.asarray(a,dtype=float)
 return dict(count=len(a),mean=float(np.mean(a)),median=float(np.median(a)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),max=float(np.max(a))) if len(a) else None
report=dict(source=str(RUN),error=log['error'],episodes=[])
fig,axes=plt.subplots(3,1,figsize=(12,10),layout='constrained')
all_ms=[];binrows=[]
for e,ax in zip(log['episodes'],axes):
 rows=e['steps']+([e['failed_step']] if e.get('failed_step') else [])
 t=np.array([s['elapsed_s'] for s in rows]);plans=np.array([s['planning_time'] for s in rows])*1000
 wall=np.array([s['wall_time_s'] for s in rows])*1000
 gap=np.diff(t)*1000
 report_e=dict(episode=e['episode']+1,outcome=e['finish_reason'],published_actions=e['n_steps'],duration_s=e['manipulation_wall_time_s'],
  effective_published_hz=e['n_steps']/e['manipulation_wall_time_s'],planning_ms=stats(plans),cycle_ms=stats(wall),
  start_to_start_gap_ms=stats(gap),nonplanning_cycle_ms=stats(wall-plans),
  fraction_plans_over_64ms=float(np.mean(plans>64)),fraction_plans_over_100ms=float(np.mean(plans>100)),
  fraction_plans_over_125ms=float(np.mean(plans>125)),windows=[])
 all_ms.extend(plans)
 for start in np.arange(0,t[-1]+1e-6,10):
  selected=(t>=start)&(t<start+10)
  if not selected.any():continue
  duration=min(10,e['manipulation_wall_time_s']-start)
  sample=plans[selected]
  b=dict(episode=e['episode']+1,start_s=float(start),end_s=float(start+duration),
   plan_count=int(selected.sum()),mean_plan_ms=float(sample.mean()),p95_plan_ms=float(np.percentile(sample,95)),max_plan_ms=float(sample.max()),
   approximate_plan_hz=float(1000/sample.mean()))
  report_e['windows'].append(b);binrows.append(b)
 # Continuous longest sequence of slow calls, independent of bin boundaries.
 for threshold in (100,125,150):
  mask=plans>threshold;best=(0,None,None);first=None
  for i,is_slow in enumerate(mask):
   if is_slow and first is None:first=i
   if first is not None and ((not is_slow) or i==len(mask)-1):
    last=i if is_slow else i-1
    if last-first+1>best[0]:best=(last-first+1,first,last)
    first=None
  n,a,b=best
  report_e[f'longest_streak_over_{threshold}ms']=dict(count=n,start_s=None if a is None else float(t[a]),end_s=None if b is None else float(t[b]),
   duration_s=None if a is None else float((t[b]-t[a])+wall[b]/1000))
 if len(gap):
  worst=np.argsort(gap)[-3:][::-1]
  report_e['largest_gaps']=[dict(step=int(rows[i]['step']),start_s=float(t[i]),gap_ms=float(gap[i]),plan_ms=float(plans[i]),cycle_ms=float(wall[i])) for i in worst]
 report['episodes'].append(report_e)
 ax.plot(t,plans,color='#4363a3',alpha=.3,lw=.8,label='Each plan')
 if len(plans)>=10:
  ax.plot(t[9:],np.convolve(plans,np.ones(10)/10,mode='valid'),color='#222222',lw=1.6,label='10-plan mean')
 ax.axhline(64,color='#b52a2a',ls='--',label='Configured 64 ms')
 ax.set(xlabel='Time since MPPI start (s)',ylabel='Planning time (ms)',ylim=(0,190),title=f"Episode {e['episode']+1}: {e['finish_reason']}, {e['n_steps']} published actions")
 ax.grid(alpha=.2)
 if e.get('failed_step'):
  ax.scatter([t[-1]],[plans[-1]],marker='x',s=80,c='red',zorder=10,label='Rejected final plan')
 ax.legend(fontsize=8,loc='upper left')
fig.suptitle('Latest M6 run: planning time and slowdown',fontsize=14)
fig.savefig(OUT/'step_times.png',dpi=150)
plt.close(fig)
report['planning_ms_all']=stats(all_ms)
report['failed_step']=log['episodes'][-1].get('failed_step')
(OUT/'timing.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'ten_second_windows.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(binrows[0]));w.writeheader();w.writerows(binrows)
for e in report['episodes']:
 print('EPISODE',json.dumps(e,indent=2))
