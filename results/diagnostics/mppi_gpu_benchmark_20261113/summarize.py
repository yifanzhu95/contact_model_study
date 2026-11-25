from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/leap_gpu_benchmark_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
summaries=[]
fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
for i,name in enumerate(['M2_separate_gpu0','M6_separate_gpu0','M6_shared_vision_gpu3']):
 folder=OUT/name;r=json.loads((folder/'result.json').read_text());t=json.loads((folder/'telemetry.json').read_text())
 baseline=[s for s in t['samples'] if s['phase']=='baseline']
 # Window membership uses plan timestamps rather than sampler phase labels,
 # so the final GPU-memory sample cannot accidentally include teardown.
 begin=r['records'][0]['timestamp'];end=r['records'][-1]['timestamp']+r['records'][-1]['planning_s']
 measured=[s for s in t['samples'] if begin<=s['timestamp']<=end]
 warm=[s for s in t['samples'] if s['phase']=='warmup']
 baseline_pids={p['pid'] for s in baseline for p in s['processes']}
 unexpected=sorted({p['pid'] for s in measured for p in s['processes'] if p['pid'] not in baseline_pids and p['pid']!=t['worker_pid']})
 util=[s['gpu_util_percent'] for s in measured]
 memory=[s['worker_memory_mib'] for s in measured]
 row=dict(case=name,model=r['model'],physical_gpu_uuid=r['physical_gpu_uuid'],measurement_s=r['measurement_s'],calls=r['calls'],failed_plans=r['failed_plans'],
  horizon_steps=r['horizon_steps'],samples=r['samples'],planning_ms=r['planning_ms'],plan_rate_hz=r['plan_rate_hz'],
  telemetry_samples=len(measured),gpu_util_mean_percent=float(np.mean(util)),gpu_util_p95_percent=float(np.percentile(util,95)),gpu_util_max_percent=max(util),
  baseline_gpu_util_mean_percent=float(np.mean([s['gpu_util_percent'] for s in baseline])),
  worker_memory_mean_mib=float(np.mean(memory)),worker_memory_observed_peak_mib=max(s['worker_memory_mib'] for s in t['samples']),
  gpu_total_memory_observed_peak_mib=max(s['gpu_memory_mib'] for s in t['samples']),
  baseline_gpu_memory_mean_mib=float(np.mean([s['gpu_memory_mib'] for s in baseline])),
  unexpected_compute_processes=unexpected,warmup_s=r['warmup_s'])
 summaries.append(row)
 xs=np.array([s['timestamp']-begin for s in measured])
 axes[i,0].plot(xs,util,color='#4165a6');axes[i,0].set(xlabel='Measurement time (s)',ylabel='Whole GPU utilization (%)',ylim=(0,105),title=name)
 plan_x=[s['timestamp']-begin for s in r['records']]
 plan_ms=[s['planning_s']*1000 for s in r['records']]
 axes[i,1].plot(plan_x,plan_ms,lw=.7,color='#4d8d52');axes[i,1].axhline(64,color='#b43232',ls='--',label='Configured 64 ms')
 axes[i,1].set(xlabel='Measurement time (s)',ylabel='MPPI planning time (ms)',title='Offline saved-state replay')
 axes[i,1].legend(fontsize=8)
 for ax in axes[i]:ax.grid(alpha=.2)
fig.suptitle('Offline MPPI GPU benchmark — no hardware, camera or LCM instances',fontsize=14)
fig.savefig(OUT/'comparison.png',dpi=150)
plt.close(fig)
summary=dict(cases=summaries,settings=dict(samples=512,noise_sigma=.05,temperature=.1,iterations=1,ctrl_time_step=.064,
 horizons_s=dict(M2=.32,M6=.384)),
 method='Separate fresh processes replay the same measured states; back-to-back planning with CUDA graphs. NVML queried through nvidia-smi approximately every 0.5 seconds.',
 limitations=['Utilization is for the whole selected GPU, not per-process. Baseline desktop activity is included on GPU 0.',
 'Memory peaks are maximum observed samples, not allocation-instrumented absolute peaks.',
 'No serial, LCM, camera or interactive viewer is opened by a benchmark worker. Existing vision services remain untouched.',
 'Back-to-back planning measures saturation throughput, not a deliberately paced 15.625 Hz loop.',
 'Saved-state replay is not a closed-loop manipulation success evaluation.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for row in summaries:print(json.dumps(row,indent=2))
