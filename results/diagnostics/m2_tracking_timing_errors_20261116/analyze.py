"""Offline failure diagnosis from controller and bridge telemetry snapshots."""
from pathlib import Path
import json,csv,html,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
paths=sorted((ROOT/'results/diagnostics/m2_tracking_baseline').glob('leap_*/controller.json'))

def stats(values):
 a=np.asarray(values,float)
 return dict(count=int(a.size),mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max())) if a.size else None

runs=[]
for p in paths:
 d=json.loads(p.read_text());launch=json.loads((p.parent/'launch.json').read_text());rows=[json.loads(line) for line in (p.parent/'bridge_telemetry.jsonl').read_text().splitlines()];cycles=[r for r in rows if r.get('event')=='cycle']
 e=next(e for e in d['episodes'] if e.get('failed_step'));s=e['failed_step'];f=s['feedback'];rejection=s['rejection']
 failed_at_us=f['robot_timestamp']+int(rejection['robot_age_s']*1e6)
 preceding=[r for r in cycles if r.get('state_timestamp_us') and r['state_timestamp_us']<=failed_at_us]
 latest=preceding[-1]
 pub={s['action_timestamp_us']:s for ep in d['episodes'] for s in ep['steps'] if s.get('action_timestamp_us')}
 applied={r['written_action_timestamp_us']:r for r in cycles if r.get('new_action') and r.get('write_succeeded') and r.get('command_type')=='delta'}
 matched=pub.keys()&applied.keys()
 entries=[r for r in cycles if r.get('read_duration_s') is not None]
 overhead=[1000*((r['cycle_finished_us']-r['read_started_us'])*1e-6-sum(r.get(k,0) for k in ['read_duration_s','write_duration_s','health_duration_s'])) for r in entries]
 record=dict(run=p.parent.name,source=str(p.relative_to(ROOT)),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),episode=e['episode']+1,failed_step=s['step'],
  initial_robot_age_ms=f['robot_age']*1000,initial_object_age_ms=f['pose_age']*1000,planning_ms=s['planning_time']*1000,
  rejected_robot_age_ms=rejection['robot_age_s']*1000,rejected_object_age_ms=rejection['pose_age_s']*1000,
  budget_excess_ms=(rejection['robot_age_s']-rejection['max_robot_age'])*1000,
  newest_bridge_published_joint_age_at_rejection_ms=(failed_at_us-latest['state_timestamp_us'])/1000,
  newer_bridge_packets_during_plan=sum(f['robot_timestamp']<r.get('state_timestamp_us',0)<=failed_at_us for r in cycles),
  read_ms=stats([r['read_duration_s']*1000 for r in cycles if 'read_duration_s' in r]),
  write_ms=stats([r['write_duration_s']*1000 for r in cycles if 'write_duration_s' in r]),
  health_ms=stats([r['health_duration_s']*1000 for r in cycles if 'health_duration_s' in r]),
  other_cycle_work_ms=stats(overhead),
  publication_to_write_ms=stats([(applied[k]['write_started_us']-k)/1000 for k in matched]),
  published_deltas=len(pub),matched_written_deltas=len(matched),telemetry_status=json.loads((p.parent/'bridge_telemetry.status.json').read_text()),
  completed_episodes=[dict(episode=ep['episode']+1,outcome=ep['finish_reason'],steps=ep['n_steps']) for ep in d['episodes'] if ep.get('completed')],
  bridge_errors=[r.get('error') for r in cycles if r.get('error')],zero_return=d.get('feedback_error_return',{}).get('zero_reached'),
  mppi_gpu=launch['config'].get('gpu_assignment'))
 runs.append(record)
report=dict(runs=runs,diagnosis='Original planned joint-state age exceeded 150 ms after ~104-111 ms planning. The object age and 200 ms plan-runtime ceiling were not the rejecting conditions. Bridge continued publishing; fresh publication alone does not refresh an already computed plan.',
 recommendations=['Keep the timing limits unchanged during diagnosis.','Reduce worst-case feedback publication gaps by distributing the two blocking health-register reads across separate cycles while preserving each register monitoring frequency; currently each health() call occupies about 32 ms.','Avoid compute contention for MPPI and assess its high-percentile planning time. All three used physical GPU0.','If admitting plans only from fresher joint snapshots, use a separate start-of-plan freshness condition that reserves compute time; account for added waiting and resulting update frequency.','The later bridge watchdog timeout occurs after the controller performed supervised zero return; it is a secondary shutdown issue and not the manipulation-abort trigger.'],
 caveats=['No freshness limit, motor parameter, control law, or running service was changed by this offline analysis.','Latest bridge publication does not independently prove receipt by the controller buffer at the rejection instant.','Full historical GPU utilization traces were not captured, so GPU contention is a candidate contributor, not proven.'])
(OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
fields=['run','episode','failed_step','initial_robot_age_ms','planning_ms','rejected_robot_age_ms','budget_excess_ms','rejected_object_age_ms','newest_bridge_published_joint_age_at_rejection_ms','published_deltas','matched_written_deltas']
with (OUT/'failures.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in runs)
body=['<!doctype html><meta charset="utf-8"><title>M2 diagnostic timing failures</title><style>body{font:16px system-ui;margin:30px auto;max-width:1150px;line-height:1.5;padding:20px}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccc}</style><h1>Three M2 diagnostic timing failures</h1>',
 '<p>All three rejected the original planning joint state after its age exceeded 150 ms. These were not object freshness failures or 200 ms planning-runtime overruns.</p>',
 '<table><tr><th>Run</th><th>Episode / step</th><th>Joint age at start (ms)</th><th>Planning (ms)</th><th>Joint age at rejection (ms)</th><th>Newest bridge publication age (ms)</th></tr>']
for r in runs:body.append(f"<tr><td>{r['run']}</td><td>{r['episode']} / {r['failed_step']}</td><td>{r['initial_robot_age_ms']:.2f}</td><td>{r['planning_ms']:.2f}</td><td>{r['rejected_robot_age_ms']:.2f}</td><td>{r['newest_bridge_published_joint_age_at_rejection_ms']:.2f}</td></tr>")
body.extend(['</table><p>Each bridge published four newer joint packets between the original planning sample and rejection. The guard runs before the new command-state read and intentionally retains the original state deadline.</p>',
 '<p>Bridge reads take about 15.9 ms, health checks (two serial register reads) about 31.7 ms, and command writes about 0.09 ms. Mean state publication interval is about 32 ms; long intervals reach about 48–50 ms. The added non-serial cycle work is about 0.13 ms on average, and telemetry reports zero dropped records and no writer errors.</p>',
 '<p>All 335 published manipulation deltas in these three logs match successful bridge writes. In the main 314-action run, publication-to-write latency averaged 19.81 ms, P95 44.36 ms, maximum 47.99 ms. Thus diagnostics are functioning and reveal queue/serial-cycle latency; no manipulation delta loss is observed in this sample.</p>',
 '<p>Two episodes completed in the middle run: one drop and one success. Three active episodes aborted and remain excluded from completed-episode success metrics. All three controller logs confirm supervised zero return, followed by bridge action-watchdog timeouts and verified torque-off.</p>',
 '<h2>Proposed next steps</h2><ol>'])
for item in report['recommendations']:body.append('<li>'+html.escape(item)+'</li>')
body.extend(['</ol><p><a href="analysis.json">Detailed analysis</a> · <a href="failures.csv">Failure table</a> · <a href="analyze.py">Reproduce</a></p>'])
(OUT/'index.html').write_text('\n'.join(body))
print('Saved:',OUT/'index.html')
