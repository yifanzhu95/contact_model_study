"""Offline motor-current comparisons for matched M2 action-rate/blending settings.

No camera, serial, CUDA or simulator is opened. Exact physics/config equality is
checked within comparisons; initial placements and seeds of physical motion are
not paired. Completed manipulation windows are primary; aborted windows and
current-limit terminations are retained separately to expose survivor bias.
"""
from collections import defaultdict
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial314_shift512_latest_h0256_unpaced'
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'.mplconfig'))
GROUPS = {
    'A_rate0': ['013114_416890','013511_218698','013829_819992'],
    'A_rate10': ['020135_762474'],
    'B_rate0.1': ['014113_396846'],
    'B_rate10': ['014558_585798'],
    'C_rate10': ['020245_795756'],
    'C_rate15': ['020749_462137'],
    'D_rate15': ['021750_229900','022234_162212','023129_804674'],
    'D_rate20': ['022054_564380'],
    'E_rate15_no_blend': ['030100_212337','031523_266949'],
    'E_rate15_blend0.6': ['032413_989582','032910_880602'],
}
NATIVE_TO_MUJOCO = np.array([1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15])


def us(text):
    return int(datetime.fromisoformat(text).timestamp()*1e6)


def normalized_scene(text):
    root = ET.fromstring(text)
    for mesh in root.findall('.//mesh'):
        mesh.attrib.pop('content_type', None)
    return hashlib.sha256(ET.tostring(root)).hexdigest()


def fingerprint(d):
    complete = [e for e in d['episodes'] if e.get('completed')]
    return dict(weights=d['weights'], hardware=d['hardware_config'], reference=d['joint_reference'],
        scenes={k:normalized_scene(v['scene_xml']) for k,v in d['scenes'].items()},
        rollout=d['rollout_config'], planner={k:v for k,v in d['planner_config'].items() if k!='action_rate_weight'},
        targets=sorted({tuple(e['target_pos']) for e in complete}),
        drop_heights=sorted({e['fallen_z'] for e in complete}), camera=d['pose_transform'],
        paced=d['args']['pace_control'])


def stats(current):
    a = np.abs(current.astype(float))
    peak = a.max(axis=1)
    return dict(cycles=len(a), mean_abs_motor_current_raw=float(a.mean()),
        rms_motor_current_raw=float(np.sqrt(np.mean(a*a))),
        motor_current_p95_raw=float(np.percentile(a,95)),
        cycle_peak_p95_raw=float(np.percentile(peak,95)),
        cycle_peak_p99_raw=float(np.percentile(peak,99)),
        peak_raw=float(peak.max()),
        **{f'cycles_any_motor_over{t}_percent':float(100*np.mean(peak>t)) for t in (400,450,500,550)},
        motors=[dict(native_motor=i, mean_abs_raw=float(a[:,i].mean()),
            rms_raw=float(np.sqrt(np.mean(a[:,i]**2))),p99_raw=float(np.percentile(a[:,i],99)),
            peak_raw=float(a[:,i].max()),over500_percent=float(100*np.mean(a[:,i]>500))) for i in range(16)])


def csvout(name, rows):
    with (OUT/name).open('w',newline='') as f:
        keys=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(rows)


def plot_summary(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=report['groups']['E_rate15_no_blend'];b=report['groups']['E_rate15_blend0.6']
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.3))
    metrics=[('Motor-current RMS','raw units',a['completed_manipulation']['rms_motor_current_raw'],
              b['completed_manipulation']['rms_motor_current_raw']),
             ('Worst-motor current: p99','raw units',a['completed_manipulation']['cycle_peak_p99_raw'],
              b['completed_manipulation']['cycle_peak_p99_raw']),
             ('Largest target change: p95','degrees',a['target_max_joint_jump_deg']['p95'],
              b['target_max_joint_jump_deg']['p95'])]
    for ax,(title,unit,left,right) in zip(axes,metrics):
        bars=ax.bar([0,1],[left,right],color=['#b25b47','#2b7b92'],width=.6)
        ax.bar_label(bars,labels=[f'{left:.1f}',f'{right:.1f}'],padding=5)
        ax.set(xticks=[0,1],xticklabels=['No blending\n24 episodes','Alpha 0.6\n9 episodes'],
               ylabel=unit,title=title,ylim=(0,max(left,right)*1.22))
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Historical M2 telemetry: same settings, action-rate weight 15',fontsize=13)
    fig.text(.5,.012,'Completed manipulations; starting placements were not paired. These runs predate measured-limit clipping.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    fig.savefig(OUT/'blending_current_summary.png',dpi=170)
    fig.savefig(OUT/'blending_current_summary.pdf')
    plt.close(fig)


def main():
    configs, result, distributions, episodes_out, sources = {}, {}, {}, [], []
    for group, suffixes in GROUPS.items():
        all_current, abort_current, jumps, fingerprints = [], [], [], []
        count_complete = count_abort = successes = trips = 0
        current_group = []
        for suffix in suffixes:
            directory = BASE/('leap_experiment_20261119_'+suffix)
            raw=(directory/'controller.json').read_bytes();d=json.loads(raw)
            fingerprints.append(fingerprint(d))
            windows=[];selected_actions=set()
            for e in d['episodes']:
                steps=[s for s in e['steps'] if s.get('action_published')]
                if not steps:continue
                complete=bool(e.get('completed'))
                if complete:
                    end=us(e['finished_at_utc']);count_complete+=1
                    successes+=int(e['finish_reason']=='success')
                else:
                    count_abort+=1
                    failed=e.get('failed_step',{})
                    # End at controller fault detection, not a subsequent
                    # operator hold/reset that can last minutes.
                    end=(us(failed['timestamp_utc'])+int(1e6*(failed.get('planning_time',0)+failed.get('pacing_wait_s',0)))
                         if failed.get('timestamp_utc') else us(steps[-1]['action_published_at_utc'])+64000)
                windows.append(dict(start=us(steps[0]['action_published_at_utc']),end=end,
                    completed=complete,episode=e['episode']+1,outcome=e['finish_reason'],values=[],
                    published_actions=len(steps)))
                if complete:selected_actions.update(s['action_timestamp_us'] for s in steps)
            windows.sort(key=lambda w:w['start']);pointer=0;filehash=hashlib.sha256();matched=set()
            telemetry=directory/'bridge_telemetry.jsonl'
            print(f'{group}: streaming {suffix} ({telemetry.stat().st_size/1e6:.0f} MB)',flush=True)
            with telemetry.open('rb') as f:
                for payload in f:
                    filehash.update(payload);x=json.loads(payload)
                    if x.get('event')!='cycle':continue
                    stamp=x.get('read_finished_us')
                    if stamp is not None:
                        while pointer<len(windows) and stamp>windows[pointer]['end']:pointer+=1
                        if (pointer<len(windows) and stamp>=windows[pointer]['start']
                                and x.get('held_command_type')=='delta' and 'current_raw_native' in x):
                            v=np.asarray(x['current_raw_native'],dtype=np.float32)
                            assert v.shape==(16,) and np.isfinite(v).all()
                            windows[pointer]['values'].append(v)
                    token=x.get('written_action_timestamp_us')
                    if (token in selected_actions and token not in matched and x.get('new_action')
                            and x.get('write_succeeded') and x.get('command_type')=='delta'):
                        matched.add(token)
                        jump=np.asarray(x['written_target'])-np.asarray(x['held_target_before_read'])
                        jumps.append(float(np.rad2deg(abs(jump).max())))
            for w in windows:
                values=np.asarray(w.pop('values'),dtype=np.float32)
                if not len(values):continue
                (all_current if w['completed'] else abort_current).append(values)
                summary=stats(values)
                episodes_out.append(dict(group=group,run=directory.name,**w,
                    **{k:v for k,v in summary.items() if k!='motors'}))
            log=(directory/'bridge.log').read_text(errors='replace')
            faults=re.findall(r'Measured current exceeds hardware limit: motor (\d+)=(-?\d+) raw',log)
            trips+=len(faults)
            status_path=telemetry.with_suffix('.status.json')
            status=json.loads(status_path.read_text()) if status_path.exists() else None
            current_group.append(dict(run=directory.name,action_rate_weight=d['args']['action_rate_weight'],
                blend_alpha=d['args'].get('plan_blend_alpha',1),current_trips=faults,
                telemetry_sha256=filehash.hexdigest(),controller_sha256=hashlib.sha256(raw).hexdigest(),
                telemetry_status=status,matched_actions=len(matched),published_actions=len(selected_actions)))
            sources.append(str(directory.relative_to(ROOT)))
        assert all(x==fingerprints[0] for x in fingerprints)
        configs[group]=fingerprints[0]
        values=np.concatenate(all_current)
        summary=stats(values)
        # Joint labels below are native IDs: current vectors never use q order.
        names=d['joint_order']
        for motor in summary['motors']:
            motor['joint']=names[NATIVE_TO_MUJOCO[motor['native_motor']]]
        own_eps=[e for e in episodes_out if e['group']==group and e['completed']]
        result[group]=dict(completed_episodes=count_complete,aborted_episodes=count_abort,successes=successes,
            current_trip_runs=trips,completed_manipulation=summary,
            median_episode_rms_current_raw=float(np.median([e['rms_motor_current_raw'] for e in own_eps])),
            median_episode_peak_p99_raw=float(np.median([e['cycle_peak_p99_raw'] for e in own_eps])),
            aborted_manipulation=stats(np.concatenate(abort_current)) if abort_current else None,
            target_max_joint_jump_deg=dict(mean=float(np.mean(jumps)),p95=float(np.percentile(jumps,95)),max=float(max(jumps))),
            sources=current_group)
        distributions[group]=np.abs(values).max(axis=1)
        print(group,json.dumps({k:v for k,v in summary.items() if k!='motors'}),flush=True)
    pairs=[('A_rate0','A_rate10'),('B_rate0.1','B_rate10'),('C_rate10','C_rate15'),
           ('D_rate15','D_rate20'),('E_rate15_no_blend','E_rate15_blend0.6')]
    for a,b in pairs:assert configs[a]==configs[b],(a,b)
    report=dict(groups=result,configurations=configs,matched_pairs=pairs,
        definitions=dict(current='raw signed current register; magnitudes used for statistics, native motor-ID order',
            rms='sqrt(mean(current_raw^2)) across all 16 motors and included bridge cycles',
            peak_p99='99th percentile of maximum absolute current across the 16 motors per bridge cycle',
            windows='first action publication through episode finish, held delta command only; excludes preparation and user holds'),
        limitations=['Physical initial placements are not paired; results are observational, not randomized causal estimates.',
            'Completed-episode current is conditioned on completion; aborted windows and current-trip run counts are reported separately.',
            'Group sizes and manipulation durations differ; pooled-cycle and equal-episode summaries are both provided.',
            'All compared recordings predate the new joint-limit penalty and absolute clamp. The two latest current-trip runs have no full current telemetry.',
            'A lower current RMS or smoother target does not guarantee lower rare peaks or adequate grasp force.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    csvout('group_metrics.csv',[dict(group=g,episodes=x['completed_episodes'],aborts=x['aborted_episodes'],
        successes=x['successes'],current_trip_runs=x['current_trip_runs'],
        **{k:v for k,v in x['completed_manipulation'].items() if k!='motors'},
        median_episode_rms_current_raw=x['median_episode_rms_current_raw'],
        actual_target_jump_p95_deg=x['target_max_joint_jump_deg']['p95']) for g,x in result.items()])
    csvout('per_motor.csv',[dict(group=g,**m) for g,x in result.items() for m in x['completed_manipulation']['motors']])
    csvout('per_episode.csv',episodes_out)
    plot_summary(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,(a,b),title in zip(axs,[pairs[0],pairs[-1]],['Action-rate penalty: 0 vs 10\n7 vs 1 completed episodes',
            'Temporal blending: alpha 1 vs 0.6\n24 vs 9 episodes; action-rate weight 15']):
        for group,color in [(a,'#b25b47'),(b,'#2b7b92')]:
            x=np.sort(distributions[group]);stride=max(1,len(x)//4000)
            ax.plot(x[::stride],np.arange(1,len(x)+1)[::stride]/len(x),label=group,color=color)
        ax.axvline(550,color='black',ls='--',lw=1,label='Current cutoff')
        ax.set(xlabel='Largest motor-current magnitude in a bridge cycle (raw)',ylabel='Cumulative fraction',title=title,xlim=(0,580))
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'current_distributions.png',dpi=170);fig.savefig(OUT/'current_distributions.pdf');plt.close(fig)
    print('RESULTS',OUT,flush=True)


if __name__=='__main__':main()
