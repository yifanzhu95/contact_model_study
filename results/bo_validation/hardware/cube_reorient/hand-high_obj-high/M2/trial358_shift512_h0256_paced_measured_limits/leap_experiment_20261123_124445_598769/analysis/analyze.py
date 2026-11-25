"""Read-only hardware analysis against the explicitly designated paced baseline."""
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
ROOT=next(p for p in OUT.parents if (p/'run_leap_hardware.py').exists())
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
import numpy as np
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_RECORD=json.loads((OUT/'baseline_record.json').read_text())
BASE=ROOT/BASE_RECORD['results_directory']
HELPER=BASE.parent/'combined_12_completed_20261121_030146_030714/analyze_and_group.py'
spec=importlib.util.spec_from_file_location('previous_hardware_analysis',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
COLORS={'baseline':'#377eb8','trial358':'#e17826'}
LABELS={'baseline':'Paced baseline','trial358':'Trial 358 hardware'}


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_write(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stats(x):return helper.stats(x)


def collect(doc):
    episodes=[e for e in doc['episodes'] if e.get('completed')]
    steps=[s for e in episodes for s in e['steps'] if s.get('action_published')]
    plans=np.array([s['planning_time'] for s in steps]);wait=np.array([s.get('pacing_wait_s',0) for s in steps])
    intervals=[];changes=[];rms=[];initial_angles=[];initial_pos=[];rows=[];curves=[]
    for e in episodes:
        ss=[s for s in e['steps'] if s.get('action_published')]
        intervals.extend(np.diff([s['action_timestamp_us'] for s in ss])*1e-6)
        difference=np.diff([s['u'] for s in ss],axis=0)
        changes.extend(np.degrees(np.max(abs(difference),axis=1)))
        rms.append(float(np.sqrt(np.mean(np.sum(difference*difference,axis=1)))))
        augmented=dict(e,source_run=e.get('source_run',RUN.name),source_episode=e.get('source_episode',e['episode']+1))
        row,curve=helper.episode_analysis(augmented,doc)
        orientation=e.get('starting_orientation')
        row['top_face']=None if orientation is None else orientation['top_face']['letter']
        row['target_change_l2_rms_rad']=rms[-1]
        row['initial_xyz_mm']=(np.asarray(ss[0]['q'][16:19])*1000).tolist()
        q=np.array([s['q'] for s in ss]+[e['q_end']]);errs=[s['goal_errors'] for s in ss]+[e['goal_errors']]
        pok=np.array([r['pos']<.02 for r in errs]);aok=np.array([r['quat']<.04 for r in errs])
        supported=q[:,18]>=e['fallen_z']
        if np.any(supported & pok):
            row['best_angle_while_position_ok_and_z_ge_65mm']=float(np.min(np.array(curve['angle_deg'])[supported & pok]))
        else:row['best_angle_while_position_ok_and_z_ge_65mm']=None
        row['minimum_angle_anywhere']=min(curve['angle_deg'])
        row['samples_orientation_ok_and_position_not_ok']=int((aok & ~pok).sum())
        row['samples_position_ok_and_orientation_not_ok']=int((pok & ~aok).sum())
        row['samples_both_pose_ok']=int((pok & aok).sum())
        initial_angles.append(row['initial_angle_deg']);initial_pos.append(row['initial_position_error_mm'])
        rows.append(row);curves.append(curve)
    intervals=np.asarray(intervals);success=[e for e in episodes if e['finish_reason']=='success']
    clip=helper.clip_stats(steps,doc)
    durations=sum(e['manipulation_wall_time_s'] for e in episodes)
    command_joint=[s['feedback']['command_state']['robot_age']*1000 for s in steps]
    command_pose=[s['feedback']['command_state']['pose_age']*1000 for s in steps]
    outcomes=Counter(e['finish_reason'] for e in episodes)
    rotation={k:dict(successes=sum(r['rotation']==k and r['outcome']=='success' for r in rows),
        episodes=sum(r['rotation']==k for r in rows)) for k in ['+X','-X','+Y','-Y','+Z','-Z']}
    summary=dict(episodes=len(episodes),successes=len(success),drops=outcomes['failure'],timeouts=outcomes['timeout'],
        aborted_episodes=len(doc['episodes'])-len(episodes),success_rate=len(success)/len(episodes),actions=len(steps),
        total_manipulation_s=durations,successful_duration_s=stats([e['manipulation_wall_time_s'] for e in success]),
        successful_final_pos_mm=stats([e['goal_errors']['pos']*1000 for e in success]),
        successful_final_angle_deg=stats([math.degrees(e['quaternion_error_rad']) for e in success]),
        all_final_pos_mm=stats([e['goal_errors']['pos']*1000 for e in episodes]),
        all_final_angle_deg=stats([math.degrees(e['quaternion_error_rad']) for e in episodes]),
        planning_ms=stats(plans*1000),planning_over_64ms_count=int((plans>.064).sum()),
        planning_over_64ms_percent=float((plans>.064).mean()*100),
        paced_wait_count=int((wait>0).sum()),paced_wait_percent=float((wait>0).mean()*100),
        paced_wait_ms=stats(wait[wait>0]*1000),paced_wait_total_s=float(wait.sum()),
        publication_interval_ms=stats(intervals*1000),publication_frequency_hz=float(len(intervals)/intervals.sum()),
        instantaneous_frequency_hz=stats(1/intervals),actions_per_manipulation_second=len(steps)/durations,
        command_joint_age_ms=stats(command_joint),command_pose_age_ms=stats(command_pose),
        planning_start_joint_age_ms=stats([s['feedback']['robot_age']*1000 for s in steps]),
        planning_start_object_age_ms=stats([s['feedback']['pose_age']*1000 for s in steps]),
        initial_angle_deg=stats(initial_angles),initial_position_mm=stats(initial_pos),
        estimated_target_change_max_joint_deg=stats(changes),
        mean_episode_target_change_l2_rms_rad=float(np.mean(rms)),clipping=clip,rotations=rotation)
    return summary,rows,curves,dict(planning_ms=plans*1000,frequency=1/intervals,wait_ms=wait*1000)


def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/'plots'/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/'plots'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def plot(metrics,rows,curves,timings):
    (OUT/'plots').mkdir(exist_ok=True)
    variants=['baseline','trial358'];x=np.arange(2)
    fig,ax=plt.subplots(figsize=(6,4))
    counts=[metrics[v]['successes'] for v in variants]
    ax.bar([LABELS[v] for v in variants],np.array(counts)/12*100,color=[COLORS[v] for v in variants])
    for i,n in enumerate(counts):ax.text(i,n/12*100+2,f'{n}/12',ha='center')
    ax.set(ylim=(0,105),ylabel='Success rate (%)',title='Separate hardware batches; same six rotation categories')
    finish(fig,'success_rate')

    fig,ax=plt.subplots(figsize=(8,4));rotations=['+X','-X','+Y','-Y','+Z','-Z']
    for j,v in enumerate(variants):
        values=[metrics[v]['rotations'][r]['successes'] for r in rotations]
        ax.bar(np.arange(6)+(j-.5)*.35,values,width=.35,color=COLORS[v],label=LABELS[v])
    ax.set(xticks=np.arange(6),xticklabels=rotations,ylim=(0,2.5),yticks=[0,1,2],ylabel='Successes / two attempts',title='Starting rotation relative to fixed goal');ax.legend()
    finish(fig,'success_by_starting_rotation')

    for field,title,threshold in [('angle_deg','Orientation error (degrees)',math.degrees(2*math.asin(.2))),('position_mm','Position error (mm)',20)]:
        fig,axes=plt.subplots(4,3,figsize=(13,11))
        for ax,row,curve in zip(axes.flat,rows['trial358'],curves['trial358']):
            ax.plot(curve['time_s'],curve[field],color=COLORS['trial358'])
            ax.axhline(threshold,color='black',ls='--',lw=.8)
            ax.set(title=f'Episode {row["episode"]}: {row["rotation"]}, {row["outcome"]}',xlabel='Manipulation time (s)',ylabel=title)
            ax.grid(alpha=.2)
        finish(fig,f'episode_{field}')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for v in variants:
        a=np.sort(timings[v]['planning_ms']);axes[0].plot(a,np.arange(1,len(a)+1)/len(a),color=COLORS[v],label=LABELS[v])
        axes[1].hist(timings[v]['frequency'],bins=np.linspace(12.5,16,60),density=True,histtype='step',color=COLORS[v],label=LABELS[v])
    axes[0].axvline(64,color='black',ls='--');axes[0].set(xlabel='MPPI planning time (ms)',ylabel='Fraction of plans',title='Planning time distribution');axes[0].legend()
    axes[1].axvline(15.625,color='black',ls='--');axes[1].set(xlabel='Publication frequency (Hz)',ylabel='Density',title='Paced action publication frequency');axes[1].legend()
    finish(fig,'planning_and_control_frequency')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,v in enumerate(variants):
        values=[metrics[v]['clipping'][k] for k in ['delta_clipped_percent','absolute_clipped_percent']]
        axes[0].bar(np.arange(2)+(j-.5)*.35,values,width=.35,color=COLORS[v],label=LABELS[v])
    axes[0].set(xticks=[0,1],xticklabels=['±0.2 clamp','Joint-bound clamp'],ylabel='Controller-estimated updates clipped (%)');axes[0].legend()
    axes[1].boxplot([[r['target_change_l2_rms_rad'] for r in rows[v]] for v in variants],tick_labels=[LABELS[v] for v in variants])
    axes[1].set(ylabel='Per-episode RMS target change (16-joint L2, rad)',title='Controller target estimates; excludes first update')
    finish(fig,'clipping_and_target_changes')

    failed=[(r,c) for r,c in zip(rows['trial358'],curves['trial358']) if r['outcome']!='success']
    fig,axes=plt.subplots(len(failed),2,figsize=(11,4*len(failed)),squeeze=False)
    for (row,curve),(a,b) in zip(failed,axes):
        t=curve['time_s'];a.plot(t,curve['angle_deg'],color=COLORS['trial358']);a.axhline(math.degrees(2*math.asin(.2)),color='black',ls='--')
        a.set(title=f'Episode {row["episode"]}: {row["rotation"]}',xlabel='Time (s)',ylabel='Orientation error (degrees)')
        for j,label in enumerate('xyz'):b.plot(t,np.array(curve['position_axes_mm'])[:,j],label=label)
        b.axhline(0,color='black',ls='--',lw=.8);b.set(xlabel='Time (s)',ylabel='Position minus goal (mm)');b.legend()
    finish(fig,'failure_details')


def main():
    docs=dict(baseline=json.loads((BASE/'controller.json').read_text()),trial358=json.loads((RUN/'controller.json').read_text()))
    launch=json.loads((RUN/'launch.json').read_text())
    metrics={};rows={};curves={};timings={}
    for name,doc in docs.items():metrics[name],rows[name],curves[name],timings[name]=collect(doc)
    checks={k:docs['baseline'][k]==docs['trial358'][k] for k in ['joint_reference','rollout_config','pose_transform','task_contact_params','success_thresholds','failure_z','control_policy','fixed_goal_quat','model_ctrl_range','effective_ctrl_range']}
    assert all(checks.values()),checks
    for name in ['baseline','trial358']:assert all(metrics[name]['rotations'][r]['episodes']==2 for r in ['+X','-X','+Y','-Y','+Z','-Z'])
    p=fisher_exact([[metrics['baseline']['successes'],12-metrics['baseline']['successes']],
                    [metrics['trial358']['successes'],12-metrics['trial358']['successes']]]).pvalue
    video=json.loads((RUN/'video/recording.json').read_text())
    hardware_diff={k:[docs['baseline']['hardware_config'].get(k),docs['trial358']['hardware_config'].get(k)]
        for k in set(docs['baseline']['hardware_config'])|set(docs['trial358']['hardware_config'])
        if docs['baseline']['hardware_config'].get(k)!=docs['trial358']['hardware_config'].get(k)}
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),run=str(RUN),baseline=str(BASE),
        summaries=metrics,episodes=rows,configuration_checks=checks,hardware_config_differences=hardware_diff,
        fisher_exact_two_sided_p=float(p),usb_latency=launch.get('usb_latency_timer'),
        video=dict(status=video['status'],episodes=len(video['episodes']),queue_dropped_frames=video['queue_dropped_frames'],
                   repeated_frames=video['repeated_video_frames'],error=video['error']),
        limitations=['Separate physical placements, not matched episode pairs; 12 episodes per batch.',
            'Baseline includes one initially off-center placement (~38mm); current starts are all close to goal position.',
            'Clipping and target-change metrics use controller estimates; no bridge telemetry or motor-current recordings.',
            'All successes stop at the first qualifying state; no consecutive-step hold requirement.',
            'New video padding and USB timer lifecycle code differ; manipulation timing excludes padding/operator waits.',
            'USB baseline observation was 1ms after its batch; its runtime timer was not logged.'])
    save('report.json',report)
    csv_write('episodes.csv',rows['trial358']);csv_write('baseline_episodes.csv',rows['baseline'])
    csv_write('per_joint_clipping.csv',metrics['trial358']['clipping']['joints'])
    save('curves.json',curves['trial358'])
    plot(metrics,rows,curves,timings)
    important=['episodes','successes','drops','timeouts','actions','total_manipulation_s','publication_frequency_hz','actions_per_manipulation_second','planning_over_64ms_count','planning_over_64ms_percent','paced_wait_count','paced_wait_percent','mean_episode_target_change_l2_rms_rad']
    table=['metric,baseline,trial358']+[f'{k},{metrics["baseline"][k]},{metrics["trial358"][k]}' for k in important]
    (OUT/'metrics.csv').write_text('\n'.join(table)+'\n')
    text=['Trial 358 hardware vs designated paced M2 baseline','',json.dumps(metrics,indent=2),'',
          'Episode outcomes',json.dumps(rows['trial358'],indent=2),'',
          'Configuration comparisons',json.dumps(checks,indent=2),'',
          'Limitations',*report['limitations']]
    (OUT/'REPORT.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(summaries={name:{k:m[k] for k in important} for name,m in metrics.items()},
                         failures=[r for r in rows['trial358'] if r['outcome']!='success'],configuration_checks=checks,
                         fisher_exact_p=float(p)),indent=2))


if __name__=='__main__':main()
