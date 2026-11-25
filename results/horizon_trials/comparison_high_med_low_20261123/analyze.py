"""Recorded-data comparison of timing-matched M2 hardware configurations."""
import csv
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import shutil

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'run_leap_hardware.py').exists())
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
import numpy as np
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCES={
 'high':'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial358_shift512_h0256_paced_measured_limits/leap_experiment_20261123_124445_598769',
 'med':'results/horizon_trials/hardware/cube_reorient/hand-med_obj-high/M2/trial358_shift512_h0320_paced/combined_12_completed_20261123_211932_212743',
 'low':'results/horizon_trials/hardware/cube_reorient/hand-low_obj-high/M2/trial358_shift512_h0320_paced/combined_12_completed_20261123_220437_221201'}
COLORS={'high':'#377eb8','med':'#e17826','low':'#419c64'}
LABELS={'high':'High (4 steps)','med':'Medium (5 steps)','low':'Low (5 steps)'}
spec=importlib.util.spec_from_file_location('baseline_analysis',ROOT/SOURCES['high']/'analysis/analyze.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_save(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def wilson(k,n):
    z=1.95996398454;p=k/n;a=1+z*z/n
    c=(p+z*z/(2*n))/a;d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return [float(c-d),float(c+d)]


def figure(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/'plots'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    (OUT/'plots').mkdir(exist_ok=True)
    docs={k:json.loads((ROOT/v/'controller.json').read_text()) for k,v in SOURCES.items()}
    metrics={};episodes={};curves={};timings={};manifest={};checks={};all_rows=[]
    keys=['weights','hardware_config','joint_reference','pose_transform','planner_config','joint_limits','success_thresholds','failure_z','control_policy','fixed_goal_quat','temporal_plan_blending']
    for key,doc in docs.items():
        assert doc['args']['hand_acc']==key and doc['args']['n_samples']==512 and doc['args']['pace_control']
        checks[key]={k:doc[k]==docs['high'][k] for k in keys};assert all(checks[key].values()),checks[key]
        metrics[key],episodes[key],curves[key],timings[key]=helper.collect(doc)
        m=metrics[key];assert m['episodes']==12 and m['aborted_episodes']==0
        assert all(v['episodes']==2 for v in m['rotations'].values())
        m['success_wilson_95']=wilson(m['successes'],12)
        folder=OUT/key;folder.mkdir(exist_ok=True);(folder/'video').mkdir(exist_ok=True)
        # Independent metadata copies; video files linked without encoding.
        for name in ['controller.json','summary.json','sources.json','launch.json']:
            src=ROOT/SOURCES[key]/name
            if src.exists():shutil.copy2(src,folder/name)
        for e in episodes[key]:
            e.update(hand_accuracy=key,horizon_s=doc['args']['time_horizon'],
                     group_episode=e['episode'],video=f'{key}/video/episode_{e["episode"]:02d}.mkv')
            all_rows.append(e)
            old=ROOT/SOURCES[key]/'video'/f'episode_{e["episode"]:02d}.mkv';new=OUT/e['video']
            assert old.is_file()
            if not new.exists():
                try:os.link(old,new)
                except OSError:shutil.copy2(old,new)
            assert old.stat().st_size==new.stat().st_size
        csv_save(folder/'episodes.csv',episodes[key])
        manifest[key]=dict(source=SOURCES[key],controller_sha256=hashlib.sha256((ROOT/SOURCES[key]/'controller.json').read_bytes()).hexdigest(),
            episodes=12,horizon_s=doc['args']['time_horizon'],video_count=12,
            note='Medium/low source directories are already completed-only aggregates; their raw sources must not be counted again.')
    csv_save(OUT/'episodes.csv',all_rows)
    save(OUT/'sources.json',dict(groups=manifest,total_distinct_episodes=36,note='Derived comparison; no additional experimental episodes. Raw sources retained.'))
    fields={
      'Successes / 12':lambda m:str(m['successes']),
      'Success rate (%)':lambda m:100*m['success_rate'],
      'Drops':lambda m:m['drops'],
      'Manipulation wall time total (s)':lambda m:m['total_manipulation_s'],
      'Successful duration mean (s)':lambda m:m['successful_duration_s']['mean'],
      'Successful duration median (s)':lambda m:m['successful_duration_s']['median'],
      'Successful endpoint position mean (mm)':lambda m:m['successful_final_pos_mm']['mean'],
      'Successful endpoint angle mean (deg)':lambda m:m['successful_final_angle_deg']['mean'],
      'All endpoint position mean (mm)':lambda m:m['all_final_pos_mm']['mean'],
      'All endpoint angle mean (deg)':lambda m:m['all_final_angle_deg']['mean'],
      'Planning mean (ms)':lambda m:m['planning_ms']['mean'],
      'Planning median (ms)':lambda m:m['planning_ms']['median'],
      'Planning p95 (ms)':lambda m:m['planning_ms']['p95'],
      'Planning p99 (ms)':lambda m:m['planning_ms']['p99'],
      'Planning min (ms)':lambda m:m['planning_ms']['min'],
      'Planning max (ms)':lambda m:m['planning_ms']['max'],
      'Plans over 64ms (%)':lambda m:m['planning_over_64ms_percent'],
      'Pacing wait (%)':lambda m:m['paced_wait_percent'],
      'Pacing wait total (s)':lambda m:m['paced_wait_total_s'],
      'Control publication rate (Hz)':lambda m:m['publication_frequency_hz'],
      'Control instantaneous min (Hz)':lambda m:m['instantaneous_frequency_hz']['min'],
      'Control instantaneous max (Hz)':lambda m:m['instantaneous_frequency_hz']['max'],
      'Command joint age mean (ms)':lambda m:m['command_joint_age_ms']['mean'],
      'Command object age mean (ms)':lambda m:m['command_pose_age_ms']['mean'],
      'Delta-clipped updates (%)':lambda m:m['clipping']['delta_clipped_percent'],
      'Absolute-bound-clipped updates (%)':lambda m:m['clipping']['absolute_clipped_percent'],
      'Either clipping (%)':lambda m:m['clipping']['either_clipped_percent'],
      'Target-change L2 RMS mean (rad)':lambda m:m['mean_episode_target_change_l2_rms_rad'],
      'Max-joint movement during planning mean (deg)':lambda m:m['clipping']['joint_motion_during_planning_max_per_action_deg']['mean'],
      'Initial position offset mean (mm)':lambda m:m['initial_position_mm']['mean']}
    table=[dict(metric='Horizon (s)',**{k:d['args']['time_horizon'] for k,d in docs.items()})]
    table += [dict(metric=label,**{k:f(m) for k,m in metrics.items()}) for label,f in fields.items()]
    csv_save(OUT/'metrics.csv',table)
    failures=[e for e in all_rows if e['outcome']!='success']
    csv_save(OUT/'failures.csv',failures)
    y=[e for e in all_rows if e['rotation'][1]=='Y'];other=[e for e in all_rows if e['rotation'][1]!='Y']
    pairwise={f'{a}_vs_{b}':float(fisher_exact([[metrics[a]['successes'],12-metrics[a]['successes']],[metrics[b]['successes'],12-metrics[b]['successes']]]).pvalue) for a,b in [('high','med'),('high','low'),('med','low')]}
    report=dict(summaries=metrics,configuration_checks=checks,episodes=all_rows,pairwise_fisher_two_sided_p=pairwise,
        joint_outcomes=dict(episodes=36,successes=sum(m['successes'] for m in metrics.values()),drops=len(failures),
            y_rotation_successes=sum(e['outcome']=='success' for e in y),y_rotation_attempts=len(y),
            other_rotation_successes=sum(e['outcome']=='success' for e in other),other_rotation_attempts=len(other)),
        limitations=['Timing-matched configurations: high uses four steps, medium/low five. Geometry effects cannot be isolated from horizon differences.',
            '12 independent manual placements per configuration; balanced rotation categories but not paired physical states.',
            'Successful endpoint metrics are conditional on success and terminate at the first qualifying frame; no hold criterion.',
            'No exact bridge target/current telemetry. Clipping and target smoothness use controller estimates.',
            'Rotation-specific counts are two per configuration; patterns are exploratory.',
            'Some source launchers reported a post-completion bridge timeout; all selected episodes had completed and remain valid.'])
    save(OUT/'report.json',report)
    # Success confidence intervals and rotation-specific success counts.
    names=list(docs);fig,ax=plt.subplots(figsize=(7,4))
    for i,k in enumerate(names):
        rate=metrics[k]['success_rate']*100;lo,hi=np.array(metrics[k]['success_wilson_95'])*100
        ax.bar(i,rate,color=COLORS[k]);ax.errorbar(i,rate,yerr=[[rate-lo],[hi-rate]],color='black',capsize=5)
        ax.text(i,rate+2,f'{metrics[k]["successes"]}/12',ha='center')
    ax.set(xticks=range(3),xticklabels=[LABELS[k] for k in names],ylim=(0,110),ylabel='Success rate (%)',title='M2 hardware: 95% Wilson intervals')
    figure(fig,'success_rates')
    rotations=['+X','-X','+Y','-Y','+Z','-Z'];array=np.array([[metrics[k]['rotations'][r]['successes'] for r in rotations] for k in names])
    fig,ax=plt.subplots(figsize=(9,3));im=ax.imshow(array,vmin=0,vmax=2,cmap='RdYlGn',aspect='auto')
    for i in range(3):
        for j in range(6):ax.text(j,i,f'{array[i,j]}/2',ha='center',va='center')
    ax.set(xticks=range(6),xticklabels=rotations,yticks=range(3),yticklabels=[LABELS[k] for k in names],title='Success by signed starting rotation (~90 degrees)')
    figure(fig,'success_by_rotation')
    for name,field,threshold,xlabel in [('planning_time','planning_ms',64,'Planning time (ms)'),('control_frequency','frequency',15.625,'Publication frequency (Hz)')]:
        fig,ax=plt.subplots(figsize=(8,4))
        for k in names:
            x=np.sort(timings[k][field]);ax.plot(x,np.arange(1,len(x)+1)/len(x),label=LABELS[k],color=COLORS[k])
        ax.axvline(threshold,color='black',ls='--');ax.set(xlabel=xlabel,ylabel='Cumulative fraction');ax.legend();ax.grid(alpha=.2);figure(fig,name)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,field,label,threshold in [(axes[0],'final_position_error_mm','Successful endpoint position error (mm)',20),(axes[1],'final_angle_deg','Successful endpoint angle error (deg)',np.degrees(2*np.arcsin(.2))),(axes[2],'duration_s','Successful manipulation duration (s)',None)]:
        for i,k in enumerate(names):
            vals=[e[field] for e in episodes[k] if e['outcome']=='success']
            ax.scatter(i+np.linspace(-.10,.10,len(vals)),vals,color=COLORS[k],alpha=.8)
            ax.plot([i-.18,i+.18],[np.median(vals)]*2,color='black')
        if threshold is not None:ax.axhline(threshold,color='black',ls='--',lw=.8)
        ax.set(xticks=range(3),xticklabels=['High','Medium','Low'],ylabel=label)
    figure(fig,'successful_errors_and_duration')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,k in enumerate(names):
        axes[0].bar(np.arange(2)+(i-1)*.25,[metrics[k]['clipping'][f] for f in ['delta_clipped_percent','absolute_clipped_percent']],width=.25,color=COLORS[k],label=LABELS[k])
        axes[1].scatter(i,[metrics[k]['mean_episode_target_change_l2_rms_rad']],color=COLORS[k],s=80)
    axes[0].set(xticks=[0,1],xticklabels=['±0.2 delta bound','Absolute joint bounds'],ylabel='Updates clipped (%)');axes[0].legend()
    axes[1].set(xticks=range(3),xticklabels=['High','Medium','Low'],ylabel='Mean episode target-change L2 RMS (rad)');figure(fig,'clipping_and_target_changes')
    body=['<!doctype html><meta charset="utf-8"><title>M2 high/medium/low comparison</title>',
        '<style>body{font:16px sans-serif;max-width:1250px;margin:30px auto;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}img{max-width:100%}</style>',
        '<h1>M2 hardware: high / medium / low hand accuracy</h1>',
        '<p>36 distinct completed episodes. High: 4-step horizon; medium and low: 5 steps. All use Trial 358 weights, 512 samples and paced control. Timing-matched comparison, not a controlled geometry-only comparison.</p>',
        '<p><a href="FINDINGS.txt">Joint interpretation</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="episodes.csv">All episodes</a> · <a href="report.json">Full analysis</a> · <a href="sources.json">Source manifest</a></p>',
        '<table><tr><th>Metric</th><th>High</th><th>Medium</th><th>Low</th></tr>']
    for row in table:
        values=[f'{row[k]:.3f}' if isinstance(row[k],float) else str(row[k]) for k in names]
        body.append('<tr><td>'+html.escape(row['metric'])+'</td>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    body.append('</table>')
    for plot in ['success_rates','success_by_rotation','planning_time','control_frequency','successful_errors_and_duration','clipping_and_target_changes']:body.append(f'<p><a href="plots/{plot}.pdf">PDF</a></p><img src="plots/{plot}.png">')
    body.append('<h2>Episode videos</h2><table><tr><th>Accuracy</th><th>Episode</th><th>Rotation</th><th>Result</th><th>Video</th></tr>')
    for e in all_rows:body.append(f'<tr><td>{e["hand_accuracy"]}</td><td>{e["episode"]}</td><td>{e["rotation"]}</td><td>{e["outcome"]}</td><td><a href="{e["video"]}">Video</a></td></tr>')
    body.append('</table><h2>Limitations</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in report['limitations'])+'</ul>')
    (OUT/'index.html').write_text('\n'.join(body))
    assert len(all_rows)==36 and len(list(OUT.glob('*/video/*.mkv')))==36
    print(json.dumps(dict(out=str(OUT),joint=report['joint_outcomes'],pairwise_p=pairwise,confidence_intervals={k:m['success_wilson_95'] for k,m in metrics.items()}),indent=2))


if __name__=='__main__':main()
