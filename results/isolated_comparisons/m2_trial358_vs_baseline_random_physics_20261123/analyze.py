"""Summarize and plot the paired randomized-physics simulation evaluation."""
import csv
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path

OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG=json.loads((OUT/'config.json').read_text())
BASE=json.loads((OUT/'baseline.json').read_text())
CASES=json.loads((OUT/'scenarios.json').read_text())
INITIAL=json.loads((OUT/'initial_states.json').read_text())
VARIANTS=['baseline','trial358']
LABELS={'baseline':'M2 baseline','trial358':'BO trial 358'}
COLORS={'baseline':'#377eb8','trial358':'#e17826'}
PERIOD=.064


def angle(q):
    return np.degrees(2*np.arcsin(np.sqrt(np.clip(q,0,1))))


def collect():
    episodes={};traces={};rows=[]
    for variant in VARIANTS:
        episodes[variant]=[]
        for case in CASES:
            stem=OUT/variant/f'pair_{case["pair"]:02d}'
            summary=json.loads(stem.with_name(stem.name+'_metrics.json').read_text())
            assert summary['physics']==case['physics']
            assert summary['initial_state_sha256']==case['initial_state_sha256']
            assert summary['paired_noise_seed']==case['noise_seed']
            assert summary['noise_seed']==case['noise_seed']
            assert summary['planner_nominal_mass_kg']==.07
            assert summary['planner_nominal_friction']==CONFIG['nominal_friction']
            document=json.loads(stem.with_suffix('.json').read_text())
            pc=document['configs']['planner']['config']
            for key,value in CONFIG['variants'][variant]['params'].items():
                assert pc[key]==value,(variant,key)
            assert pc['temperature']==.1 and pc['noise_sigma']==.05 and pc['warm_start']=='shift'
            assert document['configs']['rollout_simulator']['N']==512
            assert document['configs']['rollout_simulator']['resolved']['resolved_horizon']==4
            assert document['configs']['eval_task']['model_path']==case['eval_scene']
            for role in ('rollout_task','eval_task'):
                for key,value in (document['configs'][role]['config']['cost_weights'] or {}).items():
                    assert CONFIG['variants'][variant]['weights'][key]==value,(variant,role,key)
            data=np.load(stem.parent/document['episodes'][0]['file'],allow_pickle=False)
            assert data.shape[0]==summary['n_steps']
            if len(data):
                np.testing.assert_array_equal(data['q'][0],INITIAL[case['source_index']]['q'])
                np.testing.assert_array_equal(data['q_dot'][0],INITIAL[case['source_index']]['q_dot'])
            final_q=np.asarray(summary['q_end'])
            states=np.vstack([data['q'],final_q])
            target=np.asarray(BASE['target_pos'])
            pos=np.linalg.norm(states[:,16:19]-target,axis=1)*1000
            ori=angle(1-np.square(states[:,19]))  # goal is identity WXYZ
            t=np.r_[np.arange(len(data))*PERIOD,summary['simulation_s']]
            delta=np.diff(np.vstack([BASE['hardware_config']['robot_qref'],data['u']]),axis=0)
            later=delta[1:] if len(delta)>1 else delta
            rms=float(np.sqrt(np.mean(np.sum(later**2,axis=1)))) if len(later) else 0.
            durations=np.asarray(summary['all_plan_s'])
            counts=summary['final_target_clipping'];n=counts['applied_actions']
            e=summary['goal_errors_end']
            row=dict(variant=variant,pair=case['pair'],rotation=case['rotation'],
                mass_g=case['physics']['mass_kg']*1000,**{f'mu_{k}':v for k,v in case['physics']['sliding'].items()},
                outcome=summary['finish_reason'],success=summary['success'],n_steps=summary['n_steps'],
                simulation_s=summary['simulation_s'],wall_s=summary['wall_s'],
                final_position_mm=e['pos']*1000,final_quaternion_error=e['quat'],
                final_orientation_deg=float(angle(e['quat'])),
                min_position_mm=float(pos.min()),min_orientation_deg=float(ori.min()),
                target_change_l2_rms_rad=rms,target_change_joint_rms_rad=rms/4,
                delta_clipped_steps=counts['delta_clipped'],absolute_clipped_steps=counts['absolute_clipped'],
                applied_steps=n,delta_clip_percent=100*counts['delta_clipped']/n if n else 0,
                absolute_clip_percent=100*counts['absolute_clipped']/n if n else 0,
                planning_mean_ms=float(np.mean(durations)*1000),planning_p95_ms=float(np.percentile(durations,95)*1000),
                planning_min_ms=float(np.min(durations)*1000),planning_max_ms=float(np.max(durations)*1000))
            episodes[variant].append(row);rows.append(row)
            traces[(variant,case['pair'])]=dict(t=t,pos=pos,ori=ori,target_delta=delta,planning=durations)
    return episodes,traces,rows


def statistics(episodes,traces):
    results={}
    for variant,rows in episodes.items():
        success=[r for r in rows if r['success']]
        planning=np.concatenate([traces[(variant,r['pair'])]['planning'] for r in rows])
        n=sum(r['applied_steps'] for r in rows)
        result=dict(episodes=len(rows),successes=len(success),
            drops=sum(r['outcome']=='failure' for r in rows),timeouts=sum(r['outcome']=='timeout' for r in rows),
            success_rate=len(success)/len(rows),
            mean_final_position_mm=float(np.mean([r['final_position_mm'] for r in rows])),
            mean_final_orientation_deg=float(np.mean([r['final_orientation_deg'] for r in rows])),
            mean_final_quaternion_error=float(np.mean([r['final_quaternion_error'] for r in rows])),
            mean_steps=float(np.mean([r['n_steps'] for r in rows])),
            mean_success_simulation_s=float(np.mean([r['simulation_s'] for r in success])) if success else None,
            mean_episode_wall_s=float(np.mean([r['wall_s'] for r in rows])),
            mean_episode_target_change_l2_rms_rad=float(np.mean([r['target_change_l2_rms_rad'] for r in rows])),
            delta_clip_percent=100*sum(r['delta_clipped_steps'] for r in rows)/n,
            absolute_clip_percent=100*sum(r['absolute_clipped_steps'] for r in rows)/n,
            planning_mean_ms=float(planning.mean()*1000),planning_p95_ms=float(np.percentile(planning,95)*1000),
            planning_min_ms=float(planning.min()*1000),planning_max_ms=float(planning.max()*1000))
        results[variant]=result
    a=episodes['baseline'];b=episodes['trial358']
    only_new=sum(not x['success'] and y['success'] for x,y in zip(a,b))
    only_old=sum(x['success'] and not y['success'] for x,y in zip(a,b))
    discordant=only_new+only_old
    probability=min(1.,2*sum(math.comb(discordant,k) for k in range(min(only_new,only_old)+1))/2**discordant) if discordant else 1.
    paired=dict(both_success=sum(x['success'] and y['success'] for x,y in zip(a,b)),
        baseline_only=only_old,trial358_only=only_new,
        neither_success=sum(not x['success'] and not y['success'] for x,y in zip(a,b)),
        exact_mcnemar_two_sided_p=probability)
    common=[i for i,(x,y) in enumerate(zip(a,b)) if x['success'] and y['success']]
    paired['both_success_metrics']={v:dict(
        mean_simulation_s=float(np.mean([episodes[v][i]['simulation_s'] for i in common])),
        mean_final_position_mm=float(np.mean([episodes[v][i]['final_position_mm'] for i in common])),
        mean_final_orientation_deg=float(np.mean([episodes[v][i]['final_orientation_deg'] for i in common])))
        for v in VARIANTS} if common else {}
    paired['trial358_fewer_steps_pairs']=sum(y['n_steps']<x['n_steps'] for x,y in zip(a,b))
    paired['trial358_fewer_steps_both_success']=sum(b[i]['n_steps']<a[i]['n_steps'] for i in common)
    return results,paired


def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/'plots'/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'plots'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def plots(episodes,traces,metrics):
    (OUT/'plots').mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,4))
    bottom=np.zeros(2)
    for key,color,label in [('successes','#4daf4a','Success'),('drops','#e41a1c','Drop'),('timeouts','#999999','Timeout')]:
        values=np.array([metrics[v][key] for v in VARIANTS])
        ax.bar([LABELS[v] for v in VARIANTS],values,bottom=bottom,color=color,label=label)
        for i,n in enumerate(values):
            if n:ax.text(i,bottom[i]+n/2,str(n),ha='center',va='center',color='white',weight='bold')
        bottom+=values
    ax.set(ylim=(0,10.5),ylabel='Episodes',title='Paired physical-mismatch evaluation (10 each)');ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    finish(fig,'outcomes')

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,threshold,title in zip(axes,['final_position_mm','final_orientation_deg'],[20,float(angle(.04))],['Final position error (mm)','Final orientation error (degrees)']):
        for i in range(10):
            values=[episodes[v][i][key] for v in VARIANTS]
            ax.plot([0,1],values,color='#aaaaaa',lw=.9,zorder=1)
            for j,v in enumerate(VARIANTS):ax.scatter(j,values[j],color=COLORS[v],s=30,zorder=2)
        ax.axhline(threshold,color='black',ls='--',lw=1,label='Success threshold')
        ax.set(xticks=[0,1],xticklabels=[LABELS[v] for v in VARIANTS],ylabel=title);ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Each gray line connects the same paired episode',fontsize=11)
    finish(fig,'paired_final_errors')

    for field,title,threshold in [('ori','Orientation error (degrees)',float(angle(.04))),('pos','Position error (mm)',20)]:
        fig,axes=plt.subplots(5,2,figsize=(11,13),squeeze=False)
        for ax,case in zip(axes.flat,CASES):
            pair=case['pair']
            for v in VARIANTS:
                trace=traces[(v,pair)];row=episodes[v][pair-1]
                ax.plot(trace['t'],trace[field],color=COLORS[v],label=f'{LABELS[v]}: {row["outcome"]}')
            ax.axhline(threshold,color='black',ls='--',lw=.8)
            ax.set_title(f'Pair {pair}: {case["rotation"]}, {case["physics"]["mass_kg"]*1000:.1f} g',fontsize=10)
            ax.set(xlabel='Simulated time (s)',ylabel=title);ax.legend(fontsize=7);ax.grid(alpha=.2)
        finish(fig,f'{field}_trajectories_by_pair')

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i in range(10):
        axes[0].plot([0,1],[episodes[v][i]['target_change_l2_rms_rad'] for v in VARIANTS],color='#aaaaaa',lw=.9)
        for j,v in enumerate(VARIANTS):axes[0].scatter(j,episodes[v][i]['target_change_l2_rms_rad'],color=COLORS[v],s=25)
    axes[0].set(xticks=[0,1],xticklabels=[LABELS[v] for v in VARIANTS],ylabel='Target-change L2 RMS (rad / update)',title='Applied command changes (first update excluded)')
    for j,v in enumerate(VARIANTS):
        axes[1].bar(np.arange(2)+(j-.5)*.35,[metrics[v]['delta_clip_percent'],metrics[v]['absolute_clip_percent']],width=.35,color=COLORS[v],label=LABELS[v])
    axes[1].set(xticks=[0,1],xticklabels=['±0.2 delta clamp','Joint-bound clamp'],ylabel='Applied updates clipped (%)',title='Two clamp stages can overlap');axes[1].legend()
    finish(fig,'smoothness_and_clipping')


def main():
    episodes,traces,rows=collect();metrics,paired=statistics(episodes,traces)
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),metrics=metrics,paired_outcomes=paired,
        episodes=episodes,scope='Only evaluation mass and sliding friction randomized; both rollout models nominal.',
        limitations=['Ten pairs only; initial poses appeared in BO, physical/noise draws are new.',
            'Initial hand/object penetrations and recorded preferred-limit offsets retained from baseline.',
            'Virtual 64ms application delay; no tracking noise, USB latency, or GPU-time-dependent physics.',
            'Planning times measured on a shared GPU; not a hardware control-frequency benchmark.',
            'Both methods stop at their own first success/drop/timeout; final-error averages mix these outcomes.'])
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['metric']+VARIANTS)
        for key in metrics['baseline']:writer.writerow([key]+[metrics[v][key] for v in VARIANTS])
    plots(episodes,traces,metrics)
    text=['Paired evaluation: baseline vs BO trial 358','',
        'Randomized evaluation physics only; planners retain nominal 70g and friction .3/.5/.4/.4.',
        '10 paired scenarios, six signed rotation directions, matched initial states/noise seeds.',
        'Mass 56-84g; independent sliding friction +/-20%; inertia scales with mass.',
        'Everything else uses the frozen M2 baseline timing, calibrated hand, P011 and clipping.',
        '',json.dumps(metrics,indent=2),'',json.dumps(paired,indent=2),'',
        'See plots/ for separate images and PDFs; episodes.csv for per-pair physical values/results.',
        'comparison.json includes definitions and limitations. All production files remain unchanged.']
    (OUT/'REPORT.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(metrics=metrics,paired=paired),indent=2))


if __name__=='__main__':main()
