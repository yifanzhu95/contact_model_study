"""Observed horizon/timing relationships from original hardware recordings.

No simulated, projected, rejected-plan, or aborted-episode samples are included.
Independent physical placements/contact states mean this is not a fixed-state
GPU benchmark or an isolated causal measurement of horizon cost.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'run_leap_hardware.py').exists())
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={'high':'#3977ad','med':'#dc852c','low':'#39865b'}
MODELCOLORS={'M1':'#3977ad','M2':'#de8836'}
LABELS={'high':'High','med':'Medium','low':'Low'}
ACCS=['high','med','low']


def normalize_xml(text):
    tree=ET.fromstring(text)
    for mesh in tree.findall('.//mesh'):mesh.attrib.pop('content_type',None)
    return ET.tostring(tree)


def savefig(fig,name):
    fig.tight_layout(rect=(0,.085,1,.96))
    for ext in ['png','pdf']:
        fig.savefig(OUT/'plots'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def make_plots():
    comparison=json.loads((OUT/'report.json').read_text())
    paths=sorted((ROOT/'results/horizon_trials/hardware/cube_reorient').glob('hand-*_obj-high/M[12]/*/leap_experiment_*/controller.json'))
    paths.append(ROOT/comparison['source_manifest']['M2_high']['controller'])
    groups={};checks={};manifest=[];reference=None;cell_reference={}
    common=['weights','hardware_config','joint_reference','pose_transform','planner_config',
            'joint_limits','success_thresholds','effective_ctrl_range','control_policy',
            'fixed_goal_quat','temporal_plan_blending','failure_z']
    for path in paths:
        doc=json.loads(path.read_text())
        episodes=[e for e in doc.get('episodes',[]) if e.get('completed') and e.get('finish_reason') in ['success','failure','timeout']]
        if not episodes:continue
        a=doc['args'];model,hand=a['rollout_model'],a['hand_acc']
        assert a['n_samples']==512 and a['pace_control'] and a['ctrl_time_step']==.064
        if reference is None:reference=doc
        check={key:doc.get(key)==reference.get(key) for key in common}
        assert all(check.values()),(str(path),check)
        rc={k:v for k,v in doc['rollout_config'].items() if k not in ['horizon','time_horizon']}
        scenes={role:normalize_xml(doc['scenes'][role]['scene_xml']) for role in ['rollout','eval']}
        cell=(model,hand)
        if cell not in cell_reference:cell_reference[cell]=(rc,scenes)
        else:
            assert rc==cell_reference[cell][0],path
            assert scenes==cell_reference[cell][1],path
        check['within_model_accuracy_rollout_config_except_horizon']=True
        check['within_model_accuracy_scene_xml']=True
        checks[str(path.relative_to(ROOT))]=check
        h=round(a['time_horizon']/.064)
        assert np.isclose(h*.064,a['time_horizon'])
        key=(model,hand,h)
        group=groups.setdefault(key,dict(plans=[],intervals=[],episodes=0,runs=[]))
        for e in episodes:
            steps=[s for s in e['steps'] if s.get('action_published')]
            group['plans'].extend(s['planning_time']*1000 for s in steps)
            group['intervals'].extend(np.diff([s['action_timestamp_us'] for s in steps])*1e-6)
            group['episodes']+=1
        group['runs'].append(str(path.parent.relative_to(ROOT)))
        manifest.append(dict(controller=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            model=model,hand_accuracy=hand,horizon_steps=h,completed_episodes=len(episodes)))
    rows=[]
    for (model,hand,h),g in sorted(groups.items()):
        p=np.array(g['plans']);intervals=np.array(g['intervals'])
        row=dict(model=model,hand_accuracy=hand,horizon_steps=h,horizon_s=h*.064,
            episodes=g['episodes'],published_plans=len(p),mean_ms=float(p.mean()),median_ms=float(np.median(p)),
            p95_ms=float(np.percentile(p,95)),p99_ms=float(np.percentile(p,99)),
            min_ms=float(p.min()),max_ms=float(p.max()),plans_over64_percent=float((p>64).mean()*100),
            publication_hz=float(len(intervals)/intervals.sum()))
        rows.append(row)
    with (OUT/'horizon_timing.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(configurations=rows,inputs=manifest,configuration_checks=checks,
        scope='Only original raw source runs, 512 samples, Trial358 settings, paced control, high object geometry. Includes supplemental horizon trials beyond the 72-episode accepted-baseline comparison.',
        limitations=['Independent manual placements and contact trajectories, not paired fixed-state timing benchmarks.',
            'Plan-weighted distributions from published actions in completed episodes only. Rejected plans and aborted episode attempts are excluded.',
            'M1 medium/low have only a three-step measurement; M2 high only four steps. No horizon trend can be inferred for these cells.',
            'Lines join measured configurations only; no interpolated or extrapolated values are reported. Sample counts differ.',
            'Common recorded settings and per-model/accuracy scene XMLs and solver configuration match apart from horizon. Continuous historical GPU utilization was not logged.'])
    (OUT/'horizon_timing_report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Same-accuracy curves make the actually measured horizon comparisons clear.
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),sharex=True,sharey=True)
    for ax,model in zip(axes,['M1','M2']):
        for hand in ACCS:
            selected=sorted([r for r in rows if r['model']==model and r['hand_accuracy']==hand],key=lambda r:r['horizon_steps'])
            x=[r['horizon_steps'] for r in selected]
            ax.plot(x,[r['mean_ms'] for r in selected],'-o',color=COLORS[hand],lw=2,ms=7,label=LABELS[hand])
            ax.plot(x,[r['p95_ms'] for r in selected],'--^',color=COLORS[hand],lw=1.5,ms=6)
        ax.axhline(64,color='black',ls=':',lw=1.5)
        ax.set(title=model,xlabel='Planning horizon (control steps)',xticks=range(2,9),xlim=(1.6,8.4),ylim=(20,120))
        top=ax.secondary_xaxis('top',functions=(lambda x:x*.064,lambda x:x/.064))
        top.set_xlabel('Horizon duration (seconds)')
        top.set_xticks(np.arange(2,9)*.064)
        top.set_xticklabels([f'{h*.064:.3f}' for h in range(2,9)],fontsize=8)
        ax.grid(alpha=.18);ax.legend(title='Hand accuracy',loc='upper left')
    axes[0].set_ylabel('Recorded planning time (ms)')
    fig.legend(handles=[Line2D([0],[0],color='#444444',marker='o',label='Mean'),
        Line2D([0],[0],color='#444444',ls='--',marker='^',label='95th percentile'),
        Line2D([0],[0],color='black',ls=':',label='64 ms control period')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.025),frameon=False)
    fig.suptitle('Observed planning time versus planning horizon — 512 samples',y=1.02,fontsize=14)
    fig.text(.5,.003,'Lines connect measured batches. Isolated points have no measured horizon sweep. Placements/contact states differ.',ha='center',fontsize=9,color='#444444')
    savefig(fig,'horizon_effect_planning_time')
    # Distribution rather than only mean; every line names its measured horizon.
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),sharey=True)
    styles={2:':',3:'-',4:'-',5:'-',6:'--',7:':',8:'-.'}
    for ax,hand in zip(axes,ACCS):
        for r in [r for r in rows if r['hand_accuracy']==hand]:
            vals=np.sort(groups[(r['model'],hand,r['horizon_steps'])]['plans'])
            ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),color=MODELCOLORS[r['model']],ls=styles[r['horizon_steps']],lw=2,
                label=f'{r["model"]}: {r["horizon_steps"]} steps / {r["horizon_s"]:.3f} s')
        ax.axvline(64,color='black',ls=':',lw=1.5)
        ax.set(title=LABELS[hand]+' hand accuracy',xlabel='Planning time (ms)',xlim=(10,150))
        ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=9)
    axes[0].set_ylabel('Cumulative fraction of published plans')
    fig.suptitle('Planning-time distributions at every measured horizon',y=1.01,fontsize=14)
    fig.text(.5,.025,'Dotted vertical line: 64 ms. Completed episodes only; rare rejected/aborted plans are excluded.',ha='center',fontsize=10,color='#444444')
    savefig(fig,'horizon_effect_planning_distributions')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharex=True,sharey=True)
    for ax,model in zip(axes,['M1','M2']):
        for hand in ACCS:
            selected=sorted([r for r in rows if r['model']==model and r['hand_accuracy']==hand],key=lambda r:r['horizon_steps'])
            ax.plot([r['horizon_steps'] for r in selected],[r['plans_over64_percent'] for r in selected],'-o',color=COLORS[hand],lw=2,label=LABELS[hand])
        ax.set(title=model,xlabel='Planning horizon (control steps)',xticks=range(2,9),xlim=(1.6,8.4),ylim=(-3,105))
        ax.legend(title='Hand accuracy');ax.grid(alpha=.2)
    axes[0].set_ylabel('Published plans taking more than 64 ms (%)')
    fig.suptitle('How often planning exceeds the nominal control period',y=1.01,fontsize=14)
    fig.text(.5,.025,'Only measured points are shown; independent recorded batches with 512 samples.',ha='center',fontsize=10,color='#444444')
    savefig(fig,'horizon_effect_overruns')
    return rows


def update_index():
    index=OUT/'index.html';text=index.read_text()
    if 'Planning horizon versus measured planning time</h2>' not in text:
        section='<h2>Planning horizon versus measured planning time</h2><p>Includes supplemental recorded horizon trials beyond the primary 72 baseline episodes. Independent placements/contact states; curves show measured configurations, not a fixed-state GPU benchmark. M1 medium/low and M2 high each have only one measured horizon.</p><p><a href="horizon_timing.csv">Timing table</a> · <a href="horizon_timing_report.json">Sources and checks</a></p>'
        if 'plots/horizon_effect_planning_time.png' not in text:
            for name in ['horizon_effect_planning_time','horizon_effect_planning_distributions','horizon_effect_overruns']:
                section+=f'<p><a href="plots/{name}.pdf">PDF</a></p><img src="plots/{name}.png">'
        text=text.replace('<h2>Episodes and videos</h2>',section+'<h2>Episodes and videos</h2>')
        index.write_text(text)


if __name__=='__main__':
    rows=make_plots();update_index();print(json.dumps(rows,indent=2))
