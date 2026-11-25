"""Reproducible, offline M2 profile. Reads logs; never imports the controller."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import csv
import hashlib
import html
import json
import math
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2'
SOURCES = {
    'Unpaced': [BASE / 'trial314_shift512' / name / 'controller.json' for name in [
        'leap_experiment_20261115_173310_247065',
        'leap_experiment_20261115_173430_512823',
        'leap_experiment_20261115_173525_919954',
        'leap_experiment_20261115_195200_470971']] + [
        BASE / 'trial314_shift512_paced/leap_experiment_20261115_214159_185760/controller.json'],
    'Paced': [BASE / 'trial314_shift512_paced' / name / 'controller.json' for name in [
        'leap_experiment_20261115_210143_507664',
        'leap_experiment_20261115_212331_181021']],
}
GROUPS = list(SOURCES)
COLORS = {'Unpaced': '#2166ac', 'Paced': '#d6604d'}
DIRECTIONS = ['-X', '+X', '-Y', '+Y', '-Z', '+Z']
FACE = ['O', 'B', 'blue H', 'purple H', 'R (−yaw)', 'R (+yaw)']
data = {g: [json.loads(p.read_text()) for p in paths] for g, paths in SOURCES.items()}
reference = data['Unpaced'][0]
scene_root = ET.fromstring(reference['scenes']['rollout']['scene_xml'])
cube_mass = float(scene_root.find(".//geom[@name='obj']").get('mass'))
position_threshold = reference['success_thresholds']['pos'] * 1000
angle_threshold = math.degrees(2 * math.asin(math.sqrt(reference['success_thresholds']['quat'])))
period = reference['control_period']
episodes = {g: [] for g in GROUPS}
step_arrays = {}
intervals = {}
waits = {}
first100 = {}
manifest = []


def normalized_xml(text):
    root = ET.fromstring(text)
    for mesh in root.findall('.//mesh'):
        mesh.attrib.pop('content_type', None)
    return ET.tostring(root)


def stats(a):
    a = np.asarray(a, float).ravel()
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(mean=None, median=None, p95=None, maximum=None, minimum=None)
    return dict(mean=float(np.mean(a)), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), maximum=float(a.max()), minimum=float(a.min()))


def wilson(successes, total):
    p = successes / total
    z = 1.959963984540054
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return center-radius, center+radius


for group, runs in data.items():
    all_steps = []
    all_intervals = []
    all_waits = []
    prefix_changes = []
    for batch, (run, path) in enumerate(zip(runs, SOURCES[group]), 1):
        # Compare the logged model snapshots, never today's potentially edited files.
        for key in ['weights', 'hardware_config', 'success_thresholds', 'joint_reference',
                    'pose_transform', 'failure_z', 'task_contact_params']:
            assert run[key] == reference[key], (group, path, key)
        for scene in ['rollout', 'eval']:
            assert normalized_xml(run['scenes'][scene]['scene_xml']) == normalized_xml(reference['scenes'][scene]['scene_xml'])
        for key in ['planner_config', 'rollout_config']:
            assert run[key] == reference[key], (group, path, key)
        assert bool(run['args'].get('pace_control', False)) == (group == 'Paced')
        manifest.append(dict(group=group, file=str(path.relative_to(ROOT)),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             status=run['status'], completed=sum(bool(e.get('completed')) for e in run['episodes'])))
        for e in run['episodes']:
            if not e.get('completed') or e['finish_reason'] not in ['success', 'failure', 'timeout']:
                continue
            steps = [s for s in e['steps'] if s.get('action_published')]
            assert len(steps) == e['n_steps'] and steps
            all_steps.extend(steps)
            q = np.asarray([s['q'] for s in steps] + [e['q_end']], float)
            t = np.asarray([s['elapsed_s'] for s in steps] + [e['manipulation_wall_time_s']], float)
            xyz = (q[:, 16:19] - e['target_pos']) * 1000
            pos = np.linalg.norm(xyz, axis=1)
            angle = np.degrees([s['quaternion_error_rad'] for s in steps] + [e['quaternion_error_rad']])
            quat = q[0, 19:23] / np.linalg.norm(q[0, 19:23])
            if quat[0] < 0:
                quat = -quat
            axis = int(np.argmax(np.abs(quat[1:])))
            direction = ('+' if quat[axis+1] >= 0 else '-') + 'XYZ'[axis]
            raw = np.asarray([s['action_delta'] for s in steps])
            bounds = run['hardware_config']['ctrl_range']
            clipped = np.clip(raw, -np.asarray(bounds['lower']), np.asarray(bounds['upper']))
            hits = np.abs(raw-clipped) > 1e-6
            target = np.asarray([s['u'] for s in steps])
            change = np.abs(np.diff(target, axis=0))
            prefix_changes.extend(change[:99].ravel())
            delay = np.array([s.get('pacing_wait_s', 0) for s in steps])
            all_waits.extend(delay)
            if all(s.get('action_published_at_utc') for s in steps):
                stamps = [datetime.fromisoformat(s['action_published_at_utc']).timestamp() for s in steps]
                gaps = np.diff(stamps)
                all_intervals.extend(gaps)
            else:
                gaps = None  # Earlier unpaced logs have no exact publication timestamps.
            episodes[group].append(dict(group=group, batch=batch, source=str(path.relative_to(ROOT)),
                source_episode=e['episode']+1, episode=len(episodes[group])+1,
                outcome=e['finish_reason'], direction=direction, steps=len(steps),
                duration=e['manipulation_wall_time_s'], t=t, pos=pos, angle=angle, xyz=xyz,
                cycle_duration=e.get('cycle_wall_time_s'),
                height=q[:, 18]*1000, raw=raw, clipped=clipped, hits=hits, change=change,
                fallen_cost_z=e['fallen_z'],
                wait=delay, intervals=gaps,
                starting_position=q[0,16:19], starting_quaternion=q[0,19:23],
                starting_joint_state=q[0,:16], target_position=e['target_pos'], target_quaternion=e['goal'],
                simultaneous_pose_ever=bool(np.any((pos < position_threshold) & (angle < angle_threshold))),
                position_ever=bool(np.any(pos < position_threshold)),
                orientation_ever=bool(np.any(angle < angle_threshold))))
    step_arrays[group] = all_steps
    waits[group] = np.asarray(all_waits)
    intervals[group] = np.asarray(all_intervals)
    first100[group] = np.asarray(prefix_changes)

assert len({(e['source'],e['source_episode']) for ee in episodes.values() for e in ee}) == 48
for g in GROUPS:
    assert len(episodes[g]) == 24, (g,len(episodes[g]))
    assert Counter(e['direction'] for e in episodes[g]) == Counter({a:4 for a in DIRECTIONS})


metrics = {}
for g in GROUPS:
    ee = episodes[g]
    ss = step_arrays[g]
    flat = lambda key: np.concatenate([e[key].ravel() for e in ee])
    raw = np.concatenate([e['raw'] for e in ee])
    clipped = np.concatenate([e['clipped'] for e in ee])
    hits = np.concatenate([e['hits'] for e in ee])
    successes = [e for e in ee if e['outcome'] == 'success']
    plans = np.asarray([s['planning_time'] for s in ss])
    loop = np.asarray([s['wall_time_s'] for s in ss])
    total_s = sum(e['duration'] for e in ee)
    velocity = np.asarray([s['q_dot'] for s in ss])
    offset = np.abs([np.asarray(s['u']) - s['feedback']['command_qpos'] for s in ss])
    m = dict(episodes=len(ee), successes=len(successes), drops=sum(e['outcome']=='failure' for e in ee),
        timeouts=sum(e['outcome']=='timeout' for e in ee), success_rate=len(successes)/len(ee),
        wilson95=wilson(len(successes), len(ee)), steps=len(ss), total_manipulation_s=total_s,
        effective_hz=len(ss)/total_s, duration_s=stats([e['duration'] for e in ee]),
        cycle_duration_s=stats([e['cycle_duration'] for e in ee if e['cycle_duration'] is not None]),
        total_logged_cycle_s=sum(e['cycle_duration'] for e in ee if e['cycle_duration'] is not None),
        total_planning_s=float(plans.sum()), total_logged_step_s=float(loop.sum()),
        success_duration_s=stats([e['duration'] for e in successes]),
        drop_duration_s=stats([e['duration'] for e in ee if e['outcome']=='failure']),
        steps_per_episode=stats([e['steps'] for e in ee]),
        start_pos_mm=stats([e['pos'][0] for e in ee]), start_angle_deg=stats([e['angle'][0] for e in ee]),
        final_pos_mm=stats([e['pos'][-1] for e in ee]), final_angle_deg=stats([e['angle'][-1] for e in ee]),
        success_final_pos_mm=stats([e['pos'][-1] for e in successes]),
        success_final_angle_deg=stats([e['angle'][-1] for e in successes]),
        logged_pos_mm=stats(np.concatenate([e['pos'][:-1] for e in ee])),
        logged_angle_deg=stats(np.concatenate([e['angle'][:-1] for e in ee])),
        episode_mean_pos_mm=stats([e['pos'][:-1].mean() for e in ee]),
        episode_mean_angle_deg=stats([e['angle'][:-1].mean() for e in ee]),
        best_pos_mm=stats([e['pos'].min() for e in ee]),
        best_angle_deg=stats([e['angle'].min() for e in ee]),
        position_ever=sum(e['position_ever'] for e in ee), orientation_ever=sum(e['orientation_ever'] for e in ee),
        simultaneous_pose_ever=sum(e['simultaneous_pose_ever'] for e in ee),
        planning_ms=stats(plans*1000), loop_ms=stats(loop*1000),
        planning_under_period=int((plans<period).sum()),
        loop_under_period=int((loop<period).sum()),
        clipped_steps=int(hits.any(axis=1).sum()), clipped_step_fraction=float(hits.any(axis=1).mean()),
        clipped_joints=int(hits.sum()), clipped_joint_fraction=float(hits.mean()),
        removed_delta_fraction=float(np.abs(raw-clipped).sum()/np.abs(raw).sum()),
        requested_delta_rad=stats(np.abs(raw)), target_change_rad=stats(flat('change')),
        first100_target_change_rad=stats(first100[g]), target_offset_rad=stats(offset),
        joint_speed_rad_s=stats(np.abs(velocity[:, :16])),
        object_linear_speed_m_s=stats(np.linalg.norm(velocity[:,16:19],axis=1)),
        object_angular_speed_rad_s=stats(np.linalg.norm(velocity[:,19:22],axis=1)),
        waited_steps=int((waits[g]>0).sum()), waited_fraction=float((waits[g]>0).mean()),
        total_wait_s=float(waits[g].sum()), wait_fraction_wall=float(waits[g].sum()/total_s),
        wait_all_ms=stats(waits[g]*1000), wait_positive_ms=stats(waits[g][waits[g]>0]*1000),
        publication_interval_ms=stats(intervals[g]*1000),
        publication_interval_episodes=sum(e['intervals'] is not None for e in ee),
        publication_interval_count=len(intervals[g]),
        clipping_by_joint=hits.mean(axis=0).tolist())
    for name, key, command in [('joint_age_ms','robot_age',False), ('object_age_ms','pose_age',False),
                               ('command_joint_age_ms','robot_age',True), ('command_object_age_ms','pose_age',True)]:
        m[name]=stats([(s['feedback']['command_state'] if command else s['feedback'])[key]*1000 for s in ss])
    for j,axis in enumerate('xyz'):
        m[f'absolute_{axis}_error_mm']=stats(np.concatenate([np.abs(e['xyz'][:-1,j]) for e in ee]))
    m['by_direction']={axis: dict(total=sum(e['direction']==axis for e in ee),
        successes=sum(e['direction']==axis and e['outcome']=='success' for e in ee)) for axis in DIRECTIONS}
    metrics[g]=m

plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'savefig.dpi':180, 'pdf.fonttype':42, 'ps.fonttype':42})
figures=[]


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=180)
    fig.savefig(OUT/f'{name}.pdf')
    figures.append((name, fig))


fig, axs=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True,gridspec_kw={'width_ratios':[1,2,1.3]})
for j,g in enumerate(GROUPS):
    m=metrics[g]; lo,hi=m['wilson95']; p=m['success_rate']
    axs[0].bar(j,p,color=COLORS[g],width=.6)
    axs[0].errorbar(j,p,yerr=[[p-lo],[hi-p]],fmt='none',ecolor='black',capsize=5)
    axs[0].text(j,1.01,f"{m['successes']}/{m['episodes']}",ha='center')
axs[0].set(xticks=[0,1],xticklabels=GROUPS,ylim=(0,1.12),title='Overall success (95% Wilson intervals)')
for j,g in enumerate(GROUPS):
    n=np.array([metrics[g]['by_direction'][a]['total'] for a in DIRECTIONS])
    s=np.array([metrics[g]['by_direction'][a]['successes'] for a in DIRECTIONS])
    x=np.arange(6)+(j-.5)*.36
    axs[1].bar(x,s/n,width=.34,color=COLORS[g],label=g)
    for xx,ss,nn in zip(x,s,n):axs[1].text(xx,ss/nn+.035,f'{ss}/{nn}',ha='center',fontsize=8)
axs[1].set(xticks=np.arange(6),xticklabels=[f'{a}\n{b}' for a,b in zip(DIRECTIONS,FACE)],ylim=(0,1.15),title='Success by initial rotation / upper face')
axs[1].legend(loc='upper right',fontsize=8)
batches=[('Unpaced cycle 1','Unpaced',episodes['Unpaced'][:6]),('Unpaced cycle 2','Unpaced',episodes['Unpaced'][6:12]),
         ('Unpaced run 3','Unpaced',episodes['Unpaced'][12:]),
         ('Paced run 1','Paced',[e for e in episodes['Paced'] if e['batch']==1]),
         ('Paced run 2','Paced',[e for e in episodes['Paced'] if e['batch']==2])]
for j,(label,g,ee) in enumerate(batches):
    n=sum(e['outcome']=='success' for e in ee)
    axs[2].bar(j,n/len(ee),color=COLORS[g]);axs[2].text(j,n/len(ee)+.03,f'{n}/{len(ee)}',ha='center')
axs[2].set(xticks=range(5),xticklabels=['U1','U2','U3','P1','P2'],ylim=(0,1.12),title='Batch variation (U = unpaced, P = paced)')
for ax in axs:ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('M2 outcomes — same settings, manual approximately 90° starts')
save(fig,'success_rates')
for g in GROUPS:
    fig,axs=plt.subplots(1,2,figsize=(10,4.4),constrained_layout=True,gridspec_kw={'width_ratios':[1,2]})
    m=metrics[g];p=m['success_rate'];lo,hi=m['wilson95']
    axs[0].bar(0,p,color=COLORS[g],width=.5)
    axs[0].errorbar(0,p,yerr=[[p-lo],[hi-p]],fmt='none',ecolor='black',capsize=5)
    axs[0].set(xticks=[0],xticklabels=[g],xlim=(-.8,.8),ylim=(0,1.15),title=f"Overall: {m['successes']}/{m['episodes']} (95% CI)")
    for j,axis in enumerate(DIRECTIONS):
        d=m['by_direction'][axis];rate=d['successes']/d['total']
        axs[1].bar(j,rate,color=COLORS[g]);axs[1].text(j,rate+.035,f"{d['successes']}/{d['total']}",ha='center')
    axs[1].set(xticks=range(6),xticklabels=[f'{a}\n{f}' for a,f in zip(DIRECTIONS,FACE)],ylim=(0,1.15),title='Initial rotation / upper face')
    for ax in axs:ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'M2 {g.lower()} — {m["episodes"]} completed episodes')
    save(fig,f'success_rates_{g.lower()}')

max_time=max(e['duration'] for ee in episodes.values() for e in ee)
max_pos=max(float(e['pos'].max()) for ee in episodes.values() for e in ee)
for g in GROUPS:
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True,constrained_layout=True,gridspec_kw={'height_ratios':[2,2,.65]})
    fig.get_layout_engine().set(rect=(0,.035,1,.965))
    grid=np.arange(0,max_time+.25,.25)
    for field,ax,threshold,label in [('pos',axs[0],position_threshold,'3D position error (mm)'),
                                     ('angle',axs[1],angle_threshold,'Orientation error (degrees)')]:
        samples=[]
        for e in episodes[g]:
            color={'success':'#278048','failure':'#b33c3c','timeout':'#a96c00'}[e['outcome']]
            ax.plot(e['t'],e[field],color=color,alpha=.23,lw=.9)
            ax.plot(e['t'][-1],e[field][-1],marker={'success':'o','failure':'x','timeout':'s'}[e['outcome']],
                    color=color,ms=4,alpha=.85)
            values=np.interp(grid,e['t'],e[field]);values[(grid<e['t'][0])|(grid>e['t'][-1])]=np.nan
            samples.append(values)
        a=np.asarray(samples);counts=np.isfinite(a).sum(axis=0);valid=counts>0
        med=np.full(len(grid),np.nan);p25=med.copy();p75=med.copy()
        med[valid]=np.nanmedian(a[:,valid],axis=0)
        p25[valid]=np.nanpercentile(a[:,valid],25,axis=0);p75[valid]=np.nanpercentile(a[:,valid],75,axis=0)
        ax.fill_between(grid,p25,p75,color=COLORS[g],alpha=.15,label='Active-episode interquartile range')
        ax.plot(grid,med,color=COLORS[g],lw=2.3,label='Active-episode median')
        ax.axhline(threshold,color='black',ls='--',lw=1,label=f'Success component threshold: {threshold:.1f}')
        ax.set(ylabel=label);ax.grid(alpha=.2)
    axs[0].set(ylim=(0,math.ceil(max_pos/50)*50),title=f'{g}: {len(episodes[g])} completed episodes — full trajectories including terminal drops')
    axs[1].set(ylim=(0,185))
    axs[0].legend(loc='upper right',fontsize=8)
    axs[2].step(grid,counts,where='post',color=COLORS[g]);axs[2].set(ylabel='Episodes\nstill active',xlabel='Manipulation wall time (s)',ylim=(0,25),xlim=(0,max_time))
    axs[2].grid(alpha=.2)
    fig.text(.11,.001,'Green = success; red = drop; ochre = timeout. Curves stop at termination; later medians represent survivors only.',fontsize=8)
    save(fig,f'errors_{g.lower()}')

fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for j,g in enumerate(GROUPS):
    m=metrics[g];p=m['waited_fraction']
    axs[0,0].bar(j,1-p,color='#c5ced3',label='No inserted wait' if j==0 else None)
    axs[0,0].bar(j,p,bottom=1-p,color=COLORS['Paced'],label='Inserted wait' if j==0 else None)
    axs[0,0].text(j,.5,f"{100*(1-p):.2f}%\nno wait",ha='center')
    axs[0,0].text(j,1.025,f"{m['waited_steps']}/{m['steps']} waited",ha='center',fontsize=9)
axs[0,0].set(xticks=[0,1],xticklabels=GROUPS,ylim=(0,1.15),title='Pacing intervention per published action')
axs[0,0].yaxis.set_major_formatter(PercentFormatter(1));axs[0,0].legend(loc='lower left',fontsize=8)
positive=waits['Paced'][waits['Paced']>0]*1000
axs[0,1].hist(positive,bins=np.arange(0,31,2),color=COLORS['Paced'],edgecolor='white')
axs[0,1].set(xlabel='Inserted wait (ms)',ylabel='Steps',title=f'Only the {len(positive)} steps that waited')
for batch in [1,2]:
    ee=[e for e in episodes['Paced'] if e['batch']==batch]
    axs[1,0].bar([e['episode'] for e in ee],[e['wait'].sum()*1000 for e in ee],
                 color=COLORS['Paced'] if batch==1 else '#efab9d',label=f'Paced run {batch}')
axs[1,0].set(xlabel='Paced episode (both runs concatenated)',ylabel='Total inserted wait (ms)',title='Wait time by episode');axs[1,0].legend(fontsize=8)
elapsed=np.cumsum([e['duration'] for e in episodes['Paced']]);cwait=np.cumsum([e['wait'].sum() for e in episodes['Paced']])
axs[1,1].plot(np.r_[0,elapsed],np.r_[0,cwait],color=COLORS['Paced'],lw=2,marker='o',ms=3)
axs[1,1].set(xlabel='Cumulative manipulation time (s)',ylabel='Cumulative inserted wait (s)',ylim=(0,max(1,float(cwait[-1])*1.2)),
             title=f'Total wait: {cwait[-1]:.3f} s / {elapsed[-1]:.1f} s ({100*cwait[-1]/elapsed[-1]:.3f}%)')
for ax in axs.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('The 64 ms pacing limiter rarely inserts a delay')
save(fig,'pacing_intervention')

fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
for g in GROUPS:
    p=np.asarray([s['planning_time']*1000 for s in step_arrays[g]])
    x=np.sort(p);axs[0].plot(x,np.arange(1,len(x)+1)/len(x),color=COLORS[g],label=g)
    x=np.sort([s['wall_time_s']*1000 for s in step_arrays[g]])
    axs[1].plot(x,np.arange(1,len(x)+1)/len(x),color=COLORS[g],label=g)
for g in GROUPS:
    x=np.sort(intervals[g]*1000)
    axs[2].plot(x,np.arange(1,len(x)+1)/len(x),color=COLORS[g],
                label=f'{g}: {metrics[g]["publication_interval_episodes"]}/24 episodes')
for ax,title,xlabel in zip(axs,['Planning duration','Logged controller-step duration','Exact publication intervals\n(partial unpaced coverage)'],
                          ['Planning time (ms)','Step wall time (ms)','Publication interval (ms)']):
    ax.axvline(period*1000,color='black',ls='--',lw=1,label='64 ms period')
    ax.set(xlabel=xlabel,ylabel='Cumulative fraction of steps',title=title,xlim=(0,110),ylim=(0,1.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
axs[1].set_xlim(0,max(110,math.ceil(max(m['loop_ms']['maximum'] for m in metrics.values())/10)*10))
axs[2].set_xlim(0,max(110,math.ceil(max(m['publication_interval_ms']['maximum'] for m in metrics.values())/10)*10))
fig.suptitle('Most M2 plans already take longer than the nominal control period')
save(fig,'timing_distributions')

# Starting-orientation outcome distributions. Use the first state actually
# supplied to MPPI; labels describe initial offsets, not corrective rotations.
OUTCOME_COLORS={'success':'#278048','failure':'#b33c3c','timeout':'#a96c00'}
OUTCOME_LABELS={'success':'Success','failure':'Drop','timeout':'Timeout'}
OUTCOME_MARKERS={'success':'o','failure':'X','timeout':'s'}
orientation_rows=[]
orientation_episode_rows=[]
for g in GROUPS:
    for e in episodes[g]:
        q=np.asarray(e['starting_quaternion'],float)
        goal=np.asarray(e['target_quaternion'],float)
        q=q/np.linalg.norm(q);goal=goal/np.linalg.norm(goal)
        # q_relative = q_initial * conjugate(q_goal), wxyz convention.
        rel=np.r_[q[0]*goal[0]+np.dot(q[1:],goal[1:]),
                  -q[0]*goal[1:]+goal[0]*q[1:]-np.cross(q[1:],goal[1:])]
        if rel[0]<0:rel=-rel
        theta=2*np.arctan2(np.linalg.norm(rel[1:]),rel[0])
        rotvec=np.degrees(theta)*rel[1:]/max(np.linalg.norm(rel[1:]),1e-12)
        assert np.isclose(np.degrees(theta),e['angle'][0],atol=.001)
        main=int(np.argmax(np.abs(rotvec)))
        assert ('+' if rotvec[main]>=0 else '-')+'XYZ'[main] == e['direction']
        orientation_episode_rows.append(dict(group=g,episode=e['episode'],source=e['source'],
            source_episode=e['source_episode'],initial_rotation=e['direction'],outcome=e['outcome'],
            initial_angle_deg=float(np.degrees(theta)),
            rotation_vector_x_deg=float(rotvec[0]),rotation_vector_y_deg=float(rotvec[1]),rotation_vector_z_deg=float(rotvec[2]),
            initial_quaternion_wxyz=json.dumps(e['starting_quaternion'].tolist()),
            goal_quaternion_wxyz=json.dumps(e['target_quaternion'])))
    for axis,face in zip(DIRECTIONS,FACE):
        ee=[e for e in episodes[g] if e['direction']==axis]
        count=Counter(e['outcome'] for e in ee)
        lo,hi=wilson(count['success'],len(ee))
        orientation_rows.append(dict(group=g,initial_rotation=axis,upper_face=face,
            attempts=len(ee),successes=count['success'],drops=count['failure'],timeouts=count['timeout'],
            success_rate=count['success']/len(ee),success_wilson95_low=lo,success_wilson95_high=hi,
            initial_angle_mean_deg=float(np.mean([e['angle'][0] for e in ee])),
            initial_angle_min_deg=float(min(e['angle'][0] for e in ee)),
            initial_angle_max_deg=float(max(e['angle'][0] for e in ee))))
    fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
    bottom=np.zeros(6)
    for outcome in ['success','failure','timeout']:
        counts=np.array([sum(e['direction']==axis and e['outcome']==outcome for e in episodes[g]) for axis in DIRECTIONS])
        ax.bar(range(6),counts,bottom=bottom,color=OUTCOME_COLORS[outcome],width=.65,label=OUTCOME_LABELS[outcome],edgecolor='white')
        for j,(n,b) in enumerate(zip(counts,bottom)):
            if n:ax.text(j,b+n/2,str(n),ha='center',va='center',color='white',weight='bold',fontsize=12)
        bottom+=counts
    for j,axis in enumerate(DIRECTIONS):
        n=sum(e['direction']==axis and e['outcome']=='success' for e in episodes[g])
        ax.text(j,4.13,f'{n}/4 successful',ha='center',fontsize=9)
    ax.set(xticks=range(6),xticklabels=[f'{a}\n{f}' for a,f in zip(DIRECTIONS,FACE)],
           yticks=range(5),ylim=(0,5),xlabel='Initial rotation offset from goal / upper face',
           ylabel='Completed episodes',title=f'M2 {g.lower()}: outcomes by starting orientation (four attempts each)')
    ax.legend(loc='upper right',ncol=3,fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,f'starting_orientation_outcomes_{g.lower()}')

    fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
    for j,axis in enumerate(DIRECTIONS):
        ee=[e for e in episodes[g] if e['direction']==axis]
        for shift,e in zip(np.linspace(-.21,.21,len(ee)),ee):
            ax.scatter(j+shift,e['angle'][0],s=85,color=OUTCOME_COLORS[e['outcome']],
                       marker=OUTCOME_MARKERS[e['outcome']],edgecolor='white',linewidth=.6,zorder=3)
            ax.annotate(f'E{e["episode"]}',(j+shift,e['angle'][0]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
    ax.axhline(90,color='black',ls='--',lw=1,label='Nominal 90° start')
    all_angles=[e['angle'][0] for ee in episodes.values() for e in ee]
    ax.set(xticks=range(6),xticklabels=[f'{a}\n{f}' for a,f in zip(DIRECTIONS,FACE)],
           ylim=(math.floor(min(all_angles))-2,math.ceil(max(all_angles))+3),
           xlabel='Initial rotation offset from goal / upper face (horizontal offsets separate points)',
           ylabel='Measured initial orientation error (degrees)',
           title=f'M2 {g.lower()}: measured starting angle and episode outcome')
    handles=[Line2D([],[],ls='none',marker=OUTCOME_MARKERS[o],color=OUTCOME_COLORS[o],label=OUTCOME_LABELS[o],markersize=8)
             for o in ['success','failure','timeout']]
    handles.append(Line2D([],[],ls='--',color='black',label='Nominal 90° start'))
    ax.legend(handles=handles,loc='upper left',ncol=4,fontsize=9)
    ax.grid(alpha=.2);ax.set_axisbelow(True)
    save(fig,f'starting_orientation_measured_{g.lower()}')

fig,ax=plt.subplots(figsize=(10,3.7),constrained_layout=True)
matrix=np.array([[metrics[g]['by_direction'][a]['successes']/metrics[g]['by_direction'][a]['total'] for a in DIRECTIONS] for g in GROUPS])
ax.imshow(matrix,cmap='YlGn',vmin=0,vmax=1,aspect='auto')
for i,g in enumerate(GROUPS):
    for j,a in enumerate(DIRECTIONS):
        n=metrics[g]['by_direction'][a]['successes']
        ax.text(j,i,f'{n}/4\n{100*matrix[i,j]:.0f}%',ha='center',va='center',weight='bold',fontsize=12,
                color='white' if matrix[i,j]>=.65 else '#192b38')
ax.set(xticks=range(6),xticklabels=[f'{a}\n{f}' for a,f in zip(DIRECTIONS,FACE)],yticks=[0,1],yticklabels=GROUPS,
       xlabel='Initial rotation offset from goal / upper face',title='M2 success distribution by starting orientation (four attempts per cell)')
save(fig,'starting_orientation_success_map')
for filename,records in [('starting_orientation_outcomes.csv',orientation_rows),('starting_orientation_episodes.csv',orientation_episode_rows)]:
    with (OUT/filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

# Export each scientific plot directly from Matplotlib as a standalone image.
# No screenshot cropping or modification of rendered bitmap files is involved.
individual_dir=OUT/'individual_graphs'
individual_dir.mkdir(exist_ok=True)
panel_specs={
    'success_rates': [
        ('success_overall','M2 success rate: unpaced vs paced (24 episodes each)'),
        ('success_by_direction','M2 success by initial rotation / upper face'),
        ('success_by_batch','M2 success by batch (U = unpaced, P = paced)')],
    'success_rates_unpaced': [
        ('success_overall_unpaced','M2 unpaced success rate: 24 episodes (95% Wilson interval)'),
        ('success_by_direction_unpaced','M2 unpaced success by initial rotation / upper face')],
    'success_rates_paced': [
        ('success_overall_paced','M2 paced success rate: 24 episodes (95% Wilson interval)'),
        ('success_by_direction_paced','M2 paced success by initial rotation / upper face')],
    'errors_unpaced': [
        ('position_error_unpaced','M2 unpaced: 3D position error (24 episodes)'),
        ('orientation_error_unpaced','M2 unpaced: orientation error (24 episodes)'),
        ('active_episodes_unpaced','M2 unpaced: number of episodes still active')],
    'errors_paced': [
        ('position_error_paced','M2 paced: 3D position error (24 episodes)'),
        ('orientation_error_paced','M2 paced: orientation error (24 episodes)'),
        ('active_episodes_paced','M2 paced: number of episodes still active')],
    'pacing_intervention': [
        ('pacing_wait_fraction','Pacing intervention per published action'),
        ('pacing_positive_wait_distribution','Inserted wait duration: only the 100 actions that waited'),
        ('pacing_wait_by_episode','Total inserted wait per paced episode'),
        ('pacing_cumulative_wait','Cumulative inserted wait: 0.557 s over 498.3 s of manipulation')],
    'timing_distributions': [
        ('planning_duration','M2 planning duration: unpaced vs paced'),
        ('controller_step_duration','M2 logged controller-step duration: unpaced vs paced'),
        ('publication_interval','M2 publication intervals\n(unpaced: newest 12 episodes; paced: all 24)')],
    'starting_orientation_outcomes_unpaced': [('starting_orientation_outcomes_unpaced','M2 unpaced: outcomes by starting orientation (four attempts each)')],
    'starting_orientation_outcomes_paced': [('starting_orientation_outcomes_paced','M2 paced: outcomes by starting orientation (four attempts each)')],
    'starting_orientation_measured_unpaced': [('starting_orientation_measured_unpaced','M2 unpaced: measured starting angle and episode outcome')],
    'starting_orientation_measured_paced': [('starting_orientation_measured_paced','M2 paced: measured starting angle and episode outcome')],
    'starting_orientation_success_map': [('starting_orientation_success_map','M2 success distribution by starting orientation (four attempts per cell)')],
}
individual_plots=[]
for figure_name,figure in figures:
    engine=figure.get_layout_engine()
    figure.set_layout_engine(None)
    for ax,(name,title) in zip(figure.axes,panel_specs[figure_name]):
        axis_visibility=[a.get_visible() for a in figure.axes]
        text_visibility=[t.get_visible() for t in figure.texts]
        old_title=ax.get_title();old_xlabel=ax.get_xlabel()
        temporary_legend=None
        for a in figure.axes:a.set_visible(a is ax)
        for t in figure.texts:t.set_visible(False)
        ax.set_title(title)
        if name.startswith(('position_error_','orientation_error_')):
            ax.set_xlabel('Manipulation wall time (s)')
            if ax.get_legend() is None:
                temporary_legend=ax.legend(loc='upper right',fontsize=8)
        figure.canvas.draw()
        box=ax.get_tightbbox(figure.canvas.get_renderer()).transformed(figure.dpi_scale_trans.inverted()).padded(.12)
        for ext in ['png','pdf']:
            figure.savefig(individual_dir/f'{name}.{ext}',bbox_inches=box,dpi=200)
        individual_plots.append(dict(name=name,title=title.replace('\n',' '),source_figure=figure_name,
                                     png=f'{name}.png',pdf=f'{name}.pdf'))
        ax.set_title(old_title);ax.set_xlabel(old_xlabel)
        if temporary_legend is not None:temporary_legend.remove()
        for a,visible in zip(figure.axes,axis_visibility):a.set_visible(visible)
        for t,visible in zip(figure.texts,text_visibility):t.set_visible(visible)
    figure.set_layout_engine(engine)
assert len(individual_plots)==sum(len(spec) for spec in panel_specs.values())==25
(individual_dir/'manifest.json').write_text(json.dumps(individual_plots,indent=2)+'\n')
gallery=['<!doctype html><html><head><meta charset="utf-8"><title>M2 individual graphs</title>',
         '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}img{max-width:100%}section{margin:40px 0}li{margin:8px 0}</style></head><body>',
         '<h1>M2 individual graphs: 24 episodes per mode</h1>',
         '<p>Each image contains one graph. Click an image for its vector PDF. '
         'Position/orientation curves end at termination, including terminal drops. '
         'Green = success; red = drop; ochre = timeout. Thick curves show active-episode medians; '
         'shading shows the interquartile range. Later medians represent fewer surviving episodes; see the active-count graphs.</p>',
         '<p>Pacing inserted a wait on 100/6,711 actions (1.49%), totalling 0.557 s over 498.3 s (0.112%). '
         'Wait duration statistics exclude zero-wait actions only in the positive-wait histogram. '
         'The cumulative-wait plot is sampled at episode boundaries; its dots mark episode ends. '
         'Publication timestamps cover only the newest 12 unpaced episodes and all 24 paced episodes.</p>',
         '<p>Starting orientation is classified from the measured initial quaternion relative to the goal. '
         'Signed axes describe the initial offset, not the direction of the correction. Both yaw signs have the R face up. '
         'Measured-angle charts show one point per episode; horizontal offsets only separate points. '
         'Each starting-orientation group has four attempts per mode.</p>',
         '<p><a href="../M2_profile.pdf">Full report</a> · <a href="../metrics.csv">All metrics</a></p><ul>']
gallery.extend(f'<li><a href="#{p["name"]}">{html.escape(p["title"])}</a> · <a href="{p["png"]}">PNG</a></li>' for p in individual_plots)
gallery.append('</ul>')
for p in individual_plots:
    gallery.append(f'<section id="{p["name"]}"><h2>{html.escape(p["title"])}</h2><a href="{p["pdf"]}"><img src="{p["png"]}" alt="{html.escape(p["title"])}"></a></section>')
gallery.append('</body></html>')
(individual_dir/'index.html').write_text('\n'.join(gallery))
with zipfile.ZipFile(OUT/'individual_graphs.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in individual_plots:
        for ext in ['png','pdf']:
            path=individual_dir/f'{p["name"]}.{ext}'
            archive.write(path,path.relative_to(OUT))
    for name in ['index.html','manifest.json']:archive.write(individual_dir/name,Path('individual_graphs')/name)

rows=[]
def row(section,name,unpaced,paced):
    rows.append(dict(section=section,metric=name,Unpaced=str(unpaced),Paced=str(paced)))
def metric(section,name,key,fmt='.3f',scale=1):
    values=[]
    for g in GROUPS:
        v=metrics[g][key]
        values.append(format(v*scale,fmt))
    row(section,name,*values)
def stat_rows(section,label,key,unit,which=('mean','median','p95','maximum')):
    for stat in which:
        row(section,f'{label}: {stat} ({unit})',*[f'{metrics[g][key][stat]:.3f}' if metrics[g][key][stat] is not None else 'Not recorded' for g in GROUPS])
def constant(section,label,value):row(section,label,value,value)
def val(v):return json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,bool)) or v is None else str(v)

for key,label in [('episodes','Completed episodes'),('successes','Successes'),('drops','Drops'),('timeouts','Timeouts'),('steps','Published actions')]:
    metric('Outcomes',label,key,fmt='.0f')
metric('Outcomes','Success rate (%)','success_rate',scale=100,fmt='.2f')
row('Outcomes','Success rate: 95% Wilson interval (%)',*[f"{100*metrics[g]['wilson95'][0]:.1f}–{100*metrics[g]['wilson95'][1]:.1f}" for g in GROUPS])
for a,f in zip(DIRECTIONS,FACE):
    row('Outcomes',f'Successes / attempts: {a}, {f}',*[f"{metrics[g]['by_direction'][a]['successes']}/{metrics[g]['by_direction'][a]['total']}" for g in GROUPS])
for k,label in [('position_ever','Position threshold ever reached'),('orientation_ever','Orientation threshold ever reached'),('simultaneous_pose_ever','Both pose thresholds reached simultaneously')]:
    metric('Outcomes',label,k,fmt='.0f')
metric('Duration','Total manipulation time (s)','total_manipulation_s')
metric('Duration','Total time inside planning calls (s)','total_planning_s')
metric('Duration','Total logged controller-step time (s)','total_logged_step_s')
metric('Duration','Total recorded episode cycle time, including operator waits/reset (s)','total_logged_cycle_s')
for key,label,unit in [('duration_s','Episode duration','s'),('success_duration_s','Successful episode duration','s'),
                        ('drop_duration_s','Drop episode duration','s'),('cycle_duration_s','Recorded episode cycle including waits/reset','s'),('steps_per_episode','Actions per episode','steps')]:
    stat_rows('Duration',label,key,unit)
for key,label,unit in [('start_pos_mm','Initial 3D position error','mm'),('start_angle_deg','Initial orientation error','deg'),
    ('final_pos_mm','Terminal position error (all outcomes)','mm'),('final_angle_deg','Terminal orientation error (all outcomes)','deg'),
    ('success_final_pos_mm','Terminal position error (successes only)','mm'),('success_final_angle_deg','Terminal orientation error (successes only)','deg'),
    ('logged_pos_mm','Position error across published-step states','mm'),('logged_angle_deg','Orientation error across published-step states','deg'),
    ('episode_mean_pos_mm','Per-episode mean position error','mm'),('episode_mean_angle_deg','Per-episode mean orientation error','deg'),
    ('best_pos_mm','Best position error per episode','mm'),('best_angle_deg','Best orientation error per episode','deg')]:
    stat_rows('Pose errors',label,key,unit,('mean','median','p95'))
for axis in 'xyz':
    stat_rows('Pose errors',f'Absolute {axis} position error across published-step states',f'absolute_{axis}_error_mm','mm',('mean','p95','maximum'))
metric('Timing','Effective frequency: total actions / manipulation time (Hz)','effective_hz')
for key,label in [('planning_ms','Planning duration'),('loop_ms','Logged controller-step wall time'),
                   ('joint_age_ms','Joint feedback age at planning'),('object_age_ms','Object feedback age at planning'),
                   ('command_joint_age_ms','Joint feedback age at command'),('command_object_age_ms','Object feedback age at command')]:
    stat_rows('Timing',label,key,'ms')
for key,label in [('planning_under_period','Plans shorter than 64 ms'),('loop_under_period','Logged steps shorter than 64 ms')]:
    row('Timing',label,*[f"{metrics[g][key]}/{metrics[g]['steps']} ({100*metrics[g][key]/metrics[g]['steps']:.2f}%)" for g in GROUPS])
for stat in ['minimum','median','p95','maximum']:
    row('Pacing',f'Exact publication interval: {stat} (ms)',*[f"{metrics[g]['publication_interval_ms'][stat]:.3f}" for g in GROUPS])
row('Pacing','Exact publication timing: episodes with timestamps',*[f"{metrics[g]['publication_interval_episodes']}/{metrics[g]['episodes']}" for g in GROUPS])
metric('Pacing','Exact within-episode publication intervals recorded','publication_interval_count',fmt='.0f')
metric('Pacing','Steps with inserted wait','waited_steps',fmt='.0f')
metric('Pacing','Steps with inserted wait (%)','waited_fraction',scale=100)
metric('Pacing','Total inserted wait (s)','total_wait_s',fmt='.6f')
metric('Pacing','Inserted wait / manipulation time (%)','wait_fraction_wall',scale=100,fmt='.4f')
stat_rows('Pacing','Inserted wait across all actions','wait_all_ms','ms')
stat_rows('Pacing','Inserted wait on actions that waited','wait_positive_ms','ms')
for key,label in [('clipped_steps','Actions with any joint delta clipped'),('clipped_joints','Clipped joint-action entries')]:
    metric('Commands / clipping',label,key,fmt='.0f')
for key,label in [('clipped_step_fraction','Actions with any joint delta clipped (%)'),('clipped_joint_fraction','Joint-action entries clipped (%)'),('removed_delta_fraction','Absolute requested delta magnitude removed (%)')]:
    metric('Commands / clipping',label,key,scale=100)
for key,label in [('requested_delta_rad','Absolute requested action delta'),('target_offset_rad','Absolute target offset from fresh measured joints'),
                   ('target_change_rad','Absolute consecutive target change'),('first100_target_change_rad','Absolute target change in first 100 actions per episode')]:
    stat_rows('Commands / clipping',label,key,'rad')
for j,name in enumerate(reference['joint_order']):
    row('Commands / clipping',f'Joint {name}: clipped joint-actions (%)',*[f"{100*metrics[g]['clipping_by_joint'][j]:.3f}" for g in GROUPS])
for key,label,unit in [('joint_speed_rad_s','Absolute measured joint speed','rad/s'),
                     ('object_linear_speed_m_s','Estimated object linear speed','m/s'),
                     ('object_angular_speed_rad_s','Estimated object angular speed','rad/s')]:
    stat_rows('Measured motion',label,key,unit,('mean','p95','maximum'))
for key in ['task','rollout_model','hand_acc','obj_acc','n_samples','n_iterations','steps','ctrl_time_step','time_horizon','timestep',
            'eval_steps_per_rollout_step','control_mode','delta','warm_start','noise_sigma','temperature','action_rate_weight',
            'max_robot_age','max_pose_age','max_state_skew','max_plan_seconds','stop_on_success',
            'cube_friction','fingertip_friction','palm_friction','finger_friction']:
    constant('Run settings',key,val(reference['args'].get(key)))
row('Run settings','pace_control','false','true')
constant('Run settings','Nominal control frequency (Hz)',1/period)
constant('Run settings','Horizon control actions',round(reference['args']['time_horizon']/period))
constant('Run settings','Simulation steps per action',round(period/reference['args']['timestep']))
constant('Run settings','Cube mass (kg)',cube_mass)
constant('Run settings','Calibrated hand',reference['args']['calibrated_hand'])
constant('Run settings','Goal position (m)',val(episodes['Unpaced'][0]['target_position']))
constant('Run settings','Goal quaternion (wxyz)',val(episodes['Unpaced'][0]['target_quaternion']))
constant('Run settings','Joint-cost reference (rad)',val(reference['joint_reference']))
for key in ['delta_reference','ctrl_range','robot_qref','episode_preparation','max_current_raw','max_temperature','palm_quat_wxyz']:
    constant('Hardware settings',key,val(reference['hardware_config'][key]))
for key,value in reference['success_thresholds'].items():constant('Termination',f'Success threshold: {key}',value)
constant('Termination','Orientation threshold converted to degrees',f'{angle_threshold:.6f}')
constant('Termination','Failure height z (m)',reference['failure_z'])
constant('Termination','Fallen-cost height z (m)',episodes['Unpaced'][0]['fallen_cost_z'])
constant('Termination','Consecutive success hold requirement','None')
for key,value in reference['weights'].items():constant('Cost weights',key,value)
for section,key in [('Additional planner settings','planner_config'),('Rollout solver settings','rollout_config')]:
    for name,value in reference[key].items():constant(section,name,val(value))

with (OUT/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['section','metric','Unpaced','Paced']);w.writeheader();w.writerows(rows)
episode_rows=[]
for g in GROUPS:
    for e in episodes[g]:
        episode_rows.append(dict(group=g,batch=e['batch'],source=e['source'],source_episode=e['source_episode'],episode=e['episode'],
            outcome=e['outcome'],initial_rotation=e['direction'],steps=e['steps'],duration_s=e['duration'],
            initial_pos_error_mm=e['pos'][0],initial_orientation_error_deg=e['angle'][0],
            terminal_pos_error_mm=e['pos'][-1],terminal_orientation_error_deg=e['angle'][-1],
            clipped_step_fraction=e['hits'].any(axis=1).mean(),total_wait_s=e['wait'].sum(),
            initial_position_m=json.dumps(e['starting_position'].tolist()),initial_quaternion_wxyz=json.dumps(e['starting_quaternion'].tolist())))
with (OUT/'episodes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(episode_rows[0]));w.writeheader();w.writerows(episode_rows)
snapshot={g:[{k:v for k,v in run.items() if k!='episodes'} for run in runs] for g,runs in data.items()}
(OUT/'configuration_snapshots.json').write_text(json.dumps(snapshot,indent=2)+'\n')
notes=[
    'Seven explicitly listed source sessions contribute 24 unpaced and 24 paced completed episodes. The newest unpaced 12-episode batch is combined with the original 12. No duplicate configuration aggregates are counted.',
    'Both modes have four attempts per signed rotation. Overall success rates therefore have equal direction weighting. Batch and manual-placement effects remain uncontrolled.',
    'The separate session leap_experiment_20261115_213958_612763 contains one completed drop before the operator interrupted preparation for episode 2. That session is outside this fixed 24-per-mode cohort; its original log is preserved.',
    'Position error is full Euclidean distance to the logged goal in millimetres. Orientation error is the sign-invariant quaternion angular distance in degrees. Success also requires the logged velocity criterion.',
    'All plotted error trajectories include the terminal state, including drops. Curves are not continued after termination; active counts expose survivor bias in the later medians.',
    'Step statistics are pooled across published actions and therefore weight longer episodes more heavily. Per-episode mean errors and first-100-action changes are also included. Successful short episodes contribute fewer than 100 actions.',
    'A completed failure means a task drop, not a controller crash. Known final bridge watchdog messages followed zeroing and verified torque disable; all retained manipulations completed.',
    'Clipping is inferred from logged controller deltas and ±0.2 bounds, not independently measured at the motor. Consecutive target changes can exceed ±0.2 because that bound is relative to fresh measured positions.',
    'Exact publication timestamps cover all 24 paced episodes and only the newest 12 unpaced episodes. Publication-interval comparisons use those subsets; logged controller-step wall times cover all episodes and are not exact publication intervals.',
    'Inserted-wait totals measure deliberate sleep only. They do not measure the total causal runtime overhead of the pacing implementation. Faster/slower planning varies between sessions.',
    'The settings and physical scene XML snapshots are verified identical across all sessions, allowing only mesh content_type metadata differences. The pacing switch and administrative paths/counts differ.',
    'These observational trials show no benefit from pacing for this M2 configuration. They do not establish that the small inserted delays caused the outcome difference.',
]
report=dict(groups=metrics,sources=manifest,notes=notes,metrics_table=rows,batches=[dict(label=l,group=g,episodes=len(ee),successes=sum(e['outcome']=='success' for e in ee)) for l,g,ee in batches])
report['starting_orientation_distribution']=orientation_rows
(OUT/'profile.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

with PdfPages(OUT/'M2_profile.pdf') as pdf:
    fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.92,'M2 hardware profile: paced vs unpaced',fontsize=22,weight='bold')
    text='\n'.join([f"{g}: {metrics[g]['successes']}/{metrics[g]['episodes']} successes; {metrics[g]['drops']} drops; {metrics[g]['timeouts']} timeouts; {metrics[g]['effective_hz']:.2f} Hz" for g in GROUPS])
    fig.text(.06,.79,text,fontsize=13,linespacing=1.7)
    m=metrics['Paced']
    fig.text(.06,.63,f"Pacing inserted a wait on {m['waited_steps']}/{m['steps']} actions ({100*m['waited_fraction']:.2f}%).\n"
             f"Total inserted wait: {m['total_wait_s']:.3f} s across {m['total_manipulation_s']:.1f} s of manipulation ({100*m['wait_fraction_wall']:.3f}%).\n\n"
             "512 samples | shift | 0.32 s horizon | 0.064 s control period\n"
             "Temperature 0.1 | noise 0.05 | hardware bounds ±0.2 rad\n"
             "Calibrated high/high hand | 70 g cube | trial 314 weights\n\n"
             "Both modes contain 24 completed episodes, four attempts per signed rotation.\n"
             "Manual placements and run order remain uncontrolled. No causal effect of pacing is established.",
             fontsize=12,va='top',linespacing=1.7)
    pdf.savefig(fig);plt.close(fig)
    import textwrap
    for offset in range(0,len(notes),6):
        fig=plt.figure(figsize=(11.7,8.3))
        fig.text(.06,.94,'Metric definitions and interpretation',fontsize=18,weight='bold')
        y=.86
        for note in notes[offset:offset+6]:
            wrapped=textwrap.fill(note,width=118)
            fig.text(.06,y,wrapped,fontsize=10,va='top');y-=.029*(wrapped.count('\n')+1)+.026
        pdf.savefig(fig);plt.close(fig)
    for name,fig in figures:pdf.savefig(fig)
    # Landscape table pages, three columns = metric label + two comparison columns.
    for offset in range(0,len(rows),24):
        chunk=rows[offset:offset+24]
        fig,ax=plt.subplots(figsize=(16.5,11.7));ax.axis('off')
        cells=[]
        for r in chunk:
            cells.append([textwrap.fill(r['section']+' — '+r['metric'],60),
                          textwrap.fill(r['Unpaced'],64),textwrap.fill(r['Paced'],64)])
        table=ax.table(cellText=cells,colLabels=['Metric','Unpaced (24 episodes)','Paced (24 episodes)'],
                       colWidths=[.34,.33,.33],cellLoc='left',loc='upper center',bbox=[0,0,1,.96])
        table.auto_set_font_size(False);table.set_fontsize(7)
        weights=[2]+[max(c.count('\n')+1 for c in record)+.65 for record in cells]
        for i,weight in enumerate(weights):
            for j in range(3):table[(i,j)].set_height(.96*weight/sum(weights))
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#dddddd');cell.set_linewidth(.4)
            if i==0:cell.set_facecolor('#e4ebf1');cell.set_text_props(weight='bold')
            elif i%2==0:cell.set_facecolor('#f5f5f5')
        ax.set_title(f'M2 full metrics and settings — rows {offset+1}–{offset+len(chunk)} / {len(rows)}',loc='left',fontsize=14)
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)

parts=['<!doctype html><html><head><meta charset="utf-8"><title>M2 hardware profile</title>',
       '<style>body{font:15px system-ui;max-width:1300px;margin:32px auto;padding:0 20px;color:#192b38}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{position:sticky;top:0;background:#e4ebf1}tr:nth-child(even){background:#f6f7f8}.section{background:#dce5ec!important;font-weight:bold}li{margin:8px 0}</style></head><body>',
       '<h1>M2 hardware profile: paced vs unpaced</h1>',
       f'<p>Unpaced: <b>{metrics["Unpaced"]["successes"]}/24 successes ({100*metrics["Unpaced"]["success_rate"]:.1f}%)</b>. '
       f'Paced: <b>{metrics["Paced"]["successes"]}/24 successes ({100*metrics["Paced"]["success_rate"]:.1f}%)</b>. '
       'Settings: 512 samples, shift warm start, 0.32 s horizon, 0.064 s control period, '
       'temperature 0.1, noise 0.05, ±0.2 rad hardware bounds. Paced data include both 12-episode runs.</p>',
       '<p><a href="M2_profile.pdf">Complete PDF</a> · <a href="metrics.csv">Full metrics CSV</a> · '
       '<a href="episodes.csv">Episode CSV</a> · <a href="profile.json">Machine-readable profile</a> · '
       '<a href="configuration_snapshots.json">Full logged configuration snapshots</a></p>',
       f'<p><a href="individual_graphs/index.html">All {len(individual_plots)} graphs as separate images</a> · '
       '<a href="individual_graphs.zip">Download individual PNGs and PDFs</a></p>']
for name,_ in figures:parts.append(f'<h2>{html.escape(name.replace("_"," ").title())}</h2><a href="{name}.pdf"><img src="{name}.png"></a>')
parts.append('<h2>Interpretation and metric definitions</h2><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in notes)+'</ul>')
parts.append('<h2>Complete comparison table</h2><table><thead><tr><th>Metric</th><th>Unpaced — 24 episodes</th><th>Paced — 24 episodes</th></tr></thead><tbody>')
last=None
for r in rows:
    if r['section']!=last:parts.append('<tr class="section"><td colspan="3">'+html.escape(r['section'])+'</td></tr>');last=r['section']
    parts.append('<tr>'+''.join('<td>'+html.escape(r[k])+'</td>' for k in ['metric','Unpaced','Paced'])+'</tr>')
parts.append('</tbody></table><h2>Source logs</h2><ul>')
for s in manifest:parts.append('<li>'+html.escape(f"{s['group']}: {s['file']}")+'</li>')
parts.append('</ul></body></html>')
(OUT/'index.html').write_text('\n'.join(parts))
for _,fig in figures:plt.close(fig)
print(json.dumps(dict(output=str(OUT),metric_rows=len(rows),groups={g:{k:v for k,v in m.items() if k in ['episodes','successes','drops','timeouts','steps','effective_hz','waited_steps','waited_fraction','total_wait_s','wait_fraction_wall']} for g,m in metrics.items()}),indent=2))
