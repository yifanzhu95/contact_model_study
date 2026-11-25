"""Build separate publication-quality figures and an HTML report from offline exports."""
from pathlib import Path
import csv
import html
import json
import os
import zipfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

R = json.loads((HERE / 'report.json').read_text())
A = np.load(HERE / 'plot_data.npz')
FIG = HERE / 'individual_graphs'
FIG.mkdir(exist_ok=True)
COL = dict(current='#147d92', no_rate='#747986', success='#25945c', failure='#d15d52', timeout='#d89a28')
LABEL = dict(current='Retuned weights + rate 15 (24 episodes)', no_rate='Earlier weights + rate 0 (12 episodes)')
AT = np.degrees(2*np.arcsin(np.sqrt(.04)))
GROUPS = ['no_rate', 'current']
SHORT = ['Earlier + rate 0\n12 episodes', 'Retuned + rate 15\n24 episodes']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': .18, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42})

def readcsv(name):
    with (HERE/name).open() as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for k,v in row.items():
            if v in ['True','False']: row[k]=v=='True'
            elif v=='': row[k]=None
            else:
                try: row[k]=float(v)
                except ValueError: pass
    return rows

E = {g:readcsv(g+'_episodes.csv') for g in GROUPS}
S = {g:readcsv(g+'_steps.csv') for g in GROUPS}
B = readcsv('bridge_actions.csv')
CATALOG=[]

def fig(title, compare=False, size=(9,5)):
    f,ax=plt.subplots(figsize=size)
    f.suptitle(title, x=.09, ha='left', fontweight='bold', y=.98)
    if compare:
        f.text(.09,.915,'Observational comparison: task weights and goal position also changed.',fontsize=9,color='#735d36')
    f.subplots_adjust(top=.85 if compare else .89,bottom=.17,left=.11,right=.97)
    return f,ax

def save(f,name,caption,category):
    f.savefig(FIG/(name+'.png'),dpi=180)
    f.savefig(FIG/(name+'.pdf'))
    plt.close(f)
    CATALOG.append(dict(name=name,title=name.replace('_',' ').title(),caption=caption,category=category))

def cdf(ax,x,label,color,**kw):
    x=np.asarray(x).ravel();x=np.sort(x[np.isfinite(x)])
    ax.plot(x,np.arange(1,len(x)+1)/len(x),label=label,color=color,**kw)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Cumulative fraction')

def array(g,k): return A[g+'_'+k]
def finite(x): return np.asarray(x)[np.isfinite(x)]
def notes(f,text): f.text(.11,.035,text,fontsize=8,color='#515868')

def outcomes():
    f,ax=fig('Episode outcomes',True)
    bottom=np.zeros(2)
    for key,title in [('success','Success'),('failure','Drop'),('timeout','Timeout')]:
        n=np.array([sum(e['outcome']==key for e in E[g]) for g in GROUPS]);val=n/np.array([12,24])*100
        ax.bar([0,1],val,bottom=bottom,color=COL[key],label=title,width=.58)
        for i in range(2): ax.text(i,bottom[i]+val[i]/2,f'{n[i]} / {[12,24][i]}\n{val[i]:.1f}%',ha='center',va='center',color='white',fontweight='bold')
        bottom+=val
    ax.set(xticks=[0,1],xticklabels=SHORT,ylabel='Episodes (%)',ylim=(0,105));ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    f.subplots_adjust(right=.82)
    save(f,'01_outcomes','All completed episodes; aborted episode 17 of the first current session is excluded.','Performance')
    f,ax=fig('Success rate and uncertainty',True)
    for i,g in enumerate(GROUPS):
        x=R[g];p=x['success_rate']*100;lo,hi=np.array(x['wilson95'])*100
        ax.errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt='o',capsize=8,ms=11,color=COL[g])
        ax.text(i,p+2.5,f"{x['successes']}/{x['episodes']} ({p:.1f}%)",ha='center')
    ax.set(xticks=[0,1],xticklabels=SHORT,ylabel='Success rate (%)',ylim=(0,60),xlim=(-.5,1.5))
    notes(f,'Error bars: Wilson 95% intervals. Episodes are not randomized paired trials.')
    save(f,'02_success_uncertainty','Both batches achieved 16.7%; uncertainty is wide and intervals overlap.','Performance')
    f,ax=fig('Current episodes: outcomes and episode length')
    for key in ['success','failure','timeout']:
        rows=[e for e in E['current'] if e['outcome']==key]
        ax.bar([e['analysis_episode'] for e in rows],[e['steps'] for e in rows],color=COL[key],label=key.replace('failure','drop'))
    ax.axvline(16.5,color='#888',ls='--',lw=1);ax.axhline(1000,color='#888',ls=':',lw=1)
    ax.set(xlabel='Analysis episode (1–16: first run; 17–24: second run)',ylabel='Published MPPI actions',xticks=np.arange(1,25),ylim=(0,1120));ax.legend(ncol=3)
    save(f,'03_episode_outcomes_steps','First session: 3/16 successes; second: 1/8. Timeout means 1000 actions.','Performance')
    f,ax=fig('Outcome distribution by starting rotation')
    directions=['-X','+X','-Y','+Y','-Z','+Z'];bottom=np.zeros(6)
    for key in ['success','failure','timeout']:
        counts=np.array([sum(e['outcome']==key and e['direction']==d for e in E['current']) for d in directions])
        ax.bar(np.arange(6),counts,bottom=bottom,color=COL[key],label=key.replace('failure','drop'))
        for i,n in enumerate(counts):
            if n:ax.text(i,bottom[i]+n/2,str(n),ha='center',va='center',color='white',fontweight='bold')
        bottom+=counts
    ax.set(xticks=np.arange(6),xticklabels=directions,ylabel='Completed episodes',ylim=(0,6));ax.legend(ncol=3)
    notes(f,'Nearest signed principal axis of initial goal-relative rotation; not the cube’s printed face.')
    save(f,'04_starting_rotation_outcomes','Rotation groups have unequal counts (4,4,3,5,4,4); each starts approximately 90° from the goal.','Performance')
    f,ax=fig('Initial orientation error: approximately 90° starts')
    for e in E['current']:
        ax.scatter(e['analysis_episode'],e['initial_angle_deg'],color=COL[e['outcome']],s=50)
        ax.annotate(e['direction'],(e['analysis_episode'],e['initial_angle_deg']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax.axhline(90,color='#888',ls='--');ax.set(xlabel='Analysis episode',ylabel='Initial orientation error (degrees)',ylim=(84,97),xticks=np.arange(1,25))
    save(f,'05_initial_orientation','Initial orientation error mean 90.09°, range 86.97–93.55°.','Performance')

def errors():
    for key,label,limit,unit in [('pos','Position',20,'mm'),('angle','Orientation',AT,'degrees')]:
        f,ax=fig(label+' error during manipulation',size=(10,5.4))
        max_t=max(A[f'current_ep{i}_t'][-1] for i in range(1,25));grid=np.linspace(min(A[f'current_ep{i}_t'][0] for i in range(1,25)),max_t,350)
        vals=[]
        for i,e in enumerate(E['current'],1):
            t=A[f'current_ep{i}_t'];v=A[f'current_ep{i}_{key}']
            ax.plot(t,v,color=COL[e['outcome']],alpha=.32,lw=.8)
            y=np.interp(grid,t,v);y[(grid<t[0])|(grid>t[-1])]=np.nan;vals.append(y)
        vals=np.array(vals)
        with np.errstate(all='ignore'):
            lo,med,hi=np.nanpercentile(vals,[25,50,75],axis=0)
        ax.fill_between(grid,lo,hi,color='#303c50',alpha=.12,label='Active-episode IQR')
        ax.plot(grid,med,color='#303c50',lw=2,label='Active-episode median')
        ax.axhline(limit,color='#9b3f35',ls='--',label=f'Success threshold: {limit:.2f} {unit}')
        ax.set(xlabel='Manipulation wall time (s)',ylabel=f'{label} error ({unit})',xlim=(0,max_t));ax.legend(fontsize=8)
        notes(f,'Green: success; red: drop; amber: timeout. Curves stop at episode end; late medians describe survivors.')
        save(f,'06_position_convergence' if key=='pos' else '07_orientation_convergence','Includes terminal measurements and stops at each episode’s recorded end.','Errors')
    f,ax=fig('Episodes still active over manipulation time')
    for g in GROUPS:
        t=np.sort([e['duration_s'] for e in E[g]]);ax.step(np.r_[0,t],np.r_[len(t),len(t)-np.arange(1,len(t)+1)]/len(t)*100,where='post',color=COL[g],label=LABEL[g])
    ax.set(xlabel='Manipulation wall time (s)',ylabel='Episodes still active (%)');ax.legend(fontsize=8)
    save(f,'08_active_episodes','Both successes and failures leave the active set; this is not a survival-success curve.','Errors')
    for prefix,pkey,akey,title in [('09_final_pose_errors','final_pos_mm','final_angle_deg','Errors at the recorded episode endpoint'),('10_closest_simultaneous_pose','best_joint_pos_mm','best_joint_angle_deg','Closest simultaneous approach to both pose thresholds')]:
        f,ax=fig(title)
        ax.axvspan(0,20,color='#25945c',alpha=.08);ax.axhspan(0,AT,color='#25945c',alpha=.08)
        ax.axvline(20,color='#777',ls='--');ax.axhline(AT,color='#777',ls='--')
        for e in E['current']:
            ax.scatter(e[pkey],e[akey],color=COL[e['outcome']],s=50)
            ax.annotate(str(int(e['analysis_episode'])),(e[pkey],e[akey]),xytext=(4,3),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Position error (mm)',ylabel='Orientation error (degrees)',xlim=(0,None),ylim=(0,None))
        notes(f,'Episode numbers 1–24. Endpoint errors include dropped cubes.' if 'final' in prefix else 'Minimum over time of max(position/threshold, orientation/threshold, velocity/threshold).')
        save(f,prefix,'Shaded lower-left overlap is the joint position/orientation success region.','Errors')
    f,ax=fig('Position and orientation errors at episode end',True)
    for i,(key,unit) in enumerate([('final_pos_mm','Position (mm)'),('final_angle_deg','Orientation (degrees)')]):
        for j,g in enumerate(GROUPS):
            y=[e[key] for e in E[g]];offset=(j-.5)*.27
            ax.scatter(i+offset+np.linspace(-.05,.05,len(y)),y,color=COL[g],alpha=.6,s=25,label=LABEL[g] if i==0 else None)
            ax.plot([i+offset-.08,i+offset+.08],[np.median(y)]*2,color=COL[g],lw=3)
    ax.set(xticks=[0,1],xticklabels=['Position error (mm)','Orientation error (degrees)'],ylabel='Recorded endpoint error');ax.legend(fontsize=8)
    notes(f,'Points: episodes; thick marks: medians. Different units share a numerical axis for compact comparison.')
    save(f,'11_endpoint_comparison','Dropped-cube endpoints dominate the latest aggregate endpoint errors.','Errors')

def timing():
    for name,key,title,xlabel,threshold in [
        ('12_planning_time_cdf','planning','Planning-time distribution','Planning time (ms)',64),
        ('13_publication_interval_cdf','gaps','Action publication intervals','Consecutive publication interval (ms)',64)]:
        f,ax=fig(title,True)
        for g in GROUPS:cdf(ax,array(g,key),LABEL[g],COL[g])
        ax.axvline(threshold,color='#9b3f35',ls='--',label='64 ms nominal period')
        ax.set_xlabel(xlabel);ax.legend(fontsize=8)
        if key=='gaps':notes(f,'Full distribution shown, including one 270.6 ms interval. No intervals cross episode boundaries.')
        save(f,name,'Unpaced operation; no intentional pacing waits were recorded.','Timing')
    f,ax=fig('Instantaneous control frequency',True)
    for g in GROUPS:cdf(ax,1000/array(g,'gaps'),LABEL[g],COL[g])
    ax.axvline(15.625,color='#9b3f35',ls='--',label='Nominal 15.625 Hz');ax.set_xlabel('1 / consecutive publication interval (Hz)');ax.legend(fontsize=8)
    notes(f,'Pooled effective frequency = total actions / total manipulation time; not the mean of instantaneous Hz.')
    save(f,'14_control_frequency_cdf','Current pooled frequency 16.76 Hz; instantaneous range 3.70–33.21 Hz.','Timing')
    f,ax=fig('Per-episode control and planning timing')
    x=np.arange(1,25);ax.bar(x,[e['control_hz'] for e in E['current']],color=[COL[e['outcome']] for e in E['current']])
    ax.axhline(15.625,color='#333',ls='--',label='Nominal 15.625 Hz');ax.set(xlabel='Analysis episode',ylabel='Episode actions / manipulation seconds (Hz)',xticks=x,ylim=(0,24));ax.legend()
    save(f,'15_episode_control_frequency','Per-episode mean frequencies range from 15.93 to 20.83 Hz.','Timing')
    f,ax=fig('Planning-time distribution by episode')
    data=[]
    for i in range(1,25):data.append([s['planning_ms'] for s in S['current'] if s['episode']==i])
    ax.boxplot(data,showfliers=False,whis=(5,95),medianprops={'color':COL['current']})
    ax.axhline(64,color='#9b3f35',ls='--',label='64 ms');ax.set(xlabel='Analysis episode',ylabel='Planning time (ms)');ax.legend()
    notes(f,'Boxes: 25–75%; whiskers: 5–95%. All extrema remain in the CSV and metric table.')
    save(f,'16_planning_by_episode','966/11222 planning calls (8.61%) exceeded 64 ms.','Timing')
    f,ax=fig('Feedback freshness at command publication')
    for key,label,color in [('command_joint_age','Joint state','#147d92'),('command_pose_age','Tracked object','#a85c91')]:
        cdf(ax,array('current',key),label,color)
    ax.set_xlabel('Age of latest received measurement (ms)');ax.legend()
    notes(f,'Latest post-planning feedback; the object measurement used to start the plan is older by planning end.')
    save(f,'17_feedback_age','Mean latest joint age 1.54 ms and object age 71.66 ms.','Timing')

def clipping():
    f,ax=fig('Hardware clipping by joint',True,size=(11,5))
    x=np.arange(16)
    for j,g in enumerate(GROUPS):ax.bar(x+(j-.5)*.36,array(g,'hits').mean(axis=0)*100,width=.36,color=COL[g],label=LABEL[g])
    ax.set(xticks=x,xticklabels=A['joint_names'],ylabel='Clipped joint commands (%)');ax.tick_params(axis='x',rotation=55);ax.legend(fontsize=8)
    f.subplots_adjust(bottom=.24)
    save(f,'18_clipping_by_joint','Any-joint clipping: 14.77% of current actions; individual-joint clipping: 1.06%.','Hardware clipping')
    f,ax=fig('Hardware clipping by episode')
    ax.bar(np.arange(1,25),[100*e['clipped_action_fraction'] for e in E['current']],color=[COL[e['outcome']] for e in E['current']])
    ax.set(xlabel='Analysis episode',ylabel='Actions with at least one clipped joint (%)',xticks=np.arange(1,25))
    save(f,'19_clipping_by_episode','Clipping is computed from raw MPPI deltas against the ±0.2 rad hardware bound.','Hardware clipping')
    f,ax=fig('Raw MPPI action delta magnitudes',True)
    for g in GROUPS:cdf(ax,np.abs(np.degrees(array(g,'delta'))),LABEL[g],COL[g])
    ax.axvline(np.degrees(.2),ls='--',color='#9b3f35',label='±0.2 rad hardware bound');ax.set_xlabel('Absolute raw action delta (degrees, all joint values)');ax.legend(fontsize=8)
    save(f,'20_raw_delta_distribution','Requested deltas before hardware clipping; includes all 16 joints per action.','Hardware clipping')

def actionrate():
    f,ax=fig('Target-change tails with the action-rate cost',True)
    cdf(ax,np.degrees(array('no_rate','estimated_jump')),LABEL['no_rate'],COL['no_rate'])
    cdf(ax,np.degrees(array('current','estimated_jump')),LABEL['current'],COL['current'])
    cdf(ax,np.max(np.abs(np.degrees(A['bridge_changes'])),axis=1),'Current: bridge-written targets','#d08a31',ls='--')
    ax.axvline(10,ls=':',color='#777');ax.set_xlabel('Largest joint-target change between consecutive commands (degrees)');ax.legend(fontsize=8)
    notes(f,'Controller estimates exclude each episode’s first action. Bridge data include all 11,222 selected commands.')
    save(f,'21_action_rate_target_jump','Current estimated p95 target jump 12.74° versus 14.97° earlier; means are 8.84° versus 8.05°.','Action rate')
    f,ax=fig('Large jumps are rarer; the average is not smaller',True)
    x=np.arange(3)
    vals=[]
    for g in GROUPS:
        s=R['action_rate'][g];vals.append([s['estimated_max_joint_jump_deg']['mean'],s['estimated_max_joint_jump_deg']['p95'],100*s['estimated_jump_above_10deg_fraction']])
    for j,g in enumerate(GROUPS):
        bars=ax.bar(x+(j-.5)*.36,vals[j],width=.36,color=COL[g],label=LABEL[g])
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set(xticks=x,xticklabels=['Mean maximum jump (°)','p95 maximum jump (°)','Commands above 10° (%)'],ylabel='Value (units shown on each category)',ylim=(0,47));ax.legend(fontsize=8)
    save(f,'22_action_rate_tail_summary','Maximum means the largest absolute change over the 16 joints in one command.','Action rate')
    f,ax=fig('First-command action-rate diagnostic: planner to hardware')
    for key,label,color in [('planned_energy','Planned: pre-planning joint base','#637291'),('estimated_energy','Controller: rebased and clipped',COL['current'])]:
        cdf(ax,15*array('current',key),label,color)
    cdf(ax,15*A['bridge_actual_energy'],'Actual bridge target change','#d08a31',ls='--')
    ax.set_xlabel('15 × sum of squared target changes (rad² × weight)');ax.legend(fontsize=8)
    notes(f,'First weighted-mean command only. Not the full-horizon action-rate cost used to score each sample.')
    save(f,'23_action_rate_penalty_cdf','Mean first-command diagnostic: planned 1.95, estimated 1.36, bridge-written 1.34.','Action rate')
    f,ax=fig('Action-rate diagnostic before and after hardware application')
    x=15*A['bridge_planned_energy'];y=15*A['bridge_actual_energy'];mx=max(x.max(),y.max())
    h=ax.hexbin(x,y,gridsize=55,mincnt=1,bins='log',cmap='viridis');f.colorbar(h,ax=ax,label='Matched actions (log color scale)')
    ax.plot([0,mx],[0,mx],color='#c66554',ls='--',label='Same target-change energy')
    ax.set(xlabel='Planned first-command diagnostic',ylabel='Actual bridge first-command diagnostic');ax.legend(fontsize=8)
    notes(f,f"{100*R['bridge']['actual_energy_exceeds_planned_fraction']:.2f}% of matched actions have larger bridge-written change energy than the planned diagnostic.")
    save(f,'24_action_rate_planned_vs_actual','Both axes use weight 15; bridge change is relative to its actual previously held target.','Action rate')
    f,ax=fig('First-command penalty by episode')
    x=np.arange(1,25)
    means=[];p95=[]
    for i in x:
        v=15*A['bridge_actual_energy'][A['bridge_action_episode']==i];means.append(v.mean());p95.append(np.percentile(v,95))
    ax.plot(x,[e['planned_first_command_cost_mean'] for e in E['current']],'o-',label='Planned diagnostic: mean',color='#637291',ms=4)
    ax.plot(x,means,'o-',label='Bridge-written diagnostic: mean',color=COL['current'],ms=4)
    ax.plot(x,p95,'--',label='Bridge-written diagnostic: p95',color='#d08a31')
    ax.set(xlabel='Analysis episode',ylabel='First-command target-change penalty',xticks=x);ax.legend(fontsize=8)
    save(f,'25_action_rate_by_episode','This is a command-continuity diagnostic, not a decomposition of sampled rollout costs.','Action rate')
    f,ax=fig('Equivalent target-change speed',True)
    for g in GROUPS:cdf(ax,np.degrees(array(g,'estimated_jump')/array(g,'delta_t')),LABEL[g],COL[g])
    ax.set_xlabel('Largest target change / publication interval (degrees/s)');ax.legend(fontsize=8)
    notes(f,'This is commanded-target motion, not the physical velocity achieved by a motor.')
    save(f,'26_target_change_speed','Current p95 221°/s versus 255°/s earlier; mean 149°/s versus 135°/s.','Action rate')
    f,ax=fig('Measured joint velocity',True)
    for g in GROUPS:cdf(ax,np.degrees(array(g,'joint_speed')),LABEL[g],COL[g])
    ax.set_xlabel('Absolute measured joint velocity (degrees/s, all joints)');ax.legend(fontsize=8)
    save(f,'27_measured_joint_speed','Hardware velocity feedback at planning start; differs from target-change speed.','Action rate')
    f,ax=fig('Target changes introduced after planning')
    cdf(ax,np.degrees(array('current','rebase')),'Controller: pre/post-plan rebasing','#637291')
    cdf(ax,np.degrees(array('current','clipping_shift')),'Controller: hardware clipping shift','#b26459')
    cdf(ax,np.max(abs(np.degrees(A['bridge_extra'])),axis=1),'Bridge target vs controller estimate',COL['current'])
    ax.set_xlabel('Largest absolute difference over the 16 joints (degrees)');ax.legend(fontsize=8)
    notes(f,'These are distinct differences, not additive independent errors. Each compares two target estimates.')
    save(f,'28_post_planning_target_changes','Rebasing shift mean 5.27°; bridge-vs-estimate difference mean 0.36°.','Action rate')
    f,ax=fig('First second: target motion and orientation progress',True)
    for g in GROUPS:
        ax.scatter([e['first1s_target_step_mean_deg'] for e in E[g]],[e['first1s_angle_improvement_deg'] for e in E[g]],label=LABEL[g],color=COL[g],alpha=.8,s=45)
    ax.axhline(0,color='#777',ls='--');ax.set(xlabel='First-second mean absolute target change (degrees, all joints)',ylabel='Orientation improvement after 1 s (degrees)');ax.legend(fontsize=8)
    notes(f,'One point per episode. Positive improvement means closer to goal. Short episodes use their recorded endpoint.')
    save(f,'29_early_motion_orientation','Less first-second target motion does not necessarily imply faster or more reliable rotation.','Action rate')

def bridge():
    f,ax=fig('Command-following error during completed manipulations',size=(11,5))
    x=np.arange(16)
    ax.bar(x,A['bridge_joint_error_mean'],label='Mean absolute error',color=COL['current'])
    ax.plot(x,A['bridge_joint_error_p95'],'o-',label='p95 absolute error',color='#d08a31',ms=4)
    ax.set(xticks=x,xticklabels=A['joint_names'],ylabel='Held target − measured joint position (degrees, absolute)');ax.tick_params(axis='x',rotation=55);ax.legend(fontsize=8);f.subplots_adjust(bottom=.25)
    save(f,'30_command_following_error','Pre-write measured feedback paired with the already held target; includes dynamic tracking lag and load offsets.','Bridge diagnostics')
    f,ax=fig('Motor current peaks in the 24 completed episodes')
    x=np.arange(16);ax.bar(x,A['bridge_current_peak'],color=COL['current']);ax.axhline(550,color='#b4463c',ls='--',label='Configured cutoff 550 raw')
    ax.set(xticks=x,xlabel='Native motor ID (different from MuJoCo joint order)',ylabel='Peak absolute measured current (raw units)',ylim=(0,590));ax.legend(fontsize=8)
    notes(f,'Aborted episode 17 is excluded: it subsequently reached −554 raw at motor 14 and stopped the session.')
    save(f,'31_motor_currents','Completed episodes peak at 535 raw; not evidence that current limits can be increased.','Bridge diagnostics')
    f,ax=fig('LCM publication to first motor write')
    cdf(ax,[b['publish_to_write_ms'] for b in B],f"Matched actions: {R['bridge']['matched_actions']} / {R['bridge']['selected_actions']}",COL['current'])
    ax.set_xlabel('Publication to write start (ms)');ax.legend()
    save(f,'32_publish_to_write_latency','Mean 1.62 ms, p95 2.99 ms. This excludes the preceding 59.1 ms planning step.','Bridge diagnostics')

def additional_errors():
    f,ax=fig('Quaternion success metric over manipulation time',size=(10,5.4))
    for i,e in enumerate(E['current'],1):
        angle=np.radians(A[f'current_ep{i}_angle']);metric=np.sin(angle/2)**2
        ax.plot(A[f'current_ep{i}_t'],metric,color=COL[e['outcome']],alpha=.45,lw=.9)
    ax.axhline(.04,color='#9b3f35',ls='--',label='Quaternion threshold: 0.04')
    ax.set(xlabel='Manipulation wall time (s)',ylabel='Quaternion error: 1 − (q · goal)²',ylim=(0,1.03));ax.legend()
    notes(f,'Green: success; red: drop; amber: timeout. Equivalent orientation angle threshold: 23.07°.')
    save(f,'33_quaternion_metric','Quaternion-metric version of the orientation curves; sign-invariant for q and −q.','Errors')
    for axis in range(3):
        f,ax=fig(f'Signed {"XYZ"[axis]} position error over manipulation time',size=(10,5.4))
        for i,e in enumerate(E['current'],1):ax.plot(A[f'current_ep{i}_t'],A[f'current_ep{i}_xyz'][:,axis],color=COL[e['outcome']],alpha=.45,lw=.9)
        ax.axhline(0,color='#333',ls='--');ax.axhspan(-20,20,color='#25945c',alpha=.06)
        ax.set(xlabel='Manipulation wall time (s)',ylabel=f'Object {"xyz"[axis]} − target {"xyz"[axis]} (mm)')
        notes(f,'Shading shows ±20 mm per axis for reference; actual position success uses the 3D Euclidean norm <20 mm.')
        save(f,f'{34+axis:02d}_position_{"xyz"[axis]}','Signed errors in MuJoCo world coordinates; no per-axis success criteria are substituted.','Errors')

def metrics():
    rows=[]
    def add(label,a,b):rows.append({'metric':label,'earlier_rate0_12':a,'retuned_rate15_24':b})
    def both(label,path,fmt='{:.3f}',scale=1):
        vals=[]
        for g in GROUPS:
            x=R[g]
            for k in path.split('.'):x=x[k]
            vals.append(fmt.format(x*scale))
        add(label,*vals)
    for k in ['episodes','successes','drops','timeouts','steps']:both(k,k,'{:.0f}')
    both('success_rate_percent','success_rate',scale=100)
    add('success_95pct_Wilson_interval',*[f"{R[g]['wilson95'][0]*100:.2f}–{R[g]['wilson95'][1]*100:.2f}%" for g in GROUPS])
    for key in ['total_manipulation_s','effective_hz','duration_s.mean','duration_s.median','success_duration_s.mean','success_duration_s.median']:
        both(key,key)
    for key in ['planning_ms','publication_interval_ms','instantaneous_hz','episode_hz']:
        for st in ['min','mean','std','median','p95','p99','max']:both(key+'.'+st,'timing.'+key+'.'+st)
    both('pacing_waited_steps','waited_steps','{:.0f}');both('pacing_total_wait_s','total_wait_s')
    both('planning_over64_count','timing.planning_over64_count','{:.0f}');both('planning_over64_percent','timing.planning_over64_fraction',scale=100)
    both('publication_over64_count','timing.publication_interval_over64_count','{:.0f}');both('publication_over64_percent','timing.publication_interval_over64_fraction',scale=100)
    for key in ['clipped_action_fraction','clipped_joint_fraction']:both(key+'_percent',key,scale=100)
    for key in ['initial_pos_mm','initial_angle_deg','final_pos_mm','final_angle_deg','best_angle_deg','episode_mean_pos','episode_mean_angle','success_final_position_mm','success_final_orientation_deg']:
        for st in ['mean','median']:both(key+'.'+st,key+'.'+st)
    for k in ['orientation_ever','both_pose_ever','near_position_misses']:both(k,k,'{:.0f}')
    for key in ['estimated_max_joint_jump_deg','equivalent_target_change_speed_deg_s','measured_abs_joint_velocity_deg_s','planned_first_command_energy_rad2','estimated_first_command_energy_rad2','rebasing_shift_max_deg','command_joint_age_ms','command_pose_age_ms']:
        for st in ['mean','p95','max']:add('action.'+key+'.'+st,*[f"{R['action_rate'][g][key][st]:.4f}" for g in GROUPS])
    for k in ['estimated_jump_above_10deg_fraction','estimated_jump_above_20deg_fraction']:add(k+'_percent',*[f"{R['action_rate'][g][k]*100:.4f}" for g in GROUPS])
    for k in ['actual_max_joint_jump_deg','actual_first_command_penalty','estimated_target_discrepancy_deg','publication_to_write_ms','holding_error_deg','holding_max_joint_error_deg','read_ms','write_ms']:
        for st in ['mean','p95','max']:add('bridge.'+k+'.'+st,'not extracted',f"{R['bridge'][k][st]:.4f}")
    with (HERE/'metrics.csv').open('w',newline='') as s:
        w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

def table(rows,headers):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'

def write_report(rows):
    root=ROOT/R['current_sources'][0]['path'];d=json.loads(root.read_text())
    config={k:d[k] for k in ['args','argv','execution','scenes','effective_ctrl_range','joint_reference','rollout_config','planner_config','pose_transform','weights','action_rate_cost','temporal_plan_blending','failure_z']}
    config['scenes']={k:{'source_scene':v.get('source_scene')} for k,v in d['scenes'].items()}
    body=ET.fromstring(d['scenes']['rollout']['scene_xml']).find(".//body[@name='obj']")
    config['object_geom_snapshot']=body.find('geom').attrib
    config['actual_episode_goals']=R['current_configuration']['actual_goal_positions_m']
    config['actual_episode_drop_heights_m']=R['current_configuration']['actual_drop_heights_m']
    (HERE/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    sections=['''<!doctype html><html lang="en"><meta charset="utf-8"><title>M2 latest 24 episodes — action-rate analysis</title>
    <style>body{font:16px/1.6 system-ui;max-width:1180px;margin:35px auto;padding:0 22px;color:#253247;background:#fafbfd}h1,h2{line-height:1.2}a{color:#08778b}table{border-collapse:collapse;font-size:14px;margin:20px 0;display:block;overflow:auto}td,th{padding:7px 12px;border-bottom:1px solid #d7dce3;text-align:left}th{background:#e9edf2}img{width:100%;max-width:950px;background:white}figure{margin:25px 0 45px}figcaption{max-width:950px;font-size:14px;color:#536178}.box{padding:15px 22px;background:#eef5f7;border-left:4px solid #147d92}.warn{background:#fbf1dc;border-left-color:#b88930}code,pre{background:#edf0f3;padding:3px 6px}pre{white-space:pre-wrap}details{margin:20px 0}</style>
    <h1>M2: latest 24 completed hardware episodes</h1>
    <p>Two sessions: <strong>16 + 8</strong> completed episodes, 11,222 actions, 669.58 seconds of active manipulation. The aborted 17th episode of the first session is excluded throughout.</p>
    <div class="box"><strong>Result: 4 successes, 16 drops, 4 timeouts — 16.7% success.</strong><br>
    Planning: mean 59.14 ms, p95 65.26 ms; pooled effective control 16.76 Hz. Any-joint hardware clipping: 14.77% of actions.</div>
    <p><a href="metrics.csv">All metrics CSV</a> · <a href="current_episodes.csv">24-episode table</a> · <a href="current_steps.csv">Per-action metrics</a> · <a href="bridge_actions.csv">Matched bridge commands</a> · <a href="graphs_png.zip">All separate PNG images</a> · <a href="graphs_pdf.zip">All separate PDF figures</a> · <a href="report.json">Full statistics JSON</a> · <a href="manifest.json">Source manifest / exclusions</a> · <a href="configuration.json">Configuration snapshot</a> · <a href="validation.json">Validation checks</a></p>
    <h2>What changed and what the data show</h2>
    <p>The current batch uses M2, calibrated high/high hand/object geometry, 70 g cube, sliding friction cube/tips/palm/other fingers 0.3/0.5/0.4/0.4, P011 joint-cost reference, 512 samples, one iteration, horizon 0.256 s (4 steps), control timestep 0.064 s, shifted warm start, noise 0.05, temperature 0.1, latest Method-5 camera calibration, measured-position delta rebasing, ±0.2 rad clipping, and no pacing or temporal blending. Goal position is [0.025, 0.035, 0.075] m; <strong>the actual per-episode fallen_z threshold is 0.065 m</strong> in both cohorts. The top-level failure_z=0 field is not the effective episode threshold. Position success requires &lt;20 mm and quaternion metric &lt;0.04 (23.07°), together with the configured velocity check.</p>
    <div class="box warn">The earlier 12-episode rate-0 batch is a useful reference, but <strong>five task-cost weights and goal position changed together with action-rate weight 0 → 15</strong>. The goal moved from [20,35,80] to [25,35,75] mm (+5 mm X, −5 mm Z). This is not an action-rate-only experiment and cannot establish a causal benefit or harm from that term.</div>''']
    sections.append(table([(k,f"{v['no_rate']:.8g}",f"{v['current']:.8g}") for k,v in R['changed_cost_weights'].items()],['Changed task weight','Earlier / rate 0','Current / rate 15']))
    sections.append('''<p>Success is unchanged at 16.7% (2/12 earlier, 4/24 now). Drops increased from 41.7% to 66.7%, while timeouts decreased from 41.7% to 16.7%. Mean endpoint position/orientation errors are 115.1 mm / 86.5° now versus 83.4 mm / 67.6° earlier; these include dropped-cube endpoints. Per-episode mean orientation error is 70.7° versus 54.1° earlier. Only 5/24 episodes entered the orientation-success region; four also met the position criterion. The evidence does not support adopting the combined retuning as a performance improvement.</p>
    <h2>Action-rate findings</h2>
    <p>The action-rate term is <code>15 × Σh ||target[h] − previous_target||²</code>, summed over all 16 joints without timestep division. The first target is compared with the previous commanded/estimated target. It scores requested absolute targets before hardware rebasing and clipping and is a soft penalty.</p>
    <ul><li>Estimated maximum joint-target jump p95 fell from <strong>14.97° to 12.74°</strong>; fraction above 10° fell from <strong>39.43% to 27.87%</strong>. Mean maximum jump rose from <strong>8.05° to 8.84°</strong>: fewer extreme jumps, not uniformly smaller motion.</li>
    <li>Clipped-action fraction fell from <strong>20.91% to 14.77%</strong> (relative decrease 29.3%). Individual-joint clipping fell from 1.68% to 1.06%.</li>
    <li>Bridge telemetry confirms actual written-target maximum jumps: mean <strong>8.76°</strong>, p95 <strong>12.53°</strong>, maximum 20.30°. There is no hard bound on change relative to the previous target; ±0.2 bounds each delta relative to the newest measured joint position.</li>
    <li>For the first weighted-mean command only, mean action-rate diagnostic is <strong>1.95 planned, 1.36 controller-estimated and 1.34 actually written</strong>. In 7.12% of matched actions, actual change energy exceeds the planned diagnostic. These scalars are <strong>not the full-horizon sampled cost decomposition</strong>; the full action-rate contribution relative to other costs is not logged.</li>
    <li>Pre/post-planning rebasing changes the largest target component by <strong>5.27° on average</strong>; bridge vs controller estimate differs by another 0.36° on average (these quantities are not additive independent errors). This explains why the requested trajectory penalty is not an exact penalty on executed command continuity.</li>
    <li>Mean absolute measured joint speed is <strong>38.12°/s now versus 28.86°/s earlier</strong>; p95 is 103.05°/s versus 100.30°/s. Smaller target-change tails did not produce uniformly slower physical motion. These pooled means also reflect the different time spent stalled or manipulating.</li></ul>
    <p>Cross-cohort comparisons of estimated target jumps exclude the first command per episode. The bridge analysis independently matched <strong>all 11,222/11,222 published actions (100%)</strong> by their exact action timestamps, including the first delta replacing the preparation hold and final writes just after the recorded episode endpoint. Motor-cycle current and following-error diagnostics remain restricted to completed manipulation windows.</p>
    <h2>Timing and tracking</h2>
    <p>Planning min/mean/median/p95/p99/max: <strong>29.86 / 59.14 / 59.20 / 65.26 / 70.86 / 76.63 ms</strong>. 966/11,222 calls (8.61%) exceeded 64 ms. No pacing waits occurred. Publication intervals exceeded 64 ms for 1,205/11,198 intervals (10.76%). A single long 270.63 ms interval produces the minimum instantaneous frequency 3.70 Hz; this is distinct from the maximum planning time, which was only 76.63 ms.</p>
    <p>Pooled frequency is <strong>16.76 Hz</strong>; per-episode means span <strong>15.93–20.83 Hz</strong>, instantaneous min/max <strong>3.70–33.21 Hz</strong>. The arithmetic mean of instantaneous frequencies is 16.85 Hz and should not replace pooled throughput. Latest joint feedback at publication averages 1.54 ms old; latest object feedback averages 71.66 ms old. Mean publish-to-first-write latency is 1.62 ms (p95 2.99 ms).</p>
    <p>Bridge following error against the already held command, across all recorded joints/cycles during completed manipulation, is mean 2.67°, p95 7.20°. The largest joint error in a cycle averages 7.18°, p95 11.02°. These combine dynamic response and load offsets, rather than measuring settled accuracy alone.</p>
    <h2>Session health and exclusions</h2>
    <p>First session: 3 successes, 10 drops, 3 timeouts, then aborted episode 17 when native motor 14 current reached −554 raw versus cutoff 550. That aborted episode is retained in the source logs/manifest but excluded from the 24 completed episodes and current plots. Second session: 1 success, 6 drops, 1 timeout before operator interruption. Both bridge logs verify torque disabled on all 16 motors after cleanup. The maximum current within completed episodes was 535 raw. The four source logs have identical embedded rollout scene XML hashes, confirming unchanged scene physics in the rate-0 / rate-15 comparison.</p>
    <p>Initial rotations average 90.09° (86.97–93.55°). Four episodes each used −X,+X,−Z,+Z; three −Y and five +Y. Signed-axis labels describe the initial rotation, not the printed face on top.</p>
    <h2>Suggested next comparison</h2><p>Keep all task weights and the goal position fixed and compare action-rate 0 versus 15 with matched starting rotations. The present data show smaller command-change tails but no success-rate gain; changing the task weights, goal and rate penalty together prevents assigning the increased drop rate to any one change. No hardware, tracking, simulation, or production configuration was touched for this analysis.</p>''')
    sections.append('<h2>Separate graphs</h2><p>Every figure is available as an individual PNG and PDF; distributions use all logged observations, with episode-level comparisons also shown.</p>')
    for c in CATALOG:
        name=c['name'];sections.append(f'<figure id="{name}"><a href="individual_graphs/{name}.png"><img loading="lazy" src="individual_graphs/{name}.png" alt="{html.escape(c["title"])}"></a><figcaption>{html.escape(c["caption"])} <a href="individual_graphs/{name}.pdf">PDF</a></figcaption></figure>')
    sections.append('<h2>All metrics</h2>'+table([[v for v in row.values()] for row in rows],['Metric','Earlier: rate 0 / 12 episodes','Retuned: rate 15 / 24 episodes']))
    keys=['analysis_episode','source_episode','direction','outcome','steps','duration_s','final_pos_mm','final_angle_deg','clipped_action_fraction','control_hz']
    sections.append('<h2>Episode-level metrics</h2>'+table([[f'{e[k]:.3f}' if isinstance(e[k],float) else e[k] for k in keys] for e in E['current']],keys))
    sections.append('<h2>Reproduce offline</h2><pre>python '+str(HERE.relative_to(ROOT)/'analyze.py')+'\npython '+str(HERE.relative_to(ROOT)/'plot_report.py')+'</pre><p>Scripts read saved logs only. Full source paths and SHA-256 hashes are in manifest.json. CSV files are the numerical source for these plots.</p></html>')
    (HERE/'index.html').write_text('\n'.join(sections))
    (HERE/'graph_catalog.json').write_text(json.dumps(CATALOG,indent=2)+'\n')
    for suffix in ['png','pdf']:
        with zipfile.ZipFile(HERE/f'graphs_{suffix}.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for c in CATALOG:z.write(FIG/(c['name']+'.'+suffix),arcname=c['name']+'.'+suffix)

if __name__=='__main__':
    outcomes();errors();timing();clipping();actionrate();bridge();additional_errors();rows=metrics();write_report(rows)
    print(f'Created {len(CATALOG)} figures, each PNG + PDF, {len(rows)} metrics, HTML report and two graph archives.')
