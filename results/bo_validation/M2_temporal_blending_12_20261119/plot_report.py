"""Generate individual PNG/PDF figures, an HTML report and CSV metric tables."""
from pathlib import Path
import csv
import html
import json
import os
import zipfile
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

R=json.loads((OUT/'report.json').read_text());Q=R['summary'];Z=R['smoothness']
A=np.load(OUT/'plot_data.npz')
PREV=ROOT/'results/bo_validation/M2_profile_latest_h4_rate15_24_20261119'
OLD_A=np.load(PREV/'plot_data.npz');OLD=R['previous_unblended24']
FIG=OUT/'individual_graphs';FIG.mkdir(exist_ok=True);CAT=[]
C=dict(raw='#85899b',blend='#137e91',success='#268c5a',failure='#d05b50',timeout='#d69c29',old='#a488b3')
AT=np.degrees(2*np.arcsin(np.sqrt(.04)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42,'savefig.facecolor':'white'})

def readcsv(name):
    with (OUT/name).open() as s:rows=list(csv.DictReader(s))
    for r in rows:
        for k,v in r.items():
            if v in ['True','False']:r[k]=v=='True'
            elif v=='':r[k]=None
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows
E=readcsv('episodes.csv');S=readcsv('steps.csv');B=readcsv('bridge_actions.csv')

def new(title,subtitle=None,size=(9.5,5.3)):
    f,ax=plt.subplots(figsize=size);f.suptitle(title,x=.10,ha='left',y=.98,fontweight='bold')
    if subtitle:f.text(.10,.92,subtitle,fontsize=9,color='#77623c')
    f.subplots_adjust(top=.84 if subtitle else .88,bottom=.18,left=.11,right=.96)
    return f,ax
def note(f,text):f.text(.11,.035,text,fontsize=8,color='#566071')
def save(f,name,caption):
    for suffix in ['png','pdf']:f.savefig(FIG/(name+'.'+suffix),dpi=180)
    plt.close(f);CAT.append(dict(name=name,caption=caption))
def cdf(ax,x,label,color,**kw):
    x=np.asarray(x).ravel();x=np.sort(x[np.isfinite(x)]);ax.plot(x,np.arange(1,len(x)+1)/len(x),label=label,color=color,**kw);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Cumulative fraction')
def maxdeg(x):return np.degrees(np.max(abs(x),axis=1))
def ep(key,i):return A[f'ep{i}_{key}']
def pct(x):return f'{100*x:.2f}%'
def nfmt(x):return f'{x:.2f}'

def outcomes():
    f,ax=new('Latest 12 episodes: success and drop rates')
    bars=ax.bar(['Success','Drop','Timeout'],[7/12*100,5/12*100,0],color=[C['success'],C['failure'],C['timeout']])
    for b,k in zip(bars,[7,5,0]):ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{k}/12  ({k/12*100:.1f}%)',ha='center')
    ax.set(ylabel='Completed episodes (%)',ylim=(0,75));note(f,'All 12 completed episodes included. No aborted episodes in this session.')
    save(f,'01_success_rate','7 successes, 5 drops and no timeouts. Success Wilson 95% interval: 32.0–80.7%.')
    f,ax=new('Observed success rate versus preceding 24 episodes','Two task-cost weights changed too: w_joint 0.10 → 0.15; w_pos_term 100 → 120.')
    for i,(r,k,color) in enumerate([(OLD,'Unblended\n24 episodes',C['old']),(Q,'Blended α=0.6\n12 episodes',C['blend'])]):
        lo,hi=np.array(r['wilson95'])*100;p=r['success_rate']*100
        ax.errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt='o',capsize=7,ms=10,color=color);ax.text(i,p+3,f"{r['successes']}/{r['episodes']} ({p:.1f}%)",ha='center')
    ax.set(xticks=[0,1],xticklabels=['Unblended, earlier weights\n24 episodes','Blended, updated weights\n12 episodes'],xlim=(-.5,1.5),ylim=(0,100),ylabel='Success rate (%)')
    note(f,'Wilson 95% intervals. Observational comparison; this does not isolate blending’s effect on success.')
    save(f,'02_success_vs_previous','Success rose from 16.7% to 58.3% with the combined blending/weight changes; small unequal cohorts.')
    f,ax=new('Outcomes and number of actions for each episode')
    for outcome,label in [('success','Success'),('failure','Drop')]:
        r=[e for e in E if e['outcome']==outcome];ax.bar([e['analysis_episode'] for e in r],[e['steps'] for e in r],color=C[outcome],label=label)
    ax.set(xlabel='Episode',ylabel='Published actions',xticks=range(1,13),ylim=(0,1000));ax.legend();save(f,'03_episode_outcomes','Successful episodes: 4, 5, 6, 9, 10, 11, 12. Dropped episodes: 1, 2, 3, 7, 8.')
    f,ax=new('Success by initial ±90° rotation')
    axes=['-X','+X','-Y','+Y','-Z','+Z'];s=np.array([Q['by_direction'][x]['successes'] for x in axes]);ax.bar(axes,s,color=C['success'],label='Success');ax.bar(axes,2-s,bottom=s,color=C['failure'],label='Drop')
    for i,k in enumerate(s):ax.text(i,2.08,f'{k}/2',ha='center')
    ax.set(ylabel='Completed episodes',ylim=(0,2.45),yticks=[0,1,2]);ax.legend(ncol=2);note(f,'Nearest signed initial rotation axis relative to the identity goal; not printed cube face identification.')
    save(f,'04_success_by_starting_rotation','Two episodes per rotation. −X and +Y failed both attempts; +X succeeded once; −Y and ±Z succeeded twice.')

def smoothness():
    subtitle='Same recorded states and plans; raw traces are conditional alternatives, not a separate hardware run.'
    f,ax=new('Incremental-action continuity before and after blending',subtitle)
    cdf(ax,A['raw_delta_jump_deg'],'Raw MPPI deltas',C['raw']);cdf(ax,A['blend_delta_jump_deg'],'Published blended deltas',C['blend'])
    ax.set_xlabel('Largest joint delta change between consecutive replans (degrees)');ax.legend();note(f,'First actions are excluded from consecutive-action differences. Blending occurs before hardware clipping.')
    save(f,'05_delta_continuity','Consecutive delta-change p95: raw 16.65° → blended 10.09°; mean 10.97° → 6.56°.')
    f,ax=new('Actual bridge target continuity versus raw conditional command',subtitle)
    cdf(ax,maxdeg(A['raw_bridge_changes']),'If raw delta were sent: same bridge base and prior target',C['raw'])
    cdf(ax,maxdeg(A['actual_bridge_changes']),'Actual blended command written to motors',C['blend'])
    ax.axvline(10,color='#9c5a4f',ls=':',label='10° reference');ax.set_xlabel('Largest target change over the 16 joints (degrees)');ax.legend(fontsize=8)
    note(f,'All 3,426 published actions matched to bridge writes; both command options receive the same ±0.2 rad clip.')
    save(f,'06_bridge_target_smoothness','Paired same-state target jump p95: 12.03° → 8.95°. Fraction above 10°: 26.74% → 1.96%.')
    f,ax=new('Paired target-jump change from blending',subtitle)
    raw=maxdeg(A['raw_bridge_changes']);actual=maxdeg(A['actual_bridge_changes']);mx=max(raw.max(),actual.max())
    h=ax.hexbin(raw,actual,gridsize=45,mincnt=1,bins='log',cmap='viridis');f.colorbar(h,ax=ax,label='Commands (log color scale)');ax.plot([0,mx],[0,mx],ls='--',color='#c65c4b',label='No change')
    ax.set(xlabel='Conditional raw target jump (degrees)',ylabel='Actual blended target jump (degrees)');ax.legend(fontsize=8)
    note(f,f"Blending reduces the largest target jump in {pct(Z['paired_bridge_smaller_fraction'])} of commands; increases it in {pct(Z['paired_bridge_larger_fraction'])}.")
    save(f,'07_paired_target_jumps','A smaller target jump is typical but not guaranteed on every action.')
    f,ax=new('Conditional command-change energy at the bridge',subtitle)
    for key,label,color in [('raw_bridge_changes','Raw conditional command',C['raw']),('actual_bridge_changes','Actual blended command',C['blend'])]:cdf(ax,np.sum(A[key]**2,axis=1),label,color)
    ax.set_xlabel('Sum of squared target changes across 16 joints (rad²)');ax.legend();note(f,'Relative to the same previous held target; not the full-horizon action-rate cost scored by MPPI.')
    save(f,'08_command_change_energy','Mean squared target-change energy: 0.08854 → 0.04231 rad² (52.2% lower).')
    f,ax=new('Clipping before versus after temporal blending',subtitle)
    vals=[Z['raw_clipped_action_fraction']*100,Z['blended_clipped_action_fraction']*100]
    bars=ax.bar(['Raw MPPI output','Blended output'],vals,color=[C['raw'],C['blend']]);ax.bar_label(bars,fmt='%.2f%%',padding=4)
    ax.set(ylabel='Actions with at least one clipped joint (%)',ylim=(0,20));note(f,'Same observations, same planner, same ±0.2 rad limit. Only the post-planning delta blending differs.')
    save(f,'09_clipping_intervention','Clipped actions: 15.59% raw conditional → 3.27% actual blended; individual-joint values: 1.11% → 0.22%.')
    f,ax=new('Command smoothness by episode')
    for k,label,color in [('raw_delta_jump_p95_deg','Raw delta changes: p95',C['raw']),('blend_delta_jump_p95_deg','Blended delta changes: p95',C['blend']),('estimated_target_jump_p95_deg','Estimated absolute target changes: p95','#b78836')]:
        ax.plot(range(1,13),[e[k] for e in E],'o-',label=label,color=color,ms=4)
    ax.set(xlabel='Episode',ylabel='Largest joint change (degrees), 95th percentile',xticks=range(1,13));ax.legend(fontsize=8)
    save(f,'10_smoothness_by_episode','Shows whether the smoothing behavior is consistent across the 12 episodes; delta and target changes are distinct quantities.')
    # Use a predeclared complete successful episode, not a maximum-benefit window.
    i=12;joints=[0,4,8,14];f,axs=plt.subplots(4,1,figsize=(10,9),sharex=True)
    f.suptitle('Episode 12: raw and blended action deltas',x=.10,ha='left',fontweight='bold');t=ep('sample_t',i);mask=t<=6
    for ax,j in zip(axs,joints):
        ax.plot(t[mask],np.degrees(ep('raw',i)[mask,j]),color=C['raw'],label='Raw MPPI delta',lw=1.2)
        ax.plot(t[mask],np.degrees(ep('blend',i)[mask,j]),color=C['blend'],label='Published blended delta',lw=1.4)
        ax.set_ylabel(str(A['joint_names'][j])+'\nDelta (°)');ax.axhline(0,color='#999',lw=.6)
    axs[0].legend(fontsize=8,ncol=2);axs[-1].set_xlabel('Manipulation wall time (s)');f.subplots_adjust(top=.94,bottom=.09,left=.12,hspace=.16)
    f.text(.12,.025,'First six seconds, one joint per finger. These deltas are added to measured joints; they are not absolute targets.',fontsize=8)
    save(f,'11_delta_time_traces','Episode 12, first six seconds; raw oscillations are visibly attenuated without changing MPPI noise sigma=0.05.')
    f,axs=plt.subplots(4,1,figsize=(10,9),sharex=True);f.suptitle('Episode 12: absolute target and measured joint response',x=.10,ha='left',fontweight='bold')
    for ax,j in zip(axs,joints):
        ax.plot(t[mask],np.degrees(ep('raw_target',i)[mask,j]),color=C['raw'],label='Conditional unblended target',lw=.9,alpha=.65)
        ax.plot(t[mask],np.degrees(ep('target',i)[mask,j]),color=C['blend'],label='Blended target estimate',lw=1.3)
        ax.plot(t[mask],np.degrees(ep('measured',i)[mask,j]),color='#b28135',label='Measured at planning start',lw=1.2)
        ax.set_ylabel(str(A['joint_names'][j])+'\nPosition (°)')
    axs[0].legend(fontsize=8,ncol=3);axs[-1].set_xlabel('Manipulation wall time (s)');f.subplots_adjust(top=.94,bottom=.10,left=.12,hspace=.16)
    f.text(.12,.025,'Recorded measurements are from the blended experiment. A hypothetical unblended physical response is not inferred.',fontsize=8)
    save(f,'12_targets_and_measured_motion','Recorded-state traces expose residual target-following lag; smoothing does not imply exact tracking.')

def alignment():
    f,ax=new('Which old horizon row was actually blended?')
    slots=R['old_slot_counts'];n=sum(slots.values());vals=[slots.get('0',0),slots.get('1',0)]
    bars=ax.bar(['Old row 0\nage <64 ms','Old row 1\nage 64–128 ms'],np.array(vals)/n*100,color=[C['blend'],'#b28135'])
    for b,k in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{k:,} ({k/n*100:.1f}%)',ha='center')
    ax.set(ylabel='Overlapping plans (%)',ylim=(0,108));note(f,'3,413 overlapping plans; 12 first plans bypassed blending; one history expired after a 285.7 ms gap.')
    save(f,'13_temporal_alignment_slots','92.1% use old row 0. Time alignment is based on elapsed publication time rather than always shifting one row.')
    f,ax=new('Old-plan age at blending')
    cdf(ax,A['age_ms'],'Observed age since previous publication',C['blend']);ax.axvline(64,color='#ad6f35',ls='--',label='One nominal control slot: 64 ms');ax.axvline(256,color='#b24f47',ls=':',label='Four-slot horizon expires: 256 ms')
    ax.set_xlabel('Age (ms)');ax.legend(fontsize=8);save(f,'14_old_plan_age','One expired plan uses the fresh delta unchanged; first plans have no old-plan age.')
    f,ax=new('Distance from the time-aligned old action')
    cdf(ax,A['aligned_raw_l2'],'Fresh raw action minus aligned old action',C['raw']);cdf(ax,A['aligned_blend_l2'],'Blended action minus aligned old action',C['blend'])
    ax.set_xlabel('L2 norm across 16 joint deltas (radians)');ax.legend(fontsize=8)
    note(f,'The 40% reduction is guaranteed by α=0.6; it alone is not evidence of improved physical manipulation.')
    save(f,'15_aligned_disagreement','This diagnostic verifies the blend formula; it is distinct from consecutive executed-target discontinuity.')
    f,ax=new('Raw within-plan variation and between-plan disagreement')
    cdf(ax,A['within_l2'],'Within raw MPPI plan: adjacent horizon rows',C['raw']);cdf(ax,A['aligned_raw_l2'],'Between raw plans: aligned first action','#a488b3');cdf(ax,A['aligned_blend_l2'],'Blended first action vs aligned old action',C['blend'])
    ax.set_xlabel('L2 difference across 16 joint deltas (radians)');ax.legend(fontsize=8)
    note(f,'Only the first command is blended. Raw sampled/optimized trajectories and shifted warm starts stay unchanged.')
    save(f,'16_within_and_between_plan_variation','Blending smooths the command selected for hardware; it does not smooth the entire optimized horizon.')
    f,ax=new('Measured computation time for blending')
    cdf(ax,A['blend_compute_us'],'blending.prepare()',C['blend']);ax.set_xlabel('Compute time (microseconds)');ax.legend()
    note(f,'This measures the array blend only; diagnostic formatting and JSON serialization are not timed separately.')
    save(f,'17_blending_compute_overhead','Mean 23.0 µs, p95 30.4 µs, versus mean planning 58.2 ms; about 0.04% of a control cycle.')

def physical_and_timing():
    f,ax=new('Measured motor speeds versus preceding unblended batch','Observed physical motion; the two changed task weights also affect this comparison.')
    cdf(ax,np.degrees(OLD_A['current_joint_speed']),'Earlier unblended 24 episodes',C['old']);cdf(ax,A['measured_joint_speed_deg_s'],'Latest blended 12 episodes',C['blend'])
    ax.set_xlabel('Absolute measured joint velocity (degrees/s, all joints)');ax.legend(fontsize=8)
    save(f,'18_measured_joint_speed','Mean measured absolute speed: 38.12°/s earlier → 30.37°/s now; p95 103.05°/s → 86.56°/s.')
    f,ax=new('Planning time and action publication interval')
    cdf(ax,A['planning_ms'],'Planning',C['blend']);cdf(ax,A['gaps_ms'],'Publication intervals','#b28135');ax.axvline(64,color='#ad5650',ls='--',label='64 ms nominal period')
    ax.set_xlabel('Duration (ms)');ax.legend(fontsize=8);note(f,'Unpaced; no intentional waits. One publication gap reaches 285.9 ms, beyond the maximum planning call.')
    save(f,'19_timing_distribution','Planning mean/p95 58.21/64.50 ms; pooled control 16.99 Hz. 6.30% of planning calls exceed 64 ms.')
    f,ax=new('Per-episode effective control frequency')
    ax.bar(range(1,13),[e['control_hz'] for e in E],color=[C[e['outcome']] for e in E]);ax.axhline(15.625,color='#777',ls='--',label='Nominal 15.625 Hz')
    ax.set(xlabel='Episode',ylabel='Actions / manipulation seconds (Hz)',xticks=range(1,13),ylim=(0,21));ax.legend();save(f,'20_episode_control_frequency','Per-episode frequencies: 16.43–18.73 Hz. Green successful; red dropped.')
    for key,title,unit,limit,num in [('position','Position error','mm',20,21),('angle','Orientation error','degrees',AT,22)]:
        f,ax=new(title+' over manipulation time',size=(10,5.3))
        for i,e in enumerate(E,1):ax.plot(ep('t',i),ep(key,i),color=C[e['outcome']],alpha=.65,lw=1)
        ax.axhline(limit,color='#a44e44',ls='--',label=f'Success threshold {limit:.2f} {unit}');ax.set(xlabel='Manipulation wall time (s)',ylabel=title+f' ({unit})');ax.legend(fontsize=8)
        note(f,'Green: eventual success; red: eventual drop. Curves stop at episode end; no padding with held terminal values.')
        save(f,f'{num:02d}_'+key+'_error','All 12 trajectories, including terminal measurements; successful endpoints are not averaged with ongoing episodes.')

def table(rows,headers):return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'

def report():
    metrics=[]
    def add(name,old,new):metrics.append(dict(metric=name,previous_unblended24=old,latest_blended12=new))
    for k in ['episodes','successes','drops','timeouts','steps','effective_hz','total_manipulation_s']:add(k,OLD[k],Q[k])
    for k in ['success_rate','clipped_action_fraction','clipped_joint_fraction']:add(k+'_percent',100*OLD[k],100*Q[k])
    for key in ['planning_ms','publication_interval_ms','instantaneous_hz','episode_hz']:
        for stat in ['min','mean','median','p95','p99','max']:add(key+'.'+stat,OLD['timing'][key][stat],Q['timing'][key][stat])
    for key in ['initial_pos_mm','initial_angle_deg','final_pos_mm','final_angle_deg','success_final_position_mm','success_final_orientation_deg','success_duration_s']:
        for stat in ['mean','median']:add(key+'.'+stat,OLD[key][stat],Q[key][stat])
    for key in ['orientation_ever','both_pose_ever']:add(key,OLD[key],Q[key])
    for key in ['actual_bridge_jump_deg','conditional_raw_bridge_jump_deg','blend_compute_us','raw_delta_jump_deg','blend_delta_jump_deg','holding_error_deg','publish_to_write_ms','measured_joint_speed_deg_s']:
        for stat in ['mean','median','p95','max']:add(key+'.'+stat,'not extracted here',Z[key][stat])
    for k in ['raw_clipped_action_fraction','blended_clipped_action_fraction','paired_bridge_smaller_fraction','actual_jump_above10_fraction']:add(k+'_percent','not applicable',100*Z[k])
    with (OUT/'metrics.csv').open('w',newline='') as s:
        w=csv.DictWriter(s,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><title>M2 temporal blending — 12 episodes</title>
    <style>body{font:16px/1.6 system-ui;max-width:1160px;margin:35px auto;padding:0 22px;background:#fafbfd;color:#253247}a{color:#08778b}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px;margin:20px 0}th,td{padding:7px 12px;border-bottom:1px solid #d8dde5;text-align:left}th{background:#e9edf3}img{width:100%;max-width:960px}figure{margin:25px 0 45px}figcaption{font-size:14px;color:#566171}.box{padding:15px 22px;background:#eaf4f6;border-left:4px solid #137e91}.warn{background:#fcf2df;border-color:#b58a38}pre{white-space:pre-wrap;background:#edf0f4;padding:12px}</style>
    <h1>M2 temporal blending: latest 12 completed episodes</h1>
    <p>Source: <code>leap_experiment_20261119_034105_014853</code>. All 12 episodes completed; no aborted episodes excluded. All 3,426 actions matched to actual bridge writes.</p>
    <div class="box"><strong>7/12 successes (58.3%), 5 drops, 0 timeouts.</strong><br>Actual bridge target-jump p95 8.95°, compared with 12.03° for the same-state raw alternative. Planning mean 58.21 ms; pooled control 16.99 Hz.</div>
    <p><a href="graphs_png.zip">All individual PNG graphs</a> · <a href="graphs_pdf.zip">All PDF figures</a> · <a href="metrics.csv">Metric table CSV</a> · <a href="episodes.csv">Episode CSV</a> · <a href="steps.csv">Per-command blending metrics</a> · <a href="bridge_actions.csv">Bridge command comparisons</a> · <a href="report.json">Complete statistics and configuration</a> · <a href="manifest.json">Sources and hashes</a> · <a href="validation.json">Validation</a></p>
    <h2>Configuration and comparison limits</h2>
    <p>M2, calibrated hand, high/high geometry, cube mass 0.07 kg, friction cube/tips/palm/other fingers 0.3/0.5/0.4/0.4, 512 samples, one iteration, horizon 0.256 s (4 slots), nominal timestep 0.064 s, shifted warm start, noise 0.05, temperature 0.1, action-rate penalty 15, temporal blend α=0.6, latest Method-5 calibration, P011 joint-cost reference, measured-position delta rebasing, ±0.2 rad hardware clip, unpaced. Goal [25,35,75] mm, fallen-z 65 mm. Success: position &lt;20 mm, quaternion metric &lt;0.04 (23.07°), plus configured velocity check.</p>
    <div class="box warn">The previous unblended 24 episodes achieved 4/24 (16.7%), but <strong>w_joint changed from 0.10 to 0.15 and w_pos_term from 100 to 120</strong>. Matching scene physics, goal, preparation, camera calibration and clipping were verified from the logs. The higher success rate is evidence for the combined settings, not proof that blending alone caused the gain.</div>
    <h2>What blending changed directly</h2>
    <p>The implemented rule mixes <strong>60% fresh raw delta + 40% old time-aligned raw delta</strong> before hardware clipping. It only changes the command sent; the raw optimized horizon, sampling noise and shifted warm start are not modified. Old history stores the previous <em>raw</em> plan, not a recursively blended plan.</p>''']
    before=Z['conditional_raw_bridge_jump_deg'];after=Z['actual_bridge_jump_deg']
    paired=[['Maximum joint-target jump: mean',f"{before['mean']:.2f}°",f"{after['mean']:.2f}°"],['Maximum joint-target jump: p95',f"{before['p95']:.2f}°",f"{after['p95']:.2f}°"],['Commands with target jump >10°',pct(Z['conditional_raw_jump_above10_fraction']),pct(Z['actual_jump_above10_fraction'])],['Mean target-change squared norm',f"{Z['conditional_raw_bridge_energy_rad2']['mean']:.5f} rad²",f"{Z['actual_bridge_energy_rad2']['mean']:.5f} rad²"],['Actions with hardware clipping',pct(Z['raw_clipped_action_fraction']),pct(Z['blended_clipped_action_fraction'])],['Consecutive delta-change p95',f"{Z['raw_delta_jump_deg']['p95']:.2f}°",f"{Z['blend_delta_jump_deg']['p95']:.2f}°"]]
    parts.append(table(paired,['Same-state diagnostic','Raw conditional command','Actual blended command']))
    parts.append(f'''<p>The target-jump p95 reduction is <strong>{100*(1-after['p95']/before['p95']):.1f}%</strong>; mean target-change squared norm is <strong>{100*(1-Z['actual_bridge_energy_rad2']['mean']/Z['conditional_raw_bridge_energy_rad2']['mean']):.1f}% lower</strong>. Blending reduced the largest joint-target jump on <strong>{pct(Z['paired_bridge_smaller_fraction'])}</strong> of individual commands. These paired comparisons reuse the same actual bridge application base and actual previously held target, including float32 LCM conversion and the same clipping.</p>
    <p><strong>The raw alternative is conditional.</strong> It is not an unblended hardware trajectory: without blending, subsequent measured states, plans and object outcomes would change. No counterfactual success rate or physical motor velocity is claimed.</p>
    <h2>Time alignment and overhead</h2>
    <p>Of 3,413 overlapping replans, 3,143 (92.1%) used old horizon row 0 and 270 (7.9%) row 1. The code selects <code>floor(age / 64 ms)</code>. Because actual replanning commonly took less than 64 ms, the current implementation usually mixes consecutive first actions, rather than always the old second action. There were 12 first-plan bypasses and one expired history at age 285.7 ms; those actions used the fresh delta.</p>
    <p>Distance from the aligned old delta is exactly 60% of the fresh-plan disagreement by construction. That formula check is separate from the empirical reduction in consecutive command jumps. Within-plan adjacent rows remain raw and unchanged.</p>
    <p>Measured blend-array computation averaged <strong>{Z['blend_compute_us']['mean']:.1f} µs</strong>, p95 <strong>{Z['blend_compute_us']['p95']:.1f} µs</strong>, about 0.04% of the control cycle. This timer excludes separately unmeasured diagnostic formatting/JSON work. Overall planning averaged 58.21 ms (p95 64.50 ms), with 216/3426 calls (6.30%) over 64 ms. No intentional pacing waits occurred. Effective control averaged 16.99 Hz; per-episode means 16.43–18.73 Hz; instantaneous 3.50–30.61 Hz, including one 285.9 ms publication gap.</p>
    <h2>Physical motion and remaining failures</h2>
    <p>Measured absolute joint speed averaged <strong>30.37°/s</strong>, p95 <strong>86.56°/s</strong>, versus 38.12°/s and 103.05°/s in the preceding unblended 24 episodes. Physical motion was slower in this batch, with the task-weight changes and episode mix also contributing. Bridge target-following absolute error averaged 2.09°, p95 5.65°, versus 2.67° / 7.20° previously; these include motion/load effects rather than settled accuracy.</p>
    <p>Two episodes were run for each ±90° starting rotation. −X and +Y produced 0/2 successes each; +X 1/2; −Y, −Z and +Z 2/2 each. Thus four of the five drops occurred in the −X/+Y groups, though two samples per group are too few for a firm directional conclusion. Initial orientation errors were 87.51–93.35°. Nine episodes entered the orientation threshold at least once, while seven met the full success criteria. Orientation-only entries in dropped episodes do not establish a stable grasp.</p>
    <p>Successful episodes took a mean 14.77 s (median 12.74 s). Successful final errors averaged 15.51 mm and 20.44°. The controller was interrupted after 12 completed episodes; bridge cleanup logged an action watchdog timeout and verified torque off on all 16 motors. There are no aborted episodes in this selected session.</p>
    <h2>Interpretation</h2><p>Blending demonstrably reduces command discontinuity and clipping on the recorded states, at very small measured array-computation cost. The combined configuration is promising compared with the preceding 24-episode batch. A matched run with these same weights and α=1 would isolate whether that improvement in command continuity translates into more reliable object manipulation. No production settings, hardware, tracking or simulation were changed or run during this analysis.</p>
    <h2>Individual graphs</h2>''')
    for c in CAT:
        n=c['name'];parts.append(f'<figure><a href="individual_graphs/{n}.png"><img loading="lazy" src="individual_graphs/{n}.png" alt="{html.escape(n)}"></a><figcaption>{html.escape(c["caption"])} <a href="individual_graphs/{n}.pdf">PDF</a></figcaption></figure>')
    parts.append('<h2>All metrics</h2>'+table([[f'{v:.5f}' if isinstance(v,float) else v for v in r.values()] for r in metrics],['Metric','Earlier unblended 24','Latest blended 12']))
    keys=['analysis_episode','direction','outcome','steps','duration_s','final_pos_mm','final_angle_deg','clipped_action_fraction','control_hz']
    parts.append('<h2>All 12 episodes</h2>'+table([[f'{e[k]:.3f}' if isinstance(e[k],float) else e[k] for k in keys] for e in E],keys))
    parts.append('<h2>Reproduce offline</h2><pre>python '+str(OUT.relative_to(ROOT)/'analyze.py')+'\npython '+str(OUT.relative_to(ROOT)/'plot_report.py')+'</pre></html>')
    (OUT/'index.html').write_text('\n'.join(parts));(OUT/'graph_catalog.json').write_text(json.dumps(CAT,indent=2)+'\n')
    for suffix in ['png','pdf']:
        with zipfile.ZipFile(OUT/f'graphs_{suffix}.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for c in CAT:z.write(FIG/(c['name']+'.'+suffix),arcname=c['name']+'.'+suffix)
    print('Created',len(CAT),'individual PNG + PDF figures and',len(metrics),'metrics.')

if __name__=='__main__':
    outcomes();smoothness();alignment();physical_and_timing();report()
