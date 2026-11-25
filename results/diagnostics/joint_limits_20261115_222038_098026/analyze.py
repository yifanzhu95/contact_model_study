"""Inspect archived measured hardware positions against embedded run-time MJCF.

Read-only data analysis. No hardware, camera, tracker or simulation is started.
Primary denominator: one pre-planning measured q per published action. Other
state streams, terminal samples and preparation samples are kept separate.
"""
from pathlib import Path
from collections import Counter
import csv,hashlib,html,json,os,zipfile
import xml.etree.ElementTree as ET
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ZIP=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial314_shift512_paced_video/leap_experiment_20261115_222038_098026_logs_only.zip'
EPS=1e-6
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as s:
        w=csv.DictWriter(s,fieldnames=keys);w.writeheader();w.writerows(rows)
def ranges(xml,names):
    root=ET.fromstring(xml);defaults={x.get('class'):x for x in root.findall('.//default') if x.get('class')};jr=[];cr=[]
    for name in names:
        joint=root.find(f".//worldbody//joint[@name='{name}']");default=defaults.get(joint.get('class'))
        rg=joint.get('range')
        if rg is None:rg=default.find('joint').get('range')
        actuator=next(a for a in root.find('actuator') if a.get('joint')==name);ctrl=actuator.get('ctrlrange')
        if ctrl is None:ctrl=next(x.get('ctrlrange') for x in defaults[actuator.get('class')] if x.tag in ['general','position'])
        jr.append(list(map(float,rg.split())));cr.append(list(map(float,ctrl.split())))
    return np.array(jr),np.array(cr)
def outside(a,b):return (a<b[:,0]-EPS)|(a>b[:,1]+EPS)
def stats(a,b,weights=None):
    lo=a<b[:,0]-EPS;hi=a>b[:,1]+EPS;m=lo|hi;depth=np.maximum(np.maximum(b[:,0]-a,a-b[:,1]),0.)
    result=[]
    for j in range(16):
        r=dict(lower_rad=float(b[j,0]),upper_rad=float(b[j,1]),measured_min_rad=float(a[:,j].min()),measured_max_rad=float(a[:,j].max()),
            below_count=int(lo[:,j].sum()),above_count=int(hi[:,j].sum()),outside_count=int(m[:,j].sum()),outside_fraction=float(m[:,j].mean()),
            max_violation_rad=float(depth[:,j].max()),max_violation_deg=float(np.degrees(depth[:,j].max())))
        if weights is not None:r.update(estimated_outside_s=float(weights[m[:,j]].sum()),estimated_outside_time_fraction=float(weights[m[:,j]].sum()/weights.sum()))
        result.append(r)
    return result

def main():
    with zipfile.ZipFile(ZIP) as z:
        prefix=next(n.removesuffix('/controller.json') for n in z.namelist() if n.endswith('/controller.json'));raw=z.read(prefix+'/controller.json');d=json.loads(raw)
        preparation=[json.loads(x) for x in z.read(prefix+'/preparation_trace.jsonl').splitlines()]
        bridge_log=z.read(prefix+'/bridge.log').decode();archive_files=z.namelist()
    names=d['joint_order'];jr,cr=ranges(d['scenes']['rollout']['scene_xml'],names)
    je,ce=ranges(d['scenes']['eval']['scene_xml'],names);np.testing.assert_allclose(jr,je);np.testing.assert_allclose(cr,ce)
    # model_ctrl_range is an intersection, not the raw MJCF actuator range.
    intersection=np.column_stack((np.maximum(jr[:,0],cr[:,0]),np.minimum(jr[:,1],cr[:,1])))
    np.testing.assert_allclose(intersection[:,0],d['model_ctrl_range']['lower']);np.testing.assert_allclose(intersection[:,1],d['model_ctrl_range']['upper'])
    q=[];post=[];u=[];planned=[];requested=[];deltas=[];weights=[];locations=[];episode_rows=[];spells=[];terminal=[];timelines=[]
    for e in d['episodes']:
        assert e['completed'];ss=[s for s in e['steps'] if s.get('action_published')];assert len(ss)==e['n_steps']
        a=np.array([s['q'][:16] for s in ss]);p=np.array([s['feedback']['command_qpos'] for s in ss]);cmd=np.array([s['u'] for s in ss]);pl=np.array([s['planned_u'] for s in ss]);rq=np.array([s['requested_target'] for s in ss]);de=np.array([s['action_delta'] for s in ss])
        t=np.array([s['elapsed_s'] for s in ss]);end=e['manipulation_wall_time_s'];dt=np.diff(np.r_[t,end]);assert np.all(dt>=0)
        q.extend(a);post.extend(p);u.extend(cmd);planned.extend(pl);requested.extend(rq);deltas.extend(de);weights.extend(dt);terminal.append(e['q_end'][:16])
        m=outside(a,jr);ep=e['episode']+1;mf=a[:,4];bad=mf<jr[4,0]-EPS
        for k,s in enumerate(ss):
            locations.append(dict(episode=ep,logged_step=s['step'],elapsed_s=float(t[k]),snapshot_utc=s['timestamp_utc'],robot_timestamp_us=s['feedback']['robot_timestamp'],command_robot_timestamp_us=s['feedback']['command_state']['robot_timestamp']))
        for j,name in enumerate(names):
            mask=m[:,j];edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            for first,last in zip(starts,ends):
                segment=a[first:last,j];spells.append(dict(episode=ep,joint=name,start_logged_step=ss[first]['step'],last_logged_step=ss[last-1]['step'],start_elapsed_s=float(t[first]),end_elapsed_s=float(t[last]) if last<len(t) else end,
                    consecutive_records=int(last-first),estimated_duration_s=float(dt[first:last].sum()),min_rad=float(segment.min()),max_rad=float(segment.max()),start_robot_timestamp_us=ss[first]['feedback']['robot_timestamp'],last_robot_timestamp_us=ss[last-1]['feedback']['robot_timestamp']))
        episode_rows.append(dict(episode=ep,outcome=e['finish_reason'],steps=len(ss),manipulation_s=end,
            any_joint_outside_steps=int(m.any(axis=1).sum()),mf_mcp_below_zero_steps=int(bad.sum()),mf_mcp_below_zero_percent=float(bad.mean()*100),
            mf_mcp_below_minus0314_steps=int((mf<cr[4,0]-EPS).sum()),mf_mcp_below_minus0314_percent=float((mf<cr[4,0]-EPS).mean()*100),
            mf_mcp_min_rad=float(mf.min()),mf_mcp_min_deg=float(np.degrees(mf.min())),mf_mcp_max_rad=float(mf.max()),mf_mcp_start_rad=float(mf[0]),mf_mcp_terminal_rad=float(e['q_end'][4]),
            mf_mcp_estimated_negative_s=float(dt[bad].sum()),mf_mcp_estimated_negative_time_percent=float(dt[bad].sum()/dt.sum()*100),
            mf_mcp_target_below_zero_steps=int((cmd[:,4]<-EPS).sum()),mf_mcp_target_below_actuator_min_steps=int((cmd[:,4]<cr[4,0]-EPS).sum())))
        timelines.append(dict(episode=ep,outcome=e['finish_reason'],t=np.r_[t,end],measured=np.r_[mf,e['q_end'][4]],target_t=t,estimated_target=cmd[:,4],planned_target=pl[:,4]))
    q=np.array(q);post=np.array(post);u=np.array(u);planned=np.array(planned);requested=np.array(requested);deltas=np.array(deltas);weights=np.array(weights);terminal=np.array(terminal)
    assert len(q)==d['summary']['mppi_time']['count']==1875
    joint_stats=stats(q,jr,weights);post_stats=stats(post,jr);target_joint=stats(u,jr);target_act=stats(u,cr);measured_act=stats(q,cr);planned_joint=stats(planned,jr)
    target_residual=np.max(abs(u-(post+np.clip(deltas,-.2,.2))))
    assert target_residual<1e-10
    flat=[]
    for j,name in enumerate(names):
        flat.append(dict(joint=name,**joint_stats[j],actuator_lower_rad=float(cr[j,0]),actuator_upper_rad=float(cr[j,1]),
            measured_min_deg=float(np.degrees(q[:,j].min())),measured_max_deg=float(np.degrees(q[:,j].max())),
            episodes_outside=int(sum(outside(np.array([s['q'][:16] for s in e['steps'] if s.get('action_published')]),jr)[:,j].any() for e in d['episodes'])),
            post_plan_outside_count=post_stats[j]['outside_count'],post_plan_outside_fraction=post_stats[j]['outside_fraction'],
            estimated_target_min_rad=float(u[:,j].min()),estimated_target_max_rad=float(u[:,j].max()),
            estimated_targets_outside_joint_count=target_joint[j]['outside_count'],estimated_targets_outside_joint_fraction=target_joint[j]['outside_fraction'],
            estimated_targets_outside_actuator_count=target_act[j]['outside_count'],estimated_targets_outside_actuator_fraction=target_act[j]['outside_fraction'],
            measured_positions_outside_actuator_interval_count=measured_act[j]['outside_count'],max_violation_location=locations[int(np.argmax(np.maximum(jr[j,0]-q[:,j],q[:,j]-jr[j,1])))]) )
    detailed=[]
    for i,loc in enumerate(locations):
        for j,name in enumerate(names):
            if q[i,j]<jr[j,0]-EPS or q[i,j]>jr[j,1]+EPS:
                depth=max(jr[j,0]-q[i,j],q[i,j]-jr[j,1]);detailed.append(dict(**loc,joint=name,side='below' if q[i,j]<jr[j,0] else 'above',position_rad=float(q[i,j]),position_deg=float(np.degrees(q[i,j])),joint_lower_rad=float(jr[j,0]),joint_upper_rad=float(jr[j,1]),violation_rad=float(depth),estimated_target_rad=float(u[i,j])))
    mf=q[:,4];mi=int(mf.argmin());belowcounts={str(deg):dict(count=int((mf < -np.radians(deg)-EPS).sum()),fraction=float((mf < -np.radians(deg)-EPS).mean())) for deg in [0,1,5,10,15,20]}
    prep_m=np.array([p['measured'] for p in preparation]);prep_t=np.array([p['target'] for p in preparation]);prep_stats=stats(prep_m,jr)
    mfspells=[s for s in spells if s['joint']=='mf_mcp'];largest=max(mfspells,key=lambda x:x['estimated_duration_s'])
    report=dict(source_archive=str(ZIP.relative_to(ROOT)),source_controller_member=prefix+'/controller.json',source_controller_sha256=hashlib.sha256(raw).hexdigest(),
        run_summary=d['summary'],run_settings=dict(model=d['args']['rollout_model'],n_samples=d['args']['n_samples'],temperature=d['args']['temperature'],horizon_s=d['args']['time_horizon'],paced=d['args']['pace_control'],delta_range=d['planner_config']['delta_range'],action_rate_weight=d['args']['action_rate_weight'],delta_reference=d['hardware_config']['delta_reference'],ctrl_range=d['hardware_config']['ctrl_range']),
        recorded_steps=len(q),unique_preplan_robot_timestamps=len({x['robot_timestamp_us'] for x in locations}),measurement_tolerance_rad=EPS,
        joints=flat,any_joint_outside_count=int(outside(q,jr).any(axis=1).sum()),any_joint_outside_fraction=float(outside(q,jr).any(axis=1).mean()),
        mf_mcp=dict(joint_range_rad=jr[4].tolist(),actuator_command_range_rad=cr[4].tolist(),measured=joint_stats[4],post_plan_measured=post_stats[4],
            estimated_target_vs_joint_range=target_joint[4],estimated_target_vs_actuator_range=target_act[4],planned_target_vs_joint_range=planned_joint[4],
            measured_positions_outside_actuator_interval=measured_act[4],below_thresholds_degrees=belowcounts,worst_location=locations[mi],
            q_at_worst_rad=float(mf[mi]),post_plan_q_at_worst_rad=float(post[mi,4]),planned_target_at_worst_rad=float(planned[mi,4]),estimated_target_at_worst_rad=float(u[mi,4]),
            initial_reference_target_rad=d['hardware_config']['robot_qref'][4],episode_start_positions_rad=[e['mf_mcp_start_rad'] for e in episode_rows],
            terminal_positions_rad=terminal[:,4].tolist(),negative_terminal_count=int((terminal[:,4]<-EPS).sum()),violation_spells=len(mfspells),longest_spell=largest,
            preparation_snapshot_statistics=prep_stats[4],preparation_snapshots=len(preparation),preparation_stage_counts=dict(Counter(p['stage'] for p in preparation))),
        episodes=episode_rows,archive_has_full_bridge_telemetry=any(n.endswith('/bridge_telemetry.jsonl') for n in archive_files),
        limits_source='Joint ranges and actuator ctrlranges extracted from embedded rollout/eval scene XML; both matched. model_ctrl_range verified as their intersection.',
        caveats=['Percentages count 1,875 pre-planning measured q records during MPPI, not every serial bridge read. Post-planning measurements, endpoints and preparation snapshots are analyzed separately.',
            'u is the controller-estimated post-clipping hardware target, not an exact bridge write readback. No full bridge telemetry is present in this archive.',
            'Outside-time and spell durations use zero-order hold between recorded elapsed times. They estimate duty time and are not continuously measured limit-crossing durations.',
            'Actuator ctrlrange constrains simulated target commands, not measured positions; measured position outside that interval is reported for comparison, not a separate physical safety violation.',
            'These are model-range exceedances in the logged MuJoCo joint convention. Logs alone do not determine whether the model range, hardware zero offset/mapping, or physical travel should be changed.',
            'The frozen run snapshot is authoritative; current source/config files are not used to determine this historical run’s limits.'])
    csvrows=[{k:json.dumps(v) if isinstance(v,dict) else v for k,v in x.items()} for x in flat]
    csvout('joint_limits_and_exceedances.csv',csvrows);csvout('mf_mcp_by_episode.csv',episode_rows);csvout('violation_intervals.csv',spells);csvout('violating_samples.csv',detailed)
    export=[]
    for i,loc in enumerate(locations):
        export.append(dict(**loc,measured_mf_mcp_rad=float(q[i,4]),post_plan_measured_mf_mcp_rad=float(post[i,4]),planned_target_mf_mcp_rad=float(planned[i,4]),estimated_clipped_target_mf_mcp_rad=float(u[i,4]),raw_delta_mf_mcp_rad=float(deltas[i,4]),joint_lower_rad=float(jr[4,0]),joint_upper_rad=float(jr[4,1]),actuator_lower_rad=float(cr[4,0]),actuator_upper_rad=float(cr[4,1])))
    csvout('mf_mcp_samples.csv',export);dump('report.json',report)
    dump('validation.json',dict(all_1875_published_steps_included=True,rollout_eval_limits_identical=True,logged_model_ctrl_range_is_intersection=True,
        relative_clipping_recomputed_max_error_rad=float(target_residual),archive_and_loose_controller_match=hashlib.sha256(raw).digest()==hashlib.sha256((ZIP.parent/prefix/'controller.json').read_bytes()).digest(),
        source='archive contents',hardware_or_simulation_started=False))
    plots(report,q,u,planned,jr,cr,timelines)
    print('SCOPE',len(q),'steps,',len(d['episodes']),'completed episodes')
    print('JOINTS',json.dumps([{k:x[k] for k in ['joint','measured_min_rad','measured_max_rad','below_count','above_count','outside_fraction','max_violation_deg']} for x in flat if x['outside_count']]))
    print('MIDDLE_MCP',json.dumps(report['mf_mcp']))

def plots(r,q,u,planned,jr,cr,timelines):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.8));values=[x['outside_fraction']*100 for x in r['joints']];ax.bar(range(16),values,color=['#c0564f' if i==4 else '#257e93' for i in range(16)])
    ax.set(xticks=range(16),xticklabels=[x['joint'] for x in r['joints']],ylabel='Pre-planning measurements outside joint range (%)',ylim=(0,110),title='Measured hardware joint-range exceedances — 1,875 control steps');ax.tick_params(axis='x',rotation=50)
    save(fig,'01_joint_limit_exceedance_rates')
    fig,axs=plt.subplots(5,2,figsize=(13,14),sharey=True);colors={'success':'#268c5a','failure':'#cc5a50','timeout':'#b8892d'}
    for ax,x in zip(axs.ravel(),timelines):
        ax.plot(x['t'],x['measured'],color=colors[x['outcome']],label='Measured joints at planning start',lw=1.3);ax.plot(x['target_t'],x['estimated_target'],color='#778392',ls=':',lw=.9,label='Controller-estimated target')
        ax.axhline(0,color='#333',ls='--',label='Joint lower limit 0');ax.axhline(-.314,color='#9a772e',ls='--',label='Actuator command lower limit −0.314')
        ax.set(title=f"Episode {x['episode']} — {x['outcome']}",xlabel='Manipulation elapsed time (s)',ylabel='Middle MCP angle (rad)',ylim=(-.65,.3))
    h,l=axs.ravel()[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,fontsize=9);fig.suptitle('Middle MCP: measured pose and estimated commands in all 10 episodes',fontweight='bold');fig.tight_layout(rect=(0,.055,1,.975));save(fig,'02_middle_mcp_all_episodes')
    fig,ax=plt.subplots(figsize=(9.5,5))
    for x,label,color in [(q[:,4],'Measured at planning start','#257e93'),(u[:,4],'Controller-estimated clipped target','#cc7052'),(planned[:,4],'Raw planner absolute target','#85839b')]:
        a=np.sort(x);ax.plot(a,np.arange(1,len(a)+1)/len(a),label=label,color=color)
    ax.axvline(0,color='#333',ls='--',label='Model joint lower limit');ax.axvline(-.314,color='#9a772e',ls='--',label='Model actuator command lower limit');ax.set(xlabel='Middle MCP angle / target (rad)',ylabel='Cumulative fraction',title='Middle MCP measurements and targets');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8);save(fig,'03_middle_mcp_distribution')
    fig,ax=plt.subplots(figsize=(9,4.5));thresholds=[0,1,5,10,15,20];counts=[r['mf_mcp']['below_thresholds_degrees'][str(i)]['fraction']*100 for i in thresholds];bars=ax.bar(range(6),counts,color='#257e93');ax.bar_label(bars,fmt='%.1f%%',padding=3);ax.set(xticks=range(6),xticklabels=[f'> {i}° below 0' for i in thresholds],ylabel='Recorded planning-start measurements (%)',ylim=(0,113),title='How far below the model lower limit was middle MCP?');save(fig,'04_middle_mcp_violation_depth')
    def table(rows,headers):return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    m=r['mf_mcp'];text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Archived hardware joint-limit audit</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 24px;color:#253247}a{color:#08778b}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{padding:7px 10px;border-bottom:1px solid #d9dfe7;text-align:left}th{background:#edf1f5}img{width:100%}code{background:#edf1f5}li{margin:8px 0}</style><h1>Hardware joint limits: archived 2026-11-15 22:20:38 run</h1>'''
    text+=f'<p>Read directly from <code>{ZIP.name}</code>; limits come from the embedded scene XML. Ten completed episodes, 1,875 planning-start measured joint states. Middle MCP was below zero in <strong>{m["measured"]["below_count"]}/1875 ({m["measured"]["outside_fraction"]*100:.2f}%)</strong>, reaching <strong>{m["q_at_worst_rad"]:.6f} rad ({np.degrees(m["q_at_worst_rad"]):.2f}°)</strong>.</p>'
    text+='<p><a href="joint_limits_and_exceedances.csv">All-joint CSV</a> · <a href="mf_mcp_by_episode.csv">Middle MCP by episode</a> · <a href="mf_mcp_samples.csv">Timestamped middle-MCP samples</a> · <a href="violation_intervals.csv">Consecutive violation intervals</a> · <a href="violating_samples.csv">Every violating joint/sample</a> · <a href="report.json">Full statistics</a> · <a href="validation.json">Validation</a></p>'
    text+='<h2>Measured angles against frozen model limits</h2>'+table([[x['joint'],f"[{x['lower_rad']:.3f}, {x['upper_rad']:.3f}]",f"[{x['actuator_lower_rad']:.3f}, {x['actuator_upper_rad']:.3f}]",f"[{x['measured_min_rad']:.6f}, {x['measured_max_rad']:.6f}]",x['below_count'],x['above_count'],f"{x['outside_fraction']*100:.2f}%",f"{x['max_violation_deg']:.2f}°"] for x in r['joints']],['Joint','Joint range rad','Actuator target range rad','Measured min/max rad','Below','Above','Outside %','Maximum excess'])
    text+='<h2>Middle MCP</h2>'
    text+=f'<p>Actuator target range: [−0.314, 2.230] rad; joint-position range: [0, 2.230] rad. The saved <code>model_ctrl_range</code> is their intersection, [0,2.230], not the raw actuator range. Hardware final targets were bounded only within ±0.2 rad of the latest measured joints; sampled delta_range was unbounded. All 1,875 estimated commands were checked against that relative clipping rule.</p><p>The reference-preparation target itself sets middle MCP to <strong>{m["initial_reference_target_rad"]:.7f} rad ({np.degrees(m["initial_reference_target_rad"]):.2f}°)</strong>, already below the declared joint minimum. All ten episode-start measurements were negative; all ten terminal measurements were negative. This is not limited to momentary planning updates at the start.</p>'
    text+=table([[x['episode'],x['outcome'],x['steps'],x['mf_mcp_below_zero_steps'],f"{x['mf_mcp_below_zero_percent']:.2f}%",f"{x['mf_mcp_min_rad']:.6f}",f"{x['mf_mcp_max_rad']:.6f}",f"{x['mf_mcp_start_rad']:.6f}",f"{x['mf_mcp_target_below_actuator_min_steps']}/{x['steps']}"] for x in r['episodes']],['Episode','Outcome','Steps','MF below zero','Fraction','MF min rad','MF max rad','MF start rad','Estimated target <−0.314'])
    text+=f'<p>Worst MF sample: episode {m["worst_location"]["episode"]}, logged step {m["worst_location"]["logged_step"]} (zero-based), elapsed {m["worst_location"]["elapsed_s"]:.3f} s; UTC {m["worst_location"]["snapshot_utc"]}. The timestamped CSV retains the underlying robot acquisition timestamp as well.</p>'
    text+='<h2>Interpretation and limits</h2><p>Many hardware states lie outside the model’s declared joint ranges in the recorded coordinate convention. This is not explained simply by actuator commands being allowed slightly below zero: several measured MCP and rotational positions also fall outside their actuator command intervals. Actuator intervals are not physical travel constraints, however, and this audit does not identify whether zero/mapping offsets, model ranges or physical travel should be revised. Rollouts initialized from these measured states can immediately begin outside their declared joint constraints.</p><ul>'+''.join('<li>'+html.escape(c)+'</li>' for c in r['caveats'])+'</ul>'
    for name in ['01_joint_limit_exceedance_rates','02_middle_mcp_all_episodes','03_middle_mcp_distribution','04_middle_mcp_violation_depth']:text+=f'<p><a href="{name}.png"><img loading="lazy" src="{name}.png" alt="{name}"></a><a href="{name}.pdf">PDF</a></p>'
    (OUT/'index.html').write_text(text+'</html>')

if __name__=='__main__':main()
