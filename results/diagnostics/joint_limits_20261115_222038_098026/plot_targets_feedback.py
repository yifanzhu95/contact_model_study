"""Overlay archived middle-MCP targets with actual bridge feedback timestamps."""
from pathlib import Path
from datetime import datetime
import csv,hashlib,json,os,zipfile
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial314_shift512_paced_video/leap_experiment_20261115_222038_098026_logs_only.zip'
def micros(text):return round(datetime.fromisoformat(text).timestamp()*1e6)
def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    with zipfile.ZipFile(SOURCE) as z:
        member=next(n for n in z.namelist() if n.endswith('/controller.json'));payload=z.read(member);d=json.loads(payload)
    streams=[];rows=[];duplicates=0
    for e in d['episodes']:
        if not e['completed']:continue
        ep=e['episode']+1;origin=micros(e['started_at_utc']);measurements={};targets=[]
        for s in e['steps']:
            if not s.get('action_published'):continue
            for stamp,value in [(s['feedback']['robot_timestamp'],s['q'][4]),(s['feedback']['command_state']['robot_timestamp'],s['feedback']['command_qpos'][4])]:
                if stamp in measurements:
                    np.testing.assert_allclose(measurements[stamp],value,rtol=0,atol=1e-9);duplicates+=1
                measurements[stamp]=float(value)
            targets.append((micros(s['action_published_at_utc']),float(s['u'][4]),float(s['planned_u'][4]),s['step']))
        m=sorted(measurements.items());mt=np.array([(stamp-origin)*1e-6 for stamp,value in m]);mq=np.array([value for stamp,value in m]);u=np.array(targets)
        t=(u[:,0]-origin)*1e-6;end=e['manipulation_wall_time_s']
        assert np.all(np.diff(mt)>0) and np.all(np.diff(t)>0)
        streams.append(dict(episode=ep,outcome=e['finish_reason'],mt=mt,mq=mq,t=t,u=u[:,1],planned=u[:,2],end=end))
        for stamp,value in m:rows.append(dict(episode=ep,record='bridge_measurement',timestamp_us=stamp,time_from_episode_start_s=(stamp-origin)*1e-6,measured_rad=value,estimated_target_rad=None,planned_target_rad=None,logged_step=None))
        for stamp,value,planned,step in targets:rows.append(dict(episode=ep,record='controller_target',timestamp_us=stamp,time_from_episode_start_s=(stamp-origin)*1e-6,measured_rad=None,estimated_target_rad=value,planned_target_rad=planned,logged_step=step))
    rows.sort(key=lambda r:(r['episode'],r['timestamp_us'],r['record']))
    with (OUT/'mf_mcp_time_aligned.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42})
    def draw(ax,x,stop=None,degree_axis=False):
        # Plot measurements at acquisition timestamps, and each estimated target
        # at publication time. Connecting measured points is visualization only.
        tt=x['t'];qtime=x['mt'];mq=x['mq'];u=x['u'];pl=x['planned']
        if stop is not None:
            mm=qtime<=stop;tm=tt<=stop;qtime=qtime[mm];mq=mq[mm];tt=tt[tm];u=u[tm];pl=pl[tm]
        end=min(x['end'],stop) if stop is not None else x['end']
        ax.plot(qtime,mq,color='#087c9b',lw=1.7,marker='.',ms=3,label='Measured joint pose received from bridge',zorder=4)
        ax.step(np.r_[tt,end],np.r_[u,u[-1]],where='post',color='#d47728',lw=1.1,label='Controller-estimated target after ±0.2 clip',alpha=.95)
        ax.step(np.r_[tt,end],np.r_[pl,pl[-1]],where='post',color='#9296a4',lw=.8,ls=':',label='Raw planner absolute target',alpha=.8)
        ax.axhline(0,color='#263444',ls='--',lw=1.3,label='Model joint lower limit: 0 rad')
        ax.axhline(-.314,color='#9d506a',ls='--',lw=1.1,label='Model actuator-input lower limit: −0.314 rad')
        ax.set(xlabel='Time from episode start (s)',ylabel='Middle MCP angle (rad)',title=f"Episode {x['episode']} — {x['outcome']}",ylim=(-.74,.40))
        ax.set_xlim(min(-.015,float(qtime.min())),end)
        if degree_axis:
            ax.spines['right'].set_visible(True);secondary=ax.secondary_yaxis('right',functions=(np.degrees,np.radians));secondary.set_ylabel('Angle (degrees)')
    fig,axs=plt.subplots(5,2,figsize=(14,14),sharey=True)
    for ax,x in zip(axs.ravel(),streams):draw(ax,x)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,fontsize=9)
    fig.suptitle('Middle MCP: target commands versus bridge-measured joint pose',fontweight='bold',fontsize=16)
    fig.tight_layout(rect=(0,.082,1,.975));fig.text(.08,.060,'Targets are aligned to controller publication; measurements use bridge acquisition timestamps.',fontsize=9)
    fig.text(.08,.044,'Target curves are controller estimates. Exact motor-write targets/times are absent from this archive.',fontsize=9)
    save(fig,'05_middle_mcp_targets_vs_bridge_all_episodes')
    for ep,stop,name in [(4,None,'06_middle_mcp_episode04_success'),(5,8.,'07_middle_mcp_episode05_first8s')]:
        x=next(x for x in streams if x['episode']==ep);fig,ax=plt.subplots(figsize=(12,5.5));draw(ax,x,stop,True)
        ax.set_title(f"Middle MCP — episode {ep} ({x['outcome']})"+(' — first 8 seconds' if stop else ''))
        ax.legend(loc='upper right',fontsize=8,ncol=2)
        fig.subplots_adjust(bottom=.20,right=.91);fig.text(.10,.055,'Blue: measured feedback. Orange: estimated published target. Gray: raw planned target. No extra smoothing.',fontsize=9)
        fig.text(.10,.025,'Actual bridge write targets/times were not logged; controller publication times are used for target steps.',fontsize=8)
        save(fig,name)
    info=dict(source_archive=str(SOURCE.relative_to(ROOT)),source_controller_sha256=hashlib.sha256(payload).hexdigest(),
        target_records=sum(len(x['t']) for x in streams),unique_bridge_measurement_records=sum(len(x['mt']) for x in streams),duplicate_measurements_removed=duplicates,
        measured_source='q[4] before planning plus feedback.command_qpos[4] after planning, deduplicated by bridge robot_timestamp',
        target_source='u[4]: controller-estimated post-clipping target; planned_u[4]: raw planner absolute target',
        alignment='Measured points at bridge robot_timestamp; targets at action_published_at_utc. Origin: episode started_at_utc.',
        motor_write_times_available=False,models_or_hardware_executed=False)
    (OUT/'target_feedback_plot_metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    index=OUT/'index.html';text=index.read_text();marker='<!-- timed-target-feedback -->'
    if marker not in text:
        section=marker+'<h2>Targets against timestamp-aligned bridge feedback</h2><p>Measurements combine pre/post-planning bridge states, deduplicated by acquisition timestamp. Estimated targets are shown at controller publication time; exact bridge-write records are unavailable.</p><p><a href="mf_mcp_time_aligned.csv">Download time-aligned data</a> · <a href="target_feedback_plot_metadata.json">Timing definitions</a></p>'
        for name in ['05_middle_mcp_targets_vs_bridge_all_episodes','06_middle_mcp_episode04_success','07_middle_mcp_episode05_first8s']:section+=f'<p><a href="{name}.png"><img loading="lazy" src="{name}.png" alt="{name}"></a><a href="{name}.pdf">PDF</a></p>'
        index.write_text(text.replace('</html>',section+'</html>'))
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
