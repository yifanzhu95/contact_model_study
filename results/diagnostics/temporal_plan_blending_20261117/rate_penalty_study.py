"""Offline action-rate/temporal-blending ablation on recorded states.

No hardware/LCM clients. Recorded q, v, observer activation and publication
times are held fixed. Each tested controller maintains its own warm-start mean
and previous clipped target; the combined variant therefore replans separately
from penalty-only. This is teacher-forced policy replay, not a physical rollout
of the counterfactual commands. Production parameters are never modified.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0,str(ROOT))
from ContactModelStudy.Hardware.Workflow import SceneTaskMixin, build_rollout
from ContactModelStudy.Hardware.PlanBlending import TemporalPlanBlend
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
from ContactModelStudy.SamplingBasedPlanners.SamplingBasedPlannerBase import _action_rate_cost_kernel

CACHE = OUT/'recorded_state_replans'
RESULTS = OUT/'rate_study'
WEIGHTS = [.005,.01,.02,.05,.1,.2]


class RateDiagnostics:
    """Measure the existing rate kernel separately without changing task cost."""
    def __init__(self, task):
        self.task, self.counter, self.planner = task, 0, None

    def __getattr__(self, name):
        return getattr(self.task,name)

    def bind(self, planner):
        self.planner=planner
        self.rate=wp.zeros(planner.N,dtype=float,device=planner.device)
        self.previous=wp.zeros((planner.N,planner.nu),dtype=float,device=planner.device)
        self.applied=wp.zeros(planner.nu,dtype=float,device=planner.device)

    def calcCosts(self, sim, terminal=False, out=None):
        result=self.task.calcCosts(sim,terminal=terminal,out=out)
        p,t=self.planner,self.counter
        if t==0:self.rate.zero_()
        wp.launch(_action_rate_cost_kernel,dim=p.N,
            inputs=[p.V_wp,p._ctrl_step_wp,self.applied,t,1,1.],
            outputs=[self.previous,self.rate])
        self.counter=(t+1)%p.horizon
        return result


def snapshot(path):
    raw=path.read_bytes();return json.loads(raw),hashlib.sha256(raw).hexdigest()


def build(data, scene, penalty):
    scene.write_text(data['scenes']['rollout']['scene_xml'])
    cls=type('RecordedCubeRate',(SceneTaskMixin,CubeReorient),{})
    task=cls(scene,LeapReorientConfig(**data['scenes']['rollout']['task_config']))
    task.params['init_ctrl']=np.asarray(data['hardware_config']['robot_qref'])
    task.params['fallen_z']=data['failure_z']
    task.setJointReference(data['joint_reference']);task.setCostWeights(data['weights'])
    args=SimpleNamespace(**{**data['args'],'device':'cuda:0'})
    bounds=task.mjm.actuator_ctrlrange
    sim=build_rollout(args,task,bounds[:,0],bounds[:,1])
    assert json.loads(json.dumps(asdict(sim.config)))=={**data['rollout_config'],'device':'cuda:0'}
    diagnostic=RateDiagnostics(task)
    cfg=MPPI_Config(**{**data['planner_config'],'action_rate_weight':penalty})
    assert not cfg.normalize_cost_by_samples and not cfg.mean_cost_over_horizon
    assert cfg.control_mode=='pos_relative' and cfg.warm_start=='shift' and cfg.n_iterations==1
    planner=MPPI(sim,diagnostic,cfg);diagnostic.bind(planner)
    return task,sim,planner,diagnostic


def summary(a):
    a=np.asarray(a,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))


def distribution(cost, temperature):
    logits=-(cost-cost.min())/temperature
    logz=np.log(np.exp(logits).sum());logp=logits-logz
    return np.exp(logp),logp


def calibrate(manifest):
    selected=[];sims={};results={str(w):[] for w in WEIGHTS}
    try:
        for meta in manifest['episodes']:
            path=ROOT/meta['source'];data,sha=snapshot(path)
            assert sha==meta['source_sha256']
            if meta['source'] not in sims:
                sims[meta['source']]=build(data,RESULTS/f'calibration_scene{len(sims)}.xml',0.)
            task,sim,p,diag=sims[meta['source']]
            ep=data['episodes'][meta['source_episode']-1]
            task.setTargetPosition(ep['target_pos']);task.setJointReference(ep['home']);task.setGoal(ep['goal'])
            z=np.load(CACHE/meta['file']);length=len(z['q'])
            indices=sorted(set([0,min(5,length-1),length//4,length//2,3*length//4,length-1]))
            for i in indices:
                state=p.SaveState()
                seed=np.zeros_like(z['mean'][0])
                if i:seed[:-1]=z['mean'][i-1,1:]
                state.update(U=seed,resample_count=meta['noise_seed_offset']+i)
                p.LoadState(state)
                previous=np.asarray(data['hardware_config']['robot_qref'] if i==0 else ep['steps'][i-1]['u'])
                diag.applied.assign(previous.astype(np.float32))
                sim.activation.assign(z['activation'][i].astype(np.float32))
                p.Plan(z['q'][i],z['v'][i],u=previous)
                assert p.last_plan_ok
                cost=p.costs_wp.numpy().astype(float);rate=diag.rate.numpy().astype(float)
                samples=p.V_wp.numpy().astype(float)
                finite=np.isfinite(cost)&np.isfinite(rate)
                assert finite.sum()>=p.N//2 and rate[finite].min()>=0
                cost,rate,samples=cost[finite],rate[finite],samples[finite]
                elite=np.argsort(cost)[:max(16,len(cost)//4)]
                iqr=float(np.percentile(rate[elite],75)-np.percentile(rate[elite],25))
                w0,log0=distribution(cost,p.config.temperature)
                mu0=np.einsum('n,nhj->hj',w0,samples)
                record=dict(source=meta['source'],episode=meta['source_episode'],step=i,
                    elite_rate_iqr=iqr,rate_mean=float(rate.mean()),
                    rate_expected_baseline=float(w0@rate),cost_expected_baseline=float(w0@cost),
                    baseline_ess=float(1/np.sum(w0*w0)))
                selected.append(record)
                for weight in WEIGHTS:
                    weights,logs=distribution(cost+weight*rate,p.config.temperature)
                    mu=np.einsum('n,nhj->hj',weights,samples)
                    results[str(weight)].append(dict(
                        kl=float(np.sum(weights*(logs-log0))),
                        elite_rate_iqr_in_temperatures=weight*iqr/p.config.temperature,
                        expected_task_cost_increase=float(weights@cost-w0@cost),
                        expected_rate_cost_change_percent=float(100*((weights@rate)/(w0@rate)-1)),
                        first_delta_change_deg=float(np.degrees(abs(mu[0]-mu0[0])).mean())))
            print('Calibrated episode',meta['source_episode'],meta['source'].split('/')[-2],flush=True)
    finally:
        for _,sim,_,_ in sims.values():sim.Close()
    stats={k:{name:summary([r[name] for r in rows]) for name in rows[0]} for k,rows in results.items()}
    # Mild perturbation: target .25 temperature units of rate-cost variation
    # among competitive candidates, and avoid large distribution changes.
    eligible=[w for w in WEIGHTS if stats[str(w)]['kl']['mean']<=.05]
    if not eligible:eligible=[WEIGHTS[0]]
    chosen=min(eligible,key=lambda w:abs(np.log(max(1e-10,stats[str(w)]['elite_rate_iqr_in_temperatures']['median'])/.25)))
    report=dict(selected_weight=chosen,blending_alpha=.6,states=len(selected),
        criterion='Closest to .25 temperature units of median elite rate-cost IQR, with mean KL<=.05; weights capped at .2 because prior 1/5 tests were poor.',
        note='A small cost-scale choice, not an optimized or hardware-validated weight.',
        candidates=stats,selected_states=selected)
    (RESULTS/'calibration.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('CHOSEN',chosen,json.dumps(stats[str(chosen)]),flush=True)


def replay(manifest, variant, step_limit):
    calibration=json.loads((RESULTS/'calibration.json').read_text())
    penalty=0. if variant=='baseline_check' else calibration['selected_weight']
    alpha=calibration['blending_alpha'] if variant=='rate_blend' else 1.
    destination=RESULTS/variant;destination.mkdir(exist_ok=True)
    report=dict(variant=variant,weight=penalty,alpha=alpha,episodes=[],
        physical_inputs='recorded q, v and reconstructed observer activation, fixed for every variant',
        command_history='variant-specific previous estimated clipped target and warm-start mean',
        timing='recorded publication times; no pacing or hardware clients',
        first_rate_reference='saved robot_qref, as in current hardware action-rate mode')
    started=time.monotonic();sims={}
    try:
        for meta in manifest['episodes']:
            path=ROOT/meta['source'];data,sha=snapshot(path);assert sha==meta['source_sha256']
            if meta['source'] not in sims:
                sims[meta['source']]=build(data,destination/f'scene{len(sims)}.xml',penalty)
            task,sim,p,diag=sims[meta['source']]
            ep=data['episodes'][meta['source_episode']-1]
            task.setTargetPosition(ep['target_pos']);task.setJointReference(ep['home']);task.setGoal(ep['goal'])
            p.Reset();state=p.SaveState();state['resample_count']=meta['noise_seed_offset'];p.LoadState(state)
            z=np.load(CACHE/meta['file'])
            n=min(len(z['q']),step_limit) if step_limit else len(z['q'])
            previous=np.asarray(data['hardware_config']['robot_qref'],float)
            blend=TemporalPlanBlend(alpha,meta['ctrl_period'])
            ticks=np.rint(z['published_s']*1e6).astype(np.int64);times=(ticks-ticks[0])*1e-6
            output={key:[] for key in ['mean','raw_delta','sent_delta','target','reference','expected_task_cost',
                'expected_rate','penalty_fraction','ess','planning_s','within_delta_l2','between_delta_l2']}
            for i in range(n):
                sim.activation.assign(z['activation'][i].astype(np.float32));diag.applied.assign(previous.astype(np.float32))
                output['reference'].append(previous.copy())
                before=time.monotonic();proposed=p.Plan(z['q'][i],z['v'][i],u=previous)
                wp.synchronize();output['planning_s'].append(time.monotonic()-before)
                assert p.last_plan_ok
                raw=np.asarray(proposed)-z['q'][i,:16]
                sent,context=blend.prepare(p.last_mean,raw,times[i]);blend.commit(context,times[i])
                record=blend.metrics(context)
                target=z['command_base'][i]+np.clip(sent,-meta['bounds']['lower'],meta['bounds']['upper'])
                weights=p.w_wp.numpy().astype(float);weights/=weights.sum()
                rate=diag.rate.numpy().astype(float);total=p.costs_wp.numpy().astype(float)
                finite=np.isfinite(total)&np.isfinite(rate)&(weights>0)
                selected=weights[finite];selected/=selected.sum();c=total[finite];r=rate[finite]
                output['expected_task_cost'].append(float(selected@(c-penalty*r)))
                output['expected_rate'].append(float(selected@r))
                output['penalty_fraction'].append(float((penalty*(selected@r))/(selected@c)))
                output['ess'].append(float(1/np.sum(weights*weights)))
                output['mean'].append(p.last_mean.copy());output['raw_delta'].append(raw)
                output['sent_delta'].append(sent);output['target'].append(target)
                output['within_delta_l2'].append(float(np.mean(record['within_plan_diff_l2'])))
                output['between_delta_l2'].append(float(np.mean(record['aligned_plan_diff_l2'])) if record['aligned_plan_diff_l2'] else np.nan)
                previous=target
            file=destination/meta['file']
            np.savez_compressed(file,**{k:np.asarray(v) for k,v in output.items()})
            report['episodes'].append(dict(source=meta['source'],source_episode=meta['source_episode'],
                file=file.name,steps=n,original_reference_outcome=meta['outcome']))
            report['elapsed_s']=time.monotonic()-started
            (destination/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print(variant,'episode',meta['source_episode'],'steps',n,'elapsed',round(report['elapsed_s'],1),flush=True)
    finally:
        for _,sim,_,_ in sims.values():sim.Close()


def paired(manifest):
    """Exact common-candidate rescoring; freezes warm starts and rate reference.

    Unlike replay(), this isolates the immediate objective change and does not
    test each controller's feedback history. No candidate is rerolled for the
    alternate cost, so numerical trajectory divergence cannot confound it.
    """
    calibration=json.loads((RESULTS/'calibration.json').read_text())
    penalty,alpha=calibration['selected_weight'],calibration['blending_alpha']
    destination=RESULTS/'paired';destination.mkdir(exist_ok=True)
    report=dict(weight=penalty,alpha=alpha,episodes=[],
        method='common-candidate one-step rescoring on frozen recorded-state contexts',
        rate_reference='recorded prior estimated target; robot_qref for the first step',
        warm_start='same shifted cached baseline mean for all variants',
        feedback_history_tested=False)
    started=time.monotonic();sims={};gpu_errors=[]
    try:
        for meta in manifest['episodes']:
            path=ROOT/meta['source'];data,sha=snapshot(path);assert sha==meta['source_sha256']
            if meta['source'] not in sims:
                sims[meta['source']]=build(data,destination/f'scene{len(sims)}.xml',0.)
            task,sim,p,diag=sims[meta['source']]
            ep=data['episodes'][meta['source_episode']-1]
            task.setTargetPosition(ep['target_pos']);task.setJointReference(ep['home']);task.setGoal(ep['goal'])
            z=np.load(CACHE/meta['file']);n=len(z['q'])
            ticks=np.rint(z['published_s']*1e6).astype(np.int64);times=(ticks-ticks[0])*1e-6
            cases=['baseline','blend','rate','rate_blend']
            blenders={c:TemporalPlanBlend(alpha if c in ('blend','rate_blend') else 1.,meta['ctrl_period']) for c in cases}
            outputs={c:{k:[] for k in ['mean','raw_delta','sent_delta','target','reference',
                'expected_task_cost','expected_rate','penalty_fraction','ess','planning_s']} for c in cases}
            for i in range(n):
                state=p.SaveState();seed=np.zeros_like(z['mean'][0])
                if i:seed[:-1]=z['mean'][i-1,1:]
                state.update(U=seed,resample_count=meta['noise_seed_offset']+i);p.LoadState(state)
                previous=np.asarray(data['hardware_config']['robot_qref'] if i==0 else ep['steps'][i-1]['u'])
                diag.applied.assign(previous.astype(np.float32));sim.activation.assign(z['activation'][i].astype(np.float32))
                before=time.monotonic();p.Plan(z['q'][i],z['v'][i],u=previous);wp.synchronize();elapsed=time.monotonic()-before
                assert p.last_plan_ok
                cost=p.costs_wp.numpy().astype(float);r=diag.rate.numpy().astype(float);samples=p.V_wp.numpy().astype(float)
                valid=np.isfinite(cost)&np.isfinite(r);cost,r,samples=cost[valid],r[valid],samples[valid]
                params={}
                for weight in [0.,penalty]:
                    w,_=distribution(cost+weight*r,p.config.temperature)
                    mean=np.einsum('n,nhj->hj',w,samples).astype(np.float32)
                    params[weight]=(mean,w)
                gpu_errors.extend(np.degrees(abs(params[0.][0]-p.last_mean)).ravel())
                for case in cases:
                    weight=penalty if case in ('rate','rate_blend') else 0.
                    mean,w=params[weight]
                    raw=(z['q'][i,:16]+mean[0]).astype(np.float32).astype(float)-z['q'][i,:16]
                    sent,context=blenders[case].prepare(mean,raw,times[i]);blenders[case].commit(context,times[i])
                    target=z['command_base'][i]+np.clip(sent,-meta['bounds']['lower'],meta['bounds']['upper'])
                    result=outputs[case]
                    for key,value in dict(mean=mean,raw_delta=raw,sent_delta=sent,target=target,reference=previous,
                        expected_task_cost=float(w@cost),expected_rate=float(w@r),
                        penalty_fraction=float((weight*(w@r))/(w@(cost+weight*r))),
                        ess=float(1/np.sum(w*w)),planning_s=elapsed).items():result[key].append(value)
            for case in cases:
                folder=destination/case;folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/meta['file'],**{k:np.asarray(v) for k,v in outputs[case].items()})
            report['episodes'].append(dict(source=meta['source'],source_episode=meta['source_episode'],file=meta['file'],steps=n))
            report['elapsed_s']=time.monotonic()-started
            report['cpu_vs_gpu_mean_error_deg']=summary(gpu_errors)
            (destination/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print('paired episode',meta['source_episode'],'steps',n,'elapsed',round(report['elapsed_s'],1),flush=True)
    finally:
        for _,sim,_,_ in sims.values():sim.Close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['calibrate','replay','paired'],required=True)
    parser.add_argument('--variant',choices=['rate','rate_blend','baseline_check'])
    parser.add_argument('--step-limit',type=int,default=0)
    args=parser.parse_args();RESULTS.mkdir(exist_ok=True)
    manifest=json.loads((CACHE/'manifest.json').read_text())
    with wp.ScopedDevice('cuda:0'):
        if args.phase=='calibrate':calibrate(manifest)
        elif args.phase=='paired':paired(manifest)
        else:
            if args.variant is None:parser.error('--variant is required for replay')
            replay(manifest,args.variant,args.step_limit)


if __name__=='__main__':main()
