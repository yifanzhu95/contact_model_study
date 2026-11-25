"""CPU/fake-process tests: no GPU polling, MPS, physics episodes, or hardware."""
from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import test_setup as setup
from gpu_watchdog import LoadPolicy
import run as supervisor
from ResumeState import atomic_json,read_checkpoint,write_checkpoint
from ContactModelStudy.Drivers import EpisodePool as engine


POLICY=dict(monitor_gpu_indices=[0,1],busy_percent=20.,idle_percent=10.,busy_seconds=10.,idle_seconds=60.)


def metrics(a,b):
    return [dict(index=0,utilization=a),dict(index=1,utilization=b)]


def test_watchdog_hysteresis_and_either_external_gpu():
    policy=LoadPolicy(POLICY)
    assert not policy.update(metrics(0,0),0)['secondary_requested']
    assert policy.update(metrics(0,0),60)['secondary_requested']
    assert policy.update(metrics(25,0),61)['secondary_requested']
    assert not policy.update(metrics(25,0),71)['secondary_requested']
    assert not policy.update(metrics(0,0),72)['secondary_requested']
    assert not policy.update(metrics(0,15),131)['secondary_requested']
    assert not policy.update(metrics(0,0),132)['secondary_requested']
    assert policy.update(metrics(0,0),192)['secondary_requested']
    assert policy.update(metrics(0,80),193)['secondary_requested']
    assert not policy.update(metrics(0,80),203)['secondary_requested']


def test_missing_stale_or_unhealthy_watchdog_only_disables_secondary():
    assert not supervisor.enable_secondary(None,100)
    assert not supervisor.enable_secondary(dict(healthy=True,updated_at_unix=79,secondary_requested=True),100)
    assert not supervisor.enable_secondary(dict(healthy=False,updated_at_unix=100,secondary_requested=True),100)
    assert supervisor.enable_secondary(dict(healthy=True,updated_at_unix=99,secondary_requested=True),100)


class FakeProcess:
    def __init__(self,pid,alive=True):self.pid,self.alive,self.starts=pid,alive,0
    def is_alive(self):return self.alive
    def join(self,timeout=None):pass
    def start(self):self.alive=True;self.starts+=1


def test_release_recreate_only_secondary_without_interrupting_workers(tmp_path,monkeypatch):
    control,ack=tmp_path/'control.json',tmp_path/'ack.json'
    monkeypatch.setenv('M2_BO_GPU_CONTROL',str(control));monkeypatch.setenv('M2_BO_GPU_ACK',str(ack))
    monkeypatch.setenv('M2_BO_SECONDARY_GPU','GPU-secondary')
    pool=engine.EpisodePool.__new__(engine.EpisodePool)
    primary=FakeProcess(100);secondary=FakeProcess(200)
    pool._planners=[primary,secondary];pool.planner_gpus=['GPU-primary','GPU-secondary']
    pool._retire_events=[threading.Event(),threading.Event()]
    pool._retiring=set();pool._control_at=0;pool._closed=False;pool._abort=threading.Event()
    pool._newPlanner=lambda i:FakeProcess(201,False)
    atomic_json(control,dict(secondary_enabled=False));pool._gpuControl()
    assert pool._retire_events[1].is_set() and not pool._retire_events[0].is_set()
    assert primary.is_alive() and secondary.is_alive() and not pool._abort.is_set()
    assert not json.loads(ack.read_text())['secondary_released']
    secondary.alive=False;pool._control_at=0;pool._gpuControl()
    assert json.loads(ack.read_text())['secondary_released']
    atomic_json(control,dict(secondary_enabled=True));pool._control_at=0;pool._gpuControl()
    assert pool._planners[0] is primary
    assert pool._planners[1].pid==201 and pool._planners[1].starts==1
    assert not pool._retire_events[1].is_set() and not pool._abort.is_set()


def test_retiring_planner_finishes_one_request_and_preserves_queued_work(monkeypatch):
    retire=threading.Event();events=queue.Queue();requests=queue.Queue()
    reply=queue.Queue();reply.cancel_join_thread=lambda:None
    request=dict(model='M2',params={},weights={},goal=np.array([1,0,0,0]),state=None,
        mean=None,noise_seed=0,activation=np.zeros(16),q=np.zeros(23),q_dot=np.zeros(22),u=None,worker=0)
    requests.put(request);requests.put(request.copy())
    class Planner:
        last_plan_start=last_action_seq=None
        last_mean=np.zeros((4,16))
        def Reset(self,*a):pass
        def SaveState(self):return {}
        def LoadState(self,*a):pass
        def Plan(self,*a,**kw):retire.set();return np.ones(16)
    closed=[]
    unit=SimpleNamespace(planner=Planner(),goal=None,apply=lambda *a:None,
        task=SimpleNamespace(setGoal=lambda g:None),
        sim=SimpleNamespace(activation=SimpleNamespace(assign=lambda v:None),Close=lambda:closed.append(True)))
    monkeypatch.setattr(engine,'_ignoreInterrupts',lambda:None)
    monkeypatch.setattr(engine,'_PlannerUnit',lambda *a:unit)
    engine._plannerMain(None,None,0,['M2'],requests,[reply],events,threading.Event(),retire)
    assert requests.qsize()==1 and reply.qsize()==1 and closed==[True]
    np.testing.assert_array_equal(reply.get()['action'],1.)


def test_checkpoint_atomic_backup_retains_rng_and_partial_results(tmp_path):
    path=tmp_path/'resume.pkl';rng=np.random.RandomState(18)
    write_checkpoint(path,dict(rng=rng,completed_episodes=[0,2],candidate=[1,2,3]))
    expected=rng.rand()
    restored=read_checkpoint(path)
    assert restored['rng'].rand()==expected and restored['completed_episodes']==[0,2]
    write_checkpoint(path,dict(completed_episodes=[0,1,2]))
    path.write_bytes(b'killed during write')
    restored=read_checkpoint(path)
    assert restored['completed_episodes']==[0,2] and restored['candidate']==[1,2,3]


def test_actual_gp_optimizer_can_be_checkpointed_with_rng(tmp_path):
    spec=setup.spec();optimizer=setup.bo.makeOptimizer(spec)
    optimizer.tell(spec.seeds[0].x,-.4)
    path=tmp_path/'real_optimizer.pkl'
    write_checkpoint(path,dict(optimizer=optimizer))
    restored=read_checkpoint(path)['optimizer']
    assert restored.Xi==optimizer.Xi and restored.yi==optimizer.yi
    np.testing.assert_array_equal(restored.rng.randint(2**20,size=10),optimizer.rng.randint(2**20,size=10))


def test_orphan_cleanup_does_not_touch_a_new_supervisors_resources(tmp_path,monkeypatch):
    import gpu_watchdog
    atomic_json(tmp_path/'owned_processes.json',dict(supervisor_pid=20,supervisor_identity='new',
        search=dict(pid=30,identity='child'),mps_pipes={'GPU-owned':'/tmp/private-test-only'}))
    calls=[]
    monkeypatch.setattr(gpu_watchdog.subprocess,'run',lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setattr(gpu_watchdog.os,'killpg',lambda *a:pytest.fail('must not signal a different owner'))
    gpu_watchdog.cleanup_orphan(tmp_path,dict(mps_bin='/not-executed'),10,'old')
    assert calls==[]


class FakeOptimizer:
    def __init__(self):self.Xi=[];self.yi=[]
    def tell(self,x,y):
        if np.ndim(y):self.Xi.extend(x);self.yi.extend(y)
        else:self.Xi.append(x);self.yi.append(y)


@pytest.mark.parametrize('interruption',['during_episode','before_trial_commit','after_trial_commit'])
def test_interrupted_partial_trial_resumes_only_missing_episodes(tmp_path,monkeypatch,interruption):
    bo=setup.bo
    original=setup.spec()
    args=SimpleNamespace(**vars(original.args));args.n_episodes=2
    spec=replace(original,args=args,n_calls=2,seeds=original.seeds[:2],n_initial_points=2)
    monkeypatch.setattr(bo,'parseCell',lambda *a:spec)
    monkeypatch.setattr(bo,'makeOptimizer',lambda *a:FakeOptimizer())
    monkeypatch.setattr(bo,'cellSummary',lambda *a:None)
    monkeypatch.setattr(bo,'scoreModel',lambda *a:dict(objective=-1.,n_episodes=2,n_success=2))
    def save_trial(spec,args,cell,t,models,J):
        record=dict(trial=t.index,status='done',source=t.source,x=t.x,objective=J,
                    success_rate=1.,mean_norm_goal_err=0.,wall_s=0.)
        if len(instances)==1 and interruption=='before_trial_commit':raise KeyboardInterrupt
        atomic_json(cell/bo.trialName(t.index)/'trial.json',record)
        if len(instances)==1 and interruption=='after_trial_commit':raise KeyboardInterrupt
        return record
    monkeypatch.setattr(bo,'saveTrial',save_trial)
    submissions=[];instances=[]
    class Pool:
        def __init__(self,args,*a,**kw):
            self.args=args;self.fatal=False;self.problems=[];self.jobs=[];self.calls=0
            self.first=not instances;instances.append(self)
        def start(self):pass
        def submit(self,job):self.jobs.append(job);submissions.append((len(instances),job.key,job.episode))
        def poll(self,*a):
            self.calls+=1
            if self.first and self.calls>1:raise KeyboardInterrupt
            if self.first:
                count=1 if interruption=='during_episode' else 2
                jobs=[self.jobs.pop(0) for _ in range(count)]
            else:jobs=list(self.jobs)
            if not self.first:self.jobs=[]
            return [engine.PoolResult(j,SimpleNamespace(extra={}),dict(finish_reason='success'),None,0) for j in jobs]
        def interrupt(self,*a):self.fatal=True;self.problems=['interrupted']
        def close(self):return []
        def printUtilization(self,*a):pass
    monkeypatch.setattr(engine,'EpisodePool',Pool)
    options=SimpleNamespace(debug=False,gpus='fixed2,fixed3',planners_per_gpu=20,workers=48)
    assert bo.runCell(0,spec.row,HERE,tmp_path,False,options)==1
    cell=tmp_path/setup.bo.batches.cellName(0,spec.row)
    checkpoint=read_checkpoint(cell/'resume.pkl')
    assert len(checkpoint['in_flight'])==2
    assert checkpoint['in_flight'][0].left==(1 if interruption=='during_episode' else 0)
    assert bo.runCell(0,spec.row,HERE,tmp_path,False,options)==0
    assert (2,0,0) not in submissions  # completed before the interruption
    expected=[(0,1),(1,0),(1,1)] if interruption=='during_episode' else [(1,0),(1,1)]
    assert sorted((t,e) for run,t,e in submissions if run==2)==expected
    assert len(bo._readTrials(cell))==2
    restored=read_checkpoint(cell/'resume.pkl')
    assert restored['in_flight']=={} and len(restored['optimizer'].yi)==2
