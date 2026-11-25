"""Fault holds, explicit reset commands, launcher takeover and concise reasons.

All serial, LCM and process interfaces are fake. No hardware is accessed.
"""
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ContactModelStudy.Hardware.LCM import (
    RobotKeepTargetAction, RobotRecoveryPositionAction, RobotDeltaAction,
    RobotPositionAction, RobotStopAction,
)
from ContactModelStudy.Hardware.FailureSummary import short_failure
from ContactModelStudy.Hardware.Recovery import can_hold_after_failure, recover_after_failure
from test_leap_lcm_bridge import make_bridge, run_until_interrupted
from test_leap_episodes import phases


@pytest.mark.parametrize('kind', [RobotKeepTargetAction, RobotRecoveryPositionAction])
def test_recovery_packets_round_trip_and_reject_legacy_data(kind):
    message=kind(123456, np.ones(16)) if kind is RobotRecoveryPositionAction else kind(123456)
    decoded=kind.decode(message.encode())
    assert decoded.timestamp==123456
    if kind is RobotRecoveryPositionAction:np.testing.assert_array_equal(decoded.action,1)
    with pytest.raises(ValueError):kind.decode(b'bad')


def test_fault_hold_preserves_target_blocks_planner_and_allows_only_explicit_reset(tmp_path):
    bridge,hand,connection=make_bridge(positions=(.2,.2,.1,.05,0,0,0,0))
    bridge.fault_status_file=tmp_path/'bridge_status.json'
    stamp=int(time.time()*1e6)
    connection.schedule={
        2:RobotPositionAction(stamp,np.full(16,.8)).encode(),
        3:RobotKeepTargetAction(stamp+1).encode(),
        4:RobotDeltaAction(stamp+2,np.full(16,10.)).encode(),
        5:RobotPositionAction(stamp+3,np.zeros(16)).encode(),
        6:RobotRecoveryPositionAction(stamp+4,np.full(16,.7)).encode(),
        7:RobotStopAction(stamp+5).encode(),
    }
    try:bridge.run()
    finally:bridge.close()
    np.testing.assert_allclose(hand.writes,np.array([.8,.8,.8,.8,.7])[:,None]*np.ones((1,16)))
    assert bridge.fault_hold
    assert hand.events[-1]=='torque off'
    assert json.loads(bridge.fault_status_file.read_text())['stage']=='software_hold'


def test_expired_pending_action_is_never_applied_when_timeout_enters_hold():
    bridge,hand,connection=make_bridge(positions=(.2,.1,0))
    traces=[]
    bridge.telemetry=SimpleNamespace(submit=lambda trace:traces.append(trace), close=lambda:None)
    bridge.hold_on_failure=True
    bridge.on_action('',RobotDeltaAction(int(time.time()*1e6),np.full(16,10.)).encode())
    bridge.received=time.monotonic()-1
    run_until_interrupted(bridge)
    assert bridge.fault_hold
    np.testing.assert_allclose(hand.writes,.2)
    assert traces
    assert all(trace['held_action_timestamp_us']==0 for trace in traces)


def test_current_fault_still_stops_immediately_during_software_hold():
    bridge,hand,connection=make_bridge(positions=(.2,.2,.2))
    original=hand.read
    def read():
        q,v,current=original()
        if hand.reads==3:current[13]=565
        return q,v,current
    hand.read=read
    connection.pending=RobotKeepTargetAction(int(time.time()*1e6)).encode()
    try:
        with pytest.raises(RuntimeError,match='motor 13=565'):
            bridge.run()
    finally:bridge.close()
    assert len(hand.writes)==1
    assert hand.events[-1]=='torque off'


def test_usb_fault_still_stops_immediately_during_software_hold():
    bridge,hand,connection=make_bridge(positions=(.2,.2,.2))
    original=hand.read
    def read():
        if hand.reads==2:raise OSError('Dynamixel receive failure')
        return original()
    hand.read=read
    connection.pending=RobotKeepTargetAction(int(time.time()*1e6)).encode()
    try:
        with pytest.raises(OSError,match='Dynamixel'):
            bridge.run()
    finally:bridge.close()
    assert hand.events[-1]=='torque off'


def test_software_exception_waits_for_enter_without_rebasing_then_resets(monkeypatch,tmp_path):
    handler,measured,actions,holds,clock=phases(monkeypatch,tmp_path)
    measured[:]=.2
    handler.last_target=np.full(16,.4)  # Estimated target is deliberately different.
    keeps,stops,resets,enters=[],[],[],[]
    handler.feedback.publish_keep_target=lambda:keeps.append(clock.now)
    handler.feedback.publish_stop=lambda:stops.append(clock.now)
    position=handler.feedback.publish_position
    def reset(target):
        resets.append(clock.now)
        position(target)
    handler.feedback.publish_recovery_position=reset
    def enter(period):
        assert not stops and not resets
        np.testing.assert_allclose(measured,.2)
        assert json.loads(handler.status_file.read_text())['stage']=='software_hold'
        enters.append(clock.now);clock.sleep(period)
        return len(enters)==60
    monkeypatch.setattr(handler,'enter_pressed',enter)
    q=handler.shutdown_after_error(0,RuntimeError('CUDA planner error'))
    np.testing.assert_allclose(q,0)
    assert len(keeps)>=60 and keeps[-1]>1.
    assert resets[0]>=enters[-1]
    assert len(stops)==1 and not holds
    assert json.loads(handler.status_file.read_text())['stage']=='software_complete'
    assert not handler.recovering


@pytest.mark.parametrize('problem',['stale','eof','reset_timeout'])
def test_recovery_failure_requests_torque_off(monkeypatch,tmp_path,problem):
    handler,measured,actions,holds,clock=phases(monkeypatch,tmp_path,moving=False)
    measured[:]=.3
    keeps,stops=[],[]
    handler.interrupt_timeout=.1
    handler.feedback.publish_keep_target=lambda:keeps.append(clock.now)
    handler.feedback.publish_stop=lambda:stops.append(clock.now)
    handler.feedback.publish_recovery_position=lambda target:None
    def enter(period):
        if problem=='eof':raise RuntimeError('stdin EOF')
        return True
    monkeypatch.setattr(handler,'enter_pressed',enter)
    if problem=='stale':
        def stale(**kwargs):raise RuntimeError('Stale robot feedback')
        handler.feedback.buffer.snapshot_robot=stale
    with pytest.raises(RuntimeError):handler.shutdown_after_error(0,'software failure')
    assert len(stops)==1
    if problem=='stale':assert not keeps
    assert json.loads(handler.status_file.read_text())['stage']=='software_failed'


def test_feedback_failure_uses_exact_hold_when_enabled(monkeypatch,tmp_path):
    handler,*_=phases(monkeypatch,tmp_path)
    handler.hold_on_failure=True
    seen=[]
    monkeypatch.setattr(handler,'shutdown_after_error',lambda e,error:seen.append((e,error)))
    handler.shutdown_after_feedback_error(2,'camera stale')
    assert seen==[(2,'camera stale')]


def fake_experiment(tmp_path,bridge_code=None,controller_code=-11):
    codes={'bridge':bridge_code,'controller':controller_code}
    events=[]
    def send(name,sig):
        events.append((name,sig));codes[name]=-sig
    experiment=SimpleNamespace(directory=tmp_path,logging_enabled=False,
        children={name:SimpleNamespace(process=SimpleNamespace(poll=lambda n=name:codes[n])) for name in codes},send=send)
    return experiment,codes,events


@pytest.mark.parametrize('scenario',['disabled','bridge_dead','operator_stop','finished_reset','startup'])
def test_launcher_never_takes_over_motor_fault_or_operator_stop(tmp_path,scenario):
    experiment,codes,events=fake_experiment(tmp_path)
    hardware={'hold_on_non_motor_failure':True};state={'stop':None,'force_stop':False}
    if scenario=='disabled':hardware['hold_on_non_motor_failure']=False
    if scenario=='bridge_dead':codes['bridge']=1
    if scenario=='operator_stop':state['stop']='SIGINT'
    if scenario=='finished_reset':(tmp_path/'episode_status.json').write_text(json.dumps({'stage':'software_complete'}))
    if scenario=='startup':del experiment.children['controller']
    assert not can_hold_after_failure(experiment,hardware,state,1)


@pytest.mark.parametrize('controller_code',[None,-11])
def test_launcher_takeover_uses_robot_only_feedback_and_stops_old_controller(tmp_path,controller_code):
    experiment,codes,events=fake_experiment(tmp_path,controller_code=controller_code)
    state={'stop':None,'force_stop':False}
    assert can_hold_after_failure(experiment,{'hold_on_non_motor_failure':True},state,1)
    feedback=SimpleNamespace(buffer=SimpleNamespace(error=None,snapshot_robot=lambda **k:(np.zeros(16),np.zeros(16))),
        publish_keep_target=lambda:events.append('keep'),publish_stop=lambda:events.append('stop'),
        close=lambda:events.append('close'))
    def factory(model,url,robot,pose,action):
        assert model is None and pose is None
        events.append('feedback');return feedback
    def episodes(*args):
        handler=SimpleNamespace(enter_pressed=lambda period:True)
        def recover(episode,error):
            handler.recovery_guard()
            assert handler.enter_pressed(.02)
            events.append('recover');return np.zeros(16)
        handler.shutdown_after_error=recover
        return handler
    config={'lcm_url':'unused','state_channel':'STATE','action_channel':'ACTION'}
    result=recover_after_failure(experiment,config,{}, {},state,'controller died',factory,episodes)
    assert result['zero_reached'] and events[-2:]==['stop','close']
    if controller_code is None:assert events[0]==('controller',signal.SIGTERM)
    assert events.index('keep')<events.index('recover')


def test_short_terminal_reason_prioritizes_motor_fault_over_stale_controller(tmp_path):
    (tmp_path/'bridge.log').write_text('Torque disabled and verified on all 16 motors.\n'
        'RuntimeError: Measured current exceeds hardware limit: motor 13=565 raw; allowed +/-550 raw; other data\n')
    (tmp_path/'controller.log').write_text('RuntimeError: Stale robot feedback (0.174s)\n')
    value=short_failure(tmp_path,'bridge exited with code 1')
    assert value=='MOTOR FAULT: motor 13 (thumb axial) current 565 exceeds +/-550 raw; torque disabled and verified.'


def test_short_terminal_reason_reports_planner_error(tmp_path):
    (tmp_path/'controller.log').write_text('RuntimeError: CUDA out of memory\n')
    assert short_failure(tmp_path,'controller exited with code 1')=='CONTROLLER FAILURE: RuntimeError: CUDA out of memory'


def test_short_terminal_reason_reports_multiple_current_faults(tmp_path):
    (tmp_path/'bridge.log').write_text('RuntimeError: Measured current exceeds hardware limit: '
        'motor 5=567, motor 13=-565 raw; allowed +/-550 raw; currents_by_motor_id_raw=[]\n')
    (tmp_path/'controller.log').write_text('RuntimeError: Stale robot feedback\n')
    value=short_failure(tmp_path,'controller exited with code 1')
    assert 'middle MCP' in value and 'thumb axial' in value
    assert '567' in value and '-565' in value and 'Stale' not in value


def test_launcher_detects_watchdog_hold_without_treating_it_as_motor_fault(tmp_path):
    import launch_leap_experiment as launcher
    experiment=launcher.Experiment(tmp_path)
    experiment.children={name:SimpleNamespace(process=SimpleNamespace(poll=lambda:None))
                         for name in ['bridge','controller']}
    (tmp_path/'bridge_status.json').write_text(json.dumps(dict(stage='software_hold',reason='controller action timeout')))
    assert experiment.finished()==('Bridge software hold: controller action timeout',1)
    (tmp_path/'episode_status.json').write_text(json.dumps(dict(stage='software_hold')))
    assert experiment.finished() is None  # Already supervised; don't start a second reset.
