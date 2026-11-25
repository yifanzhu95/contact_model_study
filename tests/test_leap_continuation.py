"""Continuation counting, immutable inputs and CPU-only launcher integration."""
import json
import os
from pathlib import Path
import sys

import pytest

from ContactModelStudy.Hardware import Continuation as continuation
import launch_leap_experiment as launcher


def put(path, value):
    path.write_text(json.dumps(value))


def source(tmp_path, outcomes=('success', 'failure', 'error'), requested=6):
    run = tmp_path/'original'; run.mkdir()
    hardware = tmp_path/'hardware.json'; put(hardware, {'usb_latency_timer_ms': None})
    camera = tmp_path/'camera.json'; put(camera, {})
    config = json.loads((launcher.ROOT/'configs/leap_experiment.json').read_text())
    config.update(hardware_config=str(hardware), camera_config=str(camera),
                  record_video=False, tracking_dir=str(tmp_path/'tracking'))
    for key in ['gpu','tracking_gpu','mppi_gpu']:config.pop(key,None)
    command = [sys.executable, '-u', str(launcher.ROOT/'run_leap_hardware.py'),
        '--results', str(run/'controller.json'), '--task', 'cube_reorient',
        '--n-samples', '512', '--n-episodes', str(requested), '--hardware',
        '--camera-config', str(camera), '--device', 'cuda:0', '--logging']
    put(run/'launch.json', dict(config=config,commands={'controller':command},status='failed'))
    episodes = [dict(episode=i,completed=o!='error',finish_reason=o,target_pos=[.025,.035,.075]) for i,o in enumerate(outcomes)]
    put(run/'controller.json',dict(args={'n_episodes':requested,'object_joint_reference':False},
        hardware_config={'usb_latency_timer_ms':None},weights={'w_quat':.01,'w_velo':0},
        fixed_goal_quat=[1,0,0,0],episodes=episodes))
    return run


def test_count_completed_failures_and_retry_aborted(tmp_path):
    run=source(tmp_path,('success','failure','timeout','error'))
    before=(run/'controller.json').read_bytes()
    plan,config,cli=continuation.load(run,launcher.ROOT)
    assert plan['completed_episodes']==3 and plan['remaining_episodes']==3
    assert len(plan['aborted_attempts'])==1
    assert '--n-samples' in cli and cli[cli.index('--n-samples')+1]=='512'
    assert '--device' not in cli and '--hardware' not in cli and '--results' not in cli
    assert 'w_velo=0' in cli and '--goal-position' in cli
    assert config['controller_args']==[]
    assert (run/'controller.json').read_bytes()==before


def test_continue_continuation_counts_ancestry_once(tmp_path):
    run=source(tmp_path)
    plan,config,cli=continuation.load(run,launcher.ROOT)
    child=tmp_path/'child';child.mkdir()
    launch=json.loads((run/'launch.json').read_text());launch['continuation']=plan
    put(child/'launch.json',launch)
    d=json.loads((run/'controller.json').read_text());d['args']['n_episodes']=4
    d['episodes']=[dict(episode=0,completed=True,finish_reason='success',target_pos=[.025,.035,.075]),
                   dict(episode=1,completed=False,finish_reason='error')]
    put(child/'controller.json',d)
    next_plan,_,_=continuation.load(child,launcher.ROOT)
    assert next_plan['target_episodes']==6
    assert next_plan['completed_episodes']==3 and next_plan['remaining_episodes']==3
    assert len(next_plan['source_runs'])==2


def test_aborted_video_without_final_controller_checkpoint_does_not_count(tmp_path):
    run=source(tmp_path,('success',));(run/'video').mkdir()
    put(run/'video/recording.json',{'episodes':[{'episode':1,'outcome':'success'},{'episode':2,'outcome':'error'}]})
    plan,_,_=continuation.load(run,launcher.ROOT)
    assert plan['remaining_episodes']==5 and plan['aborted_attempts'][0]['source_episode']==2


def test_progress_after_second_abort_preserves_completed_mapping(tmp_path):
    run=source(tmp_path);plan,_,_=continuation.load(run,launcher.ROOT)
    current=tmp_path/'current';current.mkdir()
    put(current/'controller.json',{'episodes':[{'episode':0,'completed':True,'finish_reason':'timeout'},
                                            {'episode':1,'completed':False,'finish_reason':'error'}]})
    result=continuation.progress(plan,current)
    assert result['completed_episodes']==3 and result['remaining_episodes']==3
    assert result['completed_episode_sources'][-1]['batch_episode']==3
    assert result['completed_episode_sources'][-1]['source_episode']==1


def test_changed_hardware_config_is_rejected_before_launch(tmp_path):
    run=source(tmp_path);put(tmp_path/'hardware.json',{'usb_latency_timer_ms':None,'changed':True})
    with pytest.raises(ValueError,match='Hardware configuration changed'):continuation.load(run,launcher.ROOT)


def test_live_source_launcher_cannot_be_continued(tmp_path):
    run=source(tmp_path);record=json.loads((run/'launch.json').read_text())
    record['status']='starting'
    record['launcher_process']=launcher.process_identity(os.getpid())
    put(run/'launch.json',record)
    with pytest.raises(ValueError,match='still running'):continuation.load(run,launcher.ROOT)


def test_generated_inline_options_are_removed():
    cli=continuation.controller_options(['python','run_leap_hardware.py','--results=old',
        '--n-episodes=6','--hardware','--no-logging','--temperature=.1'])
    assert cli==['--temperature=.1']


def test_saved_camera_selection_is_pinned_instead_of_resolving_latest_again(tmp_path):
    run=source(tmp_path)
    doc=json.loads((run/'controller.json').read_text())
    doc['pose_transform']={'selected_calibration':{'name':'latest'}}
    put(run/'controller.json',doc)
    launch=json.loads((run/'launch.json').read_text())
    launch['config']['camera_calibration']='latest'
    launch['input_snapshots']={'selected_camera_config':{}}
    put(run/'launch.json',launch)
    _,config,_=continuation.load(run,launcher.ROOT)
    assert config['continued_camera_config']==str(tmp_path/'camera.json')
    put(tmp_path/'camera.json',{'new_fit':True})
    with pytest.raises(ValueError,match='selection file changed'):
        continuation.load(run,launcher.ROOT)


def test_completed_batch_is_noop_without_usb_or_gpu(tmp_path,monkeypatch,capsys):
    run=source(tmp_path,('success','timeout'),requested=2)
    monkeypatch.setattr(launcher,'UsbLatencyTimer',lambda *a:pytest.fail('USB accessed'))
    monkeypatch.setattr(launcher,'select_gpu',lambda *a,**kw:pytest.fail('GPU queried'))
    assert launcher.main(['--continue-run',str(run)])==0
    assert 'already complete' in capsys.readouterr().out


def test_dry_run_restores_options_and_only_remaining_count(tmp_path,monkeypatch,capsys):
    run=source(tmp_path);new=tmp_path/'new'
    monkeypatch.setattr(launcher,'UsbLatencyTimer',lambda *a:pytest.fail('USB accessed'))
    assert launcher.main(['--continue-run',str(run),'--run-dir',str(new),'--dry-run'])==0
    output=capsys.readouterr().out
    assert '2/6 completed; 4 episodes remaining' in output
    assert '--n-episodes 4' in output and '--n-samples 512' in output
    assert not new.exists()


def test_launcher_persists_chain_and_finished_batch_is_noop(tmp_path,monkeypatch):
    run=source(tmp_path);new=tmp_path/'continued'
    def managed(experiment, commands, config, state):
        command=commands['controller']
        assert command[command.index('--n-episodes')+1]=='4'
        assert (new/'launch.json').exists() and (new/'continuation.json').exists()
        doc=json.loads((run/'controller.json').read_text())
        doc['args']['n_episodes']=4
        doc['episodes']=[dict(episode=i,completed=True,finish_reason='success',target_pos=[.025,.035,.075]) for i in range(4)]
        put(new/'controller.json',doc)
        return 'complete',0
    monkeypatch.setattr(launcher,'run_managed',managed)
    monkeypatch.setattr(launcher.Experiment,'shutdown',lambda *a:[])
    monkeypatch.setattr(launcher.ConfigurationLogs,'record',lambda *a,**kw:dict(mode='append',directory='fake'))
    assert launcher.main(['--continue-run',str(run),'--run-dir',str(new)])==0
    result=json.loads((new/'continuation.json').read_text())
    assert result['completed_episodes']==6 and result['remaining_episodes']==0
    assert len(result['completed_episode_sources'])==6
    assert launcher.main(['--continue-run',str(new)])==0


@pytest.mark.parametrize('extra',[['--n-episodes','12'],['--overwrite-logs'],['--','--temperature','.2'],['--no-logging']])
def test_experiment_changes_cannot_be_silently_mixed_into_batch(tmp_path,extra):
    run=source(tmp_path)
    with pytest.raises(SystemExit):launcher.main(['--continue-run',str(run),*extra])
