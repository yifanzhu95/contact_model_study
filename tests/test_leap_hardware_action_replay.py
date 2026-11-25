"""Recorded hardware replay must preserve timing and expose position limits."""
import json

import mujoco
import numpy as np
import pytest

import replay_leap_hardware_actions as replay


def make_recording(tmp_path, missing=False, completed=True):
    source=tmp_path/'controller.json'
    source.write_text(json.dumps({'episodes':[dict(episode=0,completed=completed,finish_reason='success',
        steps=[dict(action_published=True,action_timestamp_us=t) for t in [100,200]])]}))
    rows=[]
    for index,stamp in enumerate([100,200]):
        if missing and index==1:continue
        target=np.full(16,.5);target[0]=-2 if index==0 else 9
        rows.append(dict(write_succeeded=True,new_action=True,written_action_timestamp_us=stamp,
            write_finished_us=1000000+index*73000,written_target=target.tolist(),held_target_before_read=np.zeros(16).tolist()))
    (tmp_path/'bridge_telemetry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return source


def test_exact_written_values_and_irregular_timing_survive_with_explicit_bounds(tmp_path):
    source=make_recording(tmp_path);model=mujoco.MjModel.from_xml_path(str(replay.DEFAULT_MODEL))
    s=replay.load_schedule(source,1,model)
    np.testing.assert_allclose(s['offsets_s'],[0,.073])
    assert s['original_targets'][0][0]==-2 and s['original_targets'][1][0]==9
    lo,hi=replay.control_limits(model)
    assert s['applied_targets'][0][0]==lo[0] and s['applied_targets'][1][0]==hi[0]
    assert s['adjusted_commands']==2 and s['adjusted_joint_values']==2
    np.testing.assert_allclose(np.array(s['applied_targets'])-s['original_targets'],s['position_limit_adjustments_rad'])


def test_unmatched_written_actions_stop_before_hardware(tmp_path):
    model=mujoco.MjModel.from_xml_path(str(replay.DEFAULT_MODEL))
    with pytest.raises(ValueError,match='exact successful bridge-write'):
        replay.load_schedule(make_recording(tmp_path,missing=True),1,model)


def test_aborted_episode_cannot_be_replayed_by_accident(tmp_path):
    model=mujoco.MjModel.from_xml_path(str(replay.DEFAULT_MODEL))
    with pytest.raises(ValueError,match='completed'):
        replay.load_schedule(make_recording(tmp_path,completed=False),1,model)


def test_default_validation_does_not_start_motion(monkeypatch):
    monkeypatch.setattr(replay.Replay,'run',lambda *_:pytest.fail('No hardware in validation mode'))
    assert replay.main([])==0
