"""Offline replay must preserve written input timing and measured initial state."""
import mujoco
import numpy as np
import pytest

from validate_leap_actuator_replay import changed_writes,filter_state,replay,sample_time


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 0"/>
      <worldbody><body><joint name="joint" type="hinge" damping="0.1"/>
        <geom type="capsule" size="0.01 0.1" mass="0.1" pos="0 0 0.1"/>
      </body></worldbody>
      <actuator><general joint="joint" dyntype="filterexact" dynprm="0.02"
        gainprm="3" biastype="affine" biasprm="0 -3 -0.1"/></actuator>
      </mujoco>''')


def test_read_at_write_time_does_not_see_motion_from_future_target():
    m=model();times=np.array([0,.01,.03,.04,.1])
    q,v,n=replay(m,[.2],[0],[.2],[.2],times,[(.03,np.array([.4]))])
    np.testing.assert_allclose(q[:3,0],.2,atol=1e-12)
    assert q[-1,0]>.2 and v[-1,0]>0 and n>0
    assert m.opt.timestep==pytest.approx(.002)


def test_replay_starts_from_measurements_without_projecting_or_resetting():
    m=model();times=np.array([0,.01,.1])
    q,v,_=replay(m,[-.1],[.03],[.2],[.2],times,[])
    assert q[0,0]==-.1 and v[0,0]==.03
    assert q[-1,0]!=q[0,0]


def test_activation_history_uses_model_filter_and_past_writes_only():
    tau=np.array([.02,.05]);a=np.array([0.,0.]);first=np.array([1.,2.]);second=np.array([-.5,.4])
    events=[(10000,first),(30000,second),(100000,np.array([100.,100.]))]
    activation,target=filter_state(a,0,events,50000,tau)
    after_first=first+(a-first)*np.exp(-.02/tau)
    expected=second+(after_first-second)*np.exp(-.02/tau)
    np.testing.assert_allclose(activation,expected);np.testing.assert_array_equal(target,second)


def test_duplicate_heartbeats_dont_change_input_but_small_changes_are_preserved():
    rows=[dict(write_succeeded=True,write_finished_us=t,written_target=[value])
          for t,value in [(1,0),(2,0),(3,1e-8),(4,1e-8),(5,.2)]]
    events=changed_writes(rows,4)
    assert [t for t,u in events]==[1,3]
    assert events[-1][1][0]==1e-8


def test_read_timing_sensitivity_does_not_use_publish_timestamp():
    row=dict(read_started_us=100,read_finished_us=117,state_timestamp_us=200)
    assert sample_time(row,'start')==100
    assert sample_time(row,'midpoint')==108.5
    assert sample_time(row,'end')==117


def test_finer_time_resolution_converges_with_identical_input_events():
    m=model();times=np.arange(0,.3,.01)
    events=[(.035,np.array([.3])),(.147,np.array([-.1]))]
    coarse,_,_=replay(m,[0],[0],[0],[0],times,events,max_step=.0005)
    fine,_,_=replay(m,[0],[0],[0],[0],times,events,max_step=.00025)
    assert np.max(abs(coarse-fine))<.005
