import numpy as np
from articulated_motion import minimum_jerk,duration,model,collision_pairs,screen


def test_smooth_interpolation_obeys_speed_acceleration_and_zero_endpoint_velocity():
    start=np.zeros(16);end=np.linspace(-.5,.8,16);T=duration(start,end)
    t=np.linspace(0,T,4001);q=np.array([minimum_jerk(start,end,x/T) for x in t])
    v=np.gradient(q,t,axis=0);a=np.gradient(v,t,axis=0)
    assert abs(v).max()<=.1801 and abs(a).max()<=.3501
    assert abs(v[[0,-1]]).max()<1e-5
    np.testing.assert_allclose(q[[0,-1]],[start,end])


def test_screen_rejects_known_interfinger_collision():
    m=model();pairs=collision_pairs(m);q=np.array([.2,0,.2,.45]*3+[0,0,.55,.35])
    bad=q.copy();bad[8:12]=[.45,-.35,.2,.45]
    assert screen(m,[q,q],pairs)['passed']
    assert not screen(m,[q,bad],pairs)['passed']
