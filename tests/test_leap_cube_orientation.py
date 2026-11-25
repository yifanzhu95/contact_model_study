"""Tracked cube labels and world-axis protocol, no camera/hardware access."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ContactModelStudy.Hardware.CubeOrientation import describe_start, terminal_lines


def wxyz(r):
    return r.as_quat()[[3,0,1,2]]


@pytest.mark.parametrize('label,face,axis',[
    ('+X','B','+Y'),('-X','O','-Y'),('+Y','I','-X'),('-Y','I','+X'),
    ('+Z','R','+Z'),('-Z','R','+Z'),
])
def test_six_quarter_turns_have_correct_top_face_and_unique_orientation(label,face,axis):
    q=wxyz(Rotation.from_euler(label[1].lower(),90 if label[0]=='+' else -90,degrees=True))
    d=describe_start(q,[1,0,0,0])
    assert d['rotation_label']==label
    assert d['top_face']['letter']==face and d['top_face']['object_axis']==axis
    assert d['top_face']['tilt_deg']==pytest.approx(0,abs=1e-5)
    assert d['quarter_turn_mismatch_deg']==pytest.approx(0,abs=1e-8)
    assert not d['top_face']['ambiguous']


def test_same_top_face_distinguishes_yaw_and_two_i_faces():
    plus=describe_start(wxyz(Rotation.from_euler('z',90,degrees=True)),[1,0,0,0])
    minus=describe_start(wxyz(Rotation.from_euler('z',-90,degrees=True)),[1,0,0,0])
    assert plus['world_positive_y_face']['letter']==minus['world_positive_y_face']['letter']=='I'
    assert plus['world_positive_y_face']['color']=='blue'
    assert minus['world_positive_y_face']['color']=='pink'
    assert plus['world_positive_x_face']['letter']=='O' and minus['world_positive_x_face']['letter']=='B'


def test_sign_invariance_nonidentity_goal_and_world_axis_convention():
    goal=Rotation.from_euler('xyz',[12,23,-17],degrees=True)
    q=wxyz(Rotation.from_euler('y',90,degrees=True)*goal)
    a=describe_start(q,wxyz(goal));b=describe_start(-q,-wxyz(goal))
    assert a['rotation_label']==b['rotation_label']=='+Y'
    assert a['relative_rotvec_world_deg']==pytest.approx([0,90,0],abs=1e-8)


def test_tilted_face_flags_edge_and_non_protocol_pose():
    d=describe_start(wxyz(Rotation.from_euler('x',45,degrees=True)),[1,0,0,0])
    assert d['top_face']['ambiguous'] and d['rotation_label']=='other'
    assert 'near edge' in terminal_lines(0,d,[])[0]


def test_terminal_counts_completed_attempts_but_excludes_aborts():
    d=describe_start(wxyz(Rotation.from_euler('z',90,degrees=True)),[1,0,0,0])
    old=[dict(completed=True,starting_orientation=d),dict(completed=False,starting_orientation=d)]
    lines=terminal_lines(2,d,old)
    assert 'Episode 3 start: top R (green, +Z)' in lines[0]
    assert '+Z 90.0 deg' in lines[0] and '+Z=1' in lines[1]
    assert 'Facing world +X: O (red, -Y)' in lines[1]


@pytest.mark.parametrize('bad',[[0,0,0,0],[1,2,3],[1,np.nan,0,0]])
def test_invalid_tracking_quaternion_rejected(bad):
    with pytest.raises(ValueError):describe_start(bad,[1,0,0,0])
