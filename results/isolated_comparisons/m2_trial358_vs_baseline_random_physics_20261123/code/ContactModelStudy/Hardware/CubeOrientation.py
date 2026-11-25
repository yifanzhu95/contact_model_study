"""Describe a tracked cube in the MuJoCo world frame, once before control.

Labels verified against both iris_cube.obj and the adjusted 70-mm tracker
mesh/texture. Colors describe the physical lettering (the two I faces differ).
This classifies the tracked pose; it does not independently recognize letters.
"""
import numpy as np
from scipy.spatial.transform import Rotation


FACES = (
    ('+X', 'I', 'blue', (1, 0, 0)),
    ('-X', 'I', 'pink', (-1, 0, 0)),
    ('+Y', 'B', 'red', (0, 1, 0)),
    ('-Y', 'O', 'red', (0, -1, 0)),
    ('+Z', 'R', 'green', (0, 0, 1)),
    ('-Z', 'S', 'yellow', (0, 0, -1)),
)
ROTATIONS = ('-X', '+X', '-Y', '+Y', '-Z', '+Z')


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Cube orientation needs a finite nonzero wxyz quaternion')
    q = q / np.linalg.norm(q)
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def describe_start(quaternion, goal_quaternion):
    current, goal = rotation(quaternion), rotation(goal_quaternion)
    matrix = current.as_matrix()
    normals = np.array([face[3] for face in FACES]) @ matrix.T

    def facing(direction):
        scores = normals @ np.asarray(direction)
        order = np.argsort(scores)[::-1]
        i = int(order[0]);axis, letter, color, _ = FACES[i]
        result = dict(letter=letter, color=color, object_axis=axis,
            tilt_deg=float(np.degrees(np.arccos(np.clip(scores[i], -1, 1)))),
            ambiguous=bool(scores[order[0]]-scores[order[1]] < .1))
        if result['ambiguous']:
            other=FACES[int(order[1])]
            result['alternative']=dict(letter=other[1],color=other[2],object_axis=other[0])
        return result

    # Starting rotation relative to the goal, expressed about world axes.
    relative = current * goal.inv()
    candidates=[]
    for label in ROTATIONS:
        vector=np.zeros(3);vector['XYZ'.index(label[1])] = (1 if label[0]=='+' else -1)*np.pi/2
        candidate=Rotation.from_rotvec(vector)
        candidates.append((float(np.degrees((relative*candidate.inv()).magnitude())),label))
    residual,label=min(candidates)
    angle=float(np.degrees(relative.magnitude()))
    return dict(top_face=facing((0,0,1)),world_positive_x_face=facing((1,0,0)),
        world_positive_y_face=facing((0,1,0)),rotation_label=label if residual<=15 else 'other',
        nearest_quarter_turn=label,quarter_turn_mismatch_deg=residual,
        angle_from_goal_deg=angle,relative_rotvec_world_deg=np.degrees(relative.as_rotvec()).tolist(),
        quaternion_wxyz=(np.asarray(quaternion)/np.linalg.norm(quaternion)).tolist(),
        goal_quaternion_wxyz=(np.asarray(goal_quaternion)/np.linalg.norm(goal_quaternion)).tolist(),
        convention='Top is world +Z. Signed starting rotation from goal about MuJoCo world axes; right-hand rule.',
        source='tracked quaternion; labels assume the adjusted 70mm cube mesh frame')


def face_text(face):
    text=f"{face['letter']} ({face['color']}, {face['object_axis']})"
    if face.get('ambiguous'):
        other=face['alternative']
        text+=f" / near edge with {other['letter']} ({other['color']}, {other['object_axis']})"
    return text


def terminal_lines(episode, description, previous_episodes):
    label=description['rotation_label'];angle=description['angle_from_goal_deg']
    turn=(f"{label} {angle:.1f} deg from goal" if label!='other' else
          f"off-protocol {angle:.1f} deg from goal; nearest {description['nearest_quarter_turn']} 90 deg")
    counts={key:0 for key in ROTATIONS}
    for item in previous_episodes:
        if item.get('completed'):
            key=item.get('starting_orientation',{}).get('rotation_label')
            if key in counts:counts[key]+=1
    return [f"Episode {episode+1} start: top {face_text(description['top_face'])}; "
            f"{turn} (quarter-turn mismatch {description['quarter_turn_mismatch_deg']:.1f} deg).",
            f"  Facing world +X: {face_text(description['world_positive_x_face'])}; "
            f"+Y: {face_text(description['world_positive_y_face'])}. "
            "Completed starts this run: "+', '.join(f'{key}={counts[key]}' for key in ROTATIONS)]
