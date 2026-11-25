#!/usr/bin/env python3
"""Static package audit and independent reference-array arithmetic; no actor execution.

All generated files remain beside this script. No robot, camera, simulation,
network, PyTorch import, package-script execution, or pickle loading occurs.
"""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PACKAGE = OUT.parent
REPO = PACKAGE.parents[1]


def read(name):
    return json.loads((PACKAGE / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    size = math.sqrt(sum(v * v for v in q))
    w, x, y, z = [v / size for v in q]
    return [[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]]


def six(R):
    return [R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1]]


def encode(state, joints):
    R = rotation(state['object_quaternion_wxyz'])
    G = rotation(state['goal_quaternion_wxyz'])
    relative = [[sum(R[k][i] * G[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    q = [2 * (v - joint['joint_position_range_rad'][0]) /
         (joint['joint_position_range_rad'][1] - joint['joint_position_range_rad'][0]) - 1
         for v, joint in zip(state['joint_position_rad'], joints)]
    u = [2 * (v - joint['actuator_target_range_rad'][0]) /
         (joint['actuator_target_range_rad'][1] - joint['actuator_target_range_rad'][0]) - 1
         for v, joint in zip(state['previous_target_rad'], joints)]
    return (q + [v / 20 for v in state['joint_velocity_rad_s']] +
            [v * 10 for v in state['object_position_world_m']] + six(R) +
            [(a-b) * 10 for a, b in zip(state['goal_position_world_m'], state['object_position_world_m'])] +
            six(relative) + state['object_linear_velocity_world_m_s'] +
            [v / 10 for v in state['object_angular_velocity_world_rad_s']] +
            u + [1 - state['elapsed_steps'] / 500])


def main():
    interface, contract = read('interface.json'), read('original_policy_contract.json')
    manifest = read('SHA256SUMS.json')
    integrity = [dict(file=r['file'], ok=sha(PACKAGE / r['file']) == r['sha256']) for r in manifest['files']]
    if not all(r['ok'] for r in integrity):
        raise ValueError('Original package checksum mismatch')
    rows = []
    cases = read('reference_cases.json')['cases']
    counts = {}
    for case in cases:
        state = case['state']
        observation = encode(state, interface['joints'])
        target = [max(j['actuator_target_range_rad'][0], min(j['actuator_target_range_rad'][1],
                    u + .064 * math.tanh(a)))
                  for j, u, a in zip(interface['joints'], state['previous_target_rad'], case['raw_action'])]
        if len(observation) != 73:
            raise ValueError('Observation dimension mismatch')
        model = case['id'].split('-')[0]
        counts[model] = counts.get(model, 0) + 1
        rows.append(dict(id=case['id'], model=model, elapsed_steps=state['elapsed_steps'],
            observation_max_abs_error=max(abs(a-b) for a, b in zip(observation, case['observation'])),
            decoded_target_max_abs_error_rad=max(abs(a-b) for a, b in zip(target, case['joint_target_rad'])),
            maximum_target_increment_rad=max(abs(a-b) for a, b in zip(target, state['previous_target_rad']))))
    with (OUT / 'reference_arithmetic_checks.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    max_obs = max(r['observation_max_abs_error'] for r in rows)
    max_target = max(r['decoded_target_max_abs_error_rad'] for r in rows)
    if max_obs > 2e-6 or max_target > 2e-6:
        raise ValueError(f'Independent encoder/decoder arithmetic differs: {max_obs}, {max_target}')
    layers = []
    with zipfile.ZipFile(PACKAGE / 'policy_noDR_s1009_u0800.ts.pt') as archive:
        embedded = json.loads(archive.read('actor/extra/contract.json'))
        for name in archive.namelist():
            if '/code/' not in name or not name.endswith('.py'):
                continue
            code = archive.read(name).decode()
            inp = re.search(r'in_features : Final\[int\] = (\d+)', code)
            out = re.search(r'out_features : Final\[int\] = (\d+)', code)
            if inp and out:
                layers.append([int(inp[1]), int(out[1])])
    hardware = json.loads((REPO / 'configs/leap_hardware.json').read_text())
    train_palm = next(r for r in interface['static_training_scene_root_bodies'] if r['name'] == 'palm')
    t, h = train_palm['quaternion_wxyz'], hardware['palm_quat_wxyz']
    dot = abs(sum(a*b for a, b in zip(t, h))) / math.sqrt(sum(a*a for a in t)*sum(a*a for a in h))
    calibrated = ET.parse(REPO / 'scenes/leap/hardware/calibrated_hand.xml').getroot()
    examples = [dict(name=r.get('name'), gainprm=r.get('gainprm'), biasprm=r.get('biasprm'),
                     dyntype=r.get('dyntype')) for r in list(calibrated.find('actuator'))[:4]]
    scene_path = REPO / 'scenes/leap/env_leap_rollout_cube_high_high.xml'
    original_scene_hash = next(r['sha256'] for r in read('provenance.json')['source_checksums']
                               if r['name'] == scene_path.name)
    try:
        installed_torch = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        installed_torch = None
    evaluation = read('evaluation_summary.json')
    result = dict(analysed_at_utc=datetime.now(timezone.utc).isoformat(), package=str(PACKAGE),
        python=sys.executable, original_file_integrity=integrity,
        original_file_count=len(manifest['files']) + 1,
        embedded_torchscript_contract_equals_external=embedded == contract,
        network=dict(linear_layers=layers, activation='ELU', recurrent=False,
                     mlp_parameter_count=sum(a*b+b for a, b in layers),
                     normalization='(raw observation - saved mean) / (saved std + 0.01)'),
        reference_arithmetic=dict(cases=len(rows), by_model=counts,
            elapsed_steps=sorted({r['elapsed_steps'] for r in rows}),
            observation_max_abs_error=max_obs, decoded_target_max_abs_error_rad=max_target,
            interpretation='Independent standard-library encoder/decoder arithmetic against saved arrays, not policy forward-pass parity.'),
        training_contract=contract, interface=interface, archived_evaluation=evaluation,
        current_repository_comparison=dict(
            training_palm_tilt_deg=2 * math.degrees(math.atan2(abs(t[3]), abs(t[1]))),
            current_palm_rotation_difference_deg=2 * math.degrees(math.acos(min(1., dot))),
            current_delta_reference=hardware['delta_reference'], current_ctrl_range=hardware['ctrl_range'],
            package_position_gain_values=sorted({j['simulation_position_gain'] for j in interface['joints']}),
            calibrated_actuator_examples=examples,
            current_scene_hash_equals_recorded_training_scene=sha(scene_path) == original_scene_hash,
            caution='Hash mismatch does not establish which physical values changed. Exact original physics assets are absent.'),
        runtime=dict(torch_installed_in_audit_environment=installed_torch,
            package_recorded_runtime=read('runtime_tested.json'), actor_forward_passes=0,
            full_package_self_test_run=False, simulation_steps=0, hardware_commands=0,
            dependencies_installed=False),
        missing_training_reproduction_assets=['training source and full physics implementation',
            'original scene XML and meshes', 'full initial-state bank',
            'full PPO checkpoint including critic, optimizer and RNG',
            'numerical definitions of reward terms and strict success joint gate'],
        receiver_integration_missing=['verified motor order/sign/zero/limits mapping',
            'registration into the training world', 'real-state estimation and synchronized velocity inputs',
            'previous accepted-target feedback and fixed 64 ms policy scheduling',
            'hardware reset and experiment lifecycle'])
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    text = f'''RL POLICY PACKAGE ANALYSIS

Scope: extracted and inspected only. No actor forward passes, no simulation,
no camera use, no hardware commands and no dependency installation.
All analysis files and code are confined to this directory.

FILES AND INTEGRITY
20 original files. All 19 hashes in SHA256SUMS.json match. Both actor formats,
portable_policy.py, 112 saved input/output cases, the interface/joint order,
PPO contract, archived evaluation and provenance are included.
The TorchScript embedded contract exactly matches original_policy_contract.json.

TRAINING
M2 rollout, high/high geometry, no domain randomization; PPO seed 1009.
800 updates (188 original + 612 continuation), 8192 environments, 32 steps per
update: 209,715,200 transitions. This is the scheduled final checkpoint, not a
best-checkpoint selection. PPO: five epochs, eight minibatches, adaptive learning
rate initially 0.001, clipping 0.2, gamma 0.99, GAE lambda 0.95, entropy 0.003.
Training uses Gaussian exploration (initial scalar std 0.7, learned within
[0.05,1]); exported inference uses the deterministic mean, no exploration noise.

NETWORK AND INTERFACE
MLP 73 -> 512 -> 512 -> 256 -> 16, ELU, {result['network']['mlp_parameter_count']:,} parameters.
No recurrent/history state. Frozen observation normalization is inside the actor:
(observation - mean)/(std + 0.01). Do not normalize twice.
73 observations include measured joints/velocities, object position/orientation,
goal error, world-frame object linear/angular velocity, previous target and time.
Quaternions WXYZ; rotation6 stores the first two columns of a rotation matrix.
Joint-state and actuator-target normalization use different position ranges.

ACTION
u_next = clamp(u_previous + 0.064*tanh(raw_action), actuator_lower, actuator_upper)
The decoder returns absolute position targets in radians. The base is the
previous accepted target, NOT the measured joint position. At most 0.064 rad
(3.67 deg) requested increment per decision before absolute joint-limit clipping.
At 64 ms this corresponds to a maximum target change rate of 1 rad/s, not a
guarantee about actual joint velocity. No separate lead limiter is supplied.
Exactly one policy decision per 64 ms (15.625 Hz), target held between decisions.
The initial previous target must match the established grasp target.

TASK
Cube only, two fixed canonical local-X +/-90 deg target orientations.
Goal position [0.02,0.035,0.085] m in training world. This is not an arbitrary-goal
policy or a full six-direction rotation policy. 500 controls = 32 s.
Success is nonterminal; training encourages holding. Ends on drop or step limit.
Exact reward and strict-success gate code are not included.

ARCHIVED EVALUATION (PROVIDED, NOT RE-RUN HERE)
                         M2                M5
Evaluation episodes      128               128
Strict arrival           128/128           49/128
Strict at final step     128/128           41/128
Last150 continuous hold  128/128           0/128
Drops                    0                 6
M5 -X strict arrival: 49/64; M5 +X strict arrival: 0/64.
These are simulation results, not hardware evidence. The policy transfers poorly
even to the supplied alternative simulator evaluation; no-DR robustness should
not be inferred from the perfect in-model score.

INDEPENDENT CHECKS HERE
112 reference cases: 56 M2 + 56 M5, elapsed controls 0,1,9,49,99,349,499.
Reconstructed observations maximum absolute discrepancy: {max_obs:.9g}.
Reconstructed decoded targets maximum discrepancy: {max_target:.9g} rad.
These verify stored interface arithmetic only. Neural-network inference was NOT
run. PyTorch in the inspected mj_env: {installed_torch or 'NOT INSTALLED'}.
The supplied offline_self_test_QA.json reports both backends passing on the
sender's runtime; this has not been independently reproduced in this environment.

DIFFERENCES FROM THIS REPOSITORY'S HARDWARE SETUP
1. Training palm tilt is approximately 27 deg; current hardware configuration is
   30 deg (relative mounting rotation {result['current_repository_comparison']['current_palm_rotation_difference_deg']:.6f} deg).
2. Training specifies uniform position gain 3.0, velocity bias -0.01. Current
   calibrated hand uses per-joint gains/biases and actuator filtering. This
   package does not describe training on that calibrated model.
3. Current hardware delta_reference is {hardware['delta_reference']!r}; policy
   requires previous accepted-target accumulation. Directly reusing the MPPI
   measured-position delta path would change the policy's trained action meaning.
4. Fixed 64 ms RL scheduling is part of the contract; an unpaced MPPI-style loop
   would change target accumulation and episode timing.
5. Motor/frame/state-estimator mapping is an unfilled template. Existing LEAP
   bridge code is not integrated by this package.
6. Original training physics, cube mass, regional friction and full state-bank
   assets cannot be reconstructed from this bundle. Provenance supplies hashes,
   not the actual files. The current scene XML hash differs from its recorded
   training scene hash; this alone does not identify physical differences.

FILES TO READ
../interface.json                exact 73D/16D contract, ranges and frames
../portable_policy.py            encoder, decoder and CPU inference wrapper
../original_policy_contract.json PPO/training settings
../evaluation_summary.json       archived M2/M5 metrics
analysis.json                    detailed machine-readable audit
reference_arithmetic_checks.csv  one row per saved reference case

The archive's suggested next-step commands were treated as documentation, not
instructions to execute them. Original package files remain unchanged.
'''
    (OUT / 'analysis.txt').write_text(text)
    print(json.dumps(dict(status='STATIC_ANALYSIS_COMPLETE', checksums_passed=len(integrity),
        reference_cases=len(rows), observation_max_abs_error=max_obs,
        target_max_abs_error_rad=max_target, actor_forward_passes=0,
        report=str(OUT / 'analysis.txt')), indent=2))


if __name__ == '__main__':
    main()
