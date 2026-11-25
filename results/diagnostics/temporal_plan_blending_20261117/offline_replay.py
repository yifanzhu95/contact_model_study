"""Regenerate full MPPI plans on recorded hardware states. NO hardware/LCM clients.

Old recordings omit the final mean and actuator-observer state. Infer observer
activation from recorded estimated commands/timestamps and validate regenerated
first actions against recorded ones. This is NOT closed-loop counterfactual
evaluation: alternative commands never replace the recorded measured states.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ContactModelStudy.Hardware.Workflow import SceneTaskMixin, build_rollout
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config

DEFAULT = [
    ('results/diagnostics/m2_usb_latency_1ms/leap_experiment_20261117_020910_654466/controller.json', 6),
    ('results/diagnostics/m2_usb_latency_1ms_clip015_temp010_paced/leap_experiment_20261117_022352_015441/controller.json', 5),
]


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--step-limit', type=int, default=0)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    sources = [(str(opts.source),opts.episodes)] if opts.source else DEFAULT
    manifest = dict(method='regenerated plans on recorded measured states',
        no_hardware=True, no_lcm=True, device='cuda:0', episodes=[],
        limitations=['The original full means and observer activations were not recorded.',
                    'Observer state reconstructed from recorded estimated targets and UTC timestamps.',
                    'Recorded states are fixed for every alpha; object retention/success cannot be evaluated.'])
    started = time.monotonic()
    with wp.ScopedDevice('cuda:0'):
        for source_index, (source, count) in enumerate(sources):
            path = ROOT/source if not Path(source).is_absolute() else Path(source)
            data = json.loads(path.read_text())
            assert data['args']['rollout_model'] == 'M2'
            assert data['args']['control_mode'] == 'pos_relative'
            assert data['planner_config']['action_rate_weight'] == 0
            scene = opts.output/f'saved_scene_{source_index}.xml'
            scene.write_text(data['scenes']['rollout']['scene_xml'])
            cls = type('RecordedCube', (SceneTaskMixin, CubeReorient), {})
            task = cls(scene, LeapReorientConfig(**data['scenes']['rollout']['task_config']))
            task.params['init_ctrl'] = np.asarray(data['hardware_config']['robot_qref'])
            task.params['fallen_z'] = data['failure_z']
            task.setJointReference(data['joint_reference'])
            task.setCostWeights(data['weights'])
            args = SimpleNamespace(**{**data['args'],'device':'cuda:0'})
            bounds = task.mjm.actuator_ctrlrange
            sim = build_rollout(args,task,bounds[:,0],bounds[:,1])
            cfg = json.loads(json.dumps(asdict(sim.config)))
            saved_cfg = {**data['rollout_config'],'device':'cuda:0'}
            assert cfg == saved_cfg, (cfg,saved_cfg)
            planner = MPPI(sim,task,MPPI_Config(**data['planner_config']))
            tau = task.mjm.actuator_dynprm[:,0]
            previous_calls = 1  # Original process consumed one warmup draw.
            try:
                for episode in data['episodes'][:count]:
                    assert episode['completed'] and episode['finish_reason'] in ('success','failure','timeout')
                    task.setTargetPosition(episode['target_pos'])
                    task.setJointReference(episode['home'])
                    task.setGoal(episode['goal'])
                    planner.Reset()
                    state = planner.SaveState()
                    state['resample_count'] = previous_calls
                    planner.LoadState(state)
                    steps = episode['steps'][:opts.step_limit or None]
                    activation = np.asarray(steps[0]['q'][:16],dtype=float).copy()
                    held = activation.copy()
                    act_at = timestamp(steps[0]['timestamp_utc'])
                    means, actions, activation_history, durations = [],[],[],[]
                    for index, step in enumerate(steps):
                        at = timestamp(step['timestamp_utc'])
                        activation = held + (activation-held)*np.exp(-max(0.,at-act_at)/tau)
                        act_at = at
                        activation_history.append(activation.copy())
                        sim.activation.assign(activation.astype(np.float32))
                        before = time.monotonic()
                        proposed = planner.Plan(np.asarray(step['q']),np.asarray(step['q_dot']),u=held)
                        wp.synchronize()
                        durations.append(time.monotonic()-before)
                        if not planner.last_plan_ok:
                            raise RuntimeError(f'Offline planner failed at {path.parent.name} e{episode["episode"]+1} s{index}')
                        means.append(planner.last_mean.copy())
                        actions.append(np.asarray(proposed)-np.asarray(step['plan_base']))
                        pub = step['action_timestamp_us']*1e-6
                        activation = held+(activation-held)*np.exp(-max(0.,pub-act_at)/tau)
                        act_at = pub
                        held = np.asarray(step['u'])
                    means, actions = np.asarray(means),np.asarray(actions)
                    recorded = np.array([s['action_delta'] for s in steps])
                    error = np.degrees(abs(actions-recorded))
                    output = opts.output/f'source{source_index}_episode{episode["episode"]+1:02d}.npz'
                    np.savez_compressed(output,mean=means,regenerated_delta=actions,recorded_delta=recorded,
                        command_base=np.array([s['feedback']['command_qpos'] for s in steps]),
                        q=np.array([s['q'] for s in steps]),v=np.array([s['q_dot'] for s in steps]),
                        published_s=np.array([s['action_timestamp_us']*1e-6 for s in steps]),
                        activation=np.array(activation_history),planning_s=durations)
                    record = dict(source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        source_episode=episode['episode']+1,outcome=episode['finish_reason'],steps=len(steps),
                        full_recorded_episode=len(steps)==len(episode['steps']),file=output.name,
                        ctrl_period=data['control_period'],bounds=data['hardware_config']['ctrl_range'],
                        regeneration_error_deg=dict(mean=float(error.mean()),p95=float(np.percentile(error,95)),max=float(error.max())),
                        first_action_error_deg=error[0].tolist(),
                        offline_planning_mean_s=float(np.mean(durations)),
                        weights=data['weights'],joint_reference=episode['home'],
                        noise_seed_offset=previous_calls)
                    manifest['episodes'].append(record)
                    previous_calls += len(episode['steps'])
                    manifest['elapsed_s'] = time.monotonic()-started
                    (opts.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                    print(json.dumps({k:record[k] for k in ('source_episode','outcome','steps','regeneration_error_deg','offline_planning_mean_s')}),flush=True)
            finally:
                sim.Close()
    print('Offline regeneration complete:',opts.output,flush=True)


if __name__ == '__main__':
    main()
