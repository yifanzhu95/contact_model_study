"""Target-runtime checks without environment steps or PPO optimization."""
from dataclasses import replace
from importlib.metadata import distribution, version
from importlib.util import find_spec
import json
from pathlib import Path

import mujoco
import torch

from ContactModelStudy.RL.Execution import require_single_process
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.Provenance import source_runtime
from ContactModelStudy.Tasks.Registry import task_from_environment


def runtime_preflight(config, ppo, bank, *, check_backend=False):
    require_single_process()
    pins = {'torch': '2.14.1', 'gymnasium': '1.4.0', 'skrl': '2.1.0', 'mujoco': '3.6.0'}
    if config.device.startswith('cuda'):
        pins['warp-lang'] = '1.17.0'
    if config.physics_backend == 'kamino':
        pins['newton'] = '1.6.0'
    packages = {name: version(name) for name in pins}
    for name, expected in pins.items():
        if packages[name].split('+')[0] != expected:
            raise RuntimeError(f'{name} requires audited version {expected}, found {packages[name]}')
    optional = {'pinocchio_eval': 'pinocchio', 'drake_eval': 'pydrake'}.get(config.physics_backend)
    if optional and find_spec(optional) is None:
        raise RuntimeError(f'{config.physics_backend} requires {optional}; no model fallback')
    if config.physics_backend in ('mjwarp', 'comfree', 'xpbd'):
        from ContactModelStudy.RL.MjwarpBackend import COMFREE_COMMIT
        origin = json.loads(distribution('comfree_warp').read_text('direct_url.json') or '{}')
        if origin.get('vcs_info', {}).get('commit_id') != COMFREE_COMMIT:
            raise RuntimeError('Install the project-pinned comfree_warp commit; no model fallback')
    batch = config.num_envs * ppo.steps_per_update
    if batch < 2 or batch % ppo.mini_batches:
        raise ValueError('PPO batch must have >=2 samples and divide evenly into mini_batches')
    bank.require_task(config.task_name)
    task = task_from_environment(config)
    model = mujoco.MjModel.from_xml_path(task.getModelPath())
    if (model.nq, model.nv, model.nu) != (23, 22, 16):
        raise ValueError('The selected scene must use the existing LEAP 23/22/16 layout')
    # These tiny allocations exercise the requested device without simulating.
    value = torch.ones(1, device=config.device) + 1
    if value.item() != 2:
        raise RuntimeError('Requested Torch device failed a basic arithmetic check')
    if config.device.startswith('cuda'):
        import warp as wp
        value = wp.zeros(1, dtype=wp.float32, device=config.device)
        wp.synchronize_device(config.device)
        if not torch.isfinite(wp.to_torch(value)).all():
            raise RuntimeError('Requested Warp device failed an allocation/readout check')
    report = {'execution_status': 'passed', 'requested_environment': config.to_dict(),
              'packages': packages, 'case_bank_sha256': bank.sha256,
              'scene': str(Path(task.getModelPath()).resolve()), 'scene_layout': [model.nq, model.nv, model.nu],
              'runtime': source_runtime(Path(__file__).resolve().parents[2], config.device),
              'physics_steps': 0, 'training_transitions': 0, 'ppo_updates': 0,
              'backend_reset_checked': False,
              'qualification': 'runtime_only_not_training_throughput_or_full_batch_capacity'}
    if check_backend:
        probe = replace(config, num_envs=min(config.num_envs, 2))
        env = None
        try:
            env = LeapVectorEnv(probe, bank)
            obs = env.reset_all()
            if tuple(obs.shape) != (probe.num_envs, 73) or not torch.isfinite(obs).all():
                raise RuntimeError('Backend reset/readout produced invalid observations')
            report.update(backend_reset_checked=True, probe_num_envs=probe.num_envs,
                          probe_environment=env.describe())
        finally:
            if env is not None:
                env.close()
    return report
