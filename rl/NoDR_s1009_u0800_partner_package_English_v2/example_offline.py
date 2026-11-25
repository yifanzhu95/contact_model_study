"""Print an archived simulation example, without connecting to any hardware."""
from pathlib import Path
import json
import torch
from portable_policy import Policy

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
case=json.loads((ROOT/'reference_cases.json').read_text())['cases'][0]
policy=Policy(ROOT,backend='torchscript')
result=policy.infer_state(case['state'])
print('OFFLINE EXAMPLE ONLY: archived simulation state; no motor commands sent.')
print('Policy order: '+', '.join(j['joint_name'] for j in policy.interface['joints']))
print('Raw network output:',result['raw_action'][0].tolist())
print('Absolute joint targets [rad]:',result['joint_target_rad'][0].tolist())
