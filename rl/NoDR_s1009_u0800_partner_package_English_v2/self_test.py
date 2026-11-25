"""Run offline parity cases. This script has no robot or simulator connection."""
from pathlib import Path
import argparse,json,hashlib
import torch
from portable_policy import Policy,encode_state,decode_action

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['both','torchscript','pytorch'],default='both')
    args=p.parse_args();torch.set_num_threads(4)
    manifest=json.loads((ROOT/'SHA256SUMS.json').read_text())
    for item in manifest['files']:
        f=ROOT/item['file']
        if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:
            raise AssertionError('Checksum mismatch: '+item['file'])
    cases=json.loads((ROOT/'reference_cases.json').read_text())['cases']
    backends=['torchscript','pytorch'] if args.backend=='both' else [args.backend]
    errors={}
    for backend in backends:
        policy=Policy(ROOT,backend=backend);mxraw=mxtarget=mxobs=0.
        for case in cases:
            out=policy.infer_state(case['state'])
            raw=torch.tensor(case['raw_action'],dtype=torch.float32).reshape(1,16)
            target=torch.tensor(case['joint_target_rad'],dtype=torch.float32).reshape(1,16)
            obs=torch.tensor(case['observation'],dtype=torch.float32).reshape(1,73)
            torch.testing.assert_close(out['observation'],obs,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(out['raw_action'],raw,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(out['joint_target_rad'],target,atol=2e-6,rtol=2e-5)
            mxraw=max(mxraw,float((out['raw_action']-raw).abs().max()))
            mxtarget=max(mxtarget,float((out['joint_target_rad']-target).abs().max()))
            mxobs=max(mxobs,float((out['observation']-obs).abs().max()))
        # Meaningful interface rejection checks; no actual actuation.
        bad=dict(cases[0]['state']);bad['object_quaternion_wxyz']=[0.,0.,0.,0.]
        try:encode_state(bad,policy.interface)
        except ValueError:pass
        else:raise AssertionError('Zero quaternion not rejected')
        bad=dict(cases[0]['state']);bad['elapsed_steps']=500
        try:encode_state(bad,policy.interface)
        except ValueError:pass
        else:raise AssertionError('Expired episode not rejected')
        try:policy.raw_action([[float('nan')]*73])
        except ValueError:pass
        else:raise AssertionError('Nonfinite observation not rejected')
        errors[backend]=dict(cases=len(cases),max_observation_error=mxobs,
             max_raw_action_error=mxraw,max_joint_target_error_rad=mxtarget)
    print(json.dumps(dict(status='PASS_OFFLINE_INTERFACE_PARITY',device='cpu',
        torch=torch.__version__,backends=errors,robot_commands_sent=0,physics_steps=0,
        interpretation='Inference/encoding/decoding parity only; not a robot performance test.'),indent=2))

if __name__=='__main__':main()
