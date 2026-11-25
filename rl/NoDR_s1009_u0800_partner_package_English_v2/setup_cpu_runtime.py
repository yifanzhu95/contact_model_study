#!/usr/bin/env python3
"""Prepare a package-local CPU environment; no robot/camera access.

Run using mj_env Python. Existing NumPy/MuJoCo/SciPy/LCM come from that base
environment read-only. Only the local venv receives PyTorch and its dependencies.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    for name in ['numpy','scipy','mujoco','lcm']:
        if importlib.util.find_spec(name) is None:
            parser.error(f'Base Python needs {name}; run this with the existing mj_env interpreter')
    for name in ['tmp','pip-cache','logs','cache']:(RUNTIME/name).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(RUNTIME/'tmp'),
        PIP_CACHE_DIR=str(RUNTIME/'pip-cache'),PIP_DISABLE_PIP_VERSION_CHECK='1',
        XDG_CACHE_HOME=str(RUNTIME/'cache'))
    python=RUNTIME/'venv/bin/python'
    if not args.check_only:
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(RUNTIME/'venv')],check=True,env=env)
        with (RUNTIME/'logs/pytorch_setup.log').open('w') as log:
            subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements-hardware-cpu.txt')],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not python.is_file():parser.error('Private runtime missing; omit --check-only to create it')
    with (RUNTIME/'logs/package_self_test.log').open('w') as log:
        subprocess.run([str(python),'-B',str(ROOT/'self_test.py')],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    code='''import json,sys,importlib.metadata as m
names=['torch','numpy','scipy','mujoco','lcm','opencv-python']
print(json.dumps(dict(executable=sys.executable,base_prefix=sys.base_prefix,python=sys.version,packages={n:m.version(n) for n in names}),indent=2))'''
    result=subprocess.run([str(python),'-B','-c',code],check=True,env=env,capture_output=True,text=True)
    (RUNTIME/'environment.json').write_text(result.stdout)
    print(result.stdout)
    print('CPU runtime and both 112-case package self-tests passed. No hardware accessed.')


if __name__=='__main__':main()
