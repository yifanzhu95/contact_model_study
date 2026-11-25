"""Run globally shared N/noise comparisons; each child is simulation-only."""
import os
from pathlib import Path
import subprocess
import sys

pairs=[(256,.035),(256,.05),(256,.075),(512,.05),(1024,.05)]
for n,sigma in pairs:
    label=f'global_N{n}_s{str(sigma).replace(".","p")}'
    command=[sys.executable,str(Path(__file__).resolve().with_name('offline_tuning.py')),'--rows','2,8',
        '--candidates','smooth','--samples',str(n),'--noise',str(sigma),
        '--states','0,1','--steps','80','--temperature','.3','--label',label]
    print('START',label,flush=True)
    with Path('/tmp',label+'.log').open('w') as output:
        result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=os.environ.copy())
    print('END',label,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
