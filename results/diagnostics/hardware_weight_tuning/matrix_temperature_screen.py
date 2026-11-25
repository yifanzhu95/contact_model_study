"""Bounded offline temperature screen, two simulation workers on one idle GPU."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import os
import subprocess
import sys

BASE=Path(__file__).resolve().parent

def run(row):
    label=f'train_row_{row:02d}'
    path=BASE/(label+'.jsonl')
    if path.exists() and len(path.read_text().splitlines())>=2:
        return row,'already recorded'
    command=[sys.executable,str(Path(__file__).resolve().with_name('offline_tuning.py')),'--rows',str(row),
        '--candidates','smooth','--samples','512','--noise','.05',
        '--states','0,1','--steps','120','--temperatures','.1,.3','--label',label]
    with Path('/tmp',label+'.log').open('w') as output:
        result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=os.environ.copy())
    if result.returncode:return row,f'error {result.returncode}'
    records=[json.loads(x) for x in path.read_text().splitlines()]
    # Try a softer temperature only when the first two choices never yield
    # two successful starts. Keep the search budget explicit and bounded.
    if not any(sum(e['success'] for e in r.get('episodes',[]))==2 for r in records):
        command[command.index('--temperatures')+1]='1.0'
        with Path('/tmp',label+'.log').open('a') as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=os.environ.copy())
        if result.returncode:return row,f'fallback error {result.returncode}'
    return row,'complete'

with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(run,row) for row in range(1,16)]
    for future in as_completed(futures):
        row,status=future.result()
        print('ROW',row,status,flush=True)
