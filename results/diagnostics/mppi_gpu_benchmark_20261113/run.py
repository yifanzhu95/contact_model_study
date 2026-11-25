"""Monitor authorized offline workers with read-only nvidia-smi samples."""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import time
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PYTHON='/home/chetan/miniconda3/envs/mj_env/bin/python'
GPUS={0:'GPU-1ada3dc2-9cb0-d2f6-2b12-80346a4659e0',3:'GPU-3beb4f90-65ae-0ef8-e126-97a81e20ab17'}
cases=[('M2_separate_gpu0','M2',0,40),('M6_separate_gpu0','M6',0,40),('M6_shared_vision_gpu3','M6',3,30)]
def monitor(gpu,pid=None):
 r=subprocess.run(['nvidia-smi','--id='+GPUS[gpu],
  '--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory',
  '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
 values=next(csv.reader(r.stdout.splitlines(),skipinitialspace=True))
 record=dict(timestamp=time.time(),gpu_index=int(values[0]),gpu_memory_mib=int(values[1]),
  total_memory_mib=int(values[2]),gpu_util_percent=int(values[3]),memory_util_percent=int(values[4]))
 r=subprocess.run(['nvidia-smi','--id='+GPUS[gpu],
  '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
 processes=[dict(pid=int(row[0]),memory_mib=int(row[1])) for row in csv.reader(r.stdout.splitlines(),skipinitialspace=True) if len(row)==2]
 record['processes']=processes;record['worker_memory_mib']=next((p['memory_mib'] for p in processes if p['pid']==pid),0)
 return record
for name,model,gpu,seconds in cases:
 folder=OUT/name;folder.mkdir(exist_ok=False)
 samples=[]
 print('START',name,'GPU',gpu,flush=True)
 for _ in range(4):
  item=monitor(gpu);item['phase']='baseline';samples.append(item);time.sleep(.4)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[gpu],PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
 with (folder/'worker.log').open('w') as log:
  process=subprocess.Popen([PYTHON,str(OUT/'worker.py'),'--model',model,'--out',str(folder),'--seconds',str(seconds)],
      cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  last=time.monotonic()
  try:
   while process.poll() is None:
    item=monitor(gpu,process.pid)
    try:item['phase']=json.loads((folder/'phase.json').read_text())['phase']
    except (OSError,ValueError):item['phase']='imports'
    samples.append(item)
    if time.monotonic()-last>10:
     print('MONITOR',name,item['phase'],'util',item['gpu_util_percent'],'workerMiB',item['worker_memory_mib'],flush=True);last=time.monotonic()
    time.sleep(.4)
  except BaseException:
   process.terminate();process.wait(timeout=10);raise
 (folder/'telemetry.json').write_text(json.dumps(dict(worker_pid=process.pid,exit_code=process.returncode,samples=samples),indent=2)+'\n')
 if process.returncode:
  print('FAILED',name,(folder/'worker.log').read_text()[-3000:],flush=True);raise SystemExit(process.returncode)
 result=json.loads((folder/'result.json').read_text())
 print('DONE',name,'mean_ms',round(result['planning_ms']['mean'],2),'calls',result['calls'],flush=True)
print('ALL BENCHMARKS COMPLETE',flush=True)
