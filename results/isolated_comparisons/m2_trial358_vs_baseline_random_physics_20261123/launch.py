"""Launch/resume the simulation evaluation in a detached process."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent


def main():
    if sys.argv[1:]!=['--start']:raise SystemExit('Use launch.py --start')
    cfg=json.loads((OUT/'config.json').read_text())
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            args=(path/'cmdline').read_bytes().split(b'\0')
            if str(OUT/'evaluate.py').encode() in args:
                raise SystemExit(f'Evaluation already running: PID {path.name}')
        except OSError:pass
    values=subprocess.check_output(['nvidia-smi','-i',cfg['gpu_uuid'],
        '--query-gpu=uuid,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True)
    uuid,free,temp=[v.strip() for v in values.split(',')]
    assert uuid==cfg['gpu_uuid'] and float(free)>=4096 and float(temp)<80,values
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('CUDA_MPS_','M2_BO_')):env.pop(key)
    env.pop('PYTHONPATH',None)
    env.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        WARP_CACHE_PATH=str(OUT/'warp_cache'),CUDA_CACHE_PATH=str(OUT/'cuda_cache'),MUJOCO_GL='egl')
    command=[sys.executable,'-u',str(OUT/'evaluate.py'),'--run']
    with (OUT/'evaluation.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=OUT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=process.pid,command=command,gpu=cfg['gpu_uuid'],
                started_at_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'process.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
