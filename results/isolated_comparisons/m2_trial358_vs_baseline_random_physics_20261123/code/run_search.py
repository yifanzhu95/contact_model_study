from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"experiments"))
import run_bayes_opt as bo
from runtime import install
install(bo)
if __name__=="__main__":
    import ctypes
    import os
    import signal
    parent=os.getppid()
    expected=int(os.environ.get('M2_BO_SUPERVISOR_PID',parent))
    if parent != expected or ctypes.CDLL(None).prctl(1,signal.SIGTERM) != 0 or os.getppid()!=parent:
        raise SystemExit('Supervisor disappeared before search startup')
    raise SystemExit(bo.main())
