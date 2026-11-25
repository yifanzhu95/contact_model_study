import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
os.environ['XDG_CACHE_HOME']=str(ROOT/'cache/xdg')
