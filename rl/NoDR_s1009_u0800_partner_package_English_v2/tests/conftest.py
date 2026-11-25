import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE.parents[1]))
os.environ['XDG_CACHE_HOME'] = str(PACKAGE/'runtime/cache')
os.environ['MPLCONFIGDIR'] = str(PACKAGE/'runtime/cache/matplotlib')
