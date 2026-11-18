"""把本项目全部运行缓存留在D盘。"""
import os
import sys
from pathlib import Path


def configure():
    root = Path(__file__).resolve().parents[1]
    for key, name in (('TEMP', 'tmp'), ('TMP', 'tmp'), ('MPLCONFIGDIR', 'mpl'),
                      ('PYTHONPYCACHEPREFIX', 'pycache'), ('NUMBA_CACHE_DIR', 'numba')):
        folder = root/'runs/runtime'/name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    return root
