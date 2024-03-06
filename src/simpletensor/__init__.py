# ruff: noqa: E401, F401, F403

try:
    import cupy, cupyx

    x = cupy.array([1, 2, 3])
    y = cupy.array([4, 5, 6])
    x + y
    USE_GPU = True
    print("Cupy successfully imported, using GPU")
except Exception as e:
    if isinstance(e, ImportError):
        print("Cupy failed to import, using CPU (numpy & scipy)")
    else:
        print("Cupy caused error while importing: " + e)
    USE_GPU = False

from .tensor import *
from .operation import *
