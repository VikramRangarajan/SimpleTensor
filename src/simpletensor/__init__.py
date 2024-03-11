# ruff: noqa: E402, F403
import importlib

cupy = importlib.util.find_spec("cupy")
cupyx = importlib.util.find_spec("cupyx")

if cupy is not None and cupyx is not None:
    USE_GPU = True
    print("Cupy successfully imported, using GPU")
else:
    print("Cupy failed to import, using CPU (numpy & scipy)")
    USE_GPU = False


from .tensor import *
from .operation import *
