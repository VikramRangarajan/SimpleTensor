# ruff: noqa: E402, F403
import importlib
import warnings

cupy = importlib.util.find_spec("cupy")
cupyx = importlib.util.find_spec("cupyx")


class Backend:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        if name == "module":
            return self.module
        else:
            return getattr(self.module, name)


def use_cupy():
    """
    Switches array backend to cupy, if available
    """    
    global np, scipy
    if cupy is None:
        warnings.warn("Cupy not installed, use_cupy() function does nothing")
        return
    np = Backend(cupy)
    scipy = Backend(cupyx.scipy)


def use_numpy():
    """
    Switches array backend to numpy
    """    
    global np, scipy
    import numpy as _np
    import scipy as _scipy

    np = Backend(_np)
    scipy = Backend(_scipy)


if cupy is not None and cupyx is not None:
    print("Cupy successfully imported, using GPU (cupy)")
    use_cupy()
else:
    print("Cupy failed to import, using CPU (numpy & scipy)")
    use_numpy()


from .tensor import *
from .operation import *
from .grad_manager import *
from .functions import *
