# ruff: noqa: E402, F403
import importlib
import warnings
from importlib import util
from types import ModuleType
from typing import Any, Literal

import numpy

try:
    import cupy  # type: ignore

    get_array_module = cupy.get_array_module
except Exception:
    cupy = None

    def get_array_module(*args) -> ModuleType:
        return numpy


DeviceType = Literal["cpu", "cuda"]


class Backend(ModuleType):
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name) -> Any:
        if name == "module":
            return self.module
        else:
            return getattr(self.module, name)


np = Backend(importlib.import_module("numpy"))
fft = Backend(importlib.import_module("scipy.fft"))
fftpack = Backend(importlib.import_module("scipy.fftpack"))
linalg = Backend(importlib.import_module("numpy.linalg"))
ndimage = Backend(importlib.import_module("scipy.ndimage"))
signal = Backend(importlib.import_module("scipy.signal"))


def use_cupy():
    """
    Switches array backend to cupy, if available
    """
    if _found_cupy is None or _found_cupyx is None:
        warnings.warn("Cupy not installed, use_cupy() function does nothing")
        return
    if np.module.__name__ != "cupy":
        np.module = importlib.import_module("cupy")
        fft.module = importlib.import_module("cupyx.scipy.fft")
        fftpack.module = importlib.import_module("cupyx.scipy.fftpack")
        linalg.module = importlib.import_module("cupy.linalg")
        ndimage.module = importlib.import_module("cupyx.scipy.ndimage")
        signal.module = importlib.import_module("cupyx.scipy.signal")
        print("Successfully switched to GPU (CuPy backend)")


def use_numpy():
    """
    Switches array backend to numpy
    """
    if np.module.__name__ != "numpy":
        np.module = importlib.import_module("numpy")
        fft.module = importlib.import_module("scipy.fft")
        fftpack.module = importlib.import_module("scipy.fftpack")
        linalg.module = importlib.import_module("scipy.linalg")
        ndimage.module = importlib.import_module("scipy.ndimage")
        signal.module = importlib.import_module("scipy.signal")
        print("Successfully switched to CPU (NumPy backend)")


_found_cupy = util.find_spec("cupy")
_found_cupyx = util.find_spec("cupyx")
