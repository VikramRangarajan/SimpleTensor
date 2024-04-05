# ruff: noqa: E402, F403
import importlib
import warnings



class Backend:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        if name == "module":
            return self.module
        else:
            return getattr(self.module, name)


np = Backend(importlib.import_module("numpy"))
fft = Backend(importlib.import_module("scipy.fft"))
fftpack = Backend(importlib.import_module("scipy.fftpack"))
interpolate = Backend(importlib.import_module("scipy.interpolate"))
linalg = Backend(importlib.import_module("scipy.linalg"))
ndimage = Backend(importlib.import_module("scipy.ndimage"))
signal = Backend(importlib.import_module("scipy.signal"))


def use_cupy():
    """
    Switches array backend to cupy, if available
    """
    # global np, fft, fftpack, interpolate, linalg, ndimage, signal
    if _found_cupy is not None and _found_cupyx is not None:
        warnings.warn("Cupy not installed, use_cupy() function does nothing")
        return
    np.module = importlib.import_module("cupy")
    fft.module = importlib.import_module("cupyx.scipy.fft")
    fftpack.module = importlib.import_module("cupyx.scipy.fftpack")
    interpolate.module = importlib.import_module("cupyx.scipy.interpolate")
    linalg.module = importlib.import_module("cupyx.scipy.linalg")
    ndimage.module = importlib.import_module("cupyx.scipy.ndimage")
    signal.module = importlib.import_module("cupyx.scipy.signal")


def use_numpy():
    """
    Switches array backend to numpy
    """
    # global np, fft, fftpack, interpolate, linalg, ndimage, signal
    np.module = importlib.import_module("numpy")
    fft.module = importlib.import_module("scipy.fft")
    fftpack.module = importlib.import_module("scipy.fftpack")
    interpolate.module = importlib.import_module("scipy.interpolate")
    linalg.module = importlib.import_module("scipy.linalg")
    ndimage.module = importlib.import_module("scipy.ndimage")
    signal.module = importlib.import_module("scipy.signal")

_found_cupy = importlib.util.find_spec("cupy")
_found_cupyx = importlib.util.find_spec("cupyx")
