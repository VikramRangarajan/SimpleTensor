from functools import wraps
from typing import Callable

import numpy

from ..array_backend import cupy
from ..tensor import Tensor


def not_implemented(f: Callable):
    def _f(*args, **kwargs):
        raise NotImplementedError(f"{f.__name__} is not implemented!")

    return _f


def array_api_creation_wrap(f: Callable):
    @wraps(f)
    def _f(*args, **kwargs):
        device = kwargs.pop("device", "cpu")
        if device != "cpu":
            assert cupy is not None
            xp = cupy
        else:
            xp = numpy
        xp_func = getattr(xp, f.__name__)
        _out = xp_func(*args, **kwargs)
        assert isinstance(_out, xp.ndarray)
        out = Tensor(_out, _out.dtype, copy=False, device=_out.device)
        return out

    return _f


def numpy_wrap(f: Callable):
    _f = wraps(getattr(numpy, f.__name__))

    return _f
