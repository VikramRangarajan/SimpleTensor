# ruff: noqa: E402, F403
from simpletensor.array_backend import use_cupy, use_numpy, _found_cupy, _found_cupyx

if _found_cupy is not None and _found_cupyx is not None:
    use_cupy()
else:
    use_numpy()


from .tensor import *
from .operation import *
from .grad_manager import *
from .functions import *
from .array_backend import *
