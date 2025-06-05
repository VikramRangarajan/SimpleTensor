# ruff: noqa: E402, F403
from simpletensor.array_backend import (
    _found_cupy,
    _found_cupyx,
    use_cupy,
    use_numpy,
)
from simpletensor.array_backend import (
    get_array_module as get_array_module,
)

if _found_cupy is not None and _found_cupyx is not None:
    use_cupy()
else:
    use_numpy()


from .array_backend import *
from .functions import *
from .grad_manager import *
from .operation import *
from .tensor import *
