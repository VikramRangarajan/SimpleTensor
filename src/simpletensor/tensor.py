from . import USE_GPU
import numpy as np
from .operation import Op
import uuid

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class Tensor:
    """
    Tensor class

    Raises
    ------
    NotImplementedError
        For invalid operations
    """

    taken_names = {None}

    def __init__(self, values, dtype=None, name=None):
        dtype = dtype or np.float64
        self._array = np.array(values, dtype=dtype)

        # Gives tensor unique name
        if (name is not None and name in Tensor.taken_names) or name is None:
            n = None
            while n in Tensor.taken_names:
                n = uuid.uuid1().hex.replace("-", "")
            self.name = n

        # Numpy ndarray attributes
        self.shape = self._array.shape
        self.ndim = self._array.ndim
        self.dtype = self._array.dtype
        self.size = self._array.size
        self.grad = np.zeros_like(self._array)

        self._op = Op.NONE
        self._children = []
        self._grad_params = dict()
        self._backward = lambda: None

    def __repr__(self):
        return (
            f'Tensor(\n{str(self._array)},\nshape={self.shape},\nname="{self.name}"\n)'
        )

    def __str__(self):
        return str(self._array)

    def __getitem__(self, x):
        return astensor(self._array[x])

    def backward(self, zero_grad=True):
        if self.size != 1:
            raise NotImplementedError(
                "Can only call backward() on a single value Tensor"
            )
        topo_sorted_nodes = []
        visited = set()

        def topological_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    topological_sort(child)
                topo_sorted_nodes.append(v)

        topological_sort(self)  # compare my solution to this
        if zero_grad:
            for node in topo_sorted_nodes:
                node.grad = np.zeros_like(node.grad, dtype=node.dtype)
        self.grad = np.ones_like(self._array, dtype=self.dtype)
        for node in topo_sorted_nodes[::-1]:
            node._backward()

    """
    PYTHON MAGIC FUNCTIONS
    """

    def __add__(self, other):
        other = astensor(other)
        res = Tensor(self._array + other._array)
        res._op = Op.ADD
        res._children = [self, other]
        op_axes = _broadcasted_axes(self, other)

        def _backward():
            self.grad += np.sum(res.grad, axis=op_axes[0])
            other.grad += np.sum(res.grad, axis=op_axes[1])

        res._backward = _backward
        return res

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other):
        other = astensor(other)
        res = Tensor(self._array - other._array)
        res._op = Op.SUB
        res._children = [self, other]
        op_axes = _broadcasted_axes(self, other)

        def _backward():
            self.grad += np.sum(res.grad, axis=op_axes[0])
            other.grad -= np.sum(res.grad, axis=op_axes[1])

        res._backward = _backward
        return res

    def __rsub__(self, other):
        return -self + other

    __isub__ = __sub__

    def __mul__(self, other):
        other = astensor(other)
        res = Tensor(self._array * other._array)
        res._op = Op.MUL
        res._children = [self, other]
        op_axes = _broadcasted_axes(self, other)

        def _backward():
            self.grad += np.sum(res.grad * other._array, axis=op_axes[0])
            other.grad += np.sum(res.grad * self._array, axis=op_axes[1])

        res._backward = _backward
        return res

    __imul__ = __mul__
    __rmul__ = __mul__

    """
    TODO:
    __mul__: Testing
    __imul__: Testing
    __rmul__: Testing
    __matmul__
    __rmatmul__
    __imatmul__
    __truediv__
    __rtruediv__
    __itruediv__
    __floordiv_
    __rfloordiv__
    __ifloordiv__
    __pow__
    __rpow__
    __ipow__

    Optional:
    __abs__
    __float__, __int__, __complex__
    """

    """
    UNARY OPERATORS
    """

    def __neg__(self):
        res = Tensor(-self._array)
        res._op = Op.NEG
        res.children = [self]

        def _backward():
            self.grad += -res.grad

        res._backward = _backward
        return res

    def __pos__(self):
        return self

    def sum(self, axis=None):
        res = Tensor(self._array.sum(axis=axis))
        res._op = Op.SUM
        res._children = [self]

        def _backward():
            notnullaxis: tuple = axis or tuple(range(self.ndim - 1))
            self.grad += np.broadcast_to(
                np.expand_dims(res.grad, notnullaxis), self.shape
            )

        res._backward = _backward
        return res


def astensor(a, dtype=None):
    if isinstance(a, Tensor):
        return a
    else:
        return Tensor(a, dtype)


def _padded_broadcast_shapes(*a, max_ndim=None):
    return [(0,) * (max_ndim - arr.ndim) + arr.shape for arr in a]


def _broadcasted_axes(*a):
    res_shape = np.broadcast_shapes(*(arr.shape for arr in a))
    arr_pshapes = _padded_broadcast_shapes(*a, max_ndim=len(res_shape))
    axes = [() for _ in a]
    for axis in range(0, len(res_shape)):
        for arr_num in range(len(a)):
            if arr_pshapes[arr_num][axis] == 0 or (
                arr_pshapes[arr_num][axis] == 1 and res_shape[axis] != 1
            ):
                axes[arr_num] = axes[arr_num] + (axis,)
    return axes
