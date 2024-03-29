from . import USE_GPU
import numpy as np
from .operation import Op
from importlib import import_module

if USE_GPU:
    np = import_module("cupy")
    scipy = import_module("cupyx.scipy")
else:
    import numpy as np
    import scipy

fftconvolve = scipy.signal.fftconvolve


class Tensor:
    """
    Tensor class. This is a box around a numpy array, but with support for reverse mode automatic differentiation.

    Parameters
    ----------
    values : array-like or scalar
        Values to put into Tensor
    dtype : data type, optional
        numpy data type for underlying numpy array, by default None
    name : str, optional
        Name of tensor. If not specified, or if name is taken, a random 10-letter name is given, by default None
    """

    grad_enabled = True

    def __init__(self, values, dtype=None, copy=True, name=None):
        dtype = dtype or np.float64
        self._array = np.array(values, dtype=dtype, copy=copy)

        # Gives tensor unique name
        if name is None:
            self.name = "".join(
                [
                    np.random.choice(
                        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    )
                    for _ in range(10)
                ]
            )
        else:
            self.name = name

        if Tensor.grad_enabled:
            self.zero_grad()

        self._op = Op.NONE
        self._parents = []
        self._backward = lambda: None

    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def size(self):
        return self._array.size

    def __repr__(self):
        return (
            f'Tensor(\n{str(self._array)},\nshape={self.shape},\nname="{self.name}"\n)'
        )

    def __str__(self):
        return str(self._array)

    def backward(self, zero_grad=True):
        r"""
        Backpropagation Method. If it is called on a size 1 tensor L, then the gradient for each tensor X which
        is a part of its creation graph w.r.t. this tensor's value is calculated (:math:`\frac{\partial L}{\partial X}`),
        and stored in the tensor's .grad attribute.

        Parameters
        ----------
        zero_grad : bool, optional
            Option to set all gradients in autodiff graph to 0 before setting them.
            If False, then new gradients are added to previous ones, by default True

        Raises
        ------
        NotImplementedError
            Raised if the tensor which backward() is called on does not have a size of 1.
        """
        if self.size != 1:
            raise NotImplementedError(
                "Can only call backward() on a single value Tensor"
            )
        topo_sorted_nodes = []
        visited = set()

        def topological_sort(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    topological_sort(parent)
                topo_sorted_nodes.append(v)

        topological_sort(self)
        if zero_grad:
            for node in topo_sorted_nodes:
                node.zero_grad()
        self.grad = np.ones_like(self._array, dtype=self.dtype)
        for node in topo_sorted_nodes[::-1]:
            node._backward()

    def zero_grad(self):
        """
        Sets the gradient of the tensor to a 0 array with the same
        shape as the tensor.
        """
        self.grad = np.zeros_like(self._array, dtype=self.dtype)

    """
    BINARY OPERATIONS

    TODO: Rest of binary operations
    Optional:
    convolve
    correlate
    >, <, >=, <=, !=, ==
    
    """

    def __add__(self, other):
        """
        Adds two tensor-like objects together.

        Returns
        -------
        Tensor
            Sum of both inputs
        """
        other = astensor(other)
        res = Tensor(self._array + other._array)
        if Tensor.grad_enabled:
            res._op = Op.ADD
            res._parents = [self, other]
            op_axes = _broadcasted_axes(self, other)

            def _backward():
                self.grad += np.sum(res.grad, axis=op_axes[0])
                other.grad += np.sum(res.grad, axis=op_axes[1])

            res._backward = _backward
        return res

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other):
        """
        Subtracts two tensor-like objects.

        Returns
        -------
        Tensor
            Subtraction of both inputs
        """
        other = astensor(other)
        res = Tensor(self._array - other._array)
        if Tensor.grad_enabled:
            res._op = Op.SUB
            res._parents = [self, other]
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
        """
        Multiplies (element-wise) two tensor-like objects together.

        Returns
        -------
        Tensor
            Multiplication (Hadamard product) of two tensors
        """
        other = astensor(other)
        res = Tensor(self._array * other._array)
        if Tensor.grad_enabled:
            res._op = Op.MUL
            res._parents = [self, other]
            op_axes = _broadcasted_axes(self, other)

            def _backward():
                self.grad += np.sum(res.grad * other._array, axis=op_axes[0])
                other.grad += np.sum(res.grad * self._array, axis=op_axes[1])

            res._backward = _backward
        return res

    __imul__ = __mul__
    __rmul__ = __mul__

    def __matmul__(self, other):
        """
        Matrix Multiplication. Mimics numpy @ behavior.

        There are 4 cases:

        - Both arguments are 2D (normal matrix multiplication)
        - Either argument is >2D (broadcasting is done accordingly)
        - First Tensor is 1D (1 is added to the beginning of the tensor's shape, matrix multiplication is applied, then the added 1 is removed from the output tensor's shape)
        - Second Tensor is 1D (1 is added to the end of the tensor's shape, matrix multiplication is applied, then the added 1 is removed from the output tensor's shape)

        Parameters
        ----------
        other : array-like
            Right side of matrix multiplication

        Returns
        -------
        Tensor
            Matrix product of two input arrays
        """
        other = astensor(other)

        prepended = self.ndim == 1
        appended = other.ndim == 1

        if prepended and appended:
            return (self * other).sum()

        res = Tensor(self._array @ other._array)
        if Tensor.grad_enabled:
            res._op = Op.MATMUL
            res._parents = [self, other]

            if self.ndim == other.ndim == 2:
                # 2D Tensor @ 2D Tensor

                def _backward():
                    self.grad += res.grad @ other._array.T
                    other.grad += self._array.T @ res.grad
            else:
                if prepended:
                    # 1D Tensor @ >=2D Tensor
                    op_axes = _broadcasted_axes(np.array(0), other[..., 0, 0])

                    def _backward():
                        self.grad += (
                            (res.grad[..., None, :] @ other._array.swapaxes(-1, -2))
                            .sum(op_axes[0])
                            .squeeze(-2)
                        )
                        other.grad += (
                            self._array[:, None] @ res.grad[..., None, :]
                        ).sum(op_axes[1])

                elif appended:
                    # >=2D Tensor @ 1D Tensor
                    op_axes = _broadcasted_axes(self._array[..., 0, 0], np.array(0))

                    def _backward():
                        self.grad += (
                            res.grad[..., None] @ other._array[:, None].T
                        ).sum(op_axes[0])
                        other.grad += (
                            (self._array.swapaxes(-1, -2) @ res.grad[..., None])
                            .sum(op_axes[1])
                            .squeeze(-1)
                        )

                else:
                    # >2D Tensor @ >2D Tensor
                    op_axes = _broadcasted_axes(
                        self._array[..., 0, 0], other[..., 0, 0]
                    )

                    def _backward():
                        self.grad += (res.grad @ other._array.swapaxes(-1, -2)).sum(
                            op_axes[0]
                        )
                        other.grad += (self._array.swapaxes(-1, -2) @ res.grad).sum(
                            op_axes[1]
                        )

            res._backward = _backward
        return res

    def __rmatmul__(self, other):
        other = astensor(other)
        return other @ self

    __imatmul__ = __matmul__

    def __pow__(self, other):
        """
        One tensor raised to the power of another tensor.

        Parameters
        ----------
        other : Tensor
            Exponent tensor

        Returns
        -------
        Tensor
            Result tensor of A**B
        """
        other = astensor(other)
        res = Tensor(self._array**other._array)
        if Tensor.grad_enabled:
            res._op = Op.POW
            res._parents = [self, other]
            op_axes = _broadcasted_axes(self, other)

            def _backward():
                self.grad += np.sum(
                    res.grad * other._array * self._array ** (other._array - 1),
                    axis=op_axes[0],
                )
                other.grad += np.sum(
                    res.grad * res._array * np.log(self._array),
                    axis=op_axes[1],
                )

            res._backward = _backward
        return res

    def __rpow__(self, other):
        other = astensor(other)
        return other**self

    __ipow__ = __pow__

    def __truediv__(self, other):
        """
        Division between two Tensors. A division A / B is
        defined as A * B**-1.

        Parameters
        ----------
        other : array-like
            Divisor tensor

        Returns
        -------
        Tensor
            Result tensor of A / B
        """
        return self * other**-1

    def __rtruediv__(self, other):
        return self**-1 * other

    __itruediv__ = __truediv__

    """

    Optional:
    __abs__
    __float__, __int__, __complex__
    """

    """
    UNARY OPERATIONS

    TODO:
    log
    exp
    transpose
    squeeze
    flip
    expand_dims
    reshape
    relu

    Optional:
    sin, cos, tan
    inverse trig functions
    max
    min
    """

    def __getitem__(self, index):
        """
        Index an array, using `numpy's indexing semantics <https://numpy.org/doc/stable/user/basics.indexing.html>`_

        Parameters
        ----------
        index : Iterable or scalar index
            index used to index the tensor

        Returns
        -------
        Tensor
            Result of Tensor indexing
        """
        res = astensor(self._array[index])  # Copy will be made of underlying np array
        if Tensor.grad_enabled:
            res._op = Op.INDEX
            res._parents = [self]

            def _backward():
                np.add.at(self.grad, index, res.grad)

            res._backward = _backward
        return res

    def __setitem__(self, index, values):
        self._array[index] = values

    def __neg__(self):
        return self * -1

    def __pos__(self):
        return self

    def sum(self, axis=None):
        """
        Sums an array over given axes

        Parameters
        ----------
        axis : tuple, optional
            Axes over which to sum over, by default None

        Returns
        -------
        Tensor
            Result of sum operation over axes
        """
        res = Tensor(self._array.sum(axis=axis))
        if Tensor.grad_enabled:
            res._op = Op.SUM
            res._parents = [self]

            def _backward():
                notnullaxis: tuple = axis or tuple(range(self.ndim - 1))
                self.grad += np.broadcast_to(
                    np.expand_dims(res.grad, notnullaxis), self.shape
                )

            res._backward = _backward
        return res

    def mean(self, axis=None):
        """
        Mean of an array over given axes

        Parameters
        ----------
        axis : tuple, optional
            Axes over which to sum over, by default None

        Returns
        -------
        Tensor
            Result of mean operation over axes
        """
        size = float(self.size)
        if axis is not None:
            size = float(np.prod([self.shape[i] for i in axis]))
        return self.sum(axis=axis) / size

    def std(self, axis=None, ddof=0):
        """
        Standard deviation over given axes.

        Parameters
        ----------
        axis : tuple, optional
            Axes over which to sum over, by default None
        ddof : int, optional
            Delta degrees of freedom, where the divisor is N - ddof.
            ddof = 1 will give you sample std. By default, ddof = 0

        Returns
        -------
        Tensor
            Standard deviation of input tensor along axes
        """
        df = float(self.size - ddof)
        if axis is not None:
            df = float(np.prod([self.shape[i] for i in axis]) - ddof)
        return (((self - self.mean(axis)) ** 2).sum(axis) / df) ** 0.5

    def var(self, axis=None, ddof=0):
        """
        Variance over given axes.

        Parameters
        ----------
        axis : tuple, optional
            Axes over which to sum over, by default None
        ddof : int, optional
            Delta degrees of freedom, where the divisor is N - ddof.
            ddof = 1 will give you sample std. By default, ddof = 0

        Returns
        -------
        Tensor
            Variance of input tensor along axes
        """
        df = float(self.size - ddof)
        if axis is not None:
            df = float(np.prod([self.shape[i] for i in axis]) - ddof)
        return ((self - self.mean(axis)) ** 2).sum(axis) / df

    def max(self, axis=None):
        """
        Max over given axes.

        Parameters
        ----------
        axis : tuple, optional
            Axes over which to perform max over, by default None
        """
        all_axes = np.arange(self.ndim)
        axis = axis or all_axes
        axis = tuple(axis)
        other_axes = tuple(np.setdiff1d(all_axes, axis))
        other_shape = tuple(self.shape[i] for i in other_axes)
        other_size = np.prod(other_shape, dtype="int")
        res = self.transpose(axis + other_axes)
        res = res.reshape((-1, other_size))
        argmax = res._array.argmax(0)
        res = res[argmax, np.arange(other_size)]
        res = res.reshape(other_shape)

        return res


def astensor(a, dtype=None):
    """
    Converts input to a Tensor

    Parameters
    ----------
    a : array-like
        Iterable or scalar quantity (not jagged)
    dtype : numpy datatype, optional
        Datatype for which the underlying numpy array's datatype should be, by default None

    Returns
    -------
    Tensor
        Tensor version of input
    """
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
