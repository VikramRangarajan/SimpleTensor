from enum import IntEnum


class Op(IntEnum):
    """
    Operation enumerator. An enumeration exists for every differentiable tensor operation.
    These indicate the operation that was used to create the tensor. For example, if
    tensor._op = ADD, then tensor was created by adding two other tensors together.
    """

    ADD = 0
    """Addition of two tensors"""
    CONVOLVE = 1
    """Apply n-dimensional convolution between 2 tensors"""
    CORRELATE = 2
    """Apply n-dimensional cross correlation between 2 tensors"""
    EXP = 3
    """Apply exponential function element wise"""
    EXPAND_DIMS = 4
    """Expand a tensor's with size-1 axes"""
    LOG = 5
    """Apply natural logarithm element wise"""
    MATMUL = 6
    """Matrix Multiplication of two tensors"""
    MAX = 7
    """Max over a tensor"""
    MEAN = 8
    """Mean over a tensor"""
    MIN = 9
    """Min over a tensor"""
    MUL = 10
    """Multiplication of two tensors"""
    NONE = 11
    """No operation, default _op value of a Tensor."""
    POW = 12
    """Tensor raised to a power of another tensor"""
    RELU = 13
    """Apply ReLU function element wise"""
    RESHAPE = 14
    """Reshape a tensor"""
    SQUEEZE = 15
    """Squeeze a tensor's size 1 dimensions"""
    STD = 16
    """Standard Deviation over a tensor"""
    SUB = 17
    """Subtraction of two tensors"""
    SUM = 18
    """Sum over a tensor"""
    T = 19
    """Transpose a tensor"""

    def __str__(self):
        return str_dict[self]

    def __repr__(self):
        return str(self)


str_dict = {
    Op.NONE: "",
    Op.ADD: "+",
    Op.SUB: "-",
    Op.MUL: "*",
    Op.MATMUL: "@",
    Op.POW: "**",
    Op.SUM: "sum",
    Op.MEAN: "mean",
    Op.LOG: "log",
    Op.EXP: "exp",
    Op.T: "T",
    Op.SQUEEZE: "squeeze",
    Op.EXPAND_DIMS: "expand_dims",
    Op.RESHAPE: "reshape",
    Op.RELU: "relu",
    Op.CONVOLVE: "convolve",
    Op.CORRELATE: "correlate",
    Op.MAX: "max",
    Op.MIN: "min",
    Op.STD: "std",
}
