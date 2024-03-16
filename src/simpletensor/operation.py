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
    INDEX = 5
    """Index an array"""
    LOG = 6
    """Apply natural logarithm element wise"""
    MATMUL = 7
    """Matrix Multiplication of two tensors"""
    MAX = 8
    """Max over a tensor"""
    MEAN = 9
    """Mean over a tensor"""
    MIN = 10
    """Min over a tensor"""
    MUL = 11
    """Multiplication of two tensors"""
    NONE = 12
    """No operation, default _op value of a Tensor."""
    POW = 13
    """Tensor raised to a power of another tensor"""
    RELU = 14
    """Apply ReLU function element wise"""
    RESHAPE = 15
    """Reshape a tensor"""
    SQUEEZE = 16
    """Squeeze a tensor's size 1 dimensions"""
    STD = 17
    """Standard Deviation over a tensor"""
    SUB = 18
    """Subtraction of two tensors"""
    SUM = 19
    """Sum over a tensor"""
    T = 20
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
    Op.INDEX: "index",
}
