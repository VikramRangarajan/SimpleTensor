from enum import IntEnum


class Op(IntEnum):
    """
    Operation enumerator. An enumeration exists for every differentiable tensor operation.
    These indicate the operation that was used to create the tensor. For example, if 
    tensor._op = ADD, then tensor was created by adding two other tensors together.
    """

    NONE = 0
    """No operation, default _op value of a Tensor."""
    ADD = 1
    """Addition of two tensors"""
    SUB = 2
    """Subtraction of two tensors"""
    MUL = 3
    """Multiplication of two tensors"""
    DIV = 4
    """Division of two tensors"""
    MATMUL = 5
    """Matrix Multiplication of two tensors"""
    POW = 6
    """Tensor raised to a power of another tensor"""
    SUM = 7
    """Sum over a tensor"""
    MEAN = 8
    """Mean over a tensor"""
    LOG = 9
    """Apply natural logarithm element wise"""
    EXP = 10
    """Apply exponential function element wise"""
    T = 11
    """Transpose a tensor"""
    SQUEEZE = 12
    """Squeeze a tensor's size 1 dimensions"""
    EXPAND_DIMS = 13
    """Expand a tensor's with size-1 axes"""
    RESHAPE = 14
    """Reshape a tensor"""
    RELU = 15
    """Apply ReLU function element wise"""
    CONVOLVE = 16
    """Apply n-dimensional convolution between 2 tensors"""
    CORRELATE = 17
    """Apply n-dimensional cross correlation between 2 tensors"""
    MAX = 18
    """Max over a tensor"""
    MIN = 19
    """Min over a tensor"""
    STD = 20
    """Standard Deviation over a tensor"""
    NEG = 21
    """Negate a tensor"""

    def __str__(self):
        return str_dict[self]

    def __repr__(self):
        return str(self)


str_dict = {
    Op.NONE: "",
    Op.ADD: "+",
    Op.SUB: "-",
    Op.MUL: "*",
    Op.DIV: "/",
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
    Op.NEG: "negate",
}
