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
    FLIP = 5
    """Flip tensor over axis"""
    INDEX = 6
    """Index an array"""
    LOG = 7
    """Apply natural logarithm element wise"""
    MATMUL = 8
    """Matrix Multiplication of two tensors"""
    MUL = 9
    """Multiplication of two tensors"""
    NONE = 10
    """No operation, default _op value of a Tensor."""
    POW = 11
    """Tensor raised to a power of another tensor"""
    RELU = 12
    """Apply ReLU function element wise"""
    RESHAPE = 13
    """Reshape a tensor"""
    SQUEEZE = 14
    """Squeeze a tensor's size 1 dimensions"""
    SUB = 15
    """Subtraction of two tensors"""
    SUM = 16
    """Sum over a tensor"""
    T = 17
    """Transpose a tensor"""

    def __str__(self):
        return str_dict[self]

    def __repr__(self):
        return str(self)


str_dict = {
    Op.ADD: "+",
    Op.CONVOLVE: "convolve",
    Op.CORRELATE: "correlate",
    Op.EXP: "exp",
    Op.EXPAND_DIMS: "expand_dims",
    Op.FLIP: "flip",
    Op.INDEX: "index",
    Op.LOG: "log",
    Op.MATMUL: "@",
    Op.MUL: "*",
    Op.NONE: "",
    Op.POW: "**",
    Op.RELU: "relu",
    Op.RESHAPE: "reshape",
    Op.SQUEEZE: "squeeze",
    Op.SUB: "-",
    Op.SUM: "sum",
    Op.T: "T",
}
