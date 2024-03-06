from enum import IntEnum


class Op(IntEnum):
    """Operation enumerator"""

    NONE = 0
    ADD = 1
    SUB = 2
    MUL = 3
    """Mul"""
    DIV = 4
    """Div"""
    MATMUL = 5
    """Matmul"""
    POW = 6
    SUM = 7
    MEAN = 8
    LOG = 9
    EXP = 10
    T = 11
    SQUEEZE = 12
    EXPAND_DIMS = 13
    RESHAPE = 14
    MSE = 15
    CROSSENTROPY = 16
    TANH = 17
    SIGMOID = 18
    RELU = 19
    CONVOLVE = 20
    CORRELATE = 21
    MAX = 22
    MIN = 23
    STD = 24
    NEG = 25

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
    Op.MSE: "mse",
    Op.CROSSENTROPY: "crossentropy",
    Op.TANH: "tanh",
    Op.SIGMOID: "sigmoid",
    Op.RELU: "relu",
    Op.CONVOLVE: "convolve",
    Op.CORRELATE: "correlate",
    Op.MAX: "max",
    Op.MIN: "min",
    Op.STD: "std",
    Op.NEG: "negate",
}
