from .tensor import astensor


def add(a1, a2):
    return astensor(a1) + a2


def subtract(a1, a2):
    return astensor(a1) - a2


def multiply(a1, a2):
    return astensor(a1) * a2


def matmul(a1, a2):
    return astensor(a1) @ a2


def power(a1, a2):
    return astensor(a1) ** a2


def divide(a1, a2):
    return astensor(a1) / a2


def sum(a, axis=None):
    return a.sum(axis=axis)


def mean(a, axis=None):
    return a.mean(axis=axis)


def std(a, axis=None, ddof=0):
    return a.std(axis=axis, ddof=ddof)


def var(a, axis=None, ddof=0):
    return a.var(axis=axis, ddof=ddof)
