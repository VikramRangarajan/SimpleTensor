from .tensor import astensor
from .array_backend import np


def add(a1, a2):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.__add__`
    """
    return astensor(a1) + a2


def subtract(a1, a2):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.__sub__`
    """
    return astensor(a1) - a2


def multiply(a1, a2):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.__mul__`
    """
    return astensor(a1) * a2


def matmul(a1, a2):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.__matmul__`
    """
    return astensor(a1) @ a2


def power(a1, a2):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.__pow__`
    """
    return astensor(a1) ** a2


def divide(a1, a2):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.__truediv__`
    """
    return astensor(a1) / a2


def sum(a, axis=None, keepdims=False):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.sum`
    """
    return astensor(a).sum(axis=axis, keepdims=keepdims)


def mean(a, axis=None, keepdims=False):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.mean`
    """
    return astensor(a).mean(axis=axis, keepdims=keepdims)


def std(a, axis=None, ddof=0, keepdims=False):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.std`
    """
    return astensor(a).std(axis=axis, ddof=ddof, keepdims=keepdims)


def var(a, axis=None, ddof=0, keepdims=False):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.var`
    """
    return astensor(a).var(axis=axis, ddof=ddof, keepdims=keepdims)


def max(a, axis=None, keepdims=False):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.max`
    """
    return astensor(a).max(axis=axis, keepdims=keepdims)


def min(a, axis=None, keepdims=False):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.min`
    """
    return astensor(a).min(axis=axis, keepdims=keepdims)


def exp(a):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.exp`
    """
    return astensor(a).exp()


def logn(a, n=np.e):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.logn`
    """
    return astensor(a).logn(n=n)


def transpose(a, axis=None):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.transpose`
    """
    return astensor(a).transpose(axis=axis)


def squeeze(a, axis=None):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.squeeze`
    """
    return astensor(a).squeeze(axis=axis)


def flip(a, axis=None):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.flip`
    """
    return astensor(a).flip(axis=axis)


def expand_dims(a, axis=None):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.expand_dims`
    """
    return astensor(a).expand_dims(axis=axis)


def reshape(a, shape=None):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.reshape`
    """
    return astensor(a).reshape(shape=shape)


def relu(a):
    """
    Helper function for :py:mod:`~simpletensor.tensor.Tensor.relu`
    """
    return astensor(a).relu()


def softmax(a, axis=None):
    r"""
    Numerically stable n-dimensional softmax computation over any axes. Converts
    input into valid probability distributions.

    Softmax is defined as:

    .. math::
        \text{softmax}(\vec x) = \frac{e^{\vec x}}{\sum_{i=1}^n e^{x_i}}

    However, the numerically stable version of softmax in this implementation is defined as:

    .. math::
        \text{Let x_stable}(\vec x) = \vec x - \text{max}(\vec x) \\
        \text{softmax_stable}(\vec x) = 
        \frac{e^{\text{x_stable}(\vec x)}}{\sum_{i=1}^n e^{\text{x_stable}(\vec x)_i}}

    Parameters
    ----------
    a : Tensor
        Input tensor
    axis : tuple of ints, optional
        Axis over which values should be turned into a valid probability distribution, by default None

    

    Returns
    -------
    Tensor
        Output of softmax
    """
    a = astensor(a)
    a_max = a.max(axis=axis, keepdims=True)
    exp_a = (a - a_max).exp()
    return exp_a / exp_a.sum(axis=axis, keepdims=True)


def categorical_cross_entropy(y_true, y_pred):
    r"""
    Log loss / cross entropy loss over a batch. Defined as:

    .. math::
        \text{categorical_cross_entropy}(Y, \hat Y) = -\frac{1}{\text{batch_size}}\sum_{b=1}^{\text{batch_size}} \sum_{i=1}^n Y[b, i] \log_{b}(\hat Y[b, i])

    Parameters
    ----------
    y_true : Tensor
        2D Tensor of shape (batch_size, # features)
    y_pred : Tensor
        2D Tensor of shape (batch_size, # features)

    Returns
    -------
    _type_
        _description_
    """
    return -((y_true * y_pred.logn()).sum(axis=1).mean())


def show_graph(root):
    """
    Returns a graphviz visualization of the computation graph,
    starting at this root Tensor. It shows all the Tensors, the Tensors' names,
    shapes, dtypes, and all operations used to create all the Tensors in the
    graph.

    Parameters
    ----------
    root : Tensor
        Root tensor for the computation graph. Anything beyond the root
        is not shown.

    Returns
    -------
    Digraph
        GraphViz DiGraph object representing the computation graph
    """
    import graphviz

    dot = graphviz.Digraph("Graph", graph_attr={"rankdir": "LR"})
    queue = [root]
    visited = set()
    i = 0
    while queue != []:
        node = queue.pop()
        name = node.name
        tensor_desc = f"{name}\\nShape: {node.shape}\\nDtype: {node.dtype}"
        dot.node(name, tensor_desc, shape="record")
        if str(node._op) != "":
            dot.node(f"{str(node._op)}_{i}", str(node._op), shape="circle")
            dot.edge(f"{str(node._op)}_{i}", name)
        for child in node._parents:
            dot.edge(child.name, f"{str(node._op)}_{i}")
        for parent in node._parents:
            if parent not in visited:
                queue.append(parent)
                visited.add(parent)
        i += 1
    return dot
