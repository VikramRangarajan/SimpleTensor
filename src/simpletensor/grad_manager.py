from .tensor import Tensor


class no_grad:
    """
    Gradient context manager, similar to torch.no_grad()
    """

    def __init__(self) -> None:
        self.prev = True

    def __enter__(self) -> None:
        """
        Disables the computation graph while in the context
        """
        self.prev = Tensor.grad_enabled
        Tensor.grad_enabled = False

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        """
        Reverts gradient option to what it was previously
        """
        Tensor.grad_enabled = self.prev
