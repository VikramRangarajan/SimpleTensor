About
=====
This is our final project for CMSC421 in Spring 2024 at the University of Maryland.

It seeks to mimic Tensor objects from libraries such as PyTorch or TensorFlow.
You can use this Tensor (mostly) just like you would a numpy array, except that
you can use automatic differentiation.

The reverse mode automatic differentiation was made to mimic PyTorch's .backward()
function, and this tensor will keep track of all operations needed to create the
final single value tensor.

Note: The gradient arrays are *not* Tensor's, and are just numpy arrays.