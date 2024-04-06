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

In addition, SimpleTensor has GPU support via CuPy, which acts as a drop in
substitution for numpy, except it allows us to have arrays in GPU VRAM and
GPU accelerated operations. Note that SimpleTensor requires CuPy >=12.0.0,
as CuPy did not implement ufunc.at until version 13.0.0 (this is required
for the differentiable indexing operation, which is used in max, min, and
softmax).