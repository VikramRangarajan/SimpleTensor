Demo
=====
The demo for the project is in the form of a convolutional neural network
which trains on the MNIST digit dataset. It uses only SimpleTensor's tensors
to build up the network, showing the purpose of this project and the power
it has.

While it's not fully implemented, SimpleTensor can practically be used to fully
mimic a library like PyTorch or TensorFlow when trying to make a deep learning
model, due to its vast array of differentiable operations. Also, if an operation
is not defined, a program can simply create the gradient function and use the tensors
as is.

Usage:
An executable script is included in this package, provided you install the tqdm
module. In the command line, you can simply run

.. code-block:: console

    simpletensor-train-mnist optional arguments here

These optional arguments can be seen by running

.. code-block:: console

    simpletensor-train-mnist -h

This will need internet connection to download the MNIST dataset from an 
Amazon S3 bucket.

This script will train the model (deterministically) and store the model into
a pickle file. This model can then be loaded and used to classify real digits.