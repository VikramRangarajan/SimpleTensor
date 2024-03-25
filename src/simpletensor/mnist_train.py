from simpletensor import Tensor  # , no_grad
import numpy as np
from urllib.request import urlretrieve
import os
import importlib
import argparse

if importlib.util.find_spec("tqdm") is not None:
    tqdm = importlib.import_module("tqdm")
else:
    raise ImportError("tqdm Not Found. Please install it to use this showcase.")


def parse_args():
    """
    Parse path and hyperparameter arguments.

    Returns
    -------
    Tuple
        Tuple of options
    """
    parser = argparse.ArgumentParser(description="test description")
    parser.add_argument(
        "-dn",
        "--dense_neurons",
        type=int,
        default=512,
        help="number of neurons in dense layer",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=".",
        help="path of all files to be stored",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate of optimizer",
    )
    parser.add_argument(
        "-cf",
        "--conv_filters",
        type=int,
        default=16,
        help="number of filters in convolutional layer",
    )
    args = parser.parse_args()

    PATH = args.path
    DENSE_NEURONS = args.dense_neurons
    LR = args.lr
    CONV_FILTERS = args.conv_filters
    return PATH, DENSE_NEURONS, LR, CONV_FILTERS


class TqdmProgBar(tqdm.tqdm):
    """
    Progress bar to visualize download progress of mnist.npz
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_mnist(output_path):
    """
    Downloads MNIST dataset (mnist.npz) and stores it to file

    Parameters
    ----------
    output_path : filepath
        File path for mnist.npz to be downloaded to
    """

    if not os.path.isfile(output_path):
        with TqdmProgBar(unit="B", unit_scale=True, miniters=1, desc="mnist.npz") as t:
            urlretrieve(
                "https://s3.amazonaws.com/img-datasets/mnist.npz",
                filename=output_path,
                reporthook=t.update_to,
            )


def load_data(location):
    """
    Loads mnist.npz

    Parameters
    ----------
    location : filepath
        Location of mnist.npz

    Returns
    -------
    ((x_train, y_train), (x_test, y_test))
        Training and testing data in the form of packed tuples
    """
    data = np.load(location)
    return (
        data["x_train"].reshape((-1, 784)) / 255.0,
        data["y_train"],
    ), (
        data["x_test"].reshape((-1, 784)) / 255.0,
        data["y_test"],
    )


def main():
    """
    Runs everything
    """
    PATH, DENSE_NEURONS, LR, CONV_FILTERS = parse_args()
    os.makedirs(PATH, exist_ok=True)
    data_loc = os.path.join(PATH, "mnist.npz")

    download_mnist(data_loc)
    (x_train, y_train_categorical), (x_test, y_test_categorical) = load_data(data_loc)

    # Convert to one-hot encoded vectors
    y_train = np.zeros((60000, 10))
    y_test = np.zeros((10000, 10))
    y_train[range(60000), y_train_categorical] = 1
    y_test[range(10000), y_test_categorical] = 1
    m = Model(dense_neurons=DENSE_NEURONS, conv_filters=CONV_FILTERS, lr=LR)
    # TODO: Train model
    os.remove("mnist.npz")


class Model:
    """
    Creates CNN model
    """

    def __init__(self, **kwargs):
        self.rng = np.random.default_rng(seed=123)
        self.lr = kwargs["lr"]

        # Use Xavier normal initialization

        # First dense layer
        W1_rand = self.rng.normal(0, 1 / 784**0.5, (784, kwargs["dense_neurons"]))
        self.W1 = Tensor(W1_rand)

        b1_rand = self.rng.normal(0, 1 / 784**0.5, kwargs["dense_neurons"])
        self.b1 = Tensor(b1_rand)

    def __call__(self, batch):
        Y = Tensor(batch)
        Y = Y @ self.W1 + self.b1

        return Y


if __name__ == "__main__":
    main()
