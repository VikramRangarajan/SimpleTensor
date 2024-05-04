from simpletensor import Tensor, softmax, categorical_cross_entropy, use_cupy, use_numpy
from simpletensor.array_backend import np
import numpy  # opencv not compatible with cupy arrays
from urllib.request import urlretrieve
import os
import argparse
from pathlib import Path
import tqdm

DTYPE = "float64"
np.random.seed(123)


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
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout proportion",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        default=True,
        action="store_true",
        help="Use GPU if available",
    )
    parser.add_argument(
        "-ng",
        "--no-gpu",
        dest="gpu",
        action="store_false",
        help="Use CPU only",
    )
    args = parser.parse_args()

    PATH = Path(args.path).absolute()
    DENSE_NEURONS = args.dense_neurons
    LR = args.lr
    CONV_FILTERS = args.conv_filters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DROPOUT = args.dropout
    USE_GPU = args.gpu
    if USE_GPU:
        use_cupy()
    else:
        use_numpy()
    return PATH, DENSE_NEURONS, LR, CONV_FILTERS, BATCH_SIZE, EPOCHS, DROPOUT


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
        data["x_train"].reshape((60000, 1, 28, 28)).astype(DTYPE) / 255.0,
        data["y_train"],
    ), (
        data["x_test"].reshape((10000, 1, 28, 28)).astype(DTYPE) / 255.0,
        data["y_test"],
    )


def main(*args):
    import cv2
    import matplotlib.pyplot as plt

    """
    Runs everything
    """
    PATH, DENSE_NEURONS, LR, CONV_FILTERS, BATCH_SIZE, EPOCHS, DROPOUT = (
        args or parse_args()
    )
    os.makedirs(PATH, exist_ok=True)
    data_loc = os.path.join(PATH, "mnist.npz")

    download_mnist(data_loc)
    (X_train, y_train_categorical), (X_test, y_test_categorical) = load_data(data_loc)

    # Convert to one-hot encoded vectors
    y_train = np.zeros((60000, 10), dtype=DTYPE)
    y_test = np.zeros((10000, 10), dtype=DTYPE)
    y_train[range(60000), y_train_categorical] = 1
    y_test[range(10000), y_test_categorical] = 1

    model = Model(
        dense_neurons=DENSE_NEURONS,
        conv_filters=CONV_FILTERS,
        lr=LR,
        dropout=DROPOUT,
    )
    try:
        model.fit(
            train_data=(X_train, y_train),
            test_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
    except KeyboardInterrupt:
        pass
    history = model.history
    for metric in ["acc", "loss"]:
        for data in history.keys():
            if metric in data:
                plt.plot(range(len(history[data])), history[data], label=data)
                plt.title(f"{metric.capitalize()} vs. Epochs")
                plt.xlabel("Epochs")
                plt.ylabel(metric.capitalize())
                plt.legend()
        plt.savefig(PATH / f"{metric}_plot.png")
        plt.show()
    os.remove("mnist.npz")
    cv2.namedWindow("Draw Digit", cv2.WINDOW_NORMAL)
    canvas = numpy.zeros((280, 280), dtype="uint8")
    drawing = False
    prevx, prevy = None, None

    model.eval()

    def draw(event, x, y, flags, params):
        nonlocal drawing, prevx, prevy, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            prevx, prevy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            blank = numpy.zeros((280, 280), dtype="uint8")
            cv2.line(blank, (prevx, prevy), (x, y), color=255, thickness=28)
            blank = cv2.blur(blank, (13, 13))
            canvas = canvas | blank
            canvas = cv2.resize(
                cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_LINEAR),
                (280, 280),
                interpolation=cv2.INTER_NEAREST,
            )
            prevx, prevy = x, y

    cv2.imshow("Draw Digit", canvas)
    cv2.setMouseCallback("Draw Digit", draw)
    while True:
        cv2.imshow("Draw Digit", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            canvas = numpy.zeros((280, 280), dtype="uint8")
            prevx, prevy = None, None
        if key == ord("p"):
            image = cv2.resize(canvas, (28, 28), cv2.INTER_LINEAR)
            prediction = (
                model(np.array(image[None, None]).astype(DTYPE) / 255.0)
                ._array[0]
                .argmax()
            )
            print(prediction)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


class Model:
    """
    Creates CNN model
    """

    def __init__(self, **kwargs):
        self.rng = np.random.default_rng(seed=123)
        self.lr = kwargs["lr"]
        self.dropout = kwargs["dropout"]
        self.training = True

        # Use Xavier normal initialization

        # Convolutional layer
        K_rand = self.rng.standard_normal((kwargs["conv_filters"], 1, 3, 3)) / 3
        self.K = Tensor(
            K_rand,
            dtype=DTYPE,
            name="Conv Weights",
        )

        conv_bias_rand = self.rng.standard_normal((kwargs["conv_filters"], 1, 1)) / 3
        self.conv_bias = Tensor(
            conv_bias_rand,
            dtype=DTYPE,
            name="Conv Bias",
        )

        # First dense layer
        # It's 26 and 26 because we're doing a valid convolution between a (28, 28) image and a (3, 3) kernel
        output_size_of_conv = kwargs["conv_filters"] * 26 * 26
        W1_rand = (
            self.rng.standard_normal(
                (output_size_of_conv, kwargs["dense_neurons"]),
            )
            / output_size_of_conv**0.5
        )
        self.W1 = Tensor(
            W1_rand,
            dtype=DTYPE,
            name="Dense Weights 1",
        )

        b1_rand = (
            self.rng.standard_normal(
                kwargs["dense_neurons"],
            )
            / output_size_of_conv**0.5
        )
        self.b1 = Tensor(
            b1_rand,
            dtype=DTYPE,
            name="Dense Bias 1",
        )

        # Second dense layer
        W2_rand = (
            self.rng.standard_normal((kwargs["dense_neurons"], 10))
            / kwargs["dense_neurons"] ** 0.5
        )
        self.W2 = Tensor(
            W2_rand,
            dtype=DTYPE,
            name="Dense Weights 2",
        )

        b2_rand = self.rng.standard_normal(10) / kwargs["dense_neurons"] ** 0.5
        self.b2 = Tensor(
            b2_rand,
            dtype=DTYPE,
            name="Dense Bias 2",
        )

        # Trainable parameters
        self.parameters = [self.K, self.conv_bias, self.W1, self.b1, self.W2, self.b2]
        self.adam_m = []
        self.adam_v = []
        self.adam_t = 0
        for param in self.parameters:
            self.adam_m.append(np.zeros(param.shape, param.dtype))
            self.adam_v.append(np.zeros(param.shape, param.dtype))

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, batch):
        Y = Tensor(batch, dtype=DTYPE)
        Y.name = "inputs"
        Y = Y.convolve(self.K) + self.conv_bias  # Conv layer
        Y = Y.relu()  # Relu
        Y = Y.reshape((Y.shape[0], -1))  # Flatten output of conv layer
        if self.training:
            dropout = Tensor(
                np.random.binomial(1, 1 - self.dropout, size=Y.shape),
                dtype=Y.dtype,
                name="dropout",
            )
            Y *= dropout
        Y = Y @ self.W1 + self.b1  # Dense layer
        Y = Y.relu()  # Relu
        Y = Y @ self.W2 + self.b2  # Final dense layer
        Y.name = "logits"
        Y = softmax(Y, axis=(1,))  # Softmax
        return Y

    def fit(self, train_data, test_data, epochs, batch_size):
        """
        Fits model to training data

        Parameters
        ----------
        train_data : (ndarray, ndarray)
            Training images and labels
        test_data : (ndarray, ndarray)
            Testing images and labels
        epochs : int
            Number of epochs to train
        batch_size : int
            Minibatch size

        Returns
        -------
        dict[str, list]
            History of training and testing losses and accuracies over the epochs
        """
        X_train, y_train = train_data
        X_test, y_test = test_data
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} / {epochs}")
            for phase in ["train", "val"]:
                epoch_loss = 0
                epoch_acc = 0
                if phase == "train":
                    X = X_train
                    y = y_train
                    self.train()

                    # Shuffle Data
                    shuffle_index = np.random.permutation(np.arange(60000))
                    X = X[shuffle_index]
                    y = y[shuffle_index]
                else:
                    X = X_test
                    y = y_test
                    self.eval()
    
                progbar = tqdm.tqdm(
                    range(0, len(X), batch_size),
                    mininterval=0.25,
                    bar_format="Batches: {l_bar}{bar:30}{r_bar}",
                    colour="green" if phase == "train" else "yellow",
                )
                for start_index in progbar:
                    batch_inp = X[start_index : start_index + batch_size].copy()
                    batch_labels = y[start_index : start_index + batch_size].copy()

                    preds = self(batch_inp)
                    batch_loss = categorical_cross_entropy(Tensor(batch_labels), preds)

                    preds_categorical = preds._array.argmax(1)
                    labels_categorical = batch_labels.argmax(1)
                    correct = (preds_categorical == labels_categorical).sum()
                    epoch_acc += int(correct)
                    epoch_loss += batch_loss._array.item()

                    progbar.set_postfix(
                        {
                            f"{phase}_loss": f"{epoch_loss / (batch_size + start_index):.4f}",
                            f"{phase}_accuracy": f"{epoch_acc / (batch_size + start_index):.4f}",
                        },
                        refresh=False,
                    )

                    if phase == "train":
                        b1 = 0.9
                        b2 = 0.999
                        eps = 1e-8
                        # Optimization, the WHOLE point of this automatic differentiation library
                        batch_loss.backward()  # !!!!
                        for i in range(len(self.parameters)):
                            # Stochastic Gradient Descent
                            self.adam_t += 1
                            self.adam_m[i] = (
                                b1 * self.adam_m[i] + (1 - b1) * self.parameters[i].grad
                            )
                            self.adam_v[i] = (
                                b2 * self.adam_v[i]
                                + (1 - b2)
                                * self.parameters[i].grad
                                * self.parameters[i].grad
                            )
                            adam_m_hat = self.adam_m[i] / (1 - b1**self.adam_t)
                            adam_v_hat = self.adam_v[i] / (1 - b2**self.adam_t)
                            self.parameters[i]._array -= (
                                self.lr * adam_m_hat / (np.sqrt(adam_v_hat) + eps)
                            )

                            # param._array -= self.lr * param.grad

                epoch_loss /= len(X)
                epoch_acc /= len(X)
                print(f"{phase}_loss: {epoch_loss:.6f}, {phase}_acc: {epoch_acc:.6f}")
                self.history[f"{phase}_loss"].append(epoch_loss)
                self.history[f"{phase}_acc"].append(epoch_acc)


if __name__ == "__main__":
    main(Path(".").absolute(), 512, 0.01, 16, 32, 10, 0.2)
