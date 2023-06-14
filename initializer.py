import logging
import numpy as np
import tensorflow as tf

from scipy import signal

from tensorflow.python.framework import dtypes
from tensorflow.python.ops.init_ops_v2 import _RandomGenerator
from tensorflow.python.ops.init_ops_v2 import _assert_float_dtype

logger = logging.getLogger(__name__)

def smooth(w, k=21):

    t = np.linspace(-2.5, +2.5, k, endpoint=True)
    u, v = np.meshgrid(t, t)
    win = np.exp(-(u ** 2 + v ** 2) / 2) / (k ** 2)
    sub = lambda x: x - np.mean(x)
    return np.array([
        signal.convolve2d(sub(wi) ** 2, win, mode='same')
        for wi in w]
    )
class KlindtSTAInitializer(tf.keras.initializers.Initializer):

    def __init__(self, x, y, mean=0.0, stddev=0.001, seed=None):

        super().__init__()
        self.x = x[:, :, :, 0]
        self.y = y
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=dtypes.float32):
        """Returns a tensor object initialized as specified by the initializer.
        Arguments:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only floating point types are
                supported.
        Raises:
            ValueError: If the dtype is not floating point
        """
        # dtype = _assert_float_dtype(dtype)
        # return self._random_generator.truncated_normal(
        #     shape, self.mean, self.stddev, dtype
        # )
        logger.debug("self.x.shape: {}".format(self.x.shape))
        logger.debug("self.y.shape: {}".format(self.y.shape))
        dtype = _assert_float_dtype(dtype)
        x = (self.x - np.mean(self.x)) / np.std(self.x)
        y = (self.y - np.mean(self.y, axis=0)) / np.std(self.y, axis=0)
        w = np.tensordot(y, x, axes=([0], [0]))
        # TODO smooth?
        e = smooth(w, k=21)
        center_positions = [
            np.unravel_index(np.argmax(ei), ei.shape)
            for ei in e
        ]
        w = self.stddev * np.random.normal(size=w.shape)
        w = np.abs(w)  # i.e. initialization with non-negative weights
        for k, (i, j) in enumerate(center_positions):
            w[k, i, j] = w[k, i, j] + np.std(self.y[:, k])  # i.e. add "standard deviation of neuron's response (because the output of the convolutional layer has unit variance)".

        # Crop initialization tensor.
        k_1 = (w.shape[1] - shape[1])
        k_2 = (w.shape[2] - shape[2])
        # logger.debug("k_1: {}".format(k_1))
        # logger.debug("k_2: {}".format(k_2))
        k_1 = k_1 // 2
        k_2 = k_2 // 2
        w = w[:, +k_1:-k_1, +k_2:-k_2]  # i.e. slice
        return w

    def get_config(self):

        config = {
            'mean': self.mean,
            'stddev': self.stddev,
            'seed': self.seed
        }

        return config

    
class EckerSTAInitializer(tf.keras.initializers.Initializer):

    def __init__(self, x, y, mean=0.0, stddev=0.01, seed=None):

        super().__init__()
        self.x = x[:, :, :, 0]
        self.y = y
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)
        self._alpha = 2 * 5
        self._max_val = 0.1
        self._smooth_size = 51  # TODO correct (e.g. 21 more reasonable, or input shape dependent)?

    def __call__(self, shape, dtype=dtypes.float32):
        """Returns a tensor object initialized as specified by the initializer.
        Arguments:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only floating point types are
                supported.
        Raises:
            ValueError: If the dtype is not floating point
        """
        # dtype = _assert_float_dtype(dtype)
        # return self._random_generator.truncated_normal(
        #     shape, self.mean, self.stddev, dtype
        # )
        logger.debug("self.x.shape: {}".format(self.x.shape))
        logger.debug("self.y.shape: {}".format(self.y.shape))
        dtype = _assert_float_dtype(dtype)
        x = (self.x - np.mean(self.x)) / np.std(self.x)
        y = (self.y - np.mean(self.y, axis=0)) / np.std(self.y, axis=0)
        w = np.tensordot(y, x, axes=([0], [0]))

        # # 1st attempt.
        # # w = (w / np.max(w, axis=(1, 2), keepdims=True)) ** self._alpha
        # w = (w / np.max(np.abs(w), axis=(1, 2), keepdims=True)) ** self._alpha
        # w = self._max_val * w
        # w = w + self.stddev * np.random.normal(size=w.shape)  # TODO use a truncated normal distribution instead?

        # # 2nd attempt.
        e = smooth(w, k=self._smooth_size)
        e = (e / np.max(e, axis=(1, 2), keepdims=True)) ** self._alpha
        e *= self._max_val
        e += self.stddev * np.random.normal(size=e.shape)  # TODO use a truncated normal distibution instead?
        w = e

        # # Sanity plot.
        # import matplotlib.pyplot as plt
        # plt.matshow(w[2, :, :])
        # plt.show()

        # Crop tensor.
        k_1 = (w.shape[1] - shape[1])
        k_2 = (w.shape[2] - shape[2])
        logger.debug("k_1: {}".format(k_1))
        logger.debug("k_2: {}".format(k_2))
        k_1 = k_1 // 2
        k_2 = k_2 // 2
        w = w[:, +k_1:-k_1, +k_2:-k_2]  # i.e. slice

        return w

    def get_config(self):

        config = {
            'mean': self.mean,
            'stddev': self.stddev,
            'seed': self.seed
        }

        return config