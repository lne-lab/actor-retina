import tensorflow as tf

from tensorflow.python.ops import math_ops


class SmoothSparse2DRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, smooth_factor=0.001, sparse_factor=0.001):
        super().__init__()
        if smooth_factor:
            # self.smooth_factor = tf.cast(smooth_factor, tf.float32, name=None)  # TODO remove!
            self.smooth_factor = smooth_factor
        else:
            self.smooth_factor = None
        if sparse_factor:
            # self.sparse_factor = tf.cast(sparse_factor, tf.float32, name=None)  # TODO remove!
            self.sparse_factor = sparse_factor
        else:
            self.sparse_factor = None

    def __call__(self, x):  # TODO rename `x` to `weights`?
        regularization = tf.constant(0.0, dtype=tf.float32, shape=None, name=None)
        if self.smooth_factor:
            # lap = tf.constant([
            #     [+0.25, +0.50, +0.25],
            #     [+0.50, -3.00, +0.50],
            #     [+0.25, +0.50, +0.25],
            # ])
            # lap = tf.expand_dims(lap, tf.expand_dims(lap, 2), 3, name='laplacian_filter')
            lap = tf.constant([
                [+0.25, +0.50, +0.25],
                [+0.50, -3.00, +0.50],
                [+0.25, +0.50, +0.25],
            ], shape=(3, 3, 1, 1), name='laplacian_filter')
            # nb_kernels = x.get_shape().as_list()[2]
            # nb_kernels = x.shape[2]
            _, _, nb_kernels, _ = x.shape
            x_lap = tf.nn.depthwise_conv2d(
                tf.transpose(x, perm=(3, 0, 1, 2)),  # inputs
                tf.tile(lap, (1, 1, nb_kernels, 1)),  # filter
                (1, 1, 1, 1),  # strides
                'SAME',  # padding  # TODO check this...
            )
            smooth_regularization = math_ops.reduce_sum(
                math_ops.reduce_sum(math_ops.square(x_lap), axis=(1, 2, 3)) / (1e-8 + math_ops.reduce_sum(math_ops.square(x), axis=(0, 1, 2)))
            )
            regularization += self.smooth_factor * smooth_regularization
        if self.sparse_factor:
            sparse_regularization = math_ops.reduce_sum(
                math_ops.reduce_sum(math_ops.sqrt(math_ops.reduce_sum(math_ops.square(x), axis=(0, 1))), axis=0) / math_ops.sqrt(1e-8 + math_ops.reduce_sum(math_ops.square(x), axis=(0, 1, 2)))
            )
            regularization += self.sparse_factor * sparse_regularization
        regularization = tf.identity(regularization, name='regularization')  # TODO remove?
        return regularization

    # def get_config(self):
    #     return {
    #         'smooth_factor': self.smooth_factor,
    #         'sparse_factor': self.sparse_factor,
    #     }


# class Smooth2DRegularizer(tf.keras.regularizer.Regularizer):
#
#     def __init__(self, factor=0.001):
#         super().__init__()
#         self.__factor = factor
#
#     def __call__(self, x):
#         raise NotImplementedError
#
#     def get_config(self):
#         raise NotImplementedError


# class GroupSparsity2DRegularizer(tf.keras.regularizer.Regularizer):
#
#     def __init__(self, factor=0.001):
#         super().__init__()
#         self.__factor = factor
#
#     def __call__(self, x):
#         raise NotImplementedError
#
#     def get_config(self):
#         raise NotImplementedError


# TODO is it possible to combine regularizers?
