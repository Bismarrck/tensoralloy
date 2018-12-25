# coding=utf-8
"""
This module defines the 1x1 convolutional Op for `tensoralloy`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from typing import List

from tensoralloy.misc import Defaults
from tensoralloy.nn.utils import log_tensor

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["convolution1x1"]


_initializers = {
    'xavier': xavier_initializer(seed=Defaults.seed, dtype=tf.float64),
    'msar': xavier_initializer(seed=Defaults.seed, dtype=tf.float64),
    'zero': tf.zeros_initializer(dtype=tf.float64)
}


def convolution1x1(x: tf.Tensor, activation_fn, hidden_sizes: List[int],
                   kernel_initializer='xavier', verbose=False):
    """
    Construct a 1x1 convolutional neural network.

    Parameters
    ----------
    x : tf.Tensor
        The input of the convolutional neural network. `x` must be a tensor
        with rank 3 (conv1d), 4 (conv2d) or 5 (conv3d) of dtype `float64`.
    activation_fn : Callable
        The activation function.
    hidden_sizes : List[int]
        The size of the hidden layers.
    kernel_initializer : str
        The initialization algorithm for kernel variables.
    verbose : bool
        If True, key tensors will be logged.

    Returns
    -------
    y : tf.Tensor
        The output tensor. `x` and `y` has the same rank. The last dimension
        of `y` is 1.

    """
    kernel_initializer = _initializers[kernel_initializer]
    bias_initializer = _initializers['zero']

    rank = len(x.shape)
    if rank == 3:
        conv = tf.layers.conv1d
    elif rank == 4:
        conv = tf.layers.conv2d
    elif rank == 5:
        conv = tf.layers.conv3d
    else:
        raise ValueError(
            f"The rank of `x` should be 3, 4 or 5 but not {rank}")

    for j in range(len(hidden_sizes)):
        x = conv(
            inputs=x, filters=hidden_sizes[j],
            kernel_size=1, strides=1,
            activation=activation_fn,
            use_bias=True,
            reuse=tf.AUTO_REUSE,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(),
            name=f'{conv.__name__.capitalize()}{j + 1}')
        if verbose:
            log_tensor(x)
    y = conv(inputs=x, filters=1, kernel_size=1, strides=1, use_bias=False,
             kernel_initializer=kernel_initializer, reuse=tf.AUTO_REUSE,
             name='Output')
    if verbose:
        log_tensor(y)
    return y
