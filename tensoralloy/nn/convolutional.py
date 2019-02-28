# coding=utf-8
"""
This module defines the 1x1 convolutional Op for `tensoralloy`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import six

from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from tensorflow.python.layers import base
from tensorflow.python.keras.layers.convolutional import Conv as keras_Conv
from typing import List

from tensoralloy.utils import Defaults
from tensoralloy.nn.utils import log_tensor
from tensoralloy.dtypes import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["convolution1x1"]


_initializers = {
    'xavier': xavier_initializer(seed=Defaults.seed, dtype=get_float_dtype()),
    'msar': xavier_initializer(seed=Defaults.seed, dtype=get_float_dtype()),
    'zero': tf.zeros_initializer(dtype=get_float_dtype())
}


class Conv(keras_Conv, base.Layer):
    """
    `tf.keras.layers.convolutional.Conv` is an abstract nD convolution layer
    (private, used as implementation base).

    This is a modified implementation of `tf.keras.layers.convolutional.Conv`
    whose variables can be added to selected collections.

    """

    def __init__(self, rank, filters, kernel_size, strides=1, padding='valid',
                 data_format=None, dilation_rate=1, activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, trainable=True,
                 name=None, collections=None, **kwargs):
        super(Conv, self).__init__(
            rank, filters, kernel_size, strides=strides, padding=padding,
            data_format=data_format, dilation_rate=dilation_rate,
            activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, trainable=trainable, name=name,
            **kwargs)

        if collections is not None:
            assert isinstance(collections, list)
            _set = {tf.GraphKeys.GLOBAL_VARIABLES,
                    tf.GraphKeys.TRAINABLE_VARIABLES}
            collections = list(set(collections).difference(_set))
            if len(collections) == 0:
                collections = None
        self.collections = collections

    def build(self, input_shape):
        """
        Build the convolution layer and add the variables 'kernel' and 'bias' to
        given collections.
        """
        super(Conv, self).build(input_shape)

        if self.collections is not None:
            # `tf.add_to_collections` will not check for pre-existing membership
            # of `value` in any of the collections in `names`.
            if isinstance(self.collections, six.string_types):
                names = (self.collections, )
            else:
                names = set(self.collections)
            for name in names:
                if self.kernel not in tf.get_collection(name):
                    tf.add_to_collection(name, self.kernel)
                if self.use_bias:
                    if self.bias not in tf.get_collection(name):
                        tf.add_to_collection(name, self.bias)


def convolution1x1(x: tf.Tensor, activation_fn, hidden_sizes: List[int],
                   variable_scope, kernel_initializer='xavier', l2_weight=0.0,
                   collections=None, verbose=False):
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
    variable_scope : str or None
        The name of the variable scope.
    kernel_initializer : str
        The initialization algorithm for kernel variables.
    l2_weight : float
        The weight of the l2 regularization of kernel variables.
    collections : List[str] or None
        A list of str as the collections where the variables should be added.
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

    if l2_weight > 0.0:
        regularizer = l2_regularizer(l2_weight)
    else:
        regularizer = None

    if collections is not None:
        assert isinstance(collections, (list, tuple, set))
        collections = list(collections) + [tf.GraphKeys.MODEL_VARIABLES]

    rank = len(x.shape) - 2

    def _build(_x):
        for j in range(len(hidden_sizes)):
            layer = Conv(rank=rank, filters=hidden_sizes[j], kernel_size=1,
                         strides=1, use_bias=True, activation=activation_fn,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=regularizer,
                         bias_regularizer=regularizer,
                         name=f'Conv{rank}d{j + 1}',
                         collections=collections,
                         _reuse=tf.AUTO_REUSE)
            _x = layer.apply(_x)
            if verbose:
                log_tensor(_x)
        layer = Conv(rank=rank, filters=1, kernel_size=1,
                     strides=1, use_bias=False, activation=None,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=regularizer,
                     name='Output',
                     collections=collections,
                     _reuse=tf.AUTO_REUSE)
        return layer.apply(_x)

    if variable_scope is not None:
        with tf.variable_scope(variable_scope):
            y = _build(x)
    else:
        y = _build(x)

    if verbose:
        log_tensor(y)
    return y
