# coding=utf-8
"""
This module defines the 1x1 convolutional Op for `tensoralloy`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import six

from tensorflow.contrib.layers import l2_regularizer
from tensorflow.python.layers import base
from tensorflow.python.keras.layers.convolutional import Conv as keras_Conv
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from typing import List

from tensoralloy.nn.init_ops import get_initializer
from tensoralloy.nn.utils import log_tensor

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["convolution1x1"]


class Conv(keras_Conv, base.Layer):
    """
    `tf.keras.layers.convolutional.Conv` is an abstract nD convolution layer
    (private, used as implementation base).

    This is a modified implementation of `tf.keras.layers.convolutional.Conv`
    whose variables can be added to selected collections.

    """

    def __init__(self, rank, filters, kernel_size, strides=1, padding='valid',
                 data_format=None, dilation_rate=1, activation=None,
                 use_bias=True, fixed_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer=None, kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, trainable=True,
                 name=None, collections=None, **kwargs):

        if not isinstance(bias_initializer, str):
            _bias_initializer = 'zeros'
        else:
            _bias_initializer = bias_initializer

        super(Conv, self).__init__(
            rank, filters, kernel_size, strides=strides, padding=padding,
            data_format=data_format, dilation_rate=dilation_rate,
            activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=_bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, trainable=trainable, name=name,
            **kwargs)

        # Add a small hack since we maymodify the mean of `bias_initializer`.
        if not isinstance(bias_initializer, str):
            self.bias_initializer = bias_initializer

        if collections is not None:
            assert isinstance(collections, list)
            _set = {tf.GraphKeys.GLOBAL_VARIABLES,
                    tf.GraphKeys.TRAINABLE_VARIABLES}
            collections = list(set(collections).difference(_set))
            if len(collections) == 0:
                collections = None

        self.fixed_bias = fixed_bias
        self.collections = collections
        self.kernel = None
        self.bias = None
        self._convolution_op = None

    def build(self, input_shape):
        """
        Build the convolution layer and add the variables 'kernel' and 'bias' to
        given collections.
        """
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=(not self.fixed_bias),
                dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding,
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank + 2))
        self.built = True

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
                   variable_scope, num_out=1, kernel_initializer='he_normal',
                   l2_weight=0.0, collections=None, output_bias=False,
                   output_bias_mean=0, fixed_output_bias=False,
                   use_resnet_dt=False, verbose=False):
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
    num_out : int
        The number of outputs.
    kernel_initializer : str
        The initialization algorithm for kernel variables.
    l2_weight : float
        The weight of the l2 regularization of kernel variables.
    output_bias : bool
        A flag. If True, a bias will be applied to the output layer as well.
    output_bias_mean : float
        The bias unit of the output layer will be initialized with
        `constant_initializer`. This defines the initial value.
    use_resnet_dt : bool
        Use ResNet block (x = sigma(wx + b) + x) if True.
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
    dtype = x.dtype
    kernel_initializer = get_initializer(kernel_initializer, dtype=dtype)
    bias_initializer = get_initializer('zero', dtype=dtype)

    if output_bias:
        output_bias_initializer = get_initializer(
            'constant', value=output_bias_mean, dtype=dtype)
    else:
        output_bias_initializer = None

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
            # The condition `hidden_sizes[j] == hidden_sizes[j - 1]` is required
            # for constructing reset block.
            if j and use_resnet_dt and hidden_sizes[j] == hidden_sizes[j - 1]:
                _x = tf.math.add(layer.apply(_x), _x, name=f"Res{j}")
            else:
                _x = layer.apply(_x)
            if verbose:
                log_tensor(_x)

        layer = Conv(rank=rank, filters=num_out, kernel_size=1,
                     strides=1, use_bias=output_bias,
                     fixed_bias=fixed_output_bias,  activation=None,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=regularizer,
                     bias_initializer=output_bias_initializer,
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
