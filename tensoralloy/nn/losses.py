# coding=utf-8
"""
This module defines various loss functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.python.framework import ops

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def rmse_loss(labels, predictions, name=None, add_eps=False):
    """
    Return the root mean squared error as the loss.
    """
    with ops.name_scope(name, "rmse", [labels, predictions]) as name:
        x = tf.convert_to_tensor(labels, name='x')
        y = tf.convert_to_tensor(predictions, name='y')
        mse = tf.reduce_mean(tf.squared_difference(x, y), name='mse')
        if add_eps:
            # Add a very small 'eps' to the mean squared error to make
            # sure `mse` is always greater than zero. Otherwise NaN may
            # occur at `Sqrt_Grad`.
            with tf.name_scope("safe_sqrt"):
                eps = tf.constant(1e-14, dtype=tf.float64, name='eps')
                mse = tf.add(mse, eps)
        return tf.sqrt(mse, name=name)


def norm_loss(labels, predictions, name=None):
    """
    Return the ratio of 2-norms of `labels - predictions` and `labels` as the
    loss.
    """
    with ops.name_scope(name, "norm", [labels, predictions]) as name:
        x = tf.convert_to_tensor(labels, name='x')
        y = tf.convert_to_tensor(predictions, name='y')
        diff = tf.subtract(x, y)
        ndims = x.shape.ndims
        axis = ndims - 1
        upper = tf.linalg.norm(
            diff, ord=2, axis=axis, keepdims=False, name='l2u')
        lower = tf.linalg.norm(x, ord=2, axis=axis, keepdims=False, name='l2l')
        return tf.reduce_mean(tf.div_no_nan(upper, lower), name=name)


def mae_loss(labels, predictions, name=None):
    """
    Return the mean absolute error as the loss.
    """
    with ops.name_scope(name, "mae", [labels, predictions]) as name:
        x = tf.convert_to_tensor(labels, name='x')
        y = tf.convert_to_tensor(predictions, name='y')
        return tf.reduce_mean(tf.abs(x - y), name=name)
