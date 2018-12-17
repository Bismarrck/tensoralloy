# coding=utf-8
"""
This module defines various loss functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.python.framework import ops

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _get_weight(compositions, n_max, dtype=tf.float64, name=None):
    """
    Compute the scaling factor.
    """
    with ops.name_scope(name, "Scale", [compositions, n_max]):
        compositions = tf.convert_to_tensor(
            compositions, name='compositions')
        if compositions.shape.ndims != 2:
            raise ValueError("The ndims of `compositions` should be 2")
        n_reals = tf.reduce_sum(
            compositions, axis=1, keepdims=False, name='n_reals')
        if n_reals.dtype != dtype:
            n_reals = tf.cast(n_reals, dtype=dtype)
        n_max = tf.convert_to_tensor(n_max, dtype=dtype, name='n_max')
        one = tf.constant(1.0, dtype=tf.float64, name='one')
        return tf.div(one, tf.reduce_mean(n_reals / n_max), name='weight')


def rmse_loss(labels, predictions, compositions=None, name=None, add_eps=False):
    """
    Return the root mean squared error as the loss.
    """
    with ops.name_scope(name, "RmseLoss", [labels, predictions]) as name:
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
        if compositions is not None:
            assert x.shape.ndims >= 2
            weight = _get_weight(compositions, x.shape[1].value, x.dtype)
            mse = tf.multiply(weight, mse, name='scaled')
        return tf.sqrt(mse, name=name)


def norm_loss(labels, predictions, compositions=None, name=None):
    """
    Return the ratio of 2-norms of `labels - predictions` and `labels` as the
    loss.
    """
    with ops.name_scope(
            name, "NormLoss", [labels, predictions, compositions]) as name:
        x = tf.convert_to_tensor(labels, name='x')
        y = tf.convert_to_tensor(predictions, name='y')
        ndims = x.shape.ndims
        axis = ndims - 1
        diff = tf.subtract(x, y)

        with ops.name_scope("l2"):
            upper = tf.linalg.norm(
                diff, ord=2, axis=axis, keepdims=False, name='upper')
            lower = tf.linalg.norm(
                x, ord=2, axis=axis, keepdims=False, name='lower')

        with ops.name_scope("safe_div"):
            threshold = tf.constant(1e-2, name='threshold', dtype=x.dtype)
            idx0 = tf.where(tf.less(lower, threshold), name='idx0')
            idx1 = tf.where(tf.greater_equal(lower, threshold), name='idx1')
            shape = lower.shape
            ratio = tf.div_no_nan(upper, lower)
            norm = tf.scatter_nd(
                idx0, tf.gather_nd(upper, idx0), shape, name='norm')
            ratio = tf.scatter_nd(
                idx1, tf.gather_nd(ratio, idx1), shape, name='ratio')
            values = ratio + norm

        if compositions is not None:
            assert ndims >= 2
            weight = _get_weight(compositions, x.shape[1].value, x.dtype)
            values = tf.multiply(values, weight, name='scaled')

        return tf.reduce_mean(values, name=name)


def mae_loss(labels, predictions, compositions=None, name=None):
    """
    Return the mean absolute error as the loss.
    """
    with ops.name_scope(name, "MaeLoss", [labels, predictions]) as name:
        x = tf.convert_to_tensor(labels, name='x')
        y = tf.convert_to_tensor(predictions, name='y')
        mae = tf.reduce_mean(tf.abs(x - y))
        if compositions is not None:
            assert x.shape.ndims >= 2
            weight = _get_weight(compositions, x.shape[1].value, x.dtype)
            mae = tf.multiply(weight, mae, name='scaled')
        return tf.identity(mae, name=name)
