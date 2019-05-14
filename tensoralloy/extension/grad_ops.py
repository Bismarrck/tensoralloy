#!coding=utf-8
"""
This module defines custom gradient functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow.python.ops import array_ops, gen_array_ops, math_ops

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@tf.custom_gradient
def safe_pow(x, y):
    """
    A safe implementation of the power Op `x**y`.
    """
    z = tf.pow(x, y, name='pow/safe')

    def _safe_pow_grad(grad):
        """
        Return grad * (y*x^(y-1), z*log(x)).
        """
        with tf.name_scope("SafePowGrad"):
            sx = array_ops.shape(x, name='sx')
            sy = array_ops.shape(y, name='sy')
            rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy, name='bga')
            cx = math_ops.conj(x, name='conj/x')
            cy = math_ops.conj(y, name='conj/y')
            cz = math_ops.conj(z, name='conj/z')
            # Avoid false singularity at x = 0
            if cx.dtype.is_complex:
                # real(x) < 0 is fine for the complex case
                mask = math_ops.not_equal(cx, 0, 'mask')
            else:
                # There's no sensible real value to return if x < 0, so return 0
                mask = cx > 0
            # Another mask to avoid false singularity at x = 0 caused by
            # repeated `tf.gradients` calls.
            with tf.name_scope("dzdx"):
                if cx.dtype.name == 'float64':
                    eps_val = 1e-14
                else:
                    eps_val = 1e-7
                eps = tf.convert_to_tensor(eps_val, dtype=cx.dtype, name='eps')
                safe_cx = array_ops.where(
                    mask, cx, array_ops.ones_like(x) * eps, 'safe_cx')
                xy1 = math_ops.pow(safe_cx, cy - 1, 'xy1')
                gx = array_ops.reshape(
                    math_ops.reduce_sum(grad * cy * xy1, rx), sx, 'gx')

            with tf.name_scope("dzdy"):
                safe_log_x = array_ops.where(
                    mask, cx, array_ops.ones_like(cx), 'safe_x')
                log_x = array_ops.where(
                    mask,
                    math_ops.log(safe_log_x),
                    array_ops.zeros_like(cx),
                    name='log_x')
                gy = array_ops.reshape(
                    math_ops.reduce_sum(grad * cz * log_x, ry), sy, 'gy')
        return gx, gy

    return z, _safe_pow_grad
