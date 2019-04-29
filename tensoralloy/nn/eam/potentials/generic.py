#!coding=utf-8
"""
This module defines generic potential functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def morse(r, D, gamma, r0, name='Morse'):
    """
    The generic Morse potential.

        f(x) = D[ exp(-2 * gamma * (r - r0)) - 2 * exp(-gamma * (r - r0)) ]

    """
    with tf.name_scope(name):
        r = tf.convert_to_tensor(r, name='r')
        diff = tf.math.subtract(r, r0, name='diff')
        g_diff = tf.multiply(gamma, diff, name='g_diff')
        dtype = r.dtype
        two = tf.constant(2.0, dtype=dtype)
        c = tf.exp(-two * g_diff) - two * tf.exp(-g_diff)
        return tf.multiply(c, D, name='value')


def buckingham(r, A, rho, C, order=6, name='Buckingham'):
    """
    The generic Buckingham potential.

        f(x) = A * exp(-r / rho) - C / r**order

    """
    with tf.name_scope(name):
        r = tf.convert_to_tensor(r, name='r')
        dtype = r.dtype
        order = tf.constant(order, dtype=tf.int32, name='order')
        rs = tf.math.pow(r, order, name='rs')
        rho = tf.convert_to_tensor(rho, name='rho', dtype=dtype)
        C = tf.convert_to_tensor(C, name='C', dtype=dtype)
        return tf.math.subtract(A * tf.exp(-r / rho),
                                tf.div_no_nan(C, rs, name='right'),
                                name='value')
