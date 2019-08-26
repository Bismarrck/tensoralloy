#!coding=utf-8
"""
This module defines generic potential functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow.python.framework import ops

from tensoralloy.extension.grad_ops import safe_pow

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


def mishin_cutoff(x, name=None):
    """
    The cutoff function proposed by Y. Mishin:

        psi(x) = x**4 / (1 + x**4)  x <  0
                 0                  x >= 0

    """
    with ops.name_scope(name, "Psi", [x]):
        x = tf.nn.relu(-x, name='ix')
        four = tf.convert_to_tensor(4.0, dtype=x.dtype, name='four')
        x4 = safe_pow(x, four)
        one = tf.constant(1.0, dtype=x.dtype, name='one')
        return tf.math.truediv(x4, one + x4, name='psi')


def mishin_polar(x, p1, p2, p3, rc, h, name=None):
    """
    The polarization function proposed by Y. Mishin:

        f(x) = (p1 * exp(-p2 * x) + p3) * psi((x - rc) / h)

    """
    with ops.name_scope(name, "Polar", [x]):
        x = tf.convert_to_tensor(x, name='x')
        z = tf.math.divide(tf.math.subtract(x, rc, name='dr'), h, name='dr/h')
        with tf.name_scope("Psi"):
            psi = tf.identity(mishin_cutoff(z), name='psi')
        exp2x = tf.exp(-p2 * x, name='exp2x')
        p12 = tf.math.multiply(p1, exp2x, name='p12')
        left = tf.add(p12, p3, name='left')
        polar = tf.multiply(left, psi, name='polar')
        return polar


def zhou_exp(r, A, B, C, r_eq, name=None):
    """
    The exponential-form function proposed by Zhou.

        f(r) = A * exp(-B * (r / r_eq - 1)) / (1 + (r / r_eq - C)**20)

    """
    with ops.name_scope(name, "ZhouExp", [r]):
        r = tf.convert_to_tensor(r, name='r')
        one = tf.convert_to_tensor(1.0, dtype=r.dtype, name='one')
        order = tf.convert_to_tensor(20.0, dtype=r.dtype, name='order')
        x = tf.math.divide(r, r_eq, name='x')
        upper = tf.math.multiply(A, tf.exp(-B * (x - one)), name='upper')
        lower = tf.math.add(one, safe_pow(x - C, order))
        return tf.math.divide(upper, lower, name=name)