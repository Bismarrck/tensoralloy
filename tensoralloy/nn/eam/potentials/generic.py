#!coding=utf-8
"""
This module defines generic potential functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.extension.grad_ops import safe_pow

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def morse(r, d, gamma, r0, name=None):
    """
    The generic Morse potential.

        f(x) = d * [ exp(-2 * gamma * (r - r0)) - 2 * exp(-gamma * (r - r0)) ]

    """
    with tf.name_scope(name, 'Morse'):
        r = tf.convert_to_tensor(r, name='r')
        diff = tf.math.subtract(r, r0, name='diff')
        d = tf.convert_to_tensor(d, name='D')
        gd = tf.multiply(gamma, diff, name='g_diff')
        dtype = r.dtype
        two = tf.constant(2.0, dtype=dtype)
        c = tf.exp(-two * gd) - two * tf.exp(-gd)
        return tf.multiply(c, d, name='result')


def buckingham(r, A, rho, C, order=6, name=None):
    """
    The generic Buckingham potential.

        f(x) = A * exp(-r / rho) - C / r**order

    """
    with tf.name_scope(name, 'Buckingham'):
        r = tf.convert_to_tensor(r, name='r')
        dtype = r.dtype
        order = tf.constant(order, dtype=tf.int32, name='order')
        rs = tf.math.pow(r, order, name='rs')
        rho = tf.convert_to_tensor(rho, name='rho', dtype=dtype)
        C = tf.convert_to_tensor(C, name='C', dtype=dtype)
        return tf.math.subtract(A * tf.exp(-r / rho),
                                tf.div_no_nan(C, rs, name='right'),
                                name='result')


def mishin_cutoff(x, name=None):
    """
    The cutoff function proposed by Y. Mishin:

        psi(x) = x**4 / (1 + x**4)  x <  0
                 0                  x >= 0

    """
    with tf.name_scope(name, 'Psi'):
        x = tf.nn.relu(-x, name='ix')
        four = tf.convert_to_tensor(4.0, dtype=x.dtype, name='four')
        x4 = safe_pow(x, four)
        one = tf.constant(1.0, dtype=x.dtype, name='one')
        return tf.math.truediv(x4, one + x4, name='result')


def mishin_polar(x, p1, p2, p3, rc, h, name=None):
    """
    The polarization function proposed by Y. Mishin:

        f(x) = (p1 * exp(-p2 * x) + p3) * psi((x - rc) / h)

    """
    with tf.name_scope(name, "MishinPolar"):
        x = tf.convert_to_tensor(x, name='x')
        z = tf.math.divide(tf.math.subtract(x, rc, name='dr'), h, name='dr/h')
        with tf.name_scope("Psi"):
            psi = tf.identity(mishin_cutoff(z), name='psi')
        exp2x = tf.exp(-p2 * x, name='exp2x')
        p12 = tf.math.multiply(p1, exp2x, name='p12')
        left = tf.add(p12, p3, name='left')
        polar = tf.multiply(left, psi, name='result')
        return polar


def density_exp(r, a, b, re, name=None):
    """
    The most widely used exponential density function.

        f(r) = a * exp(-b * (r / re - 1))

    """
    with tf.name_scope(name, "DensityExp"):
        r = tf.convert_to_tensor(r, name='r')
        one = tf.convert_to_tensor(1.0, dtype=r.dtype, name='one')
        x = tf.math.divide(r, re, name='x')
        upper = tf.math.multiply(a, tf.exp(-b * (x - one)), name='result')
        return upper
    

def zhou_exp(r, a, b, c, re, order=20, name=None):
    """
    The exponential-form function proposed by Zhou.

        f(r) = a * exp(-b * (r / re - 1)) / (1 + (r / re - c)**order)
     
    The default order is 20.
    """
    with tf.name_scope(name, "ZhouExp"):
        r = tf.convert_to_tensor(r, name='r')
        one = tf.convert_to_tensor(1.0, dtype=r.dtype, name='one')
        order = tf.convert_to_tensor(order, dtype=r.dtype, name='order')
        x = tf.math.divide(r, re, name='x')
        upper = density_exp(r, a, b, re)
        lower = tf.math.add(one, safe_pow(x - c, order))
        return tf.math.divide(upper, lower, name='result')


def power_exp1(r, rl, name=None):
    """
    Special routine for `power_exp` when `pl` == 1
    """
    with tf.name_scope(name, "PExp"):
        r = tf.convert_to_tensor(r, name='r')
        rl = tf.convert_to_tensor(rl, name='rl', dtype=r.dtype)
        rrl = tf.divide(r, rl, name='rrl')
        return tf.exp(-rrl, name='pexp1')


def power_exp2(r, rl, name=None):
    """
    Special routine for `power_exp` when `pl` == 2
    """
    with tf.name_scope(name, "PExp"):
        r = tf.convert_to_tensor(r, name='r')
        rl = tf.convert_to_tensor(rl, name='rl', dtype=r.dtype)
        rrl = tf.divide(r, rl, name='rrl')
        rp = tf.multiply(rrl, rrl, name='rrl2')
        return tf.exp(-rp, name='pexp2')


def power_exp3(r, rl, name=None):
    """
    Special routine for `power_exp` when `pl` == 3
    """
    with tf.name_scope(name, "PExp"):
        r = tf.convert_to_tensor(r, name='r')
        rl = tf.convert_to_tensor(rl, name='rl', dtype=r.dtype)
        rrl = tf.divide(r, rl, name='rrl')
        rp = tf.multiply(tf.multiply(rrl, rrl), rrl, name='rrl3')
        return tf.exp(-rp, name='pexp3')


def power_exp(r, rl, pl, name=None):
    """
    The powered-exponential function used by Oganov.

        f(r) = exp(-(r / rl)**pl)

    """
    with tf.name_scope(name, "PExp"):
        r = tf.convert_to_tensor(r, name='r')
        rl = tf.convert_to_tensor(rl, name='rl', dtype=r.dtype)
        pl = tf.convert_to_tensor(pl, name='pl', dtype=r.dtype)
        rrl = tf.divide(r, rl, name='rrl')
        rp = tf.pow(rrl, pl, name='rp')
        return tf.exp(-rp, name='pexp')
