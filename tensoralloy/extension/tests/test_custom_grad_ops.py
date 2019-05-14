#!coding=utf-8
"""
Unit tests of the `grad_ops` module.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from nose.tools import assert_almost_equal, assert_true

from tensoralloy.extension.grad_ops import safe_pow
from tensoralloy.precision import set_precision, get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_safe_pow():
    """
    Test the function `safe_pow`.
    """

    def assert_equal(a, b):
        """ A wrapper. """
        return assert_almost_equal(a, b, delta=1e-12)

    with set_precision("high"):
        dtype = get_float_dtype()
        with tf.Graph().as_default():
            x = tf.placeholder(dtype, name='x')
            y = tf.placeholder(dtype, name='y')
            origin = tf.pow(x, y, name='origin')
            z = safe_pow(x, y)

            g1 = tf.gradients(origin, x)[0]
            g2 = tf.gradients(g1, x, name='g')[0]

            g1x = tf.gradients(z, x, name='g1x')[0]
            g2x = tf.gradients(g1x, x, name='g2x')[0]
            g3x = tf.gradients(g2x, x, name='g3x')[0]
            g1y = tf.gradients(z, y, name='g1y')[0]
            g2y = tf.gradients(g1y, y, name='g2y')[0]

            ops = [z, g1x, g2x, g3x, g1y, g2y]

            with tf.Session() as sess:
                results = np.asarray(sess.run(ops, feed_dict={x: 3.0, y: 4.0}))
                assert_equal(results[0], 81.0)
                assert_equal(results[1], 108.0)
                assert_equal(results[2], 108.0)
                assert_equal(results[3], 72.0)
                assert_almost_equal(
                    results[4], np.log(3) * np.exp(4 * np.log(3)), delta=1e-12)

                results = np.asarray(sess.run(ops, feed_dict={x: 0.0, y: 1.0}))
                assert_equal(results[0], 0.0)
                assert_equal(results[1], 1.0)
                assert_equal(results[2], 0.0)
                assert_equal(results[3], 0.0)
                assert_equal(results[4], 0.0)
                assert_equal(results[5], 0.0)

                assert_true(np.isnan(sess.run(g2, feed_dict={x: 0.0, y: 1.0})))
                assert_equal(sess.run(g1, feed_dict={x: 0.0, y: 1.0}), 1.0)


def test_safe_pow_1():

    with set_precision("medium"):

        dtype = get_float_dtype()

        def embed1(_rho, pow_fn, name=None):
            """
            rho < rho_n
            """
            with tf.name_scope(name):
                rho_n = tf.constant(30.0, dtype=dtype, name='rho_n')
                one = tf.constant(1.0, dtype=dtype, name='one')
                two = tf.constant(2.0, dtype=dtype, name='two')
                three = tf.constant(3.0, dtype=dtype, name='three')
                Fn0 = tf.constant(3.0, dtype=dtype, name='Fn0')
                Fn1 = tf.constant(1.0, dtype=dtype, name='Fn1')
                Fn2 = tf.constant(5.0, dtype=dtype, name='Fn2')
                Fn3 = tf.constant(4.0, dtype=dtype, name='Fn3')
                x1 = tf.subtract(tf.math.divide(_rho, rho_n), one, name='x')
                x2 = pow_fn(x1, two)
                x3 = pow_fn(x1, three)
                e11 = tf.multiply(Fn1, x1, 'Fn1e1')
                e12 = tf.multiply(Fn2, x2, 'Fn2e2')
                e13 = tf.multiply(Fn3, x3, 'Fn3e3')
                return tf.add(Fn0, e11 + e12 + e13, name='e1')

        def embed3(_rho, pow_fn, name=None):
            """
            rho_0 <= rho
            """
            with tf.name_scope(name):
                rho_s = tf.constant(30.0, dtype=dtype, name='rho_n')
                one = tf.constant(1.0, dtype=dtype, name='one')
                eta = tf.constant(0.5, dtype=dtype, name='eta')
                Fe = tf.constant(-3.0, dtype=dtype, name='Fe')
                x1 = _rho / rho_s + tf.constant(1e-8, dtype=dtype, name='eps')
                lnx = tf.log(x1)
                return tf.multiply(Fe * (one - eta * lnx),
                                   pow_fn(x1, eta), name='e3')

        with tf.Graph().as_default():

            rho = tf.constant(60.0, dtype=dtype, name='rho')
            x3 = embed3(rho, safe_pow, "safe3")
            y3 = embed3(rho, tf.pow, 'old3')
            dx3 = tf.gradients(x3, rho, name='dx3')
            dy3 = tf.gradients(y3, rho, name='dy3')

            x1 = embed1(rho, safe_pow, "safe1")
            y1 = embed1(rho, tf.pow, 'old1')
            dx1 = tf.gradients(x1, rho, name='dx1')
            dy1 = tf.gradients(y1, rho, name='dy1')

            with tf.Session() as sess:
                results = sess.run([x3, y3, dx3, dy3])
                assert_almost_equal(results[0], results[1], delta=1e-6)
                assert_almost_equal(results[3], results[2], delta=1e-6)

                results = sess.run([x1, y1, dx1, dy1])
                assert_almost_equal(results[0], results[1], delta=1e-6)
                assert_almost_equal(results[3], results[2], delta=1e-6)


if __name__ == "__main__":
    nose.run()
