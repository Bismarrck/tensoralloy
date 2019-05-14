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

            g = tf.gradients(origin, x)[0]
            g = tf.gradients(g, x, name='g')[0]

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

                assert_true(np.isnan(sess.run(g, feed_dict={x: 0.0, y: 1.0})))


if __name__ == "__main__":
    nose.run()
