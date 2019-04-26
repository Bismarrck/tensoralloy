#!coding=utf-8
"""
This module defines unit tests of the extensions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from scipy.interpolate import CubicSpline

from tensoralloy.extension.interp.cubic import CubicInterpolator
from tensoralloy.test_utils import assert_array_almost_equal

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_cubic_spline():
    """
    Test the cubic spline function.
    """
    xlist = np.linspace(0, 10.0, num=11, endpoint=True)
    ylist = np.cos(-xlist**2 / 9.0)
    scipy_func = CubicSpline(xlist, ylist, bc_type='clamped')

    rlist = np.random.uniform(1.0, 9.5, size=15)
    zlist = scipy_func(rlist)

    with tf.Graph().as_default():
        dtype = tf.float64
        x = tf.convert_to_tensor(xlist, dtype=dtype, name='x')
        y = tf.convert_to_tensor(ylist, dtype=dtype, name='y')
        r = tf.placeholder(dtype=dtype, shape=(None, ), name='r')

        clf = CubicInterpolator(x, y)
        z = clf.evaluate(r)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            values = sess.run(z, feed_dict={r: rlist})

            print(values)
            print(zlist)

            assert_array_almost_equal(values, zlist, delta=1e-6)


if __name__ == "__main__":
    nose.run()
