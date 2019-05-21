#!coding=utf-8
"""
This module defines unit tests of the extensions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from scipy.interpolate import CubicSpline
from os.path import join

from tensoralloy.extension.interp.cubic import CubicInterpolator
from tensoralloy.io.lammps import read_adp_setfl
from tensoralloy.test_utils import assert_array_almost_equal, test_dir
from tensoralloy.precision import set_precision, get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_cubic_spline_1d():
    """
    Test the cubic spline function with scalars and natural boundary type.
    """
    xlist = np.linspace(0, 10.0, num=11, endpoint=True)
    ylist = np.cos(-xlist**2 / 9.0)
    scipy_func = CubicSpline(xlist, ylist, bc_type='natural')

    rlist = np.random.uniform(1.0, 9.5, size=15)
    zlist = scipy_func(rlist)

    with tf.Graph().as_default():
        dtype = tf.float64
        x = tf.convert_to_tensor(xlist, dtype=dtype, name='x')
        y = tf.convert_to_tensor(ylist, dtype=dtype, name='y')
        r = tf.placeholder(dtype=dtype, shape=(None, ), name='r')

        clf = CubicInterpolator(x, y, natural_boundary=True)
        z = clf.evaluate(r)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            values = sess.run(z, feed_dict={r: rlist})
            assert_array_almost_equal(values, zlist, delta=1e-6)


def test_cubic_spline_2d():
    """
    Test the cubic spline function with multidimensional arrays and clamped
    boundary.
    """
    xlist = np.linspace(0, 99.0, num=100, endpoint=True).reshape((10, 10))
    ylist = np.cos(-xlist ** 2 / 9.0)
    rlist = np.random.uniform(1.0, 9.5, size=(10, 5))
    rlist[:, 2] = rlist[:, 0]
    rlist[:, 3] = rlist[:, 1]

    with tf.Graph().as_default():
        dtype = tf.float64
        x = tf.convert_to_tensor(xlist, dtype=dtype, name='x')
        y = tf.convert_to_tensor(ylist, dtype=dtype, name='y')
        r = tf.placeholder(dtype=dtype, shape=rlist.shape, name='r')

        clf = CubicInterpolator(x, y, natural_boundary=False)
        z = clf.evaluate(r)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            sess.run(z, feed_dict={r: rlist})


def test_reconstruct_lammps_adp():
    """
    Test the reconstruction of the Al-Cu ADP potential.
    """
    with set_precision('high'):
        with tf.Graph().as_default():

            filename = join(test_dir(), 'lammps', 'AlCu.adp')
            adpfl = read_adp_setfl(filename)

            dtype = get_float_dtype()

            rlist = tf.convert_to_tensor(
                adpfl.dipole['AlCu'][0], name='rlist', dtype=dtype)

            u_alcu = adpfl.dipole['AlCu'][0] * adpfl.dipole['AlCu'][1]
            u_alcu[0] = adpfl.dipole['AlCu'][1][0]

            ux = adpfl.dipole['AlCu'][0][0:-1:50]
            uy = u_alcu[0:-1:50]

            func = CubicInterpolator(ux, uy, natural_boundary=True)
            u_op = func.evaluate(rlist, name='Dipole')

            with tf.Session() as sess:
                u_reconstruct = sess.run(u_op)
                assert_array_almost_equal(u_reconstruct, u_alcu, delta=1e-5)


if __name__ == "__main__":
    nose.run()
