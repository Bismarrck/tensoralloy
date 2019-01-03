# coding=utf-8
"""
This module defines unit tests for cutoff functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from nose.tools import assert_less

from tensoralloy.descriptor.cutoff import cosine_cutoff
from tensoralloy.descriptor.cutoff import polynomial_cutoff
from tensoralloy.dtypes import get_float_dtype, set_float_precision, Precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def cosine_cutoff_simple(r: float, rc: float):
    """
    The most straightforward implementation of the cosine cutoff function:

        f(r) = 0.5 * (cos(pi * r / rc) + 1) if r <= rc
             = 0                            if r > rc

    """
    if r <= rc:
        return 0.5 * (np.cos(np.pi * r / rc) + 1.0)
    else:
        return 0.0


def test_cosine_cutoff():
    """
    Test the cosine cutoff function.
    """
    rc = 6.0
    set_float_precision(Precision.medium)
    dtype = get_float_dtype()
    r = np.linspace(1.0, 10.0, num=91, endpoint=True,
                    dtype=dtype.as_numpy_dtype)
    x = np.asarray([cosine_cutoff_simple(ri, rc) for ri in r],
                   dtype=dtype.as_numpy_dtype)

    with tf.Session() as sess:
        y = sess.run(
            cosine_cutoff(tf.convert_to_tensor(r, dtype=dtype),
                          tf.convert_to_tensor(rc, dtype=dtype),
                          name='cutoff'))

        assert_less(np.abs(x - y).max(), dtype.eps)


def polynomial_cutoff_simple(r: float, rc: float, gamma: float):
    """
    The most straightforward implementation of the polynomial cutoff function.
    """
    if r <= rc:
        div = r / rc
        return 1.0 + gamma * div**(gamma + 1.0) - (gamma + 1.0) * div**gamma
    else:
        return 0.0


def test_polynomial_cutoff():
    """
    Test the polynomial cutoff function.
    """
    rc = 6.0
    gamma = 5.0
    r = np.linspace(1.0, 10.0, num=91, endpoint=True)
    x = np.asarray([polynomial_cutoff_simple(ri, rc, gamma) for ri in r])

    with tf.Session() as sess:
        y = sess.run(
            polynomial_cutoff(tf.convert_to_tensor(r, dtype=tf.float64),
                              tf.convert_to_tensor(rc, dtype=tf.float64),
                              tf.convert_to_tensor(gamma, dtype=tf.float64),
                              name='cutoff'))

        assert_less(np.abs(x - y).max(), 1e-8)


if __name__ == "__main__":
    nose.run()
