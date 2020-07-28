# coding=utf-8
"""
This module defines unit tests for cutoff functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from nose.tools import assert_less

from tensoralloy.nn.cutoff import cosine_cutoff
from tensoralloy.nn.cutoff import polynomial_cutoff
from tensoralloy.nn.cutoff import meam_cutoff
from tensoralloy.nn.cutoff import tersoff_cutoff
from tensoralloy.precision import get_float_dtype, set_float_precision
from tensoralloy.precision import Precision

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

        assert_less(np.abs(x - y).max(), 1e-7)
    set_float_precision(Precision.high)


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


def meam_cutoff_simple(x: float):
    """
    A simple implementation of the MEAM cutoff function.
    """
    if x >= 1.0:
        return 1.0
    elif x <= 0.0:
        return 0.0
    else:
        return (1 - (1 - x)**4)**2


def test_meam_cutoff():
    """
    Test the MEAM cutoff function.
    """
    x = np.linspace(-1.0, 2.0, num=11, endpoint=True)
    y = np.asarray([meam_cutoff_simple(xi) for xi in x])

    with tf.Session() as sess:
        z = sess.run(meam_cutoff(
            tf.convert_to_tensor(x, dtype=tf.float64, name='x')))
        assert_less(np.abs(y - z).max(), 1e-8)


def tersoff_cutoff_simple(r, R, D):
    """
    A plain implementation of the Tersoff cutoff function.
    """
    if r < R - D:
        return 1.0
    elif r > R + D:
        return 0.0
    else:
        return 0.5 - 0.5 * np.sin(0.5 * np.pi * (r - R) / D)


def test_tersoff_cutoff():
    """
    Test the Tersoff cutoff function.
    """
    x = np.linspace(0.0, 4.0, num=41, endpoint=True)
    y = np.asarray([tersoff_cutoff_simple(xi, 3.0, 0.2) for xi in x])
    with tf.Session() as sess:
        z = sess.run(
            tersoff_cutoff(
                tf.convert_to_tensor(x, dtype=tf.float64, name='x'), 3.0, 0.2))
        assert_less(np.abs(y - z).max(), 1e-8)


if __name__ == "__main__":
    nose.run()
