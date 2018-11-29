# coding=utf-8
"""
This module defines unit tests related functions and vars.
"""
from __future__ import print_function, absolute_import

import numpy as np
from nose.tools import assert_less, assert_equal

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def assert_array_almost_equal(a: np.ndarray, b: np.ndarray, delta, msg=None):
    """
    Fail if the two arrays are unequal as determined by their maximum absolute
    difference rounded to the given number of decimal places (default 7) and
    comparing to zero, or by comparing that the between the two objects is more
    than the given delta.
    """
    assert_less(np.abs(a - b).max(), delta, msg)


def assert_array_equal(a: np.ndarray, b: np.ndarray, msg=None):
    """
    Fail if the two arrays are unequal with threshold 0 (int) or 1e-6 (float32)
    or 1e-12 (float64).
    """
    assert_equal(a.dtype, b.dtype)
    if np.issubdtype(a.dtype, np.int_):
        delta = 0
    elif a.dtype == np.float64:
        delta = 1e-12
    else:
        delta = 1e-6
    assert_array_almost_equal(a, b, delta, msg=msg)
