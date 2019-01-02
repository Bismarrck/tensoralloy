# coding=utf-8
"""
This module controls precision of floating-point numbers.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import enum

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["Precision", "set_float_precision", "get_float_dtype"]


class Precision(enum.Enum):
    """
    Precision options of floating-point numbers:

    high : float64
    medium : float32

    """
    high = 0
    medium = 1


_floating_point_precision = None


def set_float_precision(precision=Precision.high):
    """
    Set the default precision of all floating-point numbers.

    Parameters
    ----------
    precision : Precision
        The precision of the floating-point numbers.

    """
    global _floating_point_precision
    _floating_point_precision = precision


def get_float_dtype(numpy=False):
    """
    Return the data dtype of the floating-point numbers.

    Parameters
    ----------
    numpy : bool
        If True, return the numpy data type; Otherwise return the tensorflow
        data type.

    """
    global _floating_point_precision
    if _floating_point_precision is None:
        set_float_precision()
    assert isinstance(_floating_point_precision, Precision)
    if _floating_point_precision == Precision.medium:
        if numpy:
            return np.float32
        else:
            return tf.float32
    else:
        if numpy:
            return np.float64
        else:
            return tf.float64
