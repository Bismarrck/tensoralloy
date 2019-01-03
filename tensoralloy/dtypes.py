# coding=utf-8
"""
This module controls precision of floating-point numbers.
"""
from __future__ import print_function, absolute_import

import enum

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework.dtypes import DType as tf_DType
from typing import Union


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
    precision : Precision or str
        The precision of the floating-point numbers.

    """
    global _floating_point_precision
    if isinstance(precision, Precision):
        _floating_point_precision = precision
    else:
        _floating_point_precision = Precision[precision]


class DType(tf_DType):
    """
    A wrapper of `tensorflow.core.framework.dtypes.DType` with an added property
    `eps`.
    """

    def __init__(self, type_enum, eps: Union[float, int]):
        """
        Initialization method.
        """
        super(DType, self).__init__(type_enum=type_enum)
        self._eps = eps

    @property
    def eps(self):
        """
        Return the machine epsilon of this data type.
        """
        return self._eps


float64 = DType(types_pb2.DT_DOUBLE, eps=1e-14)
float32 = DType(types_pb2.DT_FLOAT, eps=1e-8)


def get_float_dtype():
    """
    Return the data dtype of the floating-point numbers.

    Returns
    -------
    dtype : DType
        The corresponding data type of floating-point numbers.

    """
    global _floating_point_precision
    if _floating_point_precision is None:
        set_float_precision()
    assert isinstance(_floating_point_precision, Precision)
    if _floating_point_precision == Precision.medium:
        return float32
    else:
        return float64
