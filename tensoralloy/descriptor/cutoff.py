# coding=utf-8
"""
This module defines various cutoff fucntions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["cosine_cutoff"]


def cosine_cutoff(r: tf.Tensor, rc: float, name=None):
    """
    The cosine cutoff function proposed by Jörg Behler:

        fc(r) = 0.5 * [ cos(min(r / rc) * pi) + 1 ]

    Parameters
    ----------
    r : tf.Tensor
        A `float64` or `float32` tensor.
    rc : float or tf.Tensor
        The cutoff radius. fc(r) = 0 if r > rc.
    name : str or None
        The name of this Op.

    See Also
    --------
    PRL 98, 146401 (2007)

    """
    with ops.name_scope(name, "CosCutoff", [r, rc]) as name:
        r = ops.convert_to_tensor(r, name='r')
        rc = ops.convert_to_tensor(rc, dtype=r.dtype, name="rc")
        ratio = math_ops.truediv(r, rc, name='ratio')
        one = ops.convert_to_tensor(1.0, dtype=r.dtype, name='one')
        half = ops.convert_to_tensor(0.5, dtype=r.dtype, name='half')
        z = math_ops.minimum(ratio, one, name='minimum')
        z = math_ops.cos(z * np.pi, name='cos') + one
        return math_ops.multiply(z, half, name=name)
