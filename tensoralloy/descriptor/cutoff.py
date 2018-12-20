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
    The cosine cutoff function proposed by Behler:

    f_c(r) = 0.5 * [ cos(min(r / rc) * pi) + 1 ]

    See Also
    --------
    PRL 98, 146401 (2007)

    """
    with ops.name_scope(name, "CosCutoff", [r]) as name:
        # TODO: 'TypeError: must be real number, not Tensor' will occur here if
        #  this module was cythonized.
        rc = ops.convert_to_tensor(rc, dtype=tf.float64, name="rc")
        ratio = math_ops.div(r, rc, name='ratio')
        one = ops.convert_to_tensor(1.0, dtype=tf.float64, name='one')
        half = ops.convert_to_tensor(0.5, dtype=tf.float64, name='half')
        z = math_ops.minimum(ratio, one, name='minimum')
        z = math_ops.cos(z * np.pi, name='cos') + one
        return math_ops.multiply(z, half, name=name)
