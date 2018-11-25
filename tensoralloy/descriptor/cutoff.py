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


def cosine_cutoff(r: tf.Tensor, rc: float, name=None):
    """
    The cosine cutoff function proposed by Behler:

    f_c(r) = 0.5 * [ cos(min(r / rc) * pi) + 1 ]

    See Also
    --------
    PRL 98, 146401 (2007)

    """
    with ops.name_scope(name, "fc", [r]) as name:
        rc = ops.convert_to_tensor(rc, dtype=r.dtype, name="rc")
        ratio = math_ops.div(r, rc, name='ratio')
        z = math_ops.minimum(ratio, 1.0, name='minimum')
        z = math_ops.cos(z * np.pi, name='cos') + 1.0
        return math_ops.multiply(z, 0.5, name=name)
