# coding=utf-8
"""
This module defines helper functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def cutoff(r, rc, name=None):
    """
    The cutoff function.
    """
    with ops.name_scope(name, "fc", [r]) as name:
        rc = ops.convert_to_tensor(rc, dtype=tf.float32, name="rc")
        ratio = math_ops.div(r, rc, name='ratio')
        z = math_ops.minimum(ratio, 1.0, name='minimum')
        z = math_ops.cos(z * np.pi, name='cos') + 1.0
        return math_ops.multiply(z, 0.5, name=name)


def pairwise_dist(xx, yy, name=None):
    """
    Computes the pairwise distances between each elements of xx and yy.

    Parameters
    ----------
    xx : tf.Tensor
        A float32 tensor of shape `[n, d]`.
    yy : tf.Tensor
        A float32 tensor of shape `[m, d]`.
    name : str
        The name of this op.

    Returns
    -------
    D : tf.Tensor
        A float32 tensor of shape `[m, n]` as the pairwise distances.

    """
    with ops.name_scope(name, 'pairwise_dist', [xx, yy]) as name:
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(xx), 1)
        nb = tf.reduce_sum(tf.square(yy), 1)

        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidead difference matrix
        zz = tf.maximum(na - 2 * tf.matmul(xx, yy, False, True) + nb, 0.0)
        return tf.sqrt(zz, name=name)
