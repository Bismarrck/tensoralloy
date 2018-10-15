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


def indexedslices_to_dense(sparse_delta, dense_shape=None, name=None):
    """
    Convert a sparse `IndexedSlices` tensor to a dense tensor.

    Parameters
    ----------
    sparse_delta : tf.IndexedSlices
        A sparse representation of a set of tensor slices at given indices.
        The IndexedSlices class is used principally in the definition of
        gradients for operations that have sparse gradients (e.g. tf.gather).
    dense_shape : array_like or tf.Tensor or None
        A list (array, tuple) or a `tf.Tensor` as the shape of the dense tensor.
        If None, `sparse_delta.dense_shape` will be used.
    name : str
        The name of this op.

    Returns
    -------
    dense : tf.Tensor
        The dense representation of `sparse_delta`.

    """
    with ops.name_scope(name, "sparse_to_dense", [sparse_delta]) as name:
        indices = tf.reshape(sparse_delta.indices, (-1, 1), name='reshape')
        if dense_shape is None:
            dense_shape = sparse_delta.dense_shape
        else:
            dense_shape = ops.convert_to_tensor(
                dense_shape, dtype=tf.int32, name='dense_shape')
        return tf.scatter_nd(indices, sparse_delta.values, dense_shape, name)


def cutoff(r: tf.Tensor, rc: float, name=None):
    """
    The cutoff function.

    f_c(r) = 0.5 * [ cos(min(r / rc) * pi) + 1 ]

    """
    with ops.name_scope(name, "fc", [r]) as name:
        rc = ops.convert_to_tensor(rc, dtype=r.dtype, name="rc")
        ratio = math_ops.div(r, rc, name='ratio')
        z = math_ops.minimum(ratio, 1.0, name='minimum')
        z = math_ops.cos(z * np.pi, name='cos') + 1.0
        return math_ops.multiply(z, 0.5, name=name)


def pairwise_dist(xx, yy=None, name=None):
    """
    Computes the pairwise distances between each elements of xx and yy.

    Parameters
    ----------
    xx : tf.Tensor
        A tensor of shape `[m, d]` or `[b, m, d]`.
    yy : tf.Tensor
        A tensor of shape `[n, d]` or `[b, n, d]`.
    name : str
        A `str` as the name of this op.

    Returns
    -------
    D : tf.Tensor
        A tensor of shape `[m, n]` or `[b, m, n]` as the pairwise distances.

    See Also
    --------
    https://stackoverflow.com/questions/39822276
    https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865

    """
    with ops.name_scope(name, 'pairwise_dist', [xx, yy]) as name:
        if yy is None:
            yy = xx
        if len(xx.shape) == 2:
            assert xx.shape[1] == yy.shape[1]
        elif len(xx.shape) == 3:
            assert xx.shape[0] == yy.shape[0]
        else:
            raise ValueError("The rank of the input matrices must be 2 or 3")

        dtype = xx.dtype
        two = tf.constant(2.0, dtype=dtype, name='two')
        zero = tf.constant(0.0, dtype=dtype, name='zero')

        if len(xx.shape) == 2:
            # squared norms of each row in A and B
            na = tf.reduce_sum(tf.square(xx), 1)
            nb = tf.reduce_sum(tf.square(yy), 1)

            # na as a row and nb as a column vectors
            na = tf.reshape(na, [-1, 1])
            nb = tf.reshape(nb, [1, -1])

            # xy = xx * yy.T
            xy = tf.matmul(xx, yy, transpose_a=False, transpose_b=True)
        else:
            # Get the batch size
            batch_size = xx.shape[0]

            # squared norms of each row in A and B
            na = tf.reduce_sum(tf.square(xx), 2, keep_dims=True)
            nb = tf.reduce_sum(tf.square(yy), 2, keep_dims=True)

            # na as a row and nb as a column vectors
            na = tf.reshape(na, [batch_size, -1, 1])
            nb = tf.reshape(nb, [batch_size, 1, -1])

            # Use einsum to compute matrix multiplications
            xy = tf.einsum('ijk,ilk->ijl', xx, yy, name='einsum')

        # return pairwise euclidead difference matrix
        zz = tf.maximum(na - tf.multiply(two, xy, name='2xy') + nb, zero)
        return tf.sqrt(zz, name=name)
