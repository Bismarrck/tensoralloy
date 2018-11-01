# coding=utf-8
"""
This module defines tensorflow-based functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import logging
from logging.config import dictConfig
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def batch_gather_positions(R, indices, batch_size=None, name=None):
    """
    A special batched implementation for `tf.gather` for gathering positions.

    Parameters
    ----------
    R : array_like or tf.Tensor
        An array of shape `[batch_size, n_atoms, 3]` as the atomic positions for
        several structures.
    indices : array_like or tf.Tensor
        An array of shape `[batch_size, nij_max]`. `indices[i]` denotes the
        slicing indices for structure `i`.
    batch_size : int or None
        The batch size. If None, batch size will be infered from R. However in
        some cases the inference may not work.
    name : str
        The name of this op.

    Returns
    -------
    R : tf.Tensor
        A `float64` tensor of shape `[batch_size, nij_max, 3]`.

    """
    with ops.name_scope(name, "gather_r", [R, indices]) as name:
        R = tf.convert_to_tensor(R, dtype=tf.float64, name='R')
        indices = tf.convert_to_tensor(indices, dtype=tf.int32, name='indices')
        batch_size = batch_size or R.shape[0]
        step = R.shape[1]
        delta = tf.range(0, batch_size * step, step, dtype=tf.int32)
        delta = tf.reshape(delta, (-1, 1), name='delta')
        R = tf.reshape(R, (-1, 3), name='flat_R')
        indices = tf.add(indices, delta, name='indices')
        indices = tf.reshape(indices, (-1,), name='flat_idx')
        positions = tf.gather(R, indices, name='gather')
        return tf.reshape(positions, (batch_size, -1, 3), name=name)


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


def set_logging_configs(logfile="logfile"):
    """
    Setup the logging module.
    """
    LOGGING_CONFIG = {
        "version": 1,
        "formatters": {
            'file': {
                'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
            },
        },
        "handlers": {
            'file': {
                'class': 'logging.FileHandler',
                'level': logging.INFO,
                'formatter': 'file',
                'filename': logfile,
                'mode': 'a',
            },
        },
        "root": {
            'handlers': ['file'],
            'level': logging.INFO,
        },
        "disable_existing_loggers": False
    }
    dictConfig(LOGGING_CONFIG)
