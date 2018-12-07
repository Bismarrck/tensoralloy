# coding=utf-8
"""
This module defines the input normalizer for `AtomicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from tensoralloy.nn.utils import GraphKeys

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class InputNormalizer:
    """
    A collection of funcitons for normalizing input descriptors to [0, 1].
    """

    def __init__(self, method='linear'):
        """
        Initialization method.
        """
        assert method in ('linear', 'arctan', '', None)
        if not method:
            self.method = None
            self.scope = 'Identity'
        else:
            self.method = method
            self.scope = '{}Norm'.format(method.capitalize())

    def __call__(self, x: tf.Tensor, values=None):
        """
        Apply the normalization.
        """
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if not self.method:
                return tf.identity(x, name='identity')

            if values is None:
                values = np.ones(x.shape[2], dtype=np.float64)
            alpha = tf.get_variable(
                name='alpha',
                shape=x.shape[2],
                dtype=x.dtype,
                initializer=tf.constant_initializer(values, dtype=x.dtype),
                collections=[tf.GraphKeys.TRAINABLE_VARIABLES,
                             tf.GraphKeys.GLOBAL_VARIABLES,
                             GraphKeys.NORMALIZE_VARIABLES],
                trainable=True)
            tf.summary.histogram(
                name=alpha.op.name + '/summary',
                values=alpha,
                collections=[GraphKeys.TRAIN_SUMMARY])
            x = tf.multiply(x, alpha, name='ax')
            if self.method == 'linear':
                x = tf.identity(x, name='x')
            elif self.method == 'arctan':
                x = tf.atan(x, name='x')
            else:
                raise ValueError(
                    f"Unsupported normalization method: {self.method}")
            tf.summary.histogram(
                name=x.op.name + '/summary',
                values=x,
                collections=[GraphKeys.TRAIN_SUMMARY])
            return x
