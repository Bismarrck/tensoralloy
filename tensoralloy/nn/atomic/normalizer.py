# coding=utf-8
"""
This module defines the input normalizer for `AtomicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# TODO: the optimized `alpha` may be negative. Constraints should be added.


class InputNormalizer:
    """
    A collection of funcitons for normalizing input descriptors to [0, 1].
    """

    def __init__(self, method='linear'):
        """
        Initialization method.
        """
        if not method:
            self.method = None
            self.scope = 'RawInput'
        else:
            self.method = method
            self.scope = '{}Norm'.format(method.capitalize())

    @property
    def enabled(self):
        """
        Return True if the input normalization is enabled.
        """
        return bool(self.method)

    def __call__(self, x: tf.Tensor, initial_weights=None, collections=None):
        """
        Apply the normalization.
        """
        default_collections = [tf.GraphKeys.TRAINABLE_VARIABLES,
                               tf.GraphKeys.GLOBAL_VARIABLES,
                               tf.GraphKeys.MODEL_VARIABLES]
        if collections is not None:
            for collection in collections:
                if collection not in default_collections:
                    default_collections.append(collection)

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if not self.method:
                return tf.identity(x, name='identity')

            if initial_weights is None:
                initial_weights = np.ones(x.shape[2], dtype=np.float64)
            alpha = tf.get_variable(
                name='alpha',
                shape=x.shape[2],
                dtype=x.dtype,
                initializer=tf.constant_initializer(
                    initial_weights, dtype=x.dtype),
                collections=default_collections,
                trainable=True)
            tf.summary.histogram(alpha.op.name + '/hist', alpha)
            x = tf.multiply(x, alpha, name='ax')
            if self.method == 'linear':
                x = tf.identity(x, name='x')
            elif self.method == 'arctan':
                x = tf.atan(x, name='x')
            else:
                raise ValueError(
                    f"Unsupported normalization method: {self.method}")
            tf.summary.histogram(x.op.name + '/hist', x)
            return x
