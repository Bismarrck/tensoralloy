# coding=utf-8
"""
This module defines tests of `AtomicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_equal

from tensoralloy.misc import AttributeDict
from ..atomic import AtomicNN

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_inference():
    """
    Test the inference of `AtomicNN`.
    """
    with tf.Graph().as_default():

        elements = sorted(['Al', 'Cu'])
        hidden_sizes = 32

        batch_size = 10
        max_n_al = 5
        max_n_cu = 7
        max_n_atoms = max_n_al + max_n_cu + 1
        ndim = 4

        g_al = np.random.randn(batch_size, max_n_al, ndim)
        g_cu = np.random.randn(batch_size, max_n_cu, ndim)

        nn = AtomicNN(elements, hidden_sizes, activation='tanh', forces=False)

        with tf.name_scope("Inputs"):

            descriptors = AttributeDict(
                Al=(tf.convert_to_tensor(g_al, tf.float64, 'g_al'),
                    tf.no_op('Al')),
                Cu=(tf.convert_to_tensor(g_cu, tf.float64, 'g_cu'),
                    tf.no_op('Cu')))
            positions = tf.constant(
                np.random.rand(1, max_n_atoms, 3),
                dtype=tf.float64,
                name='positions')
            mask = np.ones((batch_size, max_n_atoms), np.float64)
            features = AttributeDict(
                descriptors=descriptors, positions=positions, mask=mask)

        predictions = nn.build(features, verbose=True)

        assert_equal(predictions.energy.shape.as_list(), [batch_size, ])


if __name__ == "__main__":
    nose.run()
