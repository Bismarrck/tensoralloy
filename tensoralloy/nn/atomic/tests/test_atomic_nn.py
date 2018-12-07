# coding=utf-8
"""
This module defines tests of `AtomicNN` and its variants.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_equal, assert_list_equal

from ..atomic import AtomicNN
from ..resnet import AtomicResNN
from tensoralloy.misc import AttributeDict
from tensoralloy.transformer import SymmetryFunctionTransformer

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
            mask = tf.convert_to_tensor(
                np.ones((batch_size, max_n_atoms), np.float64))
            features = AttributeDict(
                descriptors=descriptors, positions=positions, mask=mask)

        predictions = nn.build(features, verbose=True)

        assert_equal(predictions.energy.shape.as_list(), [batch_size, ])


def test_inference_from_transformer():
    """
    Test the inference of `AtomicResNN` using `SymmetryFunctionTransformer`.
    """
    with tf.Graph().as_default():
        rc = 6.5
        elements = ['Al', 'Cu']
        clf = SymmetryFunctionTransformer(rc=rc, elements=elements, k_max=2)
        placeholders = clf.placeholders
        descriptors = clf.get_graph()
        features = AttributeDict(descriptors=descriptors,
                                 positions=placeholders.positions,
                                 n_atoms=placeholders.n_atoms,
                                 cells=placeholders.cells,
                                 composition=placeholders.composition,
                                 mask=placeholders.mask,
                                 volume=placeholders.volume)
        nn = AtomicResNN(clf.elements, forces=True)
        prediction = nn.build(features)
        assert_list_equal(prediction.energy.shape.as_list(), [])


if __name__ == "__main__":
    nose.run()
