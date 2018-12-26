# coding=utf-8
"""
This module defines tests of `AtomicNN` and its variants.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import os

from os.path import exists, join, dirname
from nose.tools import assert_equal, assert_list_equal, with_setup
from nose.tools import assert_true, assert_false

from tensoralloy.nn.atomic import AtomicNN, AtomicResNN
from tensoralloy.misc import AttributeDict, test_dir
from tensoralloy.transformer import SymmetryFunctionTransformer
from tensoralloy.nn.utils import GraphKeys

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

        nn = AtomicNN(elements, hidden_sizes,
                      normalizer='linear',
                      activation='tanh',
                      minimize_properties=['energy', ],
                      export_properties=['energy', ])

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

        collection = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        assert_equal(len(collection), 8)

        collection = tf.get_collection(GraphKeys.ATOMIC_NN_VARIABLES)
        assert_equal(len(collection), 8)


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
        nn = AtomicResNN(clf.elements, export_properties=['energy', 'forces'],
                         normalizer=None)
        prediction = nn.build(features)
        assert_list_equal(prediction.energy.shape.as_list(), [])

        collection = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        assert_equal(len(collection), 11)

        collection = tf.get_collection(GraphKeys.ATOMIC_RES_NN_VARIABLES)
        assert_equal(len(collection), 11)


output_graph_path = join(
    test_dir(), 'checkpoints', 'Ni-k2', 'Ni.belher.k2.pb')

checkpoint_path = join(
    test_dir(), 'checkpoints', 'Ni-k2', 'model.ckpt-100000')


def _delete():
    if exists(output_graph_path):
        os.remove(output_graph_path)


@with_setup(teardown=_delete)
def test_export_to_pb():
    """
    Test exporting an `AtomicNN` to a pb file.
    """

    def input_fn():
        """
        An example of the `input_fn` for exporting an `AtomicNN` to a pb file.
        """
        clf = SymmetryFunctionTransformer(
            rc=6.5, elements=['Ni'], k_max=2,
            eta=[0.05, 0.4, 2.0, 4.0, 8.0, 20.0, 40.0, 80.0])
        placeholders = clf.placeholders
        descriptors = clf.get_graph()
        features = AttributeDict(descriptors=descriptors,
                                 positions=placeholders.positions,
                                 n_atoms=placeholders.n_atoms,
                                 cells=placeholders.cells,
                                 composition=placeholders.composition,
                                 mask=placeholders.mask,
                                 volume=placeholders.volume)
        return features, None

    nn = AtomicNN(elements=['Ni'], hidden_sizes={'Ni': [64, 32]},
                  activation='leaky_relu',
                  export_properties=['energy', 'forces', 'stress'],
                  normalizer=None)
    nn.export(input_fn,
              output_graph_path=output_graph_path,
              checkpoint=checkpoint_path,
              keep_tmp_files=False)

    assert_true(exists(output_graph_path))
    assert_false(exists(join(dirname(output_graph_path), 'export')))


if __name__ == "__main__":
    nose.run()
