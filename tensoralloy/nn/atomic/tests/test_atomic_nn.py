# coding=utf-8
"""
This module defines tests of `AtomicNN` and its variants.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from ase.db import connect
from os.path import join
from nose.tools import assert_equal, assert_list_equal
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.nn.atomic import AtomicNN
from tensoralloy.test_utils import test_dir, datasets_dir
from tensoralloy.transformer import SymmetryFunctionTransformer
from tensoralloy.transformer import BatchSymmetryFunctionTransformer
from tensoralloy.utils import GraphKeys

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_as_dict():
    """
    Test the method `AtomicNN.as_dict`.
    """
    elements = ['Al', 'Cu']
    hidden_sizes = 32
    old_nn = AtomicNN(elements, hidden_sizes,
                      activation='tanh',
                      use_atomic_static_energy=False,
                      minimize_properties=['energy', ],
                      export_properties=['energy', ])

    d = old_nn.as_dict()

    assert_equal(d['class'], 'AtomicNN')
    d.pop('class')

    new_nn = AtomicNN(**d)

    assert_list_equal(new_nn.elements, old_nn.elements)
    assert_list_equal(new_nn.minimize_properties, old_nn.minimize_properties)
    assert_equal(new_nn.hidden_sizes, old_nn.hidden_sizes)


def test_as_dict_advanced():
    """
    Test the method `AtomicResNN.as_dict`.
    """
    db = connect(join(datasets_dir(), 'qm7.db'))
    max_occurs = db.metadata['max_occurs']

    eta = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 40.0]
    omega = [0.0, 3.2]
    rc = 6.5
    nij_max = 506
    batch_size = 10

    bsf = BatchSymmetryFunctionTransformer(rc=rc, max_occurs=max_occurs,
                                           nij_max=nij_max, nijk_max=0,
                                           batch_size=batch_size, eta=eta,
                                           omega=omega, use_forces=False,
                                           use_stress=False)

    nn = AtomicNN(max_occurs.keys(), use_atomic_static_energy=False)
    nn.attach_transformer(bsf)

    configs = nn.as_dict()
    assert_equal(configs.pop('class'), 'AtomicNN')

    gen = nn.__class__(**configs)
    sf = bsf.as_descriptor_transformer()
    assert isinstance(sf, SymmetryFunctionTransformer)

    assert_list_equal(nn.elements, gen.elements)
    assert_equal(bsf.rc, sf.rc)
    assert_list_equal(bsf.initial_values["omega"].tolist(),
                      sf.initial_values["omega"].tolist())
    assert_list_equal(bsf.initial_values["eta"].tolist(),
                      sf.initial_values["eta"].tolist())


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
                      activation='tanh',
                      minmax_scale=False,
                      use_atomic_static_energy=False,
                      minimize_properties=['energy', ],
                      export_properties=['energy', ])

        with tf.name_scope("Inputs"):

            descriptors = dict(
                Al=(tf.convert_to_tensor(g_al, tf.float64, 'g_al'),
                    tf.no_op('Al')),
                Cu=(tf.convert_to_tensor(g_cu, tf.float64, 'g_cu'),
                    tf.no_op('Cu')))
            positions = tf.constant(
                np.random.rand(1, max_n_atoms, 3),
                dtype=tf.float64,
                name='positions')
            cell = tf.convert_to_tensor(
                np.random.rand(batch_size, 3, 3).astype(np.float64))
            atom_masks = tf.convert_to_tensor(
                np.ones((batch_size, max_n_atoms), np.float64))
            pulay_stress = tf.zeros(batch_size, dtype=tf.float64, name='pulay')
            features = dict(positions=positions, atom_masks=atom_masks,
                            cell=cell, pulay_stress=pulay_stress)

        outputs = nn._get_model_outputs(
            features=features,
            descriptors=descriptors,
            mode=tf_estimator.ModeKeys.TRAIN,
            verbose=True)
        ops = nn._get_energy_ops(
            outputs, features, verbose=False)

        assert_equal(ops.energy.shape.as_list(), [batch_size, ])

        collection = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        assert_equal(len(collection), 6)

        collection = tf.get_collection(GraphKeys.ATOMIC_NN_VARIABLES)
        assert_equal(len(collection), 6)


def test_inference_from_transformer():
    """
    Test the inference of `AtomicResNN` using `SymmetryFunctionTransformer`.
    """
    with tf.Graph().as_default():
        rc = 6.5
        elements = ['Al', 'Cu']
        clf = SymmetryFunctionTransformer(rc=rc, elements=elements,
                                          angular=False)
        nn = AtomicNN(elements=clf.elements,
                      minmax_scale=False,
                      use_atomic_static_energy=True,
                      export_properties=['energy', 'forces'])
        nn.attach_transformer(clf)
        prediction = nn.build(features=clf.get_placeholder_features(),
                              mode=tf_estimator.ModeKeys.PREDICT)
        assert_list_equal(prediction["energy"].shape.as_list(), [])

        collection = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        assert_equal(len(collection), 17)

        assert_equal(len(tf.trainable_variables()), 12)

        collection = tf.get_collection(GraphKeys.ATOMIC_NN_VARIABLES)
        assert_equal(len(collection), 12)

        collection = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        assert_equal(len(collection), 10)


if __name__ == "__main__":
    nose.main()
