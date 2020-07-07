# coding=utf-8
"""
This module defines tests of `AtomicNN` and its variants.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import unittest
import nose

from nose.tools import assert_equal, assert_list_equal, assert_almost_equal
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy import atoms_utils
from tensoralloy.io.db import snap
from tensoralloy.nn.atomic import AtomicNN
from tensoralloy.nn.atomic.sf import SymmetryFunctionNN
from tensoralloy.transformer.universal import UniversalTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@unittest.skip
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


def test_tdsf():
    """
    Test temperature-dependent symmetry function atomistic neural network model.
    """
    db = snap()
    etemp = 0.17
    atoms = db.get_atoms(id=1)
    atoms_utils.set_electron_temperature(atoms, etemp)

    elements = sorted(list(set(atoms.get_chemical_symbols())))
    nn = SymmetryFunctionNN(elements=elements,
                            hidden_sizes=[64, 64],
                            activation='softplus',
                            temperature_dependent=True)
    clf = UniversalTransformer(elements, rcut=4.5, acut=4.5, angular=True,
                               use_computed_dists=True)
    nn.attach_transformer(clf)

    with tf.Graph().as_default():
        predictions = nn.build(clf.get_constant_features(atoms),
                               mode=tf_estimator.ModeKeys.PREDICT)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            results = sess.run(predictions)
            energy = results['energy']
            free_energy = results['free_energy']
            eentropy = results['eentropy']
            atomic = results['atomic']
            assert_almost_equal(
                free_energy, energy - etemp * eentropy, delta=1e-6)
            assert_almost_equal(sum(atomic), free_energy, delta=1e-6)


if __name__ == "__main__":
    nose.main()
