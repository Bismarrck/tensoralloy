# coding=utf-8
"""
This module defines tests of `AtomicNN` and its variants.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import unittest
import nose

from nose.tools import assert_equal, assert_list_equal, assert_almost_equal

from tensoralloy import atoms_utils
from tensoralloy.utils import ModeKeys
from tensoralloy.io.db import snap
from tensoralloy.nn.atomic import TemperatureDependentAtomicNN
from tensoralloy.nn.atomic.sf import SymmetryFunction
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
    sf = SymmetryFunction(elements)

    old_nn = TemperatureDependentAtomicNN(
        elements, sf, hidden_sizes, activation='tanh',
        use_atomic_static_energy=False, minimize_properties=['energy', ],
        export_properties=['energy', ], finite_temperature={"algorithm": "off"})

    d = old_nn.as_dict()

    assert_equal(d['class'], 'TemperatureDependentAtomicNN')
    d.pop('class')

    new_nn = TemperatureDependentAtomicNN(**d)

    assert_equal(new_nn.descriptor.name, "SF")
    assert_list_equal(new_nn.elements, old_nn.elements)
    assert_list_equal(new_nn.minimize_properties, old_nn.minimize_properties)
    assert_equal(new_nn.hidden_sizes, old_nn.hidden_sizes)
    assert_equal(new_nn.finite_temperature_options.on, False)


def test_tdsf():
    """
    Test temperature-dependent symmetry function atomistic neural network model.
    """
    db = snap()
    etemp = 0.17
    atoms = db.get_atoms(id=1)
    atoms_utils.set_electron_temperature(atoms, etemp)

    elements = sorted(list(set(atoms.get_chemical_symbols())))
    clf = UniversalTransformer(elements, rcut=4.5, acut=4.5, angular=True,
                               use_computed_dists=True)
    sf = SymmetryFunction(elements=elements)
    nn = TemperatureDependentAtomicNN(
        elements, sf, hidden_sizes=[64, 64], activation='softplus')
    nn.attach_transformer(clf)

    with tf.Graph().as_default():
        predictions = nn.build(clf.get_constant_features(atoms),
                               mode=ModeKeys.PREDICT)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            results = sess.run(predictions)
            energy = results['energy']
            free_energy = results['free_energy']
            eentropy = results['eentropy']
            free_energy_per_atom = results['free_energy/atom']
            assert_almost_equal(
                free_energy, energy - etemp * eentropy, delta=1e-6)
            assert_almost_equal(
                sum(results['eentropy/atom']), results['eentropy'], delta=1e-6)
            assert_almost_equal(
                sum(results['energy/atom']), energy, delta=1e-6)
            assert_almost_equal(
                sum(free_energy_per_atom), free_energy, delta=1e-6)


if __name__ == "__main__":
    nose.main()
