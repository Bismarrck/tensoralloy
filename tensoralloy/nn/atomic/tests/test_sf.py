#!coding=utf-8
"""
Unit tests of the symmetry function neural network (SFNN).
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from unittest import skip
from tensorflow_estimator import estimator as tf_estimator
from ase.io import read
from nose.tools import assert_less
from os.path import join
from sklearn.metrics import pairwise_distances

from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.transformer.tests.test_behler import get_radial_fingerprints_v1
from tensoralloy.transformer.tests.test_behler import get_augular_fingerprints_v1
from tensoralloy.transformer.tests.test_behler import get_radial_fingerprints_v2
from tensoralloy.transformer.tests.test_behler import legacy_symmetry_function
from tensoralloy.nn.atomic.sf import SymmetryFunctionNN
from tensoralloy.precision import precision_scope
from tensoralloy.test_utils import test_dir, assert_array_equal, Pd3O2
from tensoralloy.utils import Defaults

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@skip
def test_monoatomic_molecule():
    """
    Test computing descriptors of a single mono-atomic molecule.
    """
    atoms = read(join(test_dir(), 'B28.xyz'), index=0, format='xyz')
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc([False, False, False])
    coords = atoms.get_positions()
    rr = pairwise_distances(coords)
    rc = 6.0
    zr = get_radial_fingerprints_v1(coords, rr, rc, Defaults.eta)
    za = get_augular_fingerprints_v1(coords, rr, rc, Defaults.beta,
                                     Defaults.gamma, Defaults.zeta)
    g_init = np.hstack((zr, za))
    g_old, _, _ = legacy_symmetry_function(atoms, rc=rc)
    assert_array_equal(g_init, g_old)

    with tf.Graph().as_default():
        clf = UniversalTransformer(['B'], rcut=rc, acut=rc, angular=True,
                                   periodic=False)
        nn = SymmetryFunctionNN(['B'], eta=Defaults.eta, omega=Defaults.omega,
                                gamma=Defaults.gamma, zeta=Defaults.zeta, beta=Defaults.beta)
        nn.attach_transformer(clf)
        op = nn.get_symmetry_function_descriptors(
            clf.get_descriptors(
                clf.get_constant_features(atoms)),
            mode=tf_estimator.ModeKeys.PREDICT)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            g_new = sess.run(op)
            assert_array_equal(g_old, g_new['B'])


@skip
def test_monoatomic_molecule_with_omega():
    """
    Test computing descriptors (eta and omega) of a single mono-atomic molecule.
    """
    atoms = read(join(test_dir(), 'B28.xyz'), index=0, format='xyz')
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc([False, False, False])

    omega = np.array([0.0, 1.0, 2.0])

    coords = atoms.get_positions()
    rr = pairwise_distances(coords)
    rc = 6.0
    z = get_radial_fingerprints_v2(coords, rr, rc, Defaults.eta, omega)

    with precision_scope("high"):
        with tf.Graph().as_default():
            clf = UniversalTransformer(['B'], rcut=rc, angular=False,
                                       periodic=False)
            nn = SymmetryFunctionNN(['B'], eta=Defaults.eta, omega=omega)
            nn.attach_transformer(clf)
            op = nn.get_symmetry_function_descriptors(
                clf.get_descriptors(
                    clf.get_constant_features(atoms)),
                mode=tf_estimator.ModeKeys.PREDICT)
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                assert_array_equal(sess.run(op)['B'], z)


def test_single_structure():
    """
    Test computing descriptors of a single multi-elements periodic structure.
    """
    amp = np.load(join(test_dir(), 'amp_Pd3O2.npz'))['g']

    with precision_scope("high"):
        with tf.Graph().as_default():
            symbols = Pd3O2.get_chemical_symbols()
            elements = sorted(list(set(symbols)))
            rc = 6.5
            clf = UniversalTransformer(elements, rcut=rc, angular=True,
                                       periodic=True)
            nn = SymmetryFunctionNN(elements)
            nn.attach_transformer(clf)
            op = nn.get_symmetry_function_descriptors(
                clf.get_descriptors(
                    clf.get_constant_features(Pd3O2)),
                mode=tf_estimator.ModeKeys.PREDICT)
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                g_new = sess.run(op)
                assert_array_equal(g_new['O'][0], amp[3: 5, 0: 20])
                assert_array_equal(g_new['Pd'][0], amp[0: 3, 20: 40])


if __name__ == "__main__":
    nose.main()
