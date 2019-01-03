# coding=utf-8
"""
This module defines unit tests of `Dataset`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from ase.db import connect
from nose.tools import assert_equal, assert_dict_equal
from nose.tools import assert_less, assert_true
from os.path import join

from tensoralloy.transformer.behler import BatchSymmetryFunctionTransformer
from tensoralloy.transformer import IndexTransformer
from tensoralloy.dataset.dataset import Dataset
from tensoralloy.misc import test_dir, Defaults, AttributeDict
from tensoralloy.test_utils import qm7m

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def qm7m_compute():
    """
    Compute the reference values.
    """
    batch_size = len(qm7m.trajectory)
    sf = BatchSymmetryFunctionTransformer(rc=Defaults.rc,
                                          max_occurs=qm7m.max_occurs,
                                          nij_max=qm7m.nij_max, nijk_max=0,
                                          angular=False)
    max_n_atoms = sum(qm7m.max_occurs.values()) + 1
    g2 = []
    positions = np.zeros((batch_size, max_n_atoms, 3))
    for i, atoms in enumerate(qm7m.trajectory):
        positions[i] = sf.get_index_transformer(atoms).map_positions(
            atoms.positions)
        g2.append(sf.get_g2_indexed_slices(atoms))

    return AttributeDict(positions=positions, g2=g2)


def test_qm7m():
    """
    Test the qm7m dataset for energy only and k_max = 2.
    """
    ref = qm7m_compute()

    with tf.Graph().as_default():

        savedir = join(test_dir(), 'qm7m')
        database = connect(join(savedir, 'qm7m.db'))
        dataset = Dataset(database, 'qm7m', angular=False, serial=True)

        assert_equal(len(dataset), 3)
        assert_dict_equal(dataset.max_occurs, {'C': 5, 'H': 8, 'O': 2})

        dataset.to_records(savedir, test_size=0.33)
        assert_true(dataset.load_tfrecords(savedir))

        # random_state: 611, test_size: 0.33 -> train: 1, 2, test: 0
        next_batch = dataset.next_batch(mode=tf.estimator.ModeKeys.EVAL,
                                        batch_size=1,
                                        num_epochs=1,
                                        shuffle=False)

        with tf.Session() as sess:

            res = sess.run(next_batch)
            eps = 1e-8

            assert_equal(len(res.keys()), 12)
            assert_less(np.abs(res.positions[0] - ref.positions[0]).max(), eps)
            assert_less(np.abs(res.ilist[0] - ref.g2[0].ilist).max(), eps)
            assert_less(np.abs(res.shift[0] - ref.g2[0].shift).max(), eps)
            assert_less(np.abs(res.rv2g[0] - ref.g2[0].v2g_map).max(), eps)


def test_ethanol():
    """
    Test the ethanol MD dataset for energy and forces and k_max = 3.
    """
    with tf.Graph().as_default():

        savedir = join(test_dir(), 'ethanol')
        database = connect(join(savedir, 'ethanol.db'))
        dataset = Dataset(database, 'ethanol', descriptor='behler',
                          serial=False, angular=True)

        assert_equal(len(dataset), 10)
        assert_true(dataset.use_forces)

        dataset.to_records(savedir, test_size=0.5)
        assert_true(dataset.load_tfrecords(savedir))

        # random_state: 611, test_size: 0.5 -> train: [0, 1, 8, 2, 3]
        next_batch = dataset.next_batch(mode=tf.estimator.ModeKeys.TRAIN,
                                        batch_size=5,
                                        num_epochs=1,
                                        shuffle=False)

        atoms = database.get_atoms(id=2)

        clf = IndexTransformer(dataset.transformer.max_occurs,
                               atoms.get_chemical_symbols())
        positions = clf.map_positions(atoms.positions)
        energy = atoms.get_total_energy()
        forces = clf.map_forces(atoms.get_forces())

        with tf.Session() as sess:

            result = sess.run(next_batch)
            eps = 1e-8

            assert_less(np.abs(result.positions[1] - positions).max(), eps)
            assert_less(np.abs(result.f_true[1] - forces).max(), eps)
            assert_less(float(result.y_true[1] - energy), eps)


def test_nickel():
    """
    Test the nickel dataset for stress.
    """
    with tf.Graph().as_default():

        savedir = join(test_dir(), 'Ni')
        database = connect(join(savedir, 'Ni.db'))
        dataset = Dataset(database, 'Ni', angular=False, serial=True)

        assert_equal(len(dataset), 2)
        assert_true(dataset.use_stress)

        dataset.to_records(savedir, test_size=1)
        assert_true(dataset.load_tfrecords(savedir))

        next_batch = dataset.next_batch(mode=tf.estimator.ModeKeys.TRAIN,
                                        batch_size=1,
                                        num_epochs=1,
                                        shuffle=False)

        with tf.Session() as sess:

            result = sess.run(next_batch)
            eps = 1e-5

            # These are raw VASP output. The unit is '-eV', `-stress * volume`.
            xx, yy, zz, xy, yz, xz = \
                -0.35196, -0.24978, -0.24978, 0.13262, -0.00305, 0.13262,
            stress = [-xx, -yy, -zz, -yz, -xz, -xy]
            total_pressure = -(xx + yy + zz) / 3.0

            assert_less(np.abs(result.stress[0] - stress).max(), eps)
            assert_less(result.total_pressure[0] - total_pressure, eps)


if __name__ == "__main__":
    nose.run()
