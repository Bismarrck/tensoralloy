# coding=utf-8
"""
This module defines unit tests of `Dataset`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from tensorflow_estimator import estimator as tf_estimator
from nose.tools import assert_equal, assert_dict_equal
from nose.tools import assert_less, assert_true
from os.path import join

from tensoralloy.transformer.behler import BatchSymmetryFunctionTransformer
from tensoralloy.transformer import IndexTransformer
from tensoralloy.dataset.dataset import Dataset
from tensoralloy.utils import AttributeDict, Defaults
from tensoralloy.test_utils import qm7m, test_dir
from tensoralloy.dtypes import set_precision, get_float_dtype
from tensoralloy.io.read import read_file
from tensoralloy.io.db import connect

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

    with set_precision("high"):

        ref = qm7m_compute()

        with tf.Graph().as_default():

            savedir = join(test_dir(), 'qm7m')


            database = connect(join(savedir, 'qm7m.db'))

            rc = Defaults.rc
            nij_max = database.get_nij_max(rc, allow_calculation=True)

            clf = BatchSymmetryFunctionTransformer(
                rc=rc,
                max_occurs=database.max_occurs,
                nij_max=nij_max,
                nijk_max=0,
                angular=False)

            dataset = Dataset(
                database, name='qm7m', transformer=clf, serial=True)

            assert_equal(len(dataset), 3)
            assert_dict_equal(dataset.max_occurs, {'C': 5, 'H': 8, 'O': 2})

            dataset.to_records(savedir, test_size=0.33)
            assert_true(dataset.load_tfrecords(savedir))

            # random_state: 611, test_size: 0.33 -> train: 1, 2, test: 0
            next_batch = dataset.next_batch(mode=tf_estimator.ModeKeys.EVAL,
                                            batch_size=1,
                                            num_epochs=1,
                                            shuffle=False)

            with tf.Session() as sess:
                res = sess.run(next_batch)
                eps = 1e-8

                assert_equal(len(res.keys()), 15)
                assert_less(
                    np.abs(res.positions[0] - ref.positions[0]).max(), eps)
                assert_less(np.abs(res.ilist[0] - ref.g2[0].ilist).max(), eps)
                assert_less(np.abs(res.shift[0] - ref.g2[0].shift).max(), eps)
                assert_less(np.abs(res.rv2g[0] - ref.g2[0].v2g_map).max(), eps)
                assert_equal(res.y_conf[0], 1.0)


def test_ethanol():
    """
    Test the ethanol MD dataset for energy and forces and k_max = 3.
    """
    with set_precision('medium'):
        with tf.Graph().as_default():

            savedir = join(test_dir(), 'ethanol')
            database = connect(join(savedir, 'ethanol.db'))

            rc = Defaults.rc
            nijk_max = database.get_nijk_max(rc, allow_calculation=True)
            nij_max = database.get_nij_max(rc)
            clf = BatchSymmetryFunctionTransformer(
                rc=rc, max_occurs=database.max_occurs, nij_max=nij_max,
                nijk_max=nijk_max, angular=True
            )

            dataset = Dataset(
                database, name='ethanol', transformer=clf, serial=False)

            assert_equal(len(dataset), 10)
            assert_true(dataset.has_forces)

            dataset.to_records(savedir, test_size=0.5)
            assert_true(dataset.load_tfrecords(savedir))

            # random_state: 611, test_size: 0.5 -> train: [0, 1, 8, 2, 3]
            next_batch = dataset.next_batch(mode=tf_estimator.ModeKeys.TRAIN,
                                            batch_size=5,
                                            num_epochs=1,
                                            shuffle=False)

            atoms = database.get_atoms(id=2)

            dtype = get_float_dtype()
            np_dtype = dtype.as_numpy_dtype

            clf = IndexTransformer(dataset.transformer.max_occurs,
                                   atoms.get_chemical_symbols())
            positions = clf.map_positions(atoms.positions).astype(np_dtype)
            energy = np_dtype(atoms.get_total_energy())
            forces = clf.map_forces(atoms.get_forces()).astype(np_dtype)

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
    with set_precision("high"):
        with tf.Graph().as_default():

            savedir = join(test_dir(), 'Ni')
            database = read_file(join(savedir, "Ni.extxyz"), verbose=False)

            rc = Defaults.rc
            nij_max = database.get_nij_max(rc, allow_calculation=True)

            clf = BatchSymmetryFunctionTransformer(
                rc=rc, max_occurs=database.max_occurs, nij_max=nij_max,
                nijk_max=0, angular=False
            )

            dataset = Dataset(database, 'Ni', transformer=clf, serial=True)

            assert_equal(len(dataset), 2)
            assert_true(dataset.has_stress)

            dataset.to_records(savedir, test_size=1)
            assert_true(dataset.load_tfrecords(savedir))

            next_batch = dataset.next_batch(mode=tf_estimator.ModeKeys.TRAIN,
                                            batch_size=1,
                                            num_epochs=1,
                                            shuffle=False)

            with tf.Session() as sess:

                result = sess.run(next_batch)
                eps = 1e-5

                # These are raw VASP output in '-eV' (-stress * volume).
                xx, yy, zz, xy, yz, xz = \
                    -0.35196, -0.24978, -0.24978, 0.13262, -0.00305, 0.13262,
                volume = result.volume
                stress = np.asarray([-xx, -yy, -zz, -yz, -xz, -xy]) / volume
                total_pressure = -(xx + yy + zz) / 3.0 / volume

                assert_less(np.abs(result.stress[0] - stress).max(), eps)
                assert_less(result.total_pressure[0] - total_pressure, eps)

                assert_equal(result.y_conf[0], 0.0)
                assert_equal(result.f_conf[0], 1.0)
                assert_equal(result.s_conf[0], 0.5)


if __name__ == "__main__":
    nose.run()
