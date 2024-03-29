# coding=utf-8
"""
This module defines unit tests of `Dataset`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import unittest
import glob

from nose.tools import assert_equal, assert_dict_equal
from nose.tools import assert_less, assert_true
from os.path import join, exists
from os import remove

from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.transformer import VirtualAtomMap
from tensoralloy.train.dataset.dataset import Dataset
from tensoralloy.utils import Defaults, ModeKeys
from tensoralloy.test_utils import get_qm7m_test_dict, test_dir
from tensoralloy.precision import precision_scope, get_float_dtype
from tensoralloy.io.read import read_file
from tensoralloy.io.db import connect

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def qm7m_compute():
    """
    Compute the reference values.
    """
    qm7m = get_qm7m_test_dict()
    batch_size = len(qm7m["trajectory"])
    sf = BatchUniversalTransformer(rcut=Defaults.rc,
                                   max_occurs=qm7m["max_occurs"],
                                   nij_max=qm7m["nij_max"],
                                   nijk_max=0,
                                   angular=False)
    max_n_atoms = sum(qm7m["max_occurs"].values()) + 1
    g2 = []
    positions = np.zeros((batch_size, max_n_atoms, 3))
    for i, atoms in enumerate(qm7m["trajectory"]):
        vap = sf.get_vap_transformer(atoms)
        positions[i] = vap.map_positions(atoms.positions)
        g2.append(sf.get_metadata(atoms, vap)[0])

    return dict(positions=positions, g2=g2)


class DatasetTest(unittest.TestCase):

    def __init__(self, name, fmt, *args, **kwargs):
        """
        Initialization method.
        """
        super(DatasetTest, self).__init__(*args, **kwargs)
        self._name = name
        self._format = fmt

    def setUp(self):
        """
        The setup function.
        """
        self.save_dir = join(test_dir(), 'datasets', self._name)
        self.db_file = join(self.save_dir, f"{self._name}.db")
        if self._format == "db":
            self.database = connect(self.db_file)
        else:
            self.database = read_file(
                join(self.save_dir, f"{self._name}.{self._format}"),
                verbose=False)

    def tearDown(self):
        """
        The cleanup function.
        """
        if exists(self.db_file) and self._format != "db":
            remove(self.db_file)
        for tfrecord_file in glob.glob(f"{self.save_dir}/*.tfrecords"):
            remove(tfrecord_file)


class Qm7mTest(DatasetTest):

    def __init__(self, *args, **kwargs):
        super(Qm7mTest, self).__init__("qm7m", "db", *args, **kwargs)

    def test_qm7m(self):
        """
        Test the qm7m dataset for energy only and k_max = 2.
        """
        with precision_scope("high"):

            ref = qm7m_compute()

            with tf.Graph().as_default():

                database = self.database
                savedir = self.save_dir

                rc = Defaults.rc
                nij_max = database.get_nij_max(rc, allow_calculation=True)

                clf = BatchUniversalTransformer(
                    rcut=rc,
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
                next_batch = dataset.next_batch(mode=ModeKeys.EVAL,
                                                batch_size=1,
                                                num_epochs=1,
                                                shuffle=False)

                with tf.Session() as sess:
                    res = sess.run(next_batch)
                    eps = 1e-8

                    assert_equal(len(res.keys()), 15)
                    assert_less(
                        np.abs(res["positions"][0] - ref["positions"][0]).max(),
                        eps)
                    assert_less(
                        np.abs(res["g2.ilist"][0] - ref["g2"][0].ilist).max(),
                        eps)
                    assert_less(
                        np.abs(res["g2.n1"][0] - ref["g2"][0].n1).max(),
                        eps)
                    assert_less(
                        np.abs(
                            res["g2.v2g_map"][0] - ref["g2"][0].v2g_map).max(),
                        eps)


class EthanolTest(DatasetTest):
    """
    Test the ethanol MD dataset for energy and forces and k_max = 3.
    """

    def __init__(self, *args, **kwargs):
        super(EthanolTest, self).__init__("ethanol", "db", *args, **kwargs)

    def test_ethanol(self):
        with precision_scope('medium'):
            with tf.Graph().as_default():
                savedir = self.save_dir
                database = self.database

                rc = Defaults.rc
                nijk_max = database.get_nijk_max(rc, allow_calculation=True)
                nij_max = database.get_nij_max(rc)
                clf = BatchUniversalTransformer(
                    rcut=rc, max_occurs=database.max_occurs, nij_max=nij_max,
                    nijk_max=nijk_max, angular=True
                )

                dataset = Dataset(
                    database, name='ethanol', transformer=clf, serial=True)

                assert_equal(len(dataset), 10)
                assert_true(dataset.has_forces)

                dataset.to_records(savedir, test_size=0.5)
                assert_true(dataset.load_tfrecords(savedir))

                # random_state: 611, test_size: 0.5 -> train: [0, 1, 8, 2, 3]
                next_batch = dataset.next_batch(
                    mode=ModeKeys.TRAIN,
                    batch_size=5,
                    num_epochs=1,
                    shuffle=False)

                atoms = database.get_atoms(id=2)

                dtype = get_float_dtype()
                np_dtype = dtype.as_numpy_dtype

                clf = VirtualAtomMap(dataset.transformer.max_occurs,
                                     atoms.get_chemical_symbols())
                positions = clf.map_positions(atoms.positions).astype(np_dtype)
                energy = np_dtype(atoms.get_total_energy())
                forces = clf.map_forces(atoms.get_forces()).astype(np_dtype)

                with tf.Session() as sess:
                    result = sess.run(next_batch)
                    eps = 1e-8
                    assert_less(
                        np.abs(result["positions"][1] - positions).max(), eps)
                    assert_less(np.abs(result["forces"][1] - forces).max(), eps)
                    assert_less(float(result["energy"][1] - energy), eps)


class NickelTest(DatasetTest):

    def __init__(self, *args, **kwargs):
        super(NickelTest, self).__init__("Ni", "extxyz", *args, **kwargs)

    def test_nickel(self):
        """
        Test the nickel dataset for stress.
        """
        with precision_scope("high"):
            with tf.Graph().as_default():
                savedir = self.save_dir
                database = self.database

                rc = Defaults.rc
                nij_max = database.get_nij_max(rc, allow_calculation=True)

                clf = BatchUniversalTransformer(
                    rcut=rc, max_occurs=database.max_occurs, nij_max=nij_max,
                    nijk_max=0, angular=False
                )

                dataset = Dataset(database, 'Ni', transformer=clf, serial=True)

                assert_equal(len(dataset), 2)
                assert_true(dataset.has_stress)

                dataset.to_records(savedir, test_size=1)
                assert_true(dataset.load_tfrecords(savedir))

                next_batch = dataset.next_batch(
                    mode=ModeKeys.TRAIN,
                    batch_size=1,
                    num_epochs=1,
                    shuffle=False)

                with tf.Session() as sess:

                    result = sess.run(next_batch)
                    eps = 1e-5

                    # These are raw VASP output in '-eV' (-stress * volume).
                    xx, yy, zz, xy, yz, xz = \
                        -0.35196, -0.24978, -0.24978, \
                        0.13262, -0.00305, 0.13262,
                    volume = result["volume"]
                    stress = np.asarray([-xx, -yy, -zz, -yz, -xz, -xy]) / volume
                    total_pressure = -(xx + yy + zz) / 3.0 / volume
                    assert_less(
                        np.abs(result["stress"][0] - stress).max(), eps)
                    assert_less(
                        result["total_pressure"][0] - total_pressure, eps)


if __name__ == "__main__":
    nose.run()
