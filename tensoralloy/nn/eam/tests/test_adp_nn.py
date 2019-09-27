#!coding=utf-8
"""
This module defines unit tests of `AdpNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import unittest
import shutil

from tensorflow_estimator import estimator as tf_estimator
from os.path import join, exists
from os import makedirs
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lammpsrun import LAMMPS
from nose.tools import assert_equal, assert_almost_equal
from collections import Counter
from typing import List

from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.nn.eam.adp import AdpNN
from tensoralloy.transformer.adp import ADPTransformer, BatchADPTransformer
from tensoralloy.io.lammps import LAMMPS_COMMAND
from tensoralloy.io.db import snap
from tensoralloy.calculator import TensorAlloyCalculator
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_dynamic_partition():
    """
    Test the modified implementation of `_dynamic_partition`.
    """
    atoms = read(join(test_dir(), "crystals",
                      "Ni4Mo_mp-11507_conventional_standard.cif"))
    atoms.calc = SinglePointCalculator(
        atoms, **{'energy': 0.0, 'forces': np.zeros_like(atoms.positions)})

    rc = 6.5
    elements = ['Mo', 'Ni']

    with tf.Graph().as_default():

        nn = AdpNN(elements=elements)
        adp = ADPTransformer(rc, elements)
        adp.get_placeholder_features()

        with tf.name_scope("Symmetric"):

            op, max_occurs = nn._dynamic_partition(
                adp.get_descriptors(adp.get_placeholder_features()),
                mode=tf_estimator.ModeKeys.PREDICT,
                merge_symmetric=True)

            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                partitions = sess.run(
                    op, feed_dict=adp.get_feed_dict(atoms))

                assert_equal(len(partitions), 3)
                for key, (descriptor, mask) in partitions.items():
                    assert_equal(descriptor.shape[0], 4)
                    assert_equal(mask.shape[0], 1)

    with tf.Graph().as_default():
        nn = AdpNN(elements=elements)

        size = find_neighbor_size_of_atoms(atoms, rc)
        max_occurs = Counter(atoms.get_chemical_symbols())

        adp = BatchADPTransformer(rc=rc,
                                  max_occurs=max_occurs,
                                  nij_max=size.nij,
                                  nnl_max=size.nnl,
                                  batch_size=1)

        protobuf = tf.convert_to_tensor(adp.encode(atoms).SerializeToString())
        example = adp.decode_protobuf(protobuf)

        batch = dict()
        for key, tensor in example.items():
            batch[key] = tf.expand_dims(
                tensor, axis=0, name=tensor.op.name + '/batch')

        descriptors = adp.get_descriptors(batch)
        op, max_occurs = nn._dynamic_partition(descriptors,
                                               mode=tf_estimator.ModeKeys.TRAIN,
                                               merge_symmetric=False)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            partitions = sess.run(op)

            assert_equal(len(partitions), 4)
            for key, (descriptor, mask) in partitions.items():
                assert_equal(descriptor.shape[0], 4)
                assert_equal(mask.shape[0], 1)


class AdpTestCase(unittest.TestCase):

    def get_lammps_calculator(self, elements: List[str]):
        """
        Return a LAMMPS calculator.
        """
        pot_file = join(test_dir(), 'lammps', 'Mo.19sep21.adp')
        work_dir = join(self.work_dir, ''.join(elements))
        if not exists(work_dir):
            makedirs(work_dir)

        return LAMMPS(files=[pot_file],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=work_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="adp",
                      command=LAMMPS_COMMAND,
                      pair_coeff=[f"* * Mo.19sep21.adp {' '.join(elements)}"])

    def setUp(self):
        """
        The setup function.
        """
        self.work_dir = join(test_dir(), 'lammps', 'adp')
        if not exists(self.work_dir):
            makedirs(self.work_dir)
        self.calc = TensorAlloyCalculator(
            join(test_dir(), 'models', "Mo.adp.19sep21.pb"))

    def test_main(self):
        db = snap("Mo")
        indices = [1, 100, 200, 284]
        lmp = self.get_lammps_calculator(["Mo"])
        for idx, atoms_id in enumerate(indices):
            atoms = db.get_atoms(id=atoms_id)
            y_py = self.calc.get_potential_energy(atoms)
            y_lmp = lmp.get_potential_energy(atoms)
            assert_almost_equal(y_lmp, y_py, delta=1e-4)

    def tearDown(self):
        if exists(self.work_dir):
            shutil.rmtree(self.work_dir)


if __name__ == "__main__":
    nose.run()
