#!coding=utf-8
"""
Unit tests of Agrawal/Be eam/alloy potential functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import os
import unittest

from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from os.path import join, exists
from shutil import rmtree
from unittest import skipUnless
from nose.tools import assert_almost_equal

from tensoralloy.utils import ModeKeys
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.test_utils import test_dir, assert_array_almost_equal
from tensoralloy.io.lammps import LAMMPS_COMMAND
from tensoralloy.precision import precision_scope

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@skipUnless(exists(LAMMPS_COMMAND), f"LAMMPS not found!")
class EamBe1Test(unittest.TestCase):

    def setUp(self):
        """
        The setup function.
        """
        self.tmp_dir = join(test_dir(absolute=True), 'lammps', 'Be_1')
        if not exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def get_lammps_calculator(self):
        """
        Return a LAMMPS calculator for Ag.
        """
        eam_file = join(test_dir(True), 'lammps', 'Be_Agrawal.eam.alloy')
        return LAMMPS(files=[eam_file],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=self.tmp_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="eam/alloy",
                      command=LAMMPS_COMMAND,
                      pair_coeff=['* * Be_Agrawal.eam.alloy Be'])

    def tearDown(self):
        """
        Delete the tmp dir.
        """
        if exists(self.tmp_dir):
            rmtree(self.tmp_dir, ignore_errors=True)

    def test_eam(self):
        """
        Test energy and forces calculations of `EamAlloyNN` with `Be/1`.
        """
        rc = 5.0
        lammps = self.get_lammps_calculator()

        atoms = bulk('Be') * [2, 2, 2]
        atoms.positions += np.random.randn(len(atoms), 3) * 0.1
        atoms.calc = lammps
        elements = ['Be']
        graph = tf.Graph()

        with precision_scope("high"):
            with graph.as_default():
                clf = UniversalTransformer(rcut=rc, elements=elements)
                nn = EamAlloyNN(elements=elements,
                                custom_potentials="Be/1",
                                export_properties=["energy", "forces"])
                nn.attach_transformer(clf)
                predictions = nn.build(
                    features=clf.get_placeholder_features(),
                    mode=ModeKeys.PREDICT,
                    verbose=True)

                with tf.Session() as sess:
                    tf.global_variables_initializer().run()
                    results = sess.run(predictions,
                                       feed_dict=clf.get_feed_dict(atoms))

        atoms.calc = lammps
        lmp_energy = lammps.get_potential_energy(atoms)
        lmp_forces = lammps.get_forces(atoms)
        assert_almost_equal(lmp_energy, results["energy"], delta=1e-2)
        assert_array_almost_equal(lmp_forces, results["forces"], delta=1e-2)


if __name__ == "__main__":
    nose.main()
