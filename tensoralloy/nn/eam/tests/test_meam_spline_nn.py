#!coding=utf-8
"""
Test the implementation of MEAM/Spline.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import unittest
import nose
import os
import shutil

from nose.tools import assert_almost_equal
from enum import Enum
from os.path import dirname, join, exists
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS

from tensoralloy.utils import ModeKeys
from tensoralloy.test_utils import test_dir, assert_array_almost_equal
from tensoralloy.io.lammps import LAMMPS_COMMAND
from tensoralloy.nn.eam.meam import MeamNN
from tensoralloy.transformer.universal import UniversalTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TestSystem(Enum):
    Ti = 0
    TiO = 1


class MeamSplineTest(unittest.TestCase):

    def setUp(self):
        """
        The setup function.
        """
        work_dir = join(test_dir(), 'tersoff')
        if not exists(work_dir):
            os.makedirs(work_dir)
        self.work_dir = work_dir

    def tearDown(self):
        if exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def get_lammps_calculator(self, system: TestSystem):
        """
        Return a LAMMPS calculator.
        """
        if system == TestSystem.Ti:
            pot_file = 'Ti.meam.spline'
            pair_coeff = ['* * Ti.meam.spline Ti']
            specorder = ['Ti']
        else:
            pot_file = 'TiO.meam.spline'
            pair_coeff = ['* * TiO.meam.spline Ti O']
            specorder = ['Ti', 'O']
        return LAMMPS(files=[join(test_dir(True), 'lammps', pot_file)],
                      binary_dump=False,
                      parameters={"specorder": specorder},
                      write_velocities=False,
                      tmp_dir=self.work_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="meam/spline",
                      command=LAMMPS_COMMAND,
                      pair_coeff=pair_coeff)

    @unittest.skip
    def test_elementary_meam_spline(self):
        """
        Test the meam/spline implementation with the Ti system.
        """
        atoms = bulk('Ti')
        delta = np.random.uniform(-0.05, 0.05, size=(len(atoms), 3))
        atoms.set_positions(atoms.positions + delta)
        elements = ['Ti']
        spline = f"lspline@{join(dirname(__file__), 'Ti.meam.spline.json')}"

        calc = self.get_lammps_calculator(TestSystem.Ti)
        calc.calculate(atoms, properties=['energy', 'forces'])
        pe = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)

        nn = MeamNN(elements,
                    custom_potentials={"Ti": {"rho": spline,
                                              "fs": spline,
                                              "embed": spline},
                                       "TiTi": {"phi": spline, "gs": spline}})
        clf = UniversalTransformer(elements, rcut=5.50, acut=4.41, angular=True,
                                   symmetric=False)
        nn.attach_transformer(clf)

        with tf.Graph().as_default():
            predictions = nn.build(
                features=clf.get_constant_features(atoms),
                mode=ModeKeys.PREDICT,
                verbose=True)
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                results = sess.run(predictions)
                assert_almost_equal(results['energy'], pe,
                                    delta=1e-5)
                assert_array_almost_equal(results['forces'], forces, delta=1e-4)


if __name__ == "__main__":
    nose.main()
