#!coding=utf-8
"""
Test the implementation of MEAM/Spline.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import unittest
import nose
import os
import shutil

from enum import Enum
from os.path import dirname, join, exists
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.test_utils import test_dir
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

    def test_elementary_meam_spline(self):
        """
        Test the meam/spline implementation with the Ti system.
        """
        atoms = bulk('Ti', cubic=True)
        elements = ['Ti']
        spline = f"spline@{join(dirname(__file__), 'Ti.meam.spline.json')}"

        calc = self.get_lammps_calculator(TestSystem.Ti)
        pe = calc.get_potential_energy(atoms)

        nn = MeamNN(elements,
                    custom_potentials={"Ti": {"rho": spline,
                                              "fs": spline,
                                              "embed": spline},
                                       "TiTi": {"phi": spline, "gs": spline}})
        clf = UniversalTransformer(elements, rcut=5.5, acut=4.1, angular=True,
                                   symmetric=False)
        nn.attach_transformer(clf)

        with tf.Graph().as_default():
            predictions = nn.build(
                features=clf.get_constant_features(atoms),
                mode=tf_estimator.ModeKeys.PREDICT,
                verbose=True)
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                print(sess.run(predictions['energy']))
                print(pe)


if __name__ == "__main__":
    nose.main()
