#!coding=utf-8
"""
Test the Tersoff potential.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import unittest
import enum
import os
import shutil

from tensorflow_estimator import estimator as tf_estimator
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read
from os.path import join, exists
from nose.tools import assert_almost_equal

from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.nn.tersoff import Tersoff
from tensoralloy.precision import precision_scope
from tensoralloy.test_utils import test_dir, data_dir, assert_array_almost_equal
from tensoralloy.io.lammps import LAMMPS_COMMAND

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TestSystem(enum.Enum):
    """
    Define the test systems.
    """
    Si = 0
    SiC = 1


class TersoffTest(unittest.TestCase):

    def setUp(self):
        """
        The setup function.
        """
        work_dir = join(test_dir(), 'tersoff')
        if not exists(work_dir):
            os.makedirs(work_dir)
        self.work_dir = work_dir

    def get_lammps_calculator(self, system: TestSystem):
        """
        Return a LAMMPS calculator.
        """
        if system == TestSystem.Si:
            pot_file = 'Si.tersoff'
            pair_coeff = ['* * Si.tersoff Si']
        else:
            pot_file = 'SiC.tersoff'
            pair_coeff = ['* * Si.tersoff Si C Si']
        return LAMMPS(files=[join(test_dir(True), 'lammps', pot_file)],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=self.work_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="tersoff",
                      command=LAMMPS_COMMAND,
                      pair_coeff=pair_coeff)

    def test_silicon(self):
        """
        Test the Tersoff potential with Si.
        """
        atoms = bulk('Si', cubic=True)
        size = len(atoms)
        delta = np.random.uniform(-0.05, 0.05, size=(size, 3))
        atoms.set_positions(atoms.positions + delta)
        calc = self.get_lammps_calculator(system=TestSystem.Si)
        pe = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)

        with tf.Graph().as_default():
            with precision_scope("high"):
                elements = sorted(list(set(atoms.get_chemical_symbols())))
                nn = Tersoff(elements)
                clf = UniversalTransformer(
                    elements, rcut=3.2, acut=3.2, angular=True, symmetric=False)
                nn.attach_transformer(clf)
                predictions = nn.build(
                    features=clf.get_constant_features(atoms),
                    mode=tf_estimator.ModeKeys.PREDICT)
                with tf.Session() as sess:
                    tf.global_variables_initializer().run()
                    results = sess.run(predictions)
                    assert_almost_equal(results['energy'], pe)
                    assert_array_almost_equal(
                        results['forces'], forces, delta=1e-5)

    @unittest.skip
    def test_sic(self):
        atoms = read(join(data_dir(), "crystals", "SiC_mp-8062_primitive.cif"))
        calc = self.get_lammps_calculator(system=TestSystem.SiC)
        pe = calc.get_potential_energy(atoms)

        with tf.Graph().as_default():
            with precision_scope("high"):
                elements = list(set(atoms.get_chemical_symbols()))
                nn = Tersoff(elements)
                clf = UniversalTransformer(
                    elements, rcut=3.0, acut=3.0, angular=True, symmetric=False)
                nn.attach_transformer(clf)
                predictions = nn.build(
                    features=clf.get_constant_features(atoms),
                    mode=tf_estimator.ModeKeys.PREDICT)
                with tf.Session() as sess:
                    tf.global_variables_initializer().run()
                    print(sess.run(predictions))
                    print(pe)

    def tearDown(self):
        """
        The cleanup function.
        """
        if exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)


if __name__ == "__main__":
    nose.main()
