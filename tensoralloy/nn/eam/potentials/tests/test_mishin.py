#!coding=utf-8
"""
Unit tests of the `AdpNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose
import unittest
import shutil
import os

from tensorflow_estimator import estimator as tf_estimator
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from ase import Atoms
from os.path import join, exists
from os import makedirs
from typing import List
from nose.tools import assert_almost_equal

from tensoralloy.test_utils import test_dir, assert_array_almost_equal
from tensoralloy.transformer.adp import ADPTransformer
from tensoralloy.nn.eam.adp import AdpNN
from tensoralloy.precision import set_precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AlCuAdpTest(unittest.TestCase):

    def setUp(self):
        """
        The setup function.
        """
        self.work_dir = join(test_dir(), 'lammps', 'adp')
        if not exists(self.work_dir):
            makedirs(self.work_dir)

    def tearDown(self) -> None:
        """
        The cleanup function.
        """
        if exists(self.work_dir):
            shutil.rmtree(self.work_dir)

    def get_lammps_calculator(self, elements: List[str], use_official=False):
        """
        Return a LAMMPS calculator.
        """
        if use_official:
            pot_file = join(test_dir(), 'lammps', 'AlCu.adp')
        else:
            pot_file = join(self.work_dir, 'AlCu.adp')
        parameters = {'pair_style': 'adp',
                      'pair_coeff': [f"* * AlCu.adp {' '.join(elements)}"]}
        work_dir = join(self.work_dir, ''.join(elements))
        if not exists(work_dir):
            makedirs(work_dir)

        if 'LAMMPS_COMMAND' not in os.environ:
            LAMMPS_COMMAND = '/usr/local/bin/lmp_serial'
            os.environ['LAMMPS_COMMAND'] = LAMMPS_COMMAND

        return LAMMPS(files=[pot_file], parameters=parameters,
                      tmp_dir=work_dir, keep_tmp_files=True,
                      keep_alive=False, no_data_file=False)

    def _run(self, atoms: Atoms, rc=7.5):
        """
        Run a test.
        """
        with set_precision("high"):
            with tf.Graph().as_default():
                elements = sorted(list(set(atoms.get_chemical_symbols())))
                clf = ADPTransformer(rc, elements)
                nn = AdpNN(elements,
                           custom_potentials="adp/AlCu",
                           export_properties=['energy', 'forces', 'stress'])
                nn.attach_transformer(clf)
                nn.export_to_setfl(
                    join(self.work_dir, 'AlCu.adp'),
                    nr=10000, dr=rc / 1e4, nrho=10000, drho=0.005)

                predictions = nn.build(
                    features=clf.get_constant_features(atoms),
                    mode=tf_estimator.ModeKeys.PREDICT,
                    verbose=True)

                with tf.Session() as sess:
                    tf.global_variables_initializer().run()
                    results = sess.run(predictions)

        mishin = self.get_lammps_calculator(elements, use_official=True)
        mishin.calculate(atoms)
        export = self.get_lammps_calculator(elements, use_official=False)
        export.calculate(atoms)

        assert_almost_equal(mishin.get_potential_energy(atoms),
                            export.get_potential_energy(atoms), delta=1e-5)
        assert_array_almost_equal(mishin.get_stress(atoms),
                                  export.get_stress(atoms), delta=1e-5)
        assert_almost_equal(results.energy,
                            mishin.get_potential_energy(atoms),
                            delta=1e-5)
        assert_array_almost_equal(results.stress,
                                  mishin.get_stress(atoms),
                                  delta=1e-5)


    def test_alloy(self):
        """
        Test Al-Cu alloy ADP calculation.
        """
        atoms = bulk('Al', crystalstructure='fcc', cubic=True)
        atoms.set_chemical_symbols(['Al', 'Cu', 'Al', 'Al'])
        self._run(atoms, rc=7.5)

    def test_bulk(self):
        """
        Test Al bulk ADP calculation.
        """
        atoms = bulk('Al', crystalstructure='fcc', cubic=True)
        self._run(atoms, rc=6.5)


if __name__ == "__main__":
    nose.run()
