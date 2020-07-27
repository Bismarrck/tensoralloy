# coding=utf-8
"""
This module defines unit tests of `EamFsNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import os
import shutil
import unittest

from tensorflow_estimator import estimator as tf_estimator
from unittest import skipUnless
from nose.tools import assert_list_equal, with_setup
from nose.tools import assert_almost_equal
from os.path import exists, join
from os import remove
from ase.calculators.lammpsrun import LAMMPS
from ase.build import bulk

from tensoralloy.nn.eam import EamFsNN
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.test_utils import assert_array_almost_equal, test_dir
from tensoralloy.io.lammps import LAMMPS_COMMAND
from tensoralloy.precision import precision_scope


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def export_setfl_teardown():
    """
    Remove the generated setfl file.
    """
    for obj in ('AlFe.fs.eam',
                'Al.embed.png', 'Fe.embed.png',
                'AlAl.rho.png', 'AlFe.rho.png', 'FeFe.rho.png', 'FeAl.rho.png',
                'AlFe.phi.png', 'AlAl.phi.png', 'FeFe.phi.png'):
        if exists(obj):
            remove(obj)


@with_setup(teardown=export_setfl_teardown)
def test_export_setfl():
    """
    Test exporting eam/alloy model of AlCuZJW04 to a setfl file.
    """
    nn = EamFsNN(
        elements=['Al', 'Fe'],
        custom_potentials="msah11")

    nrho = 10000
    drho = 3.00000000000000E-2
    nr = 10000
    dr = 6.50000000000000E-4

    nn.export_to_setfl('AlFe.fs.eam', nr=nr, dr=dr, nrho=nrho, drho=drho,
                       lattice_constants={'Al': 4.04527, 'Fe': 2.855312},
                       lattice_types={'Al': 'fcc', 'Fe': 'bcc'})

    with open('AlFe.fs.eam') as fp:
        out = []
        out_key_lines = []
        for i, line in enumerate(fp):
            if i < 5:
                continue
            elif i == 5 or i == 30006:
                out_key_lines.append(line)
                continue
            out.append(float(line.strip()))
    with open(join(test_dir(), 'lammps', 'Mendelev_Al_Fe.fs.eam')) as fp:
        ref = []
        ref_key_lines = []
        for i, line in enumerate(fp):
            if i < 5:
                continue
            elif i == 5 or i == 30006:
                ref_key_lines.append(line)
                continue
            ref.append(float(line.strip()))
    assert_array_almost_equal(np.asarray(out), np.asarray(ref), delta=1e-8)
    assert_list_equal(out_key_lines, ref_key_lines)


class EamFsTest(unittest.TestCase):

    def setUp(self):
        """
        The setup function.
        """
        self.work_dir = join(test_dir(), 'lammps', 'msah11')
        if not exists(self.work_dir):
            os.mkdir(self.work_dir)

    def get_lammps_calculator(self):
        """
        Return a LAMMPS calculator for Ag.
        """
        eam_file = join(test_dir(), 'lammps', 'Mendelev_Al_Fe.fs.eam')
        return LAMMPS(files=[eam_file],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=self.work_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="eam/fs",
                      command=LAMMPS_COMMAND,
                      pair_coeff=['* * Mendelev_Al_Fe.fs.eam Al Fe'])

    def tearDown(self):
        """
        Delete the tmp dir.
        """
        if exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    @skipUnless(exists(LAMMPS_COMMAND), f"LAMMPS_COMMAND not found!")
    def test_eam_fs_msah11(self):
        """
        Test the total energy calculation of `EamFsNN` with `Msah11`.
        """
        rc = 6.5

        atoms = bulk('Fe') * [2, 2, 2]
        symbols = atoms.get_chemical_symbols()
        symbols[0: 2] = ['Al', 'Al']
        atoms.set_chemical_symbols(symbols)
        elements = sorted(set(symbols))

        with precision_scope("high"):

            with tf.Graph().as_default():
                clf = UniversalTransformer(rcut=rc, elements=elements)
                nn = EamFsNN(elements=elements,
                             export_properties=['energy', 'forces', 'stress'],
                             custom_potentials={
                                 "Al": {"embed": "msah11"},
                                 "Fe": {"embed": "msah11"},
                                 "AlAl": {"phi": "msah11", "rho": "msah11"},
                                 "AlFe": {"phi": "msah11", "rho": "msah11"},
                                 "FeFe": {"phi": "msah11", "rho": "msah11"},
                                 "FeAl": {"phi": "msah11", "rho": "msah11"}})
                nn.attach_transformer(clf)
                predictions = nn.build(
                    features=clf.get_placeholder_features(),
                    mode=tf_estimator.ModeKeys.PREDICT,
                    verbose=True)

                with tf.Session() as sess:
                    tf.global_variables_initializer().run()
                    result = sess.run(predictions,
                                      feed_dict=clf.get_feed_dict(atoms))

        lammps = self.get_lammps_calculator()
        atoms.calc = lammps
        lammps.calculate(atoms)

        assert_almost_equal(result['energy'],
                            lammps.get_potential_energy(atoms), delta=1e-6)
        assert_array_almost_equal(result['forces'],
                                  lammps.get_forces(atoms), delta=1e-9)


if __name__ == "__main__":
    nose.main()
