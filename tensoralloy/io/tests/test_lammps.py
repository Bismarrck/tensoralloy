#!coding=utf-8
"""
Unit tests of Lammps I/O helper functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import nose
import unittest

from nose.tools import assert_almost_equal, assert_equal
from os.path import join, exists
from os import remove

from tensoralloy.io.lammps import read_eam_alloy_setfl, read_adp_setfl
from tensoralloy.io.lammps import read_tersoff_file, write_tersoff_file
from tensoralloy.io.lammps import read_old_meam_spline_file
from tensoralloy.io.lammps import read_meam_spline_file
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_read_eam_alloy():
    """
    Test the function `read_eam_alloy`.
    """
    filename = join(test_dir(), 'lammps', 'Zhou_AlCu.alloy.eam')
    setfl = read_eam_alloy_setfl(filename)

    assert_equal(setfl.elements, ['Al', 'Cu'])
    assert_equal(setfl.nr, 2000)
    assert_equal(setfl.nrho, 2000)
    assert_equal(setfl.lattice_types, ['fcc', 'fcc'])
    assert_equal(setfl.lattice_constants, [0.0, 0.0])
    assert_almost_equal(setfl.atomic_masses[0], 26.98, delta=1e-3)
    assert_almost_equal(setfl.embed['Al'].y[10], -1.8490865619220642e-01,
                        delta=1e-10)
    assert_almost_equal(setfl.phi['CuCu'].y[1] * setfl.phi['CuCu'].x[1],
                        3.8671050028993639e+00,
                        delta=1e-10)


def test_read_adp_setfl():
    """
    Test the function `read_adp_setfl`.
    """
    filename = join(test_dir(), 'lammps', 'AlCu.adp')
    adpfl = read_adp_setfl(filename)

    assert_equal(adpfl.elements, ['Al', 'Cu'])
    assert_equal(adpfl.nr, 10000)
    assert_equal(adpfl.nrho, 10000)

    assert_almost_equal(adpfl.drho, 2.2770502180000001e-03, delta=1e-8)
    assert_almost_equal(adpfl.dr, 6.2872099999999995e-04, delta=1e-8)

    assert_equal(np.all(adpfl.quadrupole['CuCu'].y == 0.0), True)
    assert_equal(np.all(adpfl.quadrupole['AlAl'].y == 0.0), True)
    assert_equal(np.all(adpfl.dipole['CuCu'].y == 0.0), True)
    assert_equal(np.all(adpfl.dipole['AlAl'].y == 0.0), True)

    w = adpfl.quadrupole['AlCu'].y
    assert_almost_equal(w[0], 2.6740386039473818e-01, delta=1e-8)
    assert_almost_equal(w[1], 2.6728751808066320e-01, delta=1e-8)
    assert_almost_equal(w[2], 2.6717121145664957e-01, delta=1e-8)
    assert_almost_equal(w[3], 2.6705494051816187e-01, delta=1e-8)
    assert_almost_equal(w[4], 2.6693870526066510e-01, delta=1e-8)
    assert_almost_equal(w[5], 2.6682250567962379e-01, delta=1e-8)
    assert_almost_equal(w[6], 2.6670634177050295e-01, delta=1e-8)


class TersoffFileTest(unittest.TestCase):

    def setUp(self):
        """
        The setup function.
        """
        self.input_file = join(test_dir(), 'lammps', 'SiC.tersoff')
        self.output_file = join(test_dir(), 'lammps', 'SiC.tersoff.out')

    def test_read_tersoff(self):
        """
        Test the function `read_tersoff_file`.
        """
        p = read_tersoff_file(self.input_file)

        assert_equal(p.elements, ['C', 'Si'])
        assert_almost_equal(p.params['SiSiSi']['n'], .78734)
        assert_almost_equal(p.params['SiCC']['lambda1'], 2.9839)

    def test_write_tersoff(self):
        """
        Test the function `write_tersoff_file`.
        """
        p = read_tersoff_file(self.input_file)
        write_tersoff_file(self.output_file, p)

    def tearDown(self):
        """
        The cleanup function.
        """
        if exists(self.output_file):
            remove(self.output_file)


def test_read_old_meam_spline():
    """
    Test the function `read_meam_spline_file` for old meam/spline file.
    """
    filename = join(test_dir(), 'lammps', 'Ti.meam.spline')
    p = read_old_meam_spline_file(filename, element='Ti')

    assert_almost_equal(p.rho['Ti'].bc_start, -1.0)
    assert_almost_equal(p.fs['Ti'].y[2], 2.011336597660079661e+00, delta=1e-8)


def test_read_new_meam_spline():
    """
    Test the function `read_meam_spline_file` for new meam/spline file.
    """
    filename = join(test_dir(), 'lammps', 'TiO.meam.spline')
    p = read_meam_spline_file(filename)

    assert_almost_equal(p.phi['TiTi'].bc_start, -20.0)
    assert_almost_equal(p.rho['Ti'].x[3], 2.7620612369, delta=1e-6)


if __name__ == "__main__":
    nose.run()
