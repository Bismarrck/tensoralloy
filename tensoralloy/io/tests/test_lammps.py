#!coding=utf-8
"""
Unit tests of Lammps I/O helper functions.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_almost_equal, assert_equal
from os.path import join

from tensoralloy.io.lammps import read_eam_alloy
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_read_eam_alloy():
    """
    Test the function `read_eam_alloy`.
    """
    filename = join(test_dir(), 'lammps', 'Zhou_AlCu.alloy.eam')
    setfl = read_eam_alloy(filename)

    assert_equal(setfl.elements, ['Al', 'Cu'])
    assert_equal(setfl.nr, 2000)
    assert_equal(setfl.nrho, 2000)
    assert_equal(setfl.lattice_types, ['fcc', 'fcc'])
    assert_equal(setfl.lattice_constants, [0.0, 0.0])
    assert_almost_equal(setfl.atomic_masses[0], 26.98, delta=1e-3)
    assert_almost_equal(setfl.embed['Al'][1, 10], -1.8490865619220642e-01,
                        delta=1e-10)
    assert_almost_equal(setfl.phi['CuCu'][1, 1] * setfl.phi['CuCu'][0, 1],
                        3.8671050028993639e+00,
                        delta=1e-10)


if __name__ == "__main__":
    nose.run()
