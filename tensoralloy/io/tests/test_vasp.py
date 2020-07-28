#!coding=utf-8
"""
Test customized VASP I/O functions.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_almost_equal
from os.path import join
from ase.units import kB, eV

from tensoralloy.test_utils import test_dir
from tensoralloy.io.vasp import read_vasp_xml
from tensoralloy import atoms_utils

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_read_vasp_xml():
    vasprun = join(test_dir(), "Be_hcp_4000K_vasprun.xml")
    atoms = next(read_vasp_xml(vasprun, index=0))
    etemp = atoms_utils.get_electron_temperature(atoms)
    eentropy = atoms_utils.get_electron_entropy(atoms)
    assert_almost_equal(etemp, 4000.0 * kB / eV, delta=1e-6)
    assert_almost_equal(eentropy, 0.2210591, delta=1e-6)


if __name__ == "__main__":
    nose.main()
