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
from tensoralloy.atoms_utils import get_kinetic_energy

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_read_vasp_xml():
    vasprun = join(test_dir(), "Be_hcp_4000K_vasprun.xml")
    atoms = next(read_vasp_xml(vasprun, index=0))
    etemp = atoms_utils.get_electron_temperature(atoms)
    eentropy = atoms_utils.get_electron_entropy(atoms)
    assert_almost_equal(etemp, 4000.0 * kB / eV, delta=1e-6)
    assert_almost_equal(eentropy, 0.2210591, delta=1e-6)


def test_read_vasp_md_xml():
    vasprun = join(test_dir(), "Be_md_vasprun.xml")
    trajectory = [atoms for atoms in read_vasp_xml(vasprun, index=slice(0, 10), 
                                                   finite_temperature=True)]
    assert len(trajectory) == 10
    assert_almost_equal(get_kinetic_energy(trajectory[4]), 
                        48.64234933, delta=1e-8)


if __name__ == "__main__":
    test_read_vasp_md_xml()
