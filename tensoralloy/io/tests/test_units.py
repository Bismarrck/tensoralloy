# coding=utf-8
"""
This module defines tests for unit conversions.
"""
from __future__ import print_function, absolute_import

import nose

from ase.units import kcal, mol, Hartree, eV, GPa
from nose.tools import assert_almost_equal

from tensoralloy.io.read import read_file
from tensoralloy.io.units import get_conversion_units

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_unit_conversion():
    """
    Test the unit conversion function.
    """
    to_eV, _, to_eV_Ang3 = get_conversion_units({
        'energy': 'kcal/mol*Hartree/eV',
        'stress': '0.1*GPa',
    })
    assert_almost_equal(to_eV, (kcal / mol * Hartree / eV) / eV)
    assert_almost_equal(to_eV_Ang3, 0.1 * GPa)

    _, _, to_eV_Ang3 = get_conversion_units({
        'stress': 'kbar',
    })
    assert_almost_equal(to_eV_Ang3, 0.1 * GPa)

    _, _, to_eV_Ang3 = get_conversion_units({
        'stress': 'eV/Angstrom**3',
    })
    assert_almost_equal(to_eV_Ang3, 1.0)

    xyzfile = 'test_files/examples.extxyz'
    database = read_file(xyzfile, verbose=False,
                         units={'energy': 'kcal/mol'})
    atoms = database.get_atoms(id=2)
    thres = 1e-6
    unit = kcal / mol / eV
    assert_almost_equal(atoms.get_total_energy(), -17637.613286 * unit,
                        delta=thres)


if __name__ == "__main__":
    nose.run()
