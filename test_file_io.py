# coding=utf-8
"""
Unit tests for module `data_ops`.
"""
from __future__ import print_function, absolute_import

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

from nose import main
from nose.tools import assert_almost_equal, assert_equal, assert_dict_equal
from file_io import read, find_neighbor_size_limits, get_conversion
from ase.units import kcal, mol, eV, Hartree, GPa, kB


def test_read_xyz():
    xyzfile = 'test_files/B28.xyz'
    database = read(xyzfile, verbose=False, num_examples=2)
    atoms = database.get_atoms('id=2')
    assert_equal(len(database), 2)
    assert_almost_equal(atoms.positions[1, 1], 10.65007390)


def test_read_extxyz():
    xyzfile = 'test_files/examples.extxyz'
    database = read(xyzfile, verbose=False)
    atoms = database.get_atoms('id=2')
    thres = 1e-6
    metadata = database.metadata
    max_occurs = {'C': 10, 'H': 8, 'O': 4}
    assert_equal(len(database), 2)
    assert_equal(len(atoms), 21)
    assert_almost_equal(atoms.get_forces()[0, 2], 2.49790655, delta=thres)
    assert_almost_equal(atoms.get_total_energy(), -17637.613286, delta=thres)
    assert_dict_equal(metadata['max_occurs'], max_occurs)


def test_find_neighbor_size_limits():
    xyzfile = 'test_files/examples.extxyz'
    database = read(xyzfile, verbose=False)

    find_neighbor_size_limits(database, rc=6.0, k_max=3, n_jobs=1,
                              verbose=False)
    find_neighbor_size_limits(database, rc=6.5, k_max=2, n_jobs=1,
                              verbose=False)

    metadata = database.metadata
    assert_equal(len(metadata['neighbors']), 2)
    assert_equal(metadata['neighbors']['3']['6.00']['nij_max'], 358)


def test_unit_conversion():
    xyzfile = 'test_files/examples.extxyz'
    database = read(xyzfile, verbose=False,
                    units={'energy': 'kcal/mol'})
    atoms = database.get_atoms(id=2)
    thres = 1e-6
    unit = kcal / mol / eV
    assert_almost_equal(atoms.get_total_energy(), -17637.613286 * unit,
                        delta=thres)


def test_get_convertion():
    x, _, z = get_conversion({
        'energy': 'kcal/mol*Hartree/eV',
        'stress': '0.1*GPa',
    })
    assert_almost_equal(x, (kcal / mol * Hartree / eV) / eV)
    assert_almost_equal(z, 0.1 * GPa / kB)


if __name__ == "__main__":
    main()
