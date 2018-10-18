# coding=utf-8
"""
Unit tests for module `data_ops`.
"""
from __future__ import print_function, absolute_import

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

from nose import main
from nose.tools import assert_almost_equal, assert_equal, assert_dict_equal
from file_io import read, find_neighbor_sizes


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


def test_find_neighbors():
    xyzfile = 'test_files/examples.extxyz'
    database = read(xyzfile, verbose=False)
    find_neighbor_sizes(database, rc=6.0, n_jobs=1)
    assert_equal(database.metadata['nij_max'], 358)


if __name__ == "__main__":
    main()
