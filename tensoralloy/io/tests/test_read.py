# coding=utf-8
"""
This module defines tests for file-io functions.
"""
import nose

from nose.tools import assert_equal, assert_almost_equal, assert_true
from nose.tools import assert_dict_equal
from tensoralloy.io.read import read_file


def test_read_xyz():
    """
    Test reading normal xyz files.
    """
    xyzfile = 'test_files/B28.xyz'
    database = read_file(xyzfile, verbose=False, num_examples=2)
    atoms = database.get_atoms('id=2')
    assert_equal(len(database), 2)
    assert_almost_equal(atoms.positions[1, 1], 10.65007390)
    assert_almost_equal(atoms.get_total_energy(), -78.51063520)
    assert_true(atoms.cell.sum() > 1e-8)


def test_read_extxyz():
    """
    Test reading ext xyz files.
    """
    xyzfile = 'test_files/examples.extxyz'
    database = read_file(xyzfile, verbose=False)
    atoms = database.get_atoms('id=2')
    thres = 1e-6
    metadata = database.metadata
    max_occurs = {'C': 10, 'H': 8, 'O': 4}
    assert_equal(len(database), 2)
    assert_equal(len(atoms), 21)
    assert_almost_equal(atoms.get_forces()[0, 2], 2.49790655, delta=thres)
    assert_almost_equal(atoms.get_total_energy(), -17637.613286, delta=thres)
    assert_dict_equal(metadata['max_occurs'], max_occurs)
    assert_equal(metadata['periodic'], False)


if __name__ == '__main__':
    nose.main()
