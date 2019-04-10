# coding=utf-8
"""
This module defines tests for file-io functions.
"""
import nose

from ase import Atoms
from nose.tools import assert_equal, assert_almost_equal, assert_true
from nose.tools import assert_dict_equal
from os.path import join

from tensoralloy.io.read import read_file
from tensoralloy.test_utils import test_dir


def test_read_xyz():
    """
    Test reading normal xyz files.
    """
    xyzfile = join(test_dir(), 'B28.xyz')
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
    xyzfile = join(test_dir(), 'examples.extxyz')
    database = read_file(xyzfile, verbose=False)
    atoms = database.get_atoms('id=2', add_additional_information=True)
    weights = atoms.info['data']['weights']
    thres = 1e-6
    metadata = database.metadata
    max_occurs = {'C': 10, 'H': 8, 'O': 4}
    assert_equal(len(database), 2)
    assert_equal(len(atoms), 21)
    assert_almost_equal(atoms.get_forces()[0, 2], 2.49790655, delta=thres)
    assert_almost_equal(atoms.get_total_energy(), -17637.613286, delta=thres)
    assert_dict_equal(metadata['max_occurs'], max_occurs)
    assert_equal(metadata['periodic'], False)
    assert_equal(weights[0], 1.0)
    assert_equal(weights[1], 1.0)
    assert_equal(weights[2], 1.0)


def test_read_snap_stress():
    """
    Test reading an example SNAP/Ni extxyz file.

    The unit of the stress tensor in this file is kbar.
    """
    xyzfile = join(test_dir(), 'snap_Ni_id11.extxyz')
    database = read_file(xyzfile, units={"stress": "kbar"}, verbose=False)
    atoms = database.get_atoms(id=1, add_additional_information=True)
    stress = atoms.get_stress()
    weights = atoms.info['data']['weights']
    assert isinstance(atoms, Atoms)
    assert_almost_equal(stress[0], -0.01388831152640921, delta=1e-8)
    assert_equal(weights[0], 1.0)
    assert_equal(weights[1], 1.0)
    assert_equal(weights[2], 0.0)


if __name__ == '__main__':
    nose.main()
