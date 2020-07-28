# coding=utf-8
"""
This module defines tests for file-io functions.
"""
import nose

from ase import Atoms
from ase.units import GPa, Ha
from ase.geometry import cell_to_cellpar
from nose.tools import assert_equal, assert_almost_equal, assert_true
from nose.tools import assert_dict_equal
from os.path import join

from tensoralloy.io.read import read_file
from tensoralloy import atoms_utils
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
    assert_equal(atoms.info['key_value_pairs']['source'], '')


def test_read_snap_stress():
    """
    Test reading an example SNAP/Ni extxyz file.

    The unit of the stress tensor in this file is kbar.
    """
    xyzfile = join(test_dir(), 'snap_Ni_id11.extxyz')
    database = read_file(xyzfile, units={"stress": "kbar"}, verbose=False)
    atoms = database.get_atoms(id=1, add_additional_information=True)
    assert isinstance(atoms, Atoms)
    stress = atoms.get_stress()
    weights = atoms.info['data']['weights']
    source = atoms.info['key_value_pairs']['source']
    assert_almost_equal(stress[0], -0.01388831152640921, delta=1e-8)
    assert_equal(weights[0], 1.0)
    assert_equal(weights[1], 1.0)
    assert_equal(weights[2], 0.0)
    assert_equal(source, "Ni.AIMD.0")


def test_read_pulay_stress():
    """
    Test reading `Pu4_60GPa.extxyz`.
    """
    xyzfile = join(test_dir(), 'Pu4_60GPa.extxyz')
    database = read_file(xyzfile, verbose=False)
    atoms = database.get_atoms(id=1, add_additional_information=True)
    kbar = atoms.get_stress() / GPa * 10.0
    pulay = atoms_utils.get_pulay_stress(atoms) / GPa * 10.0 * (-1)
    net = kbar[:3].mean() - pulay
    assert_almost_equal(net, -0.78, delta=0.01)


def test_read_stepmax_xyz():
    xyzfile = join(test_dir(), "Pu8.stepmax.xyz")
    database = read_file(xyzfile, num_examples=1, file_type='stepmax')
    atoms = database.get_atoms(id=1, add_additional_information=True)
    cellpars = cell_to_cellpar(atoms.cell)
    assert_almost_equal(atoms.positions[0, 2], 3.301309)
    assert_almost_equal(cellpars[2], 6.7942123514756485, delta=1e-6)
    assert_almost_equal(atoms.get_potential_energy(), -32.4 * Ha, delta=1e-6)


def test_read_electron_temperature_and_entropy():
    """
    Test the `read_file` function for reading electron entropy and electron
    temperature.
    """
    xyzfile = join(test_dir(), 'Be_liquid_4000K_TS.extxyz')
    database = read_file(xyzfile, num_examples=3)
    atoms = database.get_atoms(id=1, add_additional_information=True)
    etemp = atoms_utils.get_electron_temperature(atoms)
    eentropy = atoms_utils.get_electron_entropy(atoms)
    assert_almost_equal(etemp, 0.34469373, delta=1e-6)
    assert_almost_equal(eentropy, 14.939843166274024, delta=1e-6)


if __name__ == '__main__':
    nose.main()
