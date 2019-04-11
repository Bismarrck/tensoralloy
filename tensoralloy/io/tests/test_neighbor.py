# codingf=utf-8
"""
This module defines tests for finding neighbor sizes.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_equal
from ase.db import connect
from os.path import join

from tensoralloy.io.neighbor import find_neighbor_size_of_atoms
from tensoralloy.io.neighbor import find_neighbor_size_maximums
from tensoralloy.io.neighbor import read_neighbor_sizes
from tensoralloy.io.read import read_file
from tensoralloy.test_utils import test_dir, datasets_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_find_neighbor_size_limits():
    """
    Test the function `find_neighbor_size_limits`.
    """
    xyzfile = join(test_dir(), 'examples.extxyz')
    database = read_file(xyzfile, verbose=False)

    find_neighbor_size_maximums(database, rc=6.0, k_max=3, n_jobs=1,
                                verbose=False)
    find_neighbor_size_maximums(database, rc=6.5, k_max=2, n_jobs=1,
                                verbose=False)

    metadata = database.metadata
    assert_equal(len(metadata['neighbors']), 2)
    assert_equal(metadata['neighbors']['3']['6.00']['nij_max'], 358)


def test_find_sizes():
    """
    Test the function `_find_sizes`.
    """
    db = connect(join(test_dir(), 'qm7m', 'qm7m.db'))

    atoms = db.get_atoms('id=2')
    nij, nijk, nnl = find_neighbor_size_of_atoms(atoms, 6.5, 2)
    assert_equal(nij, 20)
    assert_equal(nijk, 0)
    assert_equal(nnl, 4)

    atoms = db.get_atoms('id=3')
    nij, nijk, nnl = find_neighbor_size_of_atoms(atoms, 6.5, 3)
    assert_equal(nij, 56)
    assert_equal(nijk, 168)
    assert_equal(nnl, 6)

    nij, nijk, nnl = find_neighbor_size_of_atoms(atoms, 6.5, 1)
    assert_equal(nij, 32)  # 2 C-C + 5 x 6 H-H = 32
    assert_equal(nijk, 0)
    assert_equal(nnl, 5)


def test_read():
    """
    Test the function `read_neighbor_sizes`.
    """
    db = connect(join(datasets_dir(), 'snap-Ni.db'))
    nij_max, _, nnl_max = read_neighbor_sizes(db, 2, 6.5)

    assert_equal(nij_max, 14494)
    assert_equal(nnl_max, 136)


if __name__ == "__main__":
    nose.main()
