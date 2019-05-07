#!coding=utf-8
"""
Unit tests of `tensoralloy.neighbor` module.
"""
from __future__ import print_function, absolute_import

import nose

from os.path import join
from ase.db import connect
from nose.tools import assert_equal

from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_find_sizes():
    """
    Test the function `_find_sizes`.
    """
    db = connect(join(test_dir(), 'qm7m', 'qm7m.db'))

    atoms = db.get_atoms('id=2')
    size = find_neighbor_size_of_atoms(atoms, 6.5, angular=False)
    assert_equal(size.nij, 20)
    assert_equal(size.nijk, 0)
    assert_equal(size.nnl, 4)

    atoms = db.get_atoms('id=3')
    size = find_neighbor_size_of_atoms(atoms, 6.5, angular=True)
    assert_equal(size.nij, 56)
    assert_equal(size.nijk, 168)
    assert_equal(size.nnl, 6)


if __name__ == "__main__":
    nose.run()
