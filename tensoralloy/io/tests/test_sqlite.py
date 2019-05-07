# codingf=utf-8
"""
This module defines tests for finding neighbor sizes.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_equal
from os.path import join

from tensoralloy.io.db import connect
from tensoralloy.io.read import read_file
from tensoralloy.test_utils import test_dir, datasets_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_find_neighbor_size_limits():
    """
    Test the function `find_neighbor_size_limits`.
    """
    xyzfile = join(test_dir(), 'examples.extxyz')
    db = read_file(xyzfile, verbose=False)

    db.update_neighbor_meta(k_max=3, rc=6.0, n_jobs=1, verbose=False)
    db.update_neighbor_meta(k_max=2, rc=6.5, n_jobs=1, verbose=False)

    assert_equal(db.get_nij_max(rc=6.0), 358)


def test_read():
    """
    Test the function `read_neighbor_sizes`.
    """
    db = connect(join(datasets_dir(), 'snap-Ni.db'))
    assert_equal(db.get_nij_max(rc=6.5), 14494)
    assert_equal(db.get_nnl_max(rc=6.5), 136)


if __name__ == "__main__":
    nose.main()
