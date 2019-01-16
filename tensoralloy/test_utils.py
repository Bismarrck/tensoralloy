# coding=utf-8
"""
This module defines unit tests related functions and vars.
"""
from __future__ import print_function, absolute_import

import numpy as np

from collections import Counter
from os.path import join, dirname, abspath
from ase import Atoms
from ase.io import read
from nose.tools import assert_less, assert_equal

from tensoralloy.utils import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def assert_array_almost_equal(a: np.ndarray, b: np.ndarray, delta, msg=None):
    """
    Fail if the two arrays are unequal as determined by their maximum absolute
    difference rounded to the given number of decimal places (default 7) and
    comparing to zero, or by comparing that the between the two objects is more
    than the given delta.
    """
    assert_less(np.abs(a - b).max(), delta, msg)


def assert_array_equal(a: np.ndarray, b: np.ndarray, msg=None):
    """
    Fail if the two arrays are unequal with threshold 0 (int) or 1e-6 (float32)
    or 1e-12 (float64).
    """
    assert_equal(a.dtype, b.dtype)
    if np.issubdtype(a.dtype, np.int_):
        delta = 0
    elif a.dtype == np.float64:
        delta = 1e-12
    else:
        delta = 1e-6
    assert_array_almost_equal(a, b, delta, msg=msg)


# Pd3O2
Pd3O2 = Atoms(symbols='Pd3O2', pbc=np.array([True, True, False], dtype=bool),
              cell=np.array([[7.78, 0., 0.],
                             [0., 5.50129076, 0.],
                             [0., 0., 15.37532269]]),
              positions=np.array([[3.89, 0., 8.37532269],
                                  [0., 2.75064538, 8.37532269],
                                  [3.89, 2.75064538, 8.37532269],
                                  [5.835, 1.37532269, 8.5],
                                  [5.835, 7.12596807, 8.]]))


# A permutation of Pd3O2
Pd2O2Pd = Atoms(symbols='Pd2O2Pd',
                pbc=np.array([True, True, False], dtype=bool),
                cell=np.array([[7.78, 0., 0.],
                               [0., 5.50129076, 0.],
                               [0., 0., 15.37532269]]),
                positions=np.array([[3.89, 0., 8.37532269],
                                    [0., 2.75064538, 8.37532269],
                                    [5.835, 1.37532269, 8.5],
                                    [5.835, 7.12596807, 8.],
                                    [3.89, 2.75064538, 8.37532269]]))


def test_dir(absolute=False):
    """
    Return the directory of `test_files`.
    """
    path = join(dirname(__file__), "..", "test_files")
    if absolute:
        path = abspath(path)
    return path


qm7m = AttributeDict(
    max_occurs=Counter({'C': 5, 'H': 8, 'O': 2}),
    nij_max=198,
    nijk_max=1217,
    trajectory=read(join(test_dir(), 'qm7m', 'qm7m.xyz'),
                    index=':', format='xyz'),
)

for _atoms in qm7m.trajectory:
    # Setting the boundary cell is important because `neighbor_list` may give
    # totally different results.
    _atoms.set_cell(np.eye(3) * 20.0)


def datasets_dir(absolute=False):
    """
    Return the directory of `datasets`. Built-in datasets can be found here.
    """
    path = join(dirname(__file__), "..", "datasets")
    if absolute:
        path = abspath(path)
    return path


def project_dir(absolute=False):
    """
    Return the root directory of this project.
    """
    path = join(dirname(__file__), "..")
    if absolute:
        path = abspath(path)
    return path
