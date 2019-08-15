#!coding=utf-8
"""
Test voigt functions.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_equal, assert_tuple_equal

from tensoralloy.nn.constraint.voigt import voigt_notation
from tensoralloy.nn.constraint.voigt import voigt_to_ij, voigt_to_ijkl

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_voigt_functions():
    """
    Test the voigt functions.
    """
    assert_equal(voigt_notation(1, 1, is_py_index=False, return_py_index=False), 1)
    assert_equal(voigt_notation(2, 2, is_py_index=False, return_py_index=False), 2)
    assert_equal(voigt_notation(3, 3, is_py_index=False, return_py_index=False), 3)
    assert_equal(voigt_notation(1, 2, is_py_index=False, return_py_index=False), 6)
    assert_equal(voigt_notation(1, 3, is_py_index=False, return_py_index=False), 5)
    assert_equal(voigt_notation(2, 3, is_py_index=False, return_py_index=False), 4)

    assert_equal(voigt_notation(0, 0, is_py_index=True, return_py_index=True), 0)
    assert_equal(voigt_notation(0, 2, is_py_index=True, return_py_index=True), 4)

    assert_tuple_equal(voigt_to_ij(1, is_py_index=False), (0, 0))
    assert_tuple_equal(voigt_to_ij(2, is_py_index=False), (1, 1))
    assert_tuple_equal(voigt_to_ij(3, is_py_index=False), (2, 2))
    assert_tuple_equal(voigt_to_ij(4, is_py_index=False), (1, 2))
    assert_tuple_equal(voigt_to_ij(5, is_py_index=False), (0, 2))
    assert_tuple_equal(voigt_to_ij(6, is_py_index=False), (0, 1))

    assert_tuple_equal(voigt_to_ijkl(1, 2, is_py_index=False), (0, 0, 1, 1))
    assert_tuple_equal(voigt_to_ijkl(1, 2, is_py_index=True), (1, 1, 2, 2))


if __name__ == "__main__":
    nose.main()
