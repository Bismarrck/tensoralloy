#!coding=utf-8
"""
Voigt helper functions.
"""
from __future__ import print_function, absolute_import

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def voigt_notation(i, j, is_py_index=True, return_py_index=False):
    """
    Return the Voigt notation given two indices (start from zero).
    """
    if not is_py_index:
        i -= 1
        j -= 1
    if i == j:
        idx = i + 1
    elif (i == 1 and j == 2) or (i == 2 and j == 1):
        idx = 4
    elif (i == 0 and j == 2) or (i == 2 and j == 0):
        idx = 5
    else:
        idx = 6
    if return_py_index:
        return idx - 1
    else:
        return idx


def voigt_to_ij(vi: int, is_py_index=False):
    """
    Return the corresponding Python index tuple (i, j).
    """
    if not is_py_index:
        vi -= 1
    if vi < 3:
        return vi, vi
    elif vi == 3:
        return 1, 2
    elif vi == 4:
        return 0, 2
    else:
        return 0, 1


def voigt_to_ijkl(vi: int, vj: int, is_py_index=False):
    """
    Return the corresponding Python index tuple (i, j, k, l).
    """
    ijkl = []
    for val in (vi, vj):
        ijkl.extend(voigt_to_ij(val, is_py_index=is_py_index))
    return tuple(ijkl)
