#!coding=utf-8
"""
Voigt helper functions.
"""
from __future__ import print_function, absolute_import

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def voigt_notation(i, j, return_py_index=False):
    """
    Return the Voigt notation given two indices (start from zero).
    """
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


def voigt_to_ijkl(vi: int, vj: int, is_py_index=False):
    """
    Return the corresponding (i, j, k, l).
    """
    if not is_py_index:
        vi -= 1
        vj -= 1
    ijkl = []
    for val in (vi, vj):
        if val < 3:
            ijkl.extend((val, val))
        elif val == 3:
            ijkl.extend((1, 2))
        elif val == 4:
            ijkl.extend((0, 2))
        else:
            ijkl.extend((0, 1))
    return ijkl
