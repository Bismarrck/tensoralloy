# coding=utf-8
"""
This module defines indexing-related classes.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from collections import namedtuple
from typing import Union

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["G2IndexedSlices", "G4IndexedSlices"]


# noinspection PyTypeChecker,PyArgumentList
class G2IndexedSlices(namedtuple('G2IndexedSlices',
                                 ('v2g_map', 'ilist', 'jlist', 'shift'))):
    """
    A `dataclass` contains indexed slices for the atom-atom interactions.

    'v2g_map' : array_like
        A list of (atomi, etai, termi) where atomi is the index of the
        center atom, etai is the index of the `eta` and termi is the index
        of the corresponding 2-body term.
    'ilist' : array_like
        A list of first atom indices.
    'jlist' : array_like
        A list of second atom indices.
    'shift' : array_like
        The cell boundary shift vectors, `shift[k] = Slist[k] @ cell`.

    """

    def __new__(cls,
                v2g_map: Union[tf.Tensor, np.ndarray],
                ilist: Union[tf.Tensor, np.ndarray],
                jlist: Union[tf.Tensor, np.ndarray],
                shift: Union[tf.Tensor, np.ndarray]):
        return super(G2IndexedSlices, cls).__new__(
            cls, v2g_map, ilist, jlist, shift)


# noinspection PyTypeChecker,PyArgumentList
class G4IndexedSlices(namedtuple('G4IndexedSlices',
                                 ('v2g_map',
                                  'ij', 'ik', 'jk',
                                  'ij_shift', 'ik_shift', 'jk_shift'))):
    """
    A `dataclass` contains indexed slices for triple-atom interactions.

    'v2g_map' : array_like
        A list of (atomi, termi) where atomi is the index of the center atom
        and termi is the index of the corresponding 3-body term.
    'ij' : array_like
        A list of (i, j) as the indices for r_{i,j}.
    'ik' : array_like
        A list of (i, k) as the indices for r_{i,k}.
    'jk' : array_like
        A list of (j, k) as the indices for r_{j,k}.
    'ij_shift' : array_like
        The cell boundary shift vectors for all r_{i,j}.
    'ik_shift' : array_like
        The cell boundary shift vectors for all r_{i,k}.
    'jk_shift' : array_like
        The cell boundary shift vectors for all r_{j,k}.

    """

    def __new__(cls,
                v2g_map: Union[np.ndarray, tf.Tensor],
                ij: Union[np.ndarray, tf.Tensor],
                ik: Union[np.ndarray, tf.Tensor],
                jk: Union[np.ndarray, tf.Tensor],
                ij_shift: Union[np.ndarray, tf.Tensor],
                ik_shift: Union[np.ndarray, tf.Tensor],
                jk_shift: Union[np.ndarray, tf.Tensor]):
        return super(G4IndexedSlices, cls).__new__(
            cls, v2g_map, ij, ik, jk, ij_shift, ik_shift, jk_shift)
