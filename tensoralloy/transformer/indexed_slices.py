# coding=utf-8
"""
This module defines indexing-related classes.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import sys

from typing import Union

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["G2IndexedSlices", "G4IndexedSlices"]


# Backward compatibility
if sys.version_info < (3, 6):

    raise Exception("Python < 3.6 is not supported")

elif sys.version_info < (3, 7):

    from collections import namedtuple

    G2IndexedSlices = namedtuple('G2IndexedSlices',
                                 ('v2g_map', 'ilist', 'jlist', 'shift'))

    G4IndexedSlices = namedtuple('G4IndexedSlices',
                                 ('v2g_map', 'ij', 'ik', 'jk',
                                  'ij_shift', 'ik_shift', 'jk_shift'))

else:

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class G2IndexedSlices:
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
        v2g_map: Union[np.ndarray, tf.Tensor]
        ilist: Union[np.ndarray, tf.Tensor]
        jlist: Union[np.ndarray, tf.Tensor]
        shift: Union[np.ndarray, tf.Tensor]

        __slots__ = ["v2g_map", "ilist", "jlist", "shift"]


    @dataclass
    class G4IndexedSlices:
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
        v2g_map: Union[np.ndarray, tf.Tensor]
        ij: Union[np.ndarray, tf.Tensor]
        ik: Union[np.ndarray, tf.Tensor]
        jk: Union[np.ndarray, tf.Tensor]
        ij_shift: Union[np.ndarray, tf.Tensor]
        ik_shift: Union[np.ndarray, tf.Tensor]
        jk_shift: Union[np.ndarray, tf.Tensor]

        __slots__ = ["v2g_map", "ij", "ik", "jk",
                     "ij_shift", "ik_shift", "jk_shift"]
