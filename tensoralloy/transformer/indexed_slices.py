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
                                 ('v2g_map', 'ilist', 'jlist', 'n1',
                                  'rij'))):
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
    'n1' : array_like
        The cell boundary shift vectors, `shift[k] = Slist[k] @ cell`.
    'rij' : array_like or None
        The interatomic distances (rij) and their components (rijx, rijy, rijz).

    """

    def __new__(cls,
                v2g_map: Union[tf.Tensor, np.ndarray],
                ilist: Union[tf.Tensor, np.ndarray],
                jlist: Union[tf.Tensor, np.ndarray],
                n1: Union[tf.Tensor, np.ndarray],
                rij: Union[tf.Tensor, np.ndarray, None]):
        return super(G2IndexedSlices, cls).__new__(
            cls, v2g_map, ilist, jlist, n1, rij)

    def as_dict(self, use_computed_dists=True):
        """
        Return a dict representation.
        """
        if use_computed_dists:
            return {"g2.v2g_map": self.v2g_map,
                    "g2.ilist": self.ilist,
                    "g2.jlist": self.jlist,
                    "g2.n1": self.n1}
        else:
            return {"g2.v2g_map": self.v2g_map,
                    "g2.rij": self.rij}


# noinspection PyTypeChecker,PyArgumentList
class G4IndexedSlices(namedtuple('G4IndexedSlices',
                                 ('v2g_map',
                                  'ilist', 'jlist', 'klist',
                                  'n1', 'n2', 'n3',
                                  'rijk'))):
    """
    A `dataclass` contains indexed slices for triple-atom interactions.

    'v2g_map' : array_like
        A list of (atomi, termi) where atomi is the index of the center atom
        and termi is the index of the corresponding 3-body term.
    'ilist' : array_like
        A list of first atom indices.
    'jlist' : array_like
        A list of second atom indices.
    'klist' : array_like
        A list of third atom indices.
    'n1' : array_like
        The cell boundary shift vectors for all r_{i,j}.
    'n2' : array_like
        The cell boundary shift vectors for all r_{i,k}.
    'n3' : array_like
        The cell boundary shift vectors for all r_{j,k}.
    'rijk' : array_like
        Interatomic distances (rij, rik, rjk) and their components (rijx, ...).

    """

    def __new__(cls,
                v2g_map: Union[np.ndarray, tf.Tensor],
                ilist: Union[np.ndarray, tf.Tensor],
                jlist: Union[np.ndarray, tf.Tensor],
                klist: Union[np.ndarray, tf.Tensor],
                n1: Union[np.ndarray, tf.Tensor],
                n2: Union[np.ndarray, tf.Tensor],
                n3: Union[np.ndarray, tf.Tensor],
                rijk: Union[np.ndarray, tf.Tensor, None]):
        return super(G4IndexedSlices, cls).__new__(
            cls, v2g_map, ilist, jlist, klist, n1, n2, n3, rijk)

    def as_dict(self, use_computed_dists=True):
        """
        Return a dict representation.
        """
        if use_computed_dists:
            return {"g4.v2g_map": self.v2g_map,
                    "g4.ilist": self.ilist,
                    "g4.jlist": self.jlist,
                    "g4.klist": self.klist,
                    "g4.n1": self.n1,
                    "g4.n2": self.n2,
                    "g4.n3": self.n3}
        else:
            return {"g4.v2g_map": self.v2g_map,
                    "g4.rijk": self.rijk}
