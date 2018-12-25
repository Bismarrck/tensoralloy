# coding=utf-8
"""
This module defines indexing-related classes.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import sys
from collections import Counter
from typing import Union, List

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["G2IndexedSlices", "G4IndexedSlices", "IndexTransformer"]


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


class IndexTransformer:
    """
    If a dataset has different stoichiometries, a global ordered symbols list
    should be kept. This class is used to transform the local indices of the
    symbols of arbitrary `Atoms` to the global indexing system.
    """
    _ISTART = 1

    def __init__(self, max_occurs: Counter, symbols: List[str]):
        """
        Initialization method.
        """
        self._max_occurs = max_occurs
        self._symbols = symbols
        self._n_atoms = sum(max_occurs.values()) + IndexTransformer._ISTART

        istart = IndexTransformer._ISTART
        elements = sorted(max_occurs.keys())
        offsets = np.cumsum([max_occurs[e] for e in elements])[:-1]
        offsets = np.insert(offsets, 0, 0)
        delta = Counter()
        index_map = {}
        mask = np.zeros(self._n_atoms, dtype=bool)
        for i, symbol in enumerate(symbols):
            idx_old = i + istart
            idx_new = offsets[elements.index(symbol)] + delta[symbol] + istart
            index_map[idx_old] = idx_new
            delta[symbol] += 1
            mask[idx_new] = True
        reverse_map = {v: k for k, v in index_map.items()}
        index_map[0] = 0
        reverse_map[0] = 0
        self._mask = mask
        self._index_map = index_map
        self._reverse_map = reverse_map

    @property
    def n_atoms(self):
        """
        Return the number of atoms (including the virtual atom at index 0).
        """
        return self._n_atoms

    @property
    def max_occurs(self) -> Counter:
        """
        Return the maximum occurance for each type of element.
        """
        return self._max_occurs

    @property
    def symbols(self) -> List[str]:
        """
        Return a list of str as the ordered chemical symbols of the target
        stoichiometry.
        """
        return self._symbols

    @property
    def reference_symbols(self) -> List[str]:
        """
        Return a list of str as the ordered chemical symbols of the reference
        (global) stoichiometry.
        """
        return sorted(self._max_occurs.elements())

    @property
    def mask(self) -> np.ndarray:
        """
        Return a `bool` array.
        """
        return self._mask

    # FIXME: the index here should start from one. This may be confusing.
    def inplace_map_index(self, index_or_indices, reverse=False,
                          exclude_extra=False):
        """
        Do the in-place index transformation.

        Parameters
        ----------
        index_or_indices : int or List[int] or array_like
            An atom index or a list of indices. One must be aware that indices
            here start from one!
        reverse : bool, optional
            If True, the indices will be mapped to the local reference from the
            global reference.
        exclude_extra : bool
            Exclude the virtual atom when calculating the index.

        """
        if reverse:
            index_map = self._reverse_map
        else:
            index_map = self._index_map
        delta = int(exclude_extra)
        if not hasattr(index_or_indices, "__len__"):
            return index_map[index_or_indices] - delta
        else:
            for i in range(len(index_or_indices)):
                index_or_indices[i] = index_map[index_or_indices[i]] - delta
            return index_or_indices

    def map_array(self, params, reverse=False, fill=0):
        """
        Do the array transformation.

        Parameters
        ----------
        params : array_like
            A 2D or 3D array.
        reverse : bool, optional
            If True, the array will be mapped to the local reference from the
            global reference.
        fill : int or float
            The value to fill for the virtual atom.

        """
        params = np.asarray(params)
        rank = np.ndim(params)
        if rank == 2:
            params = params[np.newaxis, ...]
        if not reverse and params.shape[1] == len(self._symbols):
            params = np.insert(
                params, 0, np.asarray(fill, dtype=params.dtype), axis=1)

        indices = []
        istart = IndexTransformer._ISTART
        if reverse:
            for i in range(istart, istart + len(self._symbols)):
                indices.append(self._index_map[i])
        else:
            for i in range(self._n_atoms):
                indices.append(self._reverse_map.get(i, 0))
        output = params[:, indices]
        if rank == 2:
            output = np.squeeze(output, axis=0)
        return output

    def reverse_map_hessian(self, hessian: np.ndarray):
        """
        Transform the [Np, 3, Np, 3] Hessian matrix to [3N, 3N] where N is the
        number of atoms (excluding the virtual atom) and Np is number of atoms
        in the projected frame.
        """
        rank = np.ndim(hessian)
        if rank != 4 or hessian.shape[1] != 3 or hessian.shape[3] != 3:
            raise ValueError(
                "The input array should be a 4D matrix of shape [Np, 3, Np, 3]")

        indices = []
        istart = IndexTransformer._ISTART
        for i in range(istart, istart + len(self._symbols)):
            indices.append(self._index_map[i])

        n = len(self._symbols)
        h = np.zeros((n * 3, n * 3))

        for i in range(n):
            for alpha in range(3):
                for j in range(n):
                    for beta in range(3):
                        row = i * 3 + alpha
                        col = j * 3 + beta
                        h[row, col] = h[indices[i], alpha, indices[j], beta]

        return h
