# coding=utf-8
"""
This module defines `IndexTransformer` which is used to map arrays from local
to global reference or vice versa.
"""
from __future__ import print_function, absolute_import

import numpy as np

from collections import Counter
from typing import List

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class VirtualAtomMap:
    """
    If a dataset has different stoichiometries, a global ordered symbols list
    should be kept. This class is used to transform the local indices of the
    symbols of arbitrary `Atoms` to the global indexing system.
    """
    REAL_ATOM_START = 1

    def __init__(self, max_occurs: Counter, symbols: List[str]):
        """
        Initialization method.
        """
        self._max_occurs = max_occurs
        self._symbols = symbols
        self._max_vap_natoms = sum(max_occurs.values()) + 1

        istart = VirtualAtomMap.REAL_ATOM_START
        elements = sorted(max_occurs.keys())
        offsets = np.cumsum([max_occurs[e] for e in elements])[:-1]
        offsets = np.insert(offsets, 0, 0)
        delta = Counter()
        index_map = {}
        mask = np.zeros(self._max_vap_natoms, dtype=bool)
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
    def max_vap_natoms(self):
        """
        Return the number of atoms (including the 'virtual atom') in the global
        reference system.
        """
        return self._max_vap_natoms

    @property
    def max_occurs(self) -> Counter:
        """
        Return the maximum occurance for each type of element.
        """
        return self._max_occurs

    @property
    def atom_masks(self) -> np.ndarray:
        """
        Return a `bool` array.
        """
        return self._mask

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

    def map_array(self, array: np.ndarray, reverse=False):
        """
        Transform the array from local to global reference (reverse=False) or
        vise versa (reverse=True).

        Parameters
        ----------
        array : array_like
            A 2D or 3D array.
        reverse : bool, optional
            If True, the array will be mapped to the local reference from the
            global reference.

        """
        array = np.asarray(array)
        rank = np.ndim(array)
        if rank == 2:
            array = array[np.newaxis, ...]
        elif rank <= 1 or rank > 3:
            raise ValueError("The rank should be 2 or 3")
        if not reverse:
            if array.shape[1] == len(self._symbols):
                array = np.insert(
                    array, 0, np.asarray(0, dtype=array.dtype), axis=1)
            else:
                shape = (array.shape[0], len(self._symbols), array.shape[2])
                raise ValueError(f"The shape should be {shape}")

        indices = []
        istart = VirtualAtomMap.REAL_ATOM_START
        if reverse:
            for i in range(istart, istart + len(self._symbols)):
                indices.append(self._index_map[i])
        else:
            for i in range(self._max_vap_natoms):
                indices.append(self._reverse_map.get(i, 0))
        output = array[:, indices]
        if rank == 2:
            output = np.squeeze(output, axis=0)
        return output

    def map_positions(self, positions: np.ndarray, reverse=False):
        """ A wrapper function. """
        return self.map_array(positions, reverse=reverse)

    def map_forces(self, forces: np.ndarray, reverse=False):
        """ A wrapper function. """
        return self.map_array(forces, reverse=reverse)
