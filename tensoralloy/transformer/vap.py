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
        reverse_map = {v: k - 1 for k, v in index_map.items()}
        index_map[0] = 0
        reverse_map[0] = -1
        self._mask = mask
        self.local_to_gsl_map = index_map
        self.gsl_to_local_map = reverse_map

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
                indices.append(self.local_to_gsl_map[i])
        else:
            for i in range(self._max_vap_natoms):
                indices.append(self.gsl_to_local_map.get(i, -1) + istart)
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
