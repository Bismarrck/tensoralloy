#!coding=utf-8
"""
This module defines neighbor list related functions.
"""
from __future__ import print_function, absolute_import

import numpy as np

from dataclasses import dataclass
from collections import Counter
from enum import Enum
from ase import Atoms
from ase.neighborlist import neighbor_list
from typing import Union

from tensoralloy.utils import cantor_pairing

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["NeighborProperty", "NeighborSize", "find_neighbor_size_of_atoms"]


class NeighborProperty(Enum):
    """
    Available neighbor properties.
    """
    nnl = 0
    nij = 1
    nijk = 2


@dataclass(frozen=True)
class NeighborSize:
    """
    Simply a container class.
    """
    nnl: int
    nij: int
    nijk: int

    def __getitem__(self, item: Union[str, NeighborProperty]):
        if isinstance(item, NeighborProperty):
            item = item.name
        return self.__dict__[item]


def find_neighbor_size_of_atoms(atoms: Atoms,
                                rc: float,
                                angular=False) -> NeighborSize:
    """
    A helper function to find `nij`, `nijk` and `nnl` for the `Atoms` object.

    Parameters
    ----------
    atoms : Atoms
        The target `Atoms` object.
    rc : float
        The cutoff radius.
    angular : bool
        If True, `nijk` will also be calculated.

    Returns
    -------
    dct : NeighborSize
        The neighbor size of the target `Atoms`.

        * nij : int
            The total number of atom-atom pairs within `rc`.
        * nijk : int
            The total number of triples within `rc` or 0 if `angular` is False.
        * nnl : int
            Each atom has `n_A` A-type neighbors, `n_B` B-type neigbors, etc.
            `nnl` is the maximum of all {n_A}, {n_B}, etc.

    """
    ilist, jlist = neighbor_list('ij', atoms, cutoff=rc)
    nij = len(ilist)
    numbers = atoms.numbers
    nnl = 0
    for i in range(len(atoms)):
        indices = np.where(ilist == i)[0]
        ii = numbers[ilist[indices]]
        ij = numbers[jlist[indices]]
        if len(ii) > 0:
            nnl = max(max(Counter(cantor_pairing(ii, ij)).values()), nnl)
    if angular:
        nl = {}
        for i, atomi in enumerate(ilist):
            if atomi not in nl:
                nl[atomi] = []
            nl[atomi].append(jlist[i])
        nijk = 0
        for atomi, nlist in nl.items():
            n = len(nlist)
            nijk += (n - 1 + 1) * (n - 1) // 2
    else:
        nijk = 0
    return NeighborSize(nnl=nnl, nij=nij, nijk=nijk)
