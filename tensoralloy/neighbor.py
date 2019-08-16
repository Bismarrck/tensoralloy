#!coding=utf-8
"""
This module defines neighbor list related functions.
"""
from __future__ import print_function, absolute_import

from dataclasses import dataclass
from enum import Enum
from ase import Atoms
from ase.neighborlist import neighbor_list
from typing import Union

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["NeighborProperty", "NeighborSize", "find_neighbor_size_of_atoms"]


class NeighborProperty(Enum):
    """
    Available neighbor properties.
    """
    nij = 1
    nijk = 2


@dataclass(frozen=True)
class NeighborSize:
    """
    Simply a container class.
    """
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
    return NeighborSize(nij=nij, nijk=nijk)
