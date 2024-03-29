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

from tensoralloy.utils import cantor_pairing, szudzik_pairing

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
    ij2k = 3


@dataclass(frozen=True)
class NeighborSize:
    """
    Simply a container class.
    """
    nnl: int
    nij: int
    nijk: int
    ij2k: int

    def __getitem__(self, item: Union[str, NeighborProperty]):
        if isinstance(item, NeighborProperty):
            item = item.name
        return self.__dict__[item]


def find_neighbor_size_of_atoms(atoms: Atoms,
                                rc: float,
                                find_nijk=False,
                                find_ij2k=False) -> NeighborSize:
    """
    A helper function to find `nij`, `nijk` and `nnl` for the `Atoms` object.

    Parameters
    ----------
    atoms : Atoms
        The target `Atoms` object.
    rc : float
        The cutoff radius.
    find_nijk : bool
        If True, `nijk` will be calculated.
    find_ij2k : bool
        If True, `ij2k` will be calculated.

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
        * ij2k : int
            A necessary constant for constructing triple-atoms interactions.

    """
    ilist, jlist, nlist = neighbor_list('ijS', atoms, cutoff=rc)
    nij = len(ilist)
    numbers = atoms.numbers
    nnl = 0
    nijk = 0
    ij2k = 0
    for i in range(len(atoms)):
        indices = np.where(ilist == i)[0]
        ii = numbers[ilist[indices]]
        ij = numbers[jlist[indices]]
        if len(ii) > 0:
            nnl = max(max(Counter(cantor_pairing(ii, ij)).values()), nnl)
    if find_ij2k:
        itypes = sorted(list(set(numbers)))
        tlist = np.zeros(len(numbers), dtype=int)
        for i, numberi in enumerate(numbers):
            tlist[i] = itypes.index(numberi)
        indices = {}
        vectors = {}
        for idx, atomi in enumerate(ilist):
            if atomi not in indices:
                indices[atomi] = []
                vectors[atomi] = []
            atomj = jlist[idx]
            indices[atomi].append(atomj)
            vectors[atomi].append(nlist[idx])
        counters = {}
        ntypes = len(itypes)
        ijktypes = np.zeros((ntypes, ntypes, ntypes), dtype=int)
        ijktype = 0
        for i in range(ntypes):
            for j in range(ntypes):
                for k in range(ntypes):
                    ijktypes[i, j, k] = ijktype
                    ijktype += 1
        for atomi, nl in indices.items():
            itype = tlist[atomi]
            n = len(nl)
            nijk += (n - 1 + 1) * (n - 1) // 2
            for j, atomj in enumerate(nl):
                jtype = tlist[atomj]
                for k, atomk in enumerate(nl):
                    ktype = tlist[atomk]
                    tijk = ijktypes[itype][jtype][ktype]
                    if j == k and np.all(
                            vectors[atomi][j] == vectors[atomi][k]):
                        continue
                    n_id = szudzik_pairing(*vectors[atomi][j])
                    ijt_id = cantor_pairing(cantor_pairing(atomi, atomj), tijk)
                    if ijt_id not in counters:
                        counters[ijt_id] = Counter()
                    counters[ijt_id][n_id] += 1
                    ij2k = max(ij2k, counters[ijt_id][n_id])
    elif find_nijk:
        nl = {}
        for i, atomi in enumerate(ilist):
            if atomi not in nl:
                nl[atomi] = []
            nl[atomi].append(jlist[i])
        for atomi, nlist in nl.items():
            n = len(nlist)
            nijk += (n - 1 + 1) * (n - 1) // 2
    return NeighborSize(nnl=nnl, nij=nij, nijk=nijk, ij2k=ij2k)
