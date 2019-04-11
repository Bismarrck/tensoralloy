# coding=utf-8
"""
This module defines functions to detect the limits of the neighbor lists.
"""
from __future__ import print_function, absolute_import

import numpy as np

from collections import Counter
from ase.db.sqlite import SQLite3Database
from ase.neighborlist import neighbor_list
from joblib import Parallel, delayed

from tensoralloy.utils import cantor_pairing

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def convert_k_max_to_key(k_max):
    """ Convert `k_max` to a valid key.  """
    return "{}".format(k_max)


def convert_rc_to_key(rc):
    """ Convert `rc` to a valid key. """
    return "{:.2f}".format(round(rc, 4))


def read_neighbor_sizes(db: SQLite3Database, k_max, rc):
    """
    Read `nij_max`, `nijk_max` and `nnl_max` from a database given `k_max` and
    `rc`.
    """
    rc = convert_rc_to_key(rc)
    k_max = convert_k_max_to_key(k_max)
    details = db.metadata["neighbors"][k_max][rc]
    return details['nij_max'], details['nijk_max'], details['nnl_max']


def find_neighbor_size_of_atoms(atoms, rc, k_max):
    """
    A helper function to find `nij`, `nijk` and `nnl` for the `Atoms` object.
    """
    ilist, jlist = neighbor_list('ij', atoms, cutoff=rc)
    if k_max >= 2:
        nij = len(ilist)
    else:
        numbers = atoms.numbers
        uniques = list(set(numbers))
        inlist = numbers[ilist]
        jnlist = numbers[jlist]
        counter = Counter(cantor_pairing(inlist, jnlist))
        nij = sum([counter[x] for x in cantor_pairing(uniques, uniques)])
    numbers = atoms.numbers
    nnl = 0
    for i in range(len(atoms)):
        indices = np.where(ilist == i)[0]
        ii = numbers[ilist[indices]]
        ij = numbers[jlist[indices]]
        if k_max == 1:
            indices = np.where(ii == ij)[0]
            ii = ii[indices]
            ij = ij[indices]
        if len(ii) > 0:
            nnl = max(max(Counter(cantor_pairing(ii, ij)).values()), nnl)
    if k_max == 3:
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
    return nij, nijk, nnl


def find_neighbor_size_maximums(database: SQLite3Database, rc: float,
                                k_max=3, n_jobs=-1, verbose=True):
    """
    Find `nij_max`, `nijk_max` and 'nnl_max' of all `Atoms` objects in the
    database.

    Parameters
    ----------
    database : SQLite3Database
        The database to update. This db must be created by the function `read`.
    rc : float
        The cutoff radius.
    k_max : int
        The maximum k for the many-body expansion scheme.
    n_jobs : int
        The maximum number of concurrently running jobs. If -1 all CPUs are
        used. If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.
    verbose : bool
        If True, the progress shall be logged.

    """

    def _find_wrapper(aid):
        return find_neighbor_size_of_atoms(database.get_atoms(f'id={aid}'), rc, k_max)

    if verbose:
        print('Start finding neighbors for rc = {} and k_max = {}. This may '
              'take a very long time.'.format(rc, k_max))

    verb_level = 5 if verbose else 0
    results = Parallel(n_jobs=n_jobs, verbose=verb_level)(
        delayed(_find_wrapper)(jid) for jid in range(1, len(database) + 1)
    )

    nij_max, nijk_max, nnl_max = np.asarray(
        results, dtype=int).max(axis=0).tolist()
    rc = convert_rc_to_key(rc)
    k_max = convert_k_max_to_key(k_max)
    details = {k_max: {rc: {
        'nij_max': nij_max, 'nijk_max': nijk_max, 'nnl_max': nnl_max}}}
    metadata = dict(database.metadata)
    if 'neighbors' not in metadata:
        metadata['neighbors'] = details
    elif k_max not in metadata['neighbors']:
        metadata['neighbors'][k_max] = details[k_max]
    else:
        metadata['neighbors'][k_max][rc] = details[k_max][rc]
    database.metadata = metadata

    if verbose:
        print(f'All {len(database)} jobs are done. nij_max = {nij_max}, '
              f'nijk_max = {nijk_max}, nnl_max = {nnl_max}')
