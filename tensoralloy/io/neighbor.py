# coding=utf-8
"""
This module defines functions to detect the limits of the neighbor lists.
"""
from __future__ import print_function, absolute_import

import numpy as np
import json

from ase.db.sqlite import SQLite3Database
from ase.neighborlist import neighbor_list
from joblib import Parallel, delayed
from collections import Counter
from typing import Dict

from tensoralloy.utils import cantor_pairing, nested_get, nested_set

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _get_keypath(k_max: int, rc: float, prop: str):
    """
    A helper function to get the corresponding keypath.
    """
    return f"{k_max}.{'{:.2f}'.format(np.round(rc, decimals=2))}.{prop}"


class CoreDatabase(SQLite3Database):
    """
    A wrapper of `ase.db.sqlite.SQLite3Database` with specific functions for
    this project.
    """

    def __init__(self, filename, create_indices=True, use_lock_file=False,
                 serial=False):
        """
        Initialize a `CoreDatabase` object.
        """
        super(CoreDatabase, self).__init__(
            filename=filename,
            create_indices=create_indices,
            use_lock_file=use_lock_file,
            serial=serial)

        # Automatically initialize the metadata.
        self._initialize(self._connect())

    @property
    def metadata(self) -> dict:
        """
        Return a copy of the metadata dict.
        """
        return self._metadata.copy()

    @metadata.setter
    def metadata(self, dct):
        """
        Set the metadata.
        """
        self._metadata = dct
        self._write_metadata()

    def _write_metadata(self):
        """
        Write metadata to the sqlite3 database file.
        """
        con = self._connect()
        self._initialize(con)
        md = json.dumps(self._metadata)
        cur = con.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM information WHERE name='metadata'")

        if cur.fetchone()[0]:
            cur.execute(
                "UPDATE information SET value=? WHERE name='metadata'", [md])
        else:
            cur.execute('INSERT INTO information VALUES (?, ?)',
                        ('metadata', md))
        con.commit()

    @property
    def max_occurs(self):
        """
        Return the maximum appearance of each type of element.
        """
        return self._metadata.get('max_occurs', Counter())

    @property
    def has_forces(self):
        """
        Return True if the `Atoms` objects of this database have atomic forces.
        """
        return self._metadata.get('forces', True)

    @property
    def has_stress(self):
        """
        Return True if the `Atoms` objects of this database have stress tensors.
        """
        return self._metadata.get('stress', True)

    @property
    def atomic_static_energy(self) -> Dict[str, float]:
        """
        Return the calculated atomic static energy dict.
        """
        return self._metadata.get('atomic_static_energy', {})

    def _get_neighbor_property(self, k_max: int, rc: float, prop: str,
                               allow_calculation=False):
        """
        A helper function to get the value of a neighbor property.

        Availabel properties:
            * nnl_max
            * nij_max
            * nijk_max

        """
        keypath = _get_keypath(k_max, rc, prop)
        val = nested_get(self._metadata, keypath)
        if val is None and allow_calculation:
            val = self.update_neighbor_meta(k_max, rc)[prop]
        return val

    def get_nij_max(self, k_max: int, rc: float, allow_calculation=False):
        """
        Return the corresponding `N_ij_max`.
        """
        return self._get_neighbor_property(
            k_max, rc, 'nij_max', allow_calculation)

    def get_nijk_max(self, k_max: int, rc: float, allow_calculation=False):
        """
        Return the corresponding `N_ijk_max`.
        """
        return self._get_neighbor_property(
            k_max, rc, 'nijk_max', allow_calculation)

    def get_nnl_max(self, k_max: int, rc: float, allow_calculation=False):
        """
        Return the corresponding `N_nl_max`.
        """
        return self._get_neighbor_property(
            k_max, rc, 'nnl_max', allow_calculation)

    def update_neighbor_meta(self,
                             k_max: int,
                             rc: float,
                             n_jobs=-1,
                             verbose=False) -> Dict[str, int]:
        """
        Update the metadata of neighbor properties.
        """
        def _find(aid):
            return find_neighbor_size_of_atoms(
                self.get_atoms(f'id={aid}'), rc, k_max)

        if verbose:
            print('Start finding neighbors for rc = {} and k_max = {}. This '
                  'may take a very long time.'.format(rc, k_max))

        verb_level = 5 if verbose else 0
        results = Parallel(n_jobs=n_jobs, verbose=verb_level)(
            delayed(_find)(jid) for jid in range(1, len(self) + 1)
        )
        values = np.asarray(results, dtype=int).max(axis=0)

        for i, prop in enumerate(['nij_max', 'nijk_max', 'nnl_max']):
            nested_set(self._metadata, _get_keypath(k_max, rc, prop), values[i])
            if k_max == 3:
                if prop == 'nijk_max':
                    val = 0
                else:
                    val = values[i]
                nested_set(self._metadata,
                           _get_keypath(k_max=2, rc=rc, prop=prop),
                           val)
        self._write_metadata()

        if verbose:
            print(f'All {self} jobs are done. nij_max = {values[0]}, '
                  f'nijk_max = {values[1]}, nnl_max = {values[2]}')

        return {'nij_max': values[0],
                'nijk_max': values[1],
                'nnl_max': values[2]}


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


def _get_updated_metadata(database: SQLite3Database, k_max: int, rc: float,
                          nij_max: int, nijk_max: int, nnl_max: int):
    """
    Return the updated metadata dict.
    """
    k_max = convert_k_max_to_key(k_max)
    rc = convert_rc_to_key(rc)
    details = {k_max: {rc: {
        'nij_max': nij_max, 'nijk_max': nijk_max, 'nnl_max': nnl_max}}}
    metadata = database.metadata
    if 'neighbors' not in metadata:
        metadata['neighbors'] = details
    elif k_max not in metadata['neighbors']:
        metadata['neighbors'][k_max] = details[k_max]
    else:
        metadata['neighbors'][k_max][rc] = details[k_max][rc]
    return metadata


def find_neighbor_size_maximums(database: SQLite3Database,
                                rc: float,
                                k_max=3,
                                n_jobs=-1,
                                verbose=True):
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
        return find_neighbor_size_of_atoms(
            database.get_atoms(f'id={aid}'), rc, k_max)

    if verbose:
        print('Start finding neighbors for rc = {} and k_max = {}. This may '
              'take a very long time.'.format(rc, k_max))

    verb_level = 5 if verbose else 0
    results = Parallel(n_jobs=n_jobs, verbose=verb_level)(
        delayed(_find_wrapper)(jid) for jid in range(1, len(database) + 1)
    )

    nij_max, nijk_max, nnl_max = np.asarray(
        results, dtype=int).max(axis=0).tolist()

    database.metadata = _get_updated_metadata(
        database,
        k_max=k_max,
        rc=rc,
        nij_max=nij_max,
        nijk_max=nijk_max,
        nnl_max=nnl_max)

    if k_max == 3:
        database.metadata = _get_updated_metadata(
            database,
            k_max=2,
            rc=rc,
            nij_max=nij_max,
            nijk_max=0,
            nnl_max=nnl_max)

    if verbose:
        print(f'All {len(database)} jobs are done. nij_max = {nij_max}, '
              f'nijk_max = {nijk_max}, nnl_max = {nnl_max}')

    return nij_max, nijk_max, nnl_max
