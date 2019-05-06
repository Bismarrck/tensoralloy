# coding=utf-8
"""
This module defines functions to detect the limits of the neighbor lists.
"""
from __future__ import print_function, absolute_import

import numpy as np
import json
import os

from ase.db.sqlite import SQLite3Database
from ase.db import connect as ase_connect
from ase.utils import PurePath
from ase.parallel import world
from joblib import Parallel, delayed
from collections import Counter
from typing import Dict, List
from os.path import splitext

from tensoralloy.utils import nested_get, nested_set
from tensoralloy.utils import find_neighbor_size_of_atoms

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["connect", "CoreDatabase"]


def _get_keypath(k_max: int, rc: float, prop: str):
    """
    A helper function to get the corresponding keypath.
    """
    pm = '{:.0f}'.format(rc * 100.0)
    return f"neighbors.{k_max}.{pm}.{prop}"


def connect(name, use_lock_file=True, append=True, serial=False):
    """
    Create connection to database.

    Parameters
    ----------
    name: str
        Filename or address of database.
    use_lock_file: bool
        You can turn this off if you know what you are doing ...
    append: bool
        Use append=False to start a new database.
    serial : bool
        Let someone else handle parallelization.  Default behavior is to
        interact with the database on the master only and then distribute
        results to all slaves.

    """
    if isinstance(name, PurePath):
        name = str(name)

    if not append and world.rank == 0:
        if isinstance(name, str) and os.path.isfile(name):
            os.remove(name)

    if name is None:
        db_type = None
    elif not isinstance(name, str):
        db_type = 'json'
    elif (name.startswith('postgresql://') or
          name.startswith('postgres://')):
        db_type = 'postgresql'
    else:
        db_type = splitext(name)[1][1:]
        if db_type == '':
            raise ValueError('No file extension or database type given')

    if db_type == 'db':
        return CoreDatabase(name,
                            create_indices=True,
                            use_lock_file=use_lock_file,
                            serial=serial)
    else:
        return ase_connect(name,
                           type=db_type,
                           create_indices=True,
                           use_lock_file=use_lock_file,
                           append=append,
                           serial=serial)


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

    def __str__(self):
        return f"CoreDataBase@{self.filename}"

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
    def max_occurs(self) -> Counter:
        """
        Return the maximum appearance of each type of element.
        """
        if 'max_occurs' not in self._metadata:
            self._find_max_occurs()
        return Counter(self._metadata.get('max_occurs'))

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
    def has_periodic_structures(self):
        """
        Return True if this database has at least one periodic structure.
        """
        return self._metadata.get('periodic', True)

    def _find_max_occurs(self):
        """
        Update `max_occurs` of this database.
        """
        size = len(self)
        max_occurs = Counter()
        for atoms_id in range(1, 1 + size):
            c = Counter(self.get_atoms(id=atoms_id).get_chemical_symbols())
            for element, n in c.items():
                max_occurs[element] = max(max_occurs[element], n)
        self._metadata['max_occurs'] = max_occurs
        self._write_metadata()

    def get_atomic_static_energy(self,
                                 allow_calculation=False) -> Dict[str, float]:
        """
        Return the calculated atomic static energy dict.
        """
        key = 'atomic_static_energy'
        dct = self._metadata.get(key, {})
        if not dct and allow_calculation:
            elements = sorted(self.max_occurs.keys())
            dct = _compute_atomic_static_energy(self, elements, verbose=True)
            self._metadata[key] = dct
            self._write_metadata()
        return dct

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
            verb = 5
        else:
            verb = 0
        size = len(self)

        results = Parallel(n_jobs=n_jobs,
                           verbose=verb)(
            delayed(_find)(job_id)
            for job_id in range(1, 1 + size)
        )
        values = np.asarray(results, dtype=int).max(axis=0).tolist()

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


def _compute_atomic_static_energy(database: SQLite3Database,
                                  elements: List[str],
                                  verbose=True):
    """
    Compute the static energy for each type of element and add the results to
    the metadata.

    Parameters
    ----------
    database : SQLite3Database
        The database to update. This db must be created by the function `read`.
    elements : List[str]
        A list of str as the ordered elements.
    verbose : bool
        If True, the progress shall be logged.

    """
    n = len(database)
    n_elements = len(elements)
    id_first = 1
    col_map = {element: elements.index(element) for element in elements}
    A = np.zeros((n, n_elements), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    if verbose:
        print("Start computing atomic static energies ...")

    for aid in range(id_first, id_first + n):
        atoms = database.get_atoms(id=aid)
        row = aid - 1
        for element, count in Counter(atoms.get_chemical_symbols()).items():
            A[row, col_map[element]] = float(count)
        b[row] = atoms.get_total_energy()

    rank = np.linalg.matrix_rank(A)
    if rank == n_elements:
        x = np.dot(np.linalg.pinv(A), b)
    elif rank == 1:
        x = np.tile(np.mean(b / A.sum(axis=1)), n_elements)
    else:
        raise ValueError(f"The matrix has an invalid rank of {rank}")

    if verbose:
        print("Done.")
        for i in range(len(elements)):
            print("  * Atomic Static Energy of {:2s} = {: 12.6f}".format(
                elements[i], x[i]))
        print("")

    return {elements[i]: x[i] for i in range(len(elements))}
