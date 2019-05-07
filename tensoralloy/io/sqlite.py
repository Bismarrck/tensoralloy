# coding=utf-8
"""
This module defines functions to detect the limits of the neighbor lists.
"""
from __future__ import print_function, absolute_import

import numpy as np
import json

from ase.db.sqlite import SQLite3Database
from joblib import Parallel, delayed
from collections import Counter
from typing import Dict, List

from tensoralloy.utils import nested_get, nested_set
from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.neighbor import NeighborSize, NeighborProperty

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["CoreDatabase"]


def _get_keypath(k_max: int, rc: float, prop: NeighborProperty):
    """
    A helper function to get the corresponding keypath.
    """
    pm = '{:.0f}'.format(rc * 100.0)
    return f"neighbors.{k_max}.{pm}.{prop.name}_max"


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

    def _get_neighbor_property(self,
                               rc: float,
                               prop: NeighborProperty,
                               allow_calculation=False):
        """
        A helper function to get the value of a neighbor property.

        Availabel properties:
            * nnl_max
            * nij_max
            * nijk_max

        """
        for k_max in (2, 3):
            keypath = _get_keypath(k_max, rc, prop)
            val = nested_get(self._metadata, keypath)
            if val is not None:
                return val
        if allow_calculation:
            dct = self.update_neighbor_meta(rc=rc, angular=True, verbose=True)
            val = dct[prop]
        return val

    def get_nij_max(self, rc: float, allow_calculation=False):
        """
        Return the corresponding `N_ij_max`.
        """
        return self._get_neighbor_property(rc=rc,
                                           prop=NeighborProperty.nij,
                                           allow_calculation=allow_calculation)

    def get_nijk_max(self, rc: float, allow_calculation=False):
        """
        Return the corresponding `N_ijk_max`.
        """
        return self._get_neighbor_property(rc=rc,
                                           prop=NeighborProperty.nijk,
                                           allow_calculation=allow_calculation)

    def get_nnl_max(self, rc: float, allow_calculation=False):
        """
        Return the corresponding `N_nl_max`.
        """
        return self._get_neighbor_property(rc=rc,
                                           prop=NeighborProperty.nnl,
                                           allow_calculation=allow_calculation)

    def update_neighbor_meta(self,
                             rc: float,
                             angular=False,
                             n_jobs=-1,
                             verbose=False) -> NeighborSize:
        """
        Update the metadata of neighbor properties.
        """
        def _find(aid):
            nl = find_neighbor_size_of_atoms(
                self.get_atoms(f'id={aid}'), rc, angular=angular)
            return nl.nnl, nl.nij, nl.nijk

        if angular:
            cond = 'enabled'
            k_max = 3
        else:
            cond = 'disabled'
            k_max = 2

        if verbose:
            print('Start finding neighbors for rc = {} and angular {}. This '
                  'may take a very long time.'.format(rc, cond))
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

        # noinspection PyTypeChecker
        for i, prop in enumerate(NeighborProperty):
            nested_set(self._metadata, _get_keypath(k_max, rc, prop), values[i])
            if angular:
                if prop == NeighborProperty.nijk:
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

        return NeighborSize(nij=values[0], nijk=values[1], nnl=values[2])


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
