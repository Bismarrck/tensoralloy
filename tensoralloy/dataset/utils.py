# coding=utf-8
"""
This module defines utility functions for constructing datasets.
"""
from __future__ import print_function, absolute_import

from collections import Counter
from typing import List

import numpy as np
from ase.db.sqlite import SQLite3Database

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def compute_atomic_static_energy(database: SQLite3Database,
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

    metadata = dict(database.metadata)
    metadata["atomic_static_energy"] = {elements[i]: x[i]
                                        for i in range(len(elements))}
    database.metadata = metadata
