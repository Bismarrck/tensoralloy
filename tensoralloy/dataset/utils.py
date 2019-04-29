# coding=utf-8
"""
This module defines utility functions for constructing datasets.
"""
from __future__ import print_function, absolute_import

import numpy as np
import re
import platform

from subprocess import PIPE, Popen
from collections import Counter
from typing import List
from ase.db.sqlite import SQLite3Database

from tensoralloy.utils import get_pulay_stress

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def brange(start, stop, batch_size):
    """
    Range from `start` to `stop` given a batch size and return the start and
    stop of each batch.

    Parameters
    ----------
    start : int
        The start number of a sequence.
    stop : int,
        The end number of a sequence.
    batch_size : int
        The size of each batch.

    Yields
    ------
    istart : int
        The start number of a batch.
    istop : int
        The end number of a batch.

    """
    istart = start
    while istart < stop:
        istop = min(istart + batch_size, stop)
        yield istart, istop
        istart = istop


def should_be_serial():
    """
    Return True if the dataset should be in serial mode.

    For macOS this function always return False.
    For Linux if `glibc>=2.17` return False; otherwise return True.

    """
    if platform.system() == 'Darwin':
        return False

    pattern = re.compile(r'^GLIBC_2.([\d.]+)')
    p = Popen('strings /lib64/libc.so.6 | grep GLIBC_2.',
              shell=True, stdout=PIPE, stderr=PIPE)
    (stdout, stderr) = p.communicate()
    if stderr.decode('utf-8') != '':
        return True
    glibc_ver = 0.0
    for line in stdout.decode('utf-8').split('\n'):
        m = pattern.search(line)
        if m:
            glibc_ver = max(float(m.group(1)), glibc_ver)
    if glibc_ver >= 17.0:
        return False
    else:
        return True


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
