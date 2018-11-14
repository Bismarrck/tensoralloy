#!/usr/bin/env python3
# coding=utf-8
"""
This module defines data IO functions and classes.
"""
from __future__ import print_function, absolute_import

import numpy as np
import time
import sys
import re
import ase.units
from ase.db import connect
from ase.db.sqlite import SQLite3Database
from ase.neighborlist import neighbor_list
from ase.io.extxyz import read_xyz
from collections import Counter
from os.path import splitext
from joblib import Parallel, delayed
from argparse import ArgumentParser
from typing import Dict, List


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_conversion(units: Dict[str, str]) -> (float, float):
    """
    Return the conversion factors.
    """
    def _parse_comb(unit: str):
        if '/' in unit or '*' in unit:
            values = [getattr(ase.units, name)
                      for name in re.split(r'[/*]', unit)]
            total = values[0]
            index = 1
            for i in range(len(unit)):
                if unit[i] == '/':
                    total /= values[index]
                    index += 1
                elif unit[i] == '*':
                    total *= values[index]
                    index += 1
                if index == len(values):
                    break
            return total
        else:
            return getattr(ase.units, unit)

    eV = ase.units.eV
    Angstrom = ase.units.Angstrom

    if 'energy' not in units:
        to_eV = eV
    else:
        to_eV = _parse_comb(units['energy']) / eV

    if 'forces' not in units:
        to_eV_Angstrom = eV / Angstrom
    else:
        to_eV_Angstrom = _parse_comb(units['forces']) / eV / Angstrom

    return to_eV, to_eV_Angstrom


def _read_extxyz(filename, units, ext=True, num_examples=None,
                 verbose=True):
    """
    Read `Atoms` objects from a `xyz` or an `extxyz` file.

    Parameters
    ----------
    filename : str
        The xyz file to read.
    units : Dict[str, str]
        A dict of str as the units of the properties in the file. Supported keys
        are 'energy' and 'forces'.
    ext : bool
        The file is in `extxyz` format if True.
    num_examples : int
        An `int` indicating the maximum number of examples to read.
    verbose : bool
        If True, the reading progress shall be logged.

    Returns
    -------
    database : SQLite3Database
        The database for the given xyz file.

    """
    to_eV, to_eV_Angstrom = get_conversion(units)
    count = 0
    max_occurs = Counter()
    database = connect(name='{}.db'.format(splitext(filename)[0]),
                       append=False)

    tic = time.time()
    if verbose:
        sys.stdout.write("Extract cartesian coordinates ...\n")

    with open(filename) as fp:
        index = slice(0, num_examples, 1)
        if ext:
            reader = read_xyz(fp, index)
        else:
            def _parser(line):
                return {'energy': float(line.strip())}
            reader = read_xyz(fp, index, properties_parser=_parser)

        for atoms in reader:
            # Make sure all energies are in 'eV' and all forces are in
            # 'eV/angstroms'
            atoms.calc.results['energy'] *= to_eV
            if ext:
                atoms.calc.results['forces'] *= to_eV_Angstrom
            else:
                atoms.calc.results['forces'] = np.zeros_like(atoms.positions)
            database.write(atoms)
            count += 1

            for symbol, n in Counter(atoms.get_chemical_symbols()).items():
                max_occurs[symbol] = max(max_occurs[symbol], n)

            if verbose and (count + 1) % 100 == 0:
                speed = (count + 1) / (time.time() - tic)
                total = num_examples or -1
                sys.stdout.write(
                    "\rProgress: {:7d} / {:7d} | Speed = {:.1f}".format(
                        count + 1, total, speed))
        if verbose:
            print("")
            print("Total {} structures, time: {:.3f} sec".format(
                count, time.time() - tic))

    database.metadata = {'max_occurs': max_occurs, 'extxyz': ext,
                         'unit_conversion': {
                             'energy': to_eV,
                             'forces': to_eV_Angstrom}
                         }
    return database


def read(filename, units=None, num_examples=None, verbose=True):
    """
    Read `Atoms` objects from a file.

    Parameters
    ----------
    filename : str
        The file to read. Can be a `xyz` file, a `extxyz` file or a `db` file.
    units : Dict[str, str]
        A dict of str as the units of the properties ('energy', 'forces') in the
        file. Defaults to 'eV' for 'energy' and 'eV/Angstrom' for 'forces'.
    num_examples : int
        An `int` indicating the maximum number of examples to read.
    verbose : bool
        If True, the reading progress shall be logged.

    Returns
    -------
    database : SQLite3Database
        The database for the given xyz file.

    """
    file_type = splitext(filename)[1][1:]

    if units is None:
        units = {'energy': 'eV', 'forces': 'eV/Angstrom'}

    if file_type == 'db':
        database = connect(filename)
        validated_keys = ('max_occurs', 'ext')
        for key in validated_keys:
            if key not in database.metadata:
                print("Warning: the key '{}' is missing!".format(key))
        return database

    elif file_type == 'extxyz':
        return _read_extxyz(filename, units, True, num_examples, verbose)

    elif file_type == 'xyz':
        return _read_extxyz(filename, units, False, num_examples, verbose)

    else:
        raise ValueError("Unknown file type: {}".format(file_type))


def convert_k_max_to_key(k_max):
    """ Convert `k_max` to a valid key.  """
    return "{}".format(k_max)


def convert_rc_to_key(rc):
    """ Convert `rc` to a valid key. """
    return "{:.2f}".format(round(rc, 4))


def find_neighbor_size_limits(database: SQLite3Database, rc: float,
                              k_max: int=3, n_jobs=-1, verbose=True):
    """
    Find `nij_max` and `nijk_max` of all `Atoms` objects in the database.

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
    def _pipeline(aid):
        atoms = database.get_atoms(id=aid)
        ilist, jlist = neighbor_list('ij', atoms, cutoff=rc)
        if k_max >= 2:
            nij = len(ilist)
        else:
            symbols = atoms.get_chemical_symbols()
            nij = 0
            for k in range(len(ilist)):
                if symbols[ilist[k]] == symbols[jlist[k]]:
                   nij += 1
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
        return nij, nijk

    if verbose:
        print('Start finding neighbors for rc = {} and k_max = {}. This may '
              'take a very long time.'.format(rc, k_max))

    verb_level = 5 if verbose else 0
    results = Parallel(n_jobs=n_jobs, verbose=verb_level)(
        delayed(_pipeline)(jid) for jid in range(1, len(database) + 1)
    )

    nij_max, nijk_max = np.asarray(results, dtype=int).max(axis=0).tolist()
    rc = convert_rc_to_key(rc)
    k_max = convert_k_max_to_key(k_max)
    details = {k_max: {rc: {'nij_max': nij_max, 'nijk_max': nijk_max}}}
    metadata = dict(database.metadata)
    if 'neighbors' not in metadata:
        metadata['neighbors'] = details
    elif k_max not in metadata['neighbors']:
        metadata['neighbors'][k_max] = details[k_max]
    else:
        metadata['neighbors'][k_max][rc] = details[k_max][rc]
    database.metadata = metadata

    if verbose:
        print('All {} jobs are done. nij_max = {}, nijk_max = {}'.format(
            len(database), nij_max, nijk_max))


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


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        'filename',
        type=str,
        help="Specify the xyz or extxyz file to read.",
    )
    parser.add_argument(
        '-n', '--num-examples',
        default=None,
        type=int,
        help="Set the maximum number of examples to read."
    )
    parser.add_argument(
        '--energy-unit',
        type=str,
        default='eV',
        choices=('eV', 'Hartree', 'kcal/mol'),
        help='The unit of the energies in the file'
    )
    parser.add_argument(
        '--forces-unit',
        type=str,
        default='eV/Angstrom',
        choices=['kcal/mol/Angstrom', 'kcal/mol/Bohr', 'eV/Bohr', 'eV/Angstrom',
                 'Hartree/Bohr', 'Hartree/Angstrom'],
        help='The unit of the atomic forces in the file.'
    )
    args = parser.parse_args()
    read(args.filename,
         units={'energy': args.energy_unit,
                'forces': args.forces_unit},
         num_examples=args.num_examples,
         verbose=True)
