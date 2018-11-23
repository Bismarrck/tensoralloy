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
from utils import cantor_pairing


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_conversion(units: Dict[str, str]) -> (float, float, float):
    """
    Return the conversion factors:
        * 'energy' should be converted to 'eV'
        * 'forces' should be converted to 'eV / Angstrom'
        * 'stress' should be converted to 'GPa'

    """
    _units = {
        'kbar': 0.1 * ase.units.GPa,
    }

    def _parse_unit(unit: str):
        if hasattr(ase.units, unit):
            return getattr(ase.units, unit)
        elif unit in _units:
            return _units[unit]
        else:
            try:
                return float(unit)
            except Exception:
                raise ValueError("Unknown unit: {}".format(unit))

    def _parse_comb(comb: str):
        if '/' in comb or '*' in comb:
            values = [_parse_unit(unit) for unit in re.split(r'[/*]', comb)]
            total = values[0]
            index = 1
            for i in range(len(comb)):
                if comb[i] == '/':
                    total /= values[index]
                    index += 1
                elif comb[i] == '*':
                    total *= values[index]
                    index += 1
                if index == len(values):
                    break
            return total
        else:
            return getattr(ase.units, comb)

    eV = ase.units.eV
    Angstrom = ase.units.Angstrom
    GPa = ase.units.GPa

    if 'energy' not in units:
        to_eV = 1.0
    else:
        to_eV = _parse_comb(units['energy']) / eV

    if 'forces' not in units:
        to_eV_Angstrom = 1.0
    else:
        to_eV_Angstrom = _parse_comb(units['forces']) / eV / Angstrom

    if 'stress' not in units:
        to_GPa = 1.0
    else:
        to_GPa = _parse_comb(units['stress']) / GPa

    return to_eV, to_eV_Angstrom, to_GPa


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
    to_eV, to_eV_Angstrom, to_GPa = get_conversion(units)
    count = 0
    max_occurs = Counter()
    stress = None
    periodic = False
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
            # The default parser for normal xyz files will ignore the energies.
            # So here we implement a single parser just converting the second
            # line of each XYZ block to a float.
            def _parser(line):
                return {'energy': float(line.strip())}
            reader = read_xyz(fp, index, properties_parser=_parser)

        for atoms in reader:

            # Make sure the property `cell` is a non-zero matrix or set it based
            # the size of the `Atoms`.
            if np.abs(atoms.cell).sum() < 1e-8:
                length = 20.0 + (divmod(len(atoms), 50)[0] * 5.0)
                cell = np.eye(3) * length
                # To pass `calc.check_state` both must be set because 'cell' is
                # included in `all_changes`.
                atoms.cell, atoms.calc.atoms.cell = cell, cell

            # Scale the energies, forces and stress tensors to make sure
            # energies are in 'eV', forces in 'eV/Angstrom' and stress in 'kB'.
            atoms.calc.results['energy'] *= to_eV

            if ext:
                atoms.calc.results['forces'] *= to_eV_Angstrom
            else:
                # Structures without forces are considered to be local minima so
                # we manually set the forces to zeros.
                atoms.calc.results['forces'] = np.zeros_like(atoms.positions)

            # Check if the stress tensor is included in the results.
            if stress is None:
                stress = bool('stress' in atoms.calc.results)
            if stress:
                # Convert the unit of stress tensors to 'GPa':
                # 1 eV/Angstrom**3 = 160.21766208 GPa
                # 1 GPa = 10 kbar
                atoms.calc.results['stress'] *= to_GPa

            # `periodic` will be set to True if any of the `Atoms` is periodic.
            periodic = any(atoms.pbc) or periodic

            # Write the `Atoms` object to the database.
            database.write(atoms)
            count += 1

            # Update the dict of `max_occurs` and print the parsing progress.
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

    database.metadata = {
        'max_occurs': max_occurs,
        'extxyz': ext,
        'forces': True,
        'stress': stress,
        'periodic': periodic,
        'unit_conversion': {'energy': to_eV,
                            'forces': to_eV_Angstrom,
                            'stress': to_GPa}}
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


def _find_sizes(atoms, rc, k_max):
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


def find_neighbor_size_limits(database: SQLite3Database, rc: float,
                              k_max: int=3, n_jobs=-1, verbose=True):
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

    def _find(aid):
        return _find_sizes(database.get_atoms(f'id={aid}'), rc, k_max)

    if verbose:
        print('Start finding neighbors for rc = {} and k_max = {}. This may '
              'take a very long time.'.format(rc, k_max))

    verb_level = 5 if verbose else 0
    results = Parallel(n_jobs=n_jobs, verbose=verb_level)(
        delayed(_find)(jid) for jid in range(1, len(database) + 1)
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
    parser.add_argument(
        '--stress-unit',
        type=str,
        default='GPa',
        choices=['GPa', 'kbar'],
        help='The unit of the stress tensors in the file.',
    )
    args = parser.parse_args()
    read(args.filename,
         units={'energy': args.energy_unit,
                'forces': args.forces_unit,
                'stress': args.stress_unit},
         num_examples=args.num_examples,
         verbose=True)
