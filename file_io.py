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
from ase import Atoms, Atom
from ase.db import connect
from ase.db.sqlite import SQLite3Database
from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list
from collections import Counter
from os.path import splitext, exists
from os import remove
from joblib import Parallel, delayed
from argparse import ArgumentParser
from typing import Dict
from misc import safe_select


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class VirtualCalculator(Calculator):
    """
    A virtual calculator just returning the provided energy and forces.
    """
    implemented_properties = ["energy", "forces"]

    ENERGY_KEY = 'virt_energy'
    FORCES_KEY = 'virt_forces'

    def __init__(self, atoms=None):
        """
        Initialization method.
        """
        Calculator.__init__(self, label="virtual", atoms=atoms)

    def set_atoms(self, atoms):
        """
        Set the attached `ase.Atoms` object.
        """
        self.atoms = atoms

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """
        Set the calculation results.
        """
        super(VirtualCalculator, self).calculate(atoms, properties=properties,
                                                 system_changes=system_changes)
        zeros = np.zeros((len(self.atoms), 3))
        self.results = {
            'energy': self.atoms.info.get(self.ENERGY_KEY, 0.0),
            'forces': self.atoms.info.get(self.FORCES_KEY, zeros),
        }


def _read_cell(string: str):
    """
    Read the periodic boundary cell.
    """
    return np.reshape([float(x) for x in string.split()], (3, 3))


def _read_pbc(string: str):
    """
    Read the periodic conditions.
    """
    return [True if x == "T" else False for x in string.split()]


def _read_extxyz(filename, ext=True, unit_convertion=None, num_examples=None,
                 verbose=True):
    """
    Read `Atoms` objects from a `xyz` or an `extxyz` file.

    Parameters
    ----------
    filename : str
        The xyz file to read.
    ext : bool
        The file is in `extxyz` format if True.
    unit_convertion : Dict[str, float]
        A dict of units. Supported keys are 'energy', 'forces' and 'stress'.
    num_examples : int
        An `int` indicating the maximum number of examples to read.
    verbose : bool
        If True, the reading progress shall be logged.

    Returns
    -------
    database : SQLite3Database
        The database for the given xyz file.

    """
    if ext:
        energy_patt = re.compile(r"Lattice=\"(.*)\".*"
                                 r"energy=([\d.-]+)\s+pbc=\"(.*)\"")
        string_patt = re.compile(r"([A-Za-z]{1,2})\s+([\d.-]+)\s+([\d.-]+)"
                                 r"\s+([\d.-]+)\s+\d+\s+\d.\d+\s+\d+\s+"
                                 r"([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)")
    else:
        energy_patt = re.compile(r"([\d.-]+)")
        string_patt = re.compile(r"([A-Za-z]{1,2})\s+"
                                 r"([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)")

    logstr = "\rProgress: {:7d}  /  {:7d} | Speed = {:.1f}"
    unit_convertion = safe_select(unit_convertion, {})
    energy_unit = unit_convertion.get('energy', 1.0)
    forces_unit = unit_convertion.get('forces', 1.0)
    atoms = None
    count = 0
    stage = 0
    ai = 0
    natoms = 0
    max_occurs = Counter()
    dbfile = '{}.db'.format(splitext(filename)[0])
    if exists(dbfile):
        remove(dbfile)
    database = connect(name=dbfile)
    tic = time.time()
    if verbose:
        sys.stdout.write("Extract cartesian coordinates ...\n")
    with open(filename) as f:
        for line in f:
            if num_examples and count == num_examples:
                break
            line = line.strip()
            if line == "":
                continue
            if stage == 0:
                if line.isdigit():
                    natoms = int(line)
                    zeros = np.zeros((natoms, 3))
                    atoms = Atoms(calculator=VirtualCalculator())
                    atoms.info[VirtualCalculator.FORCES_KEY] = zeros
                    stage += 1
            elif stage == 1:
                m = energy_patt.search(line)
                if m:
                    if ext:
                        energy = float(m.group(2)) * energy_unit
                        atoms.set_cell(_read_cell(m.group(1)))
                        atoms.set_pbc(_read_pbc(m.group(3)))
                    else:
                        energy = float(m.group(1)) * energy_unit
                        atoms.set_pbc([False, False, False])
                        side_length = 20.0 + (divmod(natoms, 50)[0] * 5.0)
                        atoms.set_cell(np.eye(3) * side_length)
                    atoms.info[VirtualCalculator.ENERGY_KEY] = energy
                    stage += 1
            elif stage == 2:
                m = string_patt.search(line)
                if m:
                    if ext:
                        floats = [float(v) for v in m.groups()[1: 7]]
                        forces = [v * forces_unit for v in floats[3:]]
                        atoms.info[VirtualCalculator.FORCES_KEY][ai, :] = forces
                    else:
                        floats = [float(v) for v in m.groups()[1: 4]]
                    atoms.append(Atom(symbol=m.group(1), position=floats[:3]))
                    ai += 1
                    if ai == natoms:
                        atoms.calc.calculate()
                        database.write(atoms)
                        counter = Counter(atoms.get_chemical_symbols())
                        for symbol, n in counter.items():
                            max_occurs[symbol] = max(max_occurs[symbol], n)
                        ai = 0
                        stage = 0
                        count += 1
                        if verbose and count % 100 == 0:
                            speed = count / (time.time() - tic)
                            total = num_examples or -1
                            sys.stdout.write(logstr.format(count, total, speed))
        if verbose:
            print("")
            print("Total {} structures, time: {:.3f} sec".format(
                count, time.time() - tic))
    database.metadata = {'max_occurs': max_occurs, 'extxyz': ext,
                         'unit_conversion': {
                             'energy': energy_unit,
                             'forces': forces_unit}
                         }
    return database


def read(filename, unit_conversion=None, num_examples=None, verbose=True):
    """
    Read `Atoms` objects from a file.

    Parameters
    ----------
    filename : str
        The file to read. Can be a `xyz` file, a `extxyz` file or a `db` file.
    unit_conversion : Dict[str, float]
        A dict of units. Supported keys are 'energy', 'forces' and 'stress'.
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

    if file_type == 'db':
        database = connect(filename)
        validated_keys = ('max_occurs', 'ext')
        for key in validated_keys:
            if key not in database.metadata:
                print("Warning: the key '{}' is missing!".format(key))
        return database
    elif file_type == 'extxyz':
        return _read_extxyz(filename, ext=True, num_examples=num_examples,
                            unit_convertion=unit_conversion, verbose=verbose)
    elif file_type == 'xyz':
        return _read_extxyz(filename, ext=False, num_examples=num_examples,
                            unit_convertion=unit_conversion, verbose=verbose)
    else:
        raise ValueError("Unknown file type: {}".format(file_type))


def convert_k_max_to_key(k_max):
    """ Convert `k_max` to a valid key.  """
    return "{}".format(k_max)


def convert_rc_to_key(rc):
    """ Convert `rc` to a valid key. """
    return "{:.2f}".format(round(rc, 4))


def find_neighbor_sizes(database: SQLite3Database, rc: float, k_max: int=3,
                        n_jobs=-1, verbose=True):
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

    results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
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
        type=float,
        default=1.0,
        help='The energy conversion unit.'
    )
    parser.add_argument(
        '--forces-unit',
        type=float,
        default=1.0,
        help='The forces conversion unit.'
    )
    args = parser.parse_args()
    read(args.filename,
         unit_conversion={'energy': args.energy_unit,
                          'forces': args.forces_unit},
         num_examples=args.num_examples,
         verbose=True)
