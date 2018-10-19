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


def _read_extxyz(filename, ext=True, num_examples=None, verbose=True):
    """
    Read `Atoms` objects from a extxyz file.

    Parameters
    ----------
    filename : str
        The xyz file to read.
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
                        energy = float(m.group(2))
                        atoms.set_cell(_read_cell(m.group(1)))
                        atoms.set_pbc(_read_pbc(m.group(3)))
                    else:
                        energy = float(m.group(1))
                        atoms.set_pbc([False, False, False])
                    atoms.info[VirtualCalculator.ENERGY_KEY] = energy
                    stage += 1
            elif stage == 2:
                m = string_patt.search(line)
                if m:
                    if ext:
                        floats = [float(v) for v in m.groups()[1: 7]]
                        forces = floats[3:]
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
                        if verbose and count % 1000 == 0:
                            speed = count / (time.time() - tic)
                            sys.stdout.write(logstr.format(
                                count, num_examples, speed))
        if verbose:
            print("")
            print("Total time: %.3f s\n" % (time.time() - tic))
    database.metadata = {'max_occurs': max_occurs, 'extxyz': ext}
    return database


def read(filename, num_examples=None, verbose=True):
    """
    Read `Atoms` objects from a file.

    Parameters
    ----------
    filename : str
        The file to read. Can be a `xyz` file, a `extxyz` file or a `db` file.
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
                            verbose=verbose)
    elif file_type == 'xyz':
        return _read_extxyz(filename, ext=False, num_examples=num_examples,
                            verbose=verbose)
    else:
        raise ValueError("Unknown file type: {}".format(file_type))


def find_neighbor_sizes(database: SQLite3Database, rc: float, n_jobs=-1):
    """
    Find `nij_max` and `nijk_max` of all `Atoms` objects in the database.

    Parameters
    ----------
    database : SQLite3Database
        The database to update. This db must be created by the function `read`.
    rc : float
        The cutoff radius.
    n_jobs : int
        The maximum number of concurrently running jobs. If -1 all CPUs are
        used. If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    """

    def _pipeline(aid):
        atoms = database.get_atoms(id=aid)
        ilist, jlist = neighbor_list('ij', atoms, cutoff=rc)
        nij = len(ilist)
        nl = {}
        for i, atomi in enumerate(ilist):
            nl[atomi] = nl.get(atomi, []) + [jlist[i]]
        nijk = 0
        for atomi, nlist in nl.items():
            n = len(nlist)
            nijk += (n - 1 + 1) * (n - 1) // 2
        return nij, nijk

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_pipeline)(jid) for jid in range(1, len(database) + 1)
    )
    nij_max, nijk_max = np.asarray(results, dtype=int).max(axis=0).tolist()
    metadata = database.metadata
    metadata.update({'nij_max': nij_max, 'nijk_max': nijk_max, 'rc': rc})
    database.metadata = metadata
