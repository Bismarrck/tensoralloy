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
from collections import Counter
from os.path import splitext, exists
from os import remove


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# The regex pattern for parsing the second line.
energy_patt = re.compile(r"Lattice=\"(.*)\".*energy=([\d.-]+)\s+pbc=\"(.*)\"")

# The regex pattern for parsing the later lines.
string_patt = re.compile(r"([A-Za-z]{1,2})\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)"
                         r"\s+\d+\s+\d.\d+\s+\d+\s+([\d.-]+)\s+([\d.-]+)\s+"
                         r"([\d.-]+)")


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


def read(filename, num_examples=None, verbose=True, append=False):
    """
    Read `Atoms` objects from a xyz file.`

    Parameters
    ----------
    filename : str
        The xyz file to read.
    num_examples : int
        An `int` indicating the maximum number of examples to read.
    append : bool
        The parsed `Atoms` objects shall be appended to the database.
    verbose : bool
        If True, the reading progress shall be logged.

    Returns
    -------
    db : SQLite3Database
        The database for the given xyz file.

    """
    logstr = "\rProgress: {:7d}  /  {:7d} | Speed = {:.1f}"
    atoms = None
    count = 0
    stage = 0
    ai = 0
    natoms = 0
    max_occurs = Counter()
    dbfile = '{}.db'.format(splitext(filename)[0])
    if exists(dbfile) and not append:
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
                    energy = float(m.group(2))
                    atoms.set_cell(_read_cell(m.group(1)))
                    atoms.set_pbc(_read_pbc(m.group(3)))
                    atoms.info[VirtualCalculator.ENERGY_KEY] = energy
                    stage += 1
            elif stage == 2:
                m = string_patt.search(line)
                if m:
                    floats = [float(v) for v in m.groups()[1: 7]]
                    atoms.append(Atom(symbol=m.group(1),
                                      position=floats[:3]))
                    atoms.info[VirtualCalculator.FORCES_KEY][ai, :] = floats[3:]
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
    database.metadata = {'max_occurs': max_occurs}
    return database


def find_neighbors(db: SQLite3Database, rc: float, n_jobs=-1):
    """
    Update the neighbor lists of all `Atoms` objects in the database.

    Parameters
    ----------
    db : SQLite3Database
        The database to update. This db must be created by the function `read`.
    rc : float
        The cutoff radius.
    n_jobs : int


    """
    pass
