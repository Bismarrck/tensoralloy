# coding=utf-8
"""
This module defines file-io functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import sys
import time

from collections import Counter
from os.path import splitext
from ase.db import connect
from ase.db.sqlite import SQLite3Database
from ase.io.extxyz import read_xyz
from argparse import ArgumentParser, Namespace

from tensoralloy.io.units import get_conversion_units

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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
    to_eV, to_eV_Angstrom, to_eV_Ang3 = get_conversion_units(units)
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
                # Convert the unit of stress tensors to 'eV / Angstrom**3':
                # 1 eV/Angstrom**3 = 160.21766208 GPa
                # 1 GPa = 10 kbar
                atoms.calc.results['stress'] *= to_eV_Ang3

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
                            'stress': to_eV_Ang3}}
    return database


def read_file(filename, units=None, num_examples=None, verbose=True):
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


def config_parser(parser: ArgumentParser):
    """
    Return an `ArgumentParser` for executing this module directly.
    """
    parser.add_argument(
        'filename',
        type=str,
        help="Specify the xyz or extxyz file to read.",
    )
    parser.add_argument(
        '--num-examples',
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
        default='eV/Angstrom**3',
        choices=['GPa', 'kbar', 'eV/Angstrom**3'],
        help='The unit of the stress tensors in the file.',
    )
    return parser


def main(args: Namespace):
    """
    The main function.
    """
    read_file(args.filename,
              units={'energy': args.energy_unit,
                     'forces': args.forces_unit,
                     'stress': args.stress_unit},
              num_examples=args.num_examples,
              verbose=True)


if __name__ == "__main__":

    main(config_parser(ArgumentParser()).parse_args())
