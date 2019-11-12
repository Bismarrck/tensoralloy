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
from ase.io.extxyz import read_xyz
from ase.geometry import cellpar_to_cell
from ase.units import Hartree
from enum import Enum

from tensoralloy.io.sqlite import CoreDatabase
from tensoralloy.io.db import connect
from tensoralloy.io.units import get_conversion_units

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class XyzFormat(Enum):
    """
    Differrent xyz formats.
    """
    normal = 0
    ext = 1
    stepmax = 2


def _read_extxyz(filename, units, xyz_format=XyzFormat.ext, num_examples=None,
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
    xyz_format : XyzFormat
        The format of the xyz file.
    num_examples : int
        An `int` indicating the maximum number of examples to read.
    verbose : bool
        If True, the reading progress shall be logged.

    Returns
    -------
    database : CoreDatabase
        The database for the given xyz file.

    """
    to_eV, to_eV_Angstrom, to_eV_Ang3 = get_conversion_units(units)
    count = 0
    max_occurs = Counter()
    use_stress = None
    periodic = False
    database = connect(name='{}.db'.format(splitext(filename)[0]),
                       append=False)

    tic = time.time()
    if verbose:
        sys.stdout.write("Extract cartesian coordinates ...\n")

    with open(filename) as fp:
        index = slice(0, num_examples, 1)
        if xyz_format == XyzFormat.ext:
            reader = read_xyz(fp, index)
        elif xyz_format == XyzFormat.normal:
            # The default parser for normal xyz files will ignore the energies.
            # So here we implement a single parser just converting the second
            # line of each XYZ block to a float.
            def _parser(line: str):
                return {'energy': float(line.strip())}
            reader = read_xyz(fp, index, properties_parser=_parser)

        else:
            # Stepmax xyz files
            # The second lines contains the energy (a.u.), cell pararameters and
            # a label (should be 'Cartesian').
            def _parser(line: str):
                _splits = line.strip().split()
                _cellpars = [float(x) for x in _splits[1: 7]]
                assert len(_splits) == 8
                assert _splits[-1].lower() == 'cartesian'
                return {'energy': float(_splits[0]) * Hartree,
                        'Lattice': np.transpose(cellpar_to_cell(_cellpars))}
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

            if xyz_format == XyzFormat.ext:
                atoms.calc.results['forces'] *= to_eV_Angstrom
            else:
                # Structures without forces are considered to be local minima so
                # we manually set the forces to zeros.
                atoms.calc.results['forces'] = np.zeros_like(atoms.positions)

            # Check if the stress tensor is included in the results.
            if use_stress is None:
                use_stress = bool('stress' in atoms.calc.results)
            if use_stress:
                # Convert the unit of stress tensors to 'eV / Angstrom**3':
                # 1 eV/Angstrom**3 = 160.21766208 GPa
                # 1 GPa = 10 kbar
                atoms.calc.results['stress'] *= to_eV_Ang3

            # `periodic` will be set to True if any of the `Atoms` is periodic.
            periodic = any(atoms.pbc) or periodic

            # Write the `Atoms` object to the database.
            weights = atoms.info.get('weights', np.ones(3))
            assert len(weights) == 3
            source = atoms.info.get('source', '')
            key_value_pairs = {'source': source}
            if use_stress:
                pulay_stress = atoms.info.get('pulay_stress', 0.0) * to_eV_Ang3
                key_value_pairs['pulay_stress'] = pulay_stress

            database.write(atoms,
                           data={'weights': weights},
                           key_value_pairs=key_value_pairs)
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
        'extxyz': xyz_format == XyzFormat.ext,
        'forces': True,
        'stress': use_stress,
        'periodic': periodic,
        'unit_conversion': {'energy': to_eV,
                            'forces': to_eV_Angstrom,
                            'stress': to_eV_Ang3}}
    return database


def read_file(filename, units=None, num_examples=None, file_type=None,
              verbose=True):
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
    file_type : str
        The type of the file. 'db', 'extxyz', 'xyz' or 'stepmax' are supported.
    verbose : bool
        If True, the reading progress shall be logged.

    Returns
    -------
    database : CoreDatabase
        The database for the given xyz file.

    """
    if file_type is None:
        file_type = splitext(filename)[1][1:]

    if units is None:
        units = {'energy': 'eV', 'forces': 'eV/Angstrom'}

    if file_type == 'db':
        database = connect(filename)
        validated_keys = ('max_occurs', 'extxyz')
        for key in validated_keys:
            if key not in database.metadata:
                print("Warning: the key '{}' is missing!".format(key))
        return database

    elif file_type == 'extxyz':
        return _read_extxyz(
            filename, units, XyzFormat.ext, num_examples, verbose)

    elif file_type == 'xyz':
        return _read_extxyz(
            filename, units, XyzFormat.normal, num_examples, verbose)

    elif file_type == 'stepmax':
        return _read_extxyz(
            filename, units, XyzFormat.stepmax, num_examples, verbose)

    else:
        raise ValueError("Unknown file type: {}".format(file_type))
