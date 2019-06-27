#!coding=utf-8
"""
Custom XYZ utility functions.
"""
from __future__ import print_function, absolute_import

import numpy as np

from ase import Atoms
from ase.io.extxyz import read_xyz
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.units import Hartree

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def read_stepmax_xyz(xyzfile: str) -> Atoms:
    """
    Parse STEPMAX/XYZ file and return a `Atoms` object.
    """
    def _parser(line: str):
        _splits = line.strip().split()
        _cellpars = [float(x) for x in _splits[1: 7]]
        assert len(_splits) == 8
        assert _splits[-1].lower() == 'cartesian'
        return {'energy': float(_splits[0]) * Hartree,
                'Lattice': np.transpose(cellpar_to_cell(_cellpars))}
    with open(xyzfile) as fp:
        return next(read_xyz(fp, index=0, properties_parser=_parser))


def write_stepmax_xyz(xyzfile: str, atoms: Atoms, energy=None):
    """
    Write a `Atoms` to a STEPMAX/XYZ file.
    """
    if energy is not None:
        _energy = energy
    elif atoms.calc is not None:
        _energy = atoms.get_total_energy()
    else:
        _energy = 0.0
    _energy /= Hartree
    cellpars = ' '.join(
        map(lambda v: "{: 10.6f}".format(v), cell_to_cellpar(atoms.cell)))
    with open(xyzfile, 'w') as fp:
        fp.write(f"{len(atoms)}\n")
        fp.write(f"{_energy} {cellpars} Cartesian\n")
        for atom in atoms:
            fp.write("{:2s} {: 10.6f} {: 10.6f} {: 10.6f}\n".format(
                atom.symbol, *atom.position))
