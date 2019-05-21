#!coding=utf-8
"""
Lammps I/O helper functions.
"""
from __future__ import print_function, absolute_import

import numpy as np

from dataclasses import dataclass
from typing import List, Dict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["SetFL", "AdpFL", "read_adp_setfl", "read_eam_alloy_setfl"]


@dataclass
class SetFL:
    """
    Representation of a Lammps eam/alloy file.
    """

    elements: List[str]
    rho: Dict[str, np.ndarray]
    phi: Dict[str, np.ndarray]
    frho: Dict[str, np.ndarray]
    nr: int
    dr: float
    nrho: int
    drho: float
    rcut: float
    atomic_masses: List[float]
    lattice_constants: List[float]
    lattice_types: List[str]

    __slots__ = ("elements", "rho", "phi", "frho", "nr", "dr", "nrho", "drho",
                 "rcut", "atomic_masses", "lattice_constants", "lattice_types")


@dataclass
class AdpFL(SetFL):
    """
    Representation of a Lammps adp file.
    """
    dipole: Dict[str, np.ndarray]
    quadrupole: Dict[str, np.ndarray]

    __slots__ = ("elements", "rho", "phi", "frho", "dipole", "quadrupole",
                 "nr", "dr", "nrho", "drho", "rcut", "atomic_masses",
                 "lattice_constants", "lattice_types")


def _read_setfl(filename, is_adp=False):
    """
    Read tabulated rho, phi, F(rho), dipole and quadrupole values from a Lammps
    setfl file.
    """
    with open(filename) as fp:
        rho = {}
        frho = {}
        phi = {}
        dipole = {}
        quadrupole = {}
        n_el = None
        nrho = None
        nr = None
        rcut = None
        base = 6
        elements = None
        curr_element = None
        atomic_masses = []
        lattice_constants = []
        lattice_types = []
        ab = []
        stage = 0
        ab_cycle = 0
        nab = None
        for number, line in enumerate(fp):
            line = line.strip()
            if number < 3:
                continue
            elif number == 3:
                values = line.split()
                assert len(values) >= 2
                n_el = int(values[0])
                elements = [x for x in values[1:]]
            elif number == 4:
                values = line.split()
                assert len(values) == 5
                nrho = int(values[0])
                drho = float(values[1])
                nr = int(values[2])
                dr = float(values[3])
                rcut = float(values[4])
                for i in range(n_el):
                    el_i = elements[i]
                    rho[el_i] = np.zeros((2, nr))
                    rho[el_i][0] = np.linspace(0.0, nr * dr, nr, endpoint=False)
                    frho[el_i] = np.zeros((2, nrho))
                    frho[el_i][0] = np.linspace(0.0, nrho * drho, nrho,
                                                endpoint=False)
                    for j in range(i, n_el):
                        el_j = elements[j]
                        key = f"{el_i}{el_j}"
                        phi[key] = np.zeros((2, nr))
                        phi[key][0] = np.linspace(0.0, nr * dr, nr,
                                                  endpoint=False)
                        if is_adp:
                            dipole[key] = np.zeros((2, nr))
                            dipole[key][0] = np.linspace(0.0, nr * dr, nr,
                                                         endpoint=False)
                            quadrupole[key] = np.zeros((2, nr))
                            quadrupole[key][0] = np.linspace(0.0, nr * dr, nr,
                                                             endpoint=False)
                        ab.append(key)
                nab = len(ab)
                curr_element = elements[0]
                stage = 1
            elif 1 <= stage <= n_el:
                inc = number - base
                if inc < 0:
                    values = line.split()
                    assert len(values) == 4
                    atomic_masses.append(float(values[1]))
                    lattice_constants.append(float(values[2]))
                    lattice_types.append(values[3].strip())
                    continue
                elif inc < nrho:
                    frho[curr_element][1, inc] = float(line)
                else:
                    rho[curr_element][1, inc - nrho] = float(line)
                if inc + 1 == nr + nrho:
                    if stage < n_el:
                        stage += 1
                        curr_element = elements[stage - 1]
                        base = number + 2
                    else:
                        stage += 1
                        base = number + 1
            elif stage > n_el:
                rdiv, residual = divmod(number - base, nr)
                div = rdiv % nab
                key = ab[div]
                inc = number - base - nr * rdiv
                if ab_cycle == 0:
                    adict = phi
                elif ab_cycle == 1:
                    adict = dipole
                else:
                    adict = quadrupole
                if inc == 0:
                    adict[key][1, inc] = float(line)
                else:
                    adict[key][1, inc] = float(line) / adict[key][0, inc]
                if div + 1 == nab and inc + 1 == nr and is_adp:
                    ab_cycle += 1

        return {'elements': elements, 'rho': rho, 'phi': phi, 'frho': frho,
                'dipole': dipole, 'quadrupole': quadrupole, 'nr': nr, 'dr': dr,
                'nrho': nrho, 'drho': drho, 'rcut': rcut,
                'atomic_masses': atomic_masses,
                'lattice_constants': lattice_constants,
                'lattice_types': lattice_types}


def read_eam_alloy_setfl(filename) -> SetFL:
    """
    Read a Lammps eam/alloy setfl file.
    """
    adict = _read_setfl(filename, is_adp=False)
    adict.pop('dipole')
    adict.pop('quadrupole')
    return SetFL(**adict)


def read_adp_setfl(filename) -> AdpFL:
    """
    Read a Lammps adp setfl file.
    """
    return AdpFL(**_read_setfl(filename, is_adp=True))
