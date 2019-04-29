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


@dataclass
class SetFL:
    """
    All details of a Lammps eam/alloy file.
    """

    elements: List[str]
    rho: Dict[str, np.ndarray]
    phi: Dict[str, np.ndarray]
    embed: Dict[str, np.ndarray]
    nr: int
    dr: float
    nrho: int
    drho: float
    rcut: float
    atomic_masses: List[float]
    lattice_constants: List[float]
    lattice_types: List[str]

    __slots__ = ("elements", "rho", "phi", "embed", "nr", "dr", "nrho", "drho",
                 "rcut", "atomic_masses", "lattice_constants", "lattice_types")


def read_eam_alloy(filename) -> SetFL:
    """
    Read tabulated rho, phi and F(rho) values from a Lammps eam/alloy file.
    """
    with open(filename) as fp:
        rho = {}
        frho = {}
        phi = {}
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
                        ab.append(key)
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
                div, residual = divmod(number - base, nr)
                key = ab[div]
                inc = number - base - nr * div
                if inc == 0:
                    phi[key][1, inc] = float(line)
                else:
                    phi[key][1, inc] = float(line) / phi[key][0, inc]

        return SetFL(elements, rho, phi, frho, nr, dr, nrho, drho, rcut,
                     atomic_masses=atomic_masses,
                     lattice_constants=lattice_constants,
                     lattice_types=lattice_types)
