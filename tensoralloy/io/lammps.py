#!coding=utf-8
"""
Lammps I/O helper functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import sys
import shutil
import os

from io import StringIO
from dataclasses import dataclass
from typing import List, Dict
from atsim.potentials import writeSetFL

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["LAMMPS_COMMAND", "SetFL", "read_adp_setfl",
           "read_eam_alloy_setfl", "write_adp_setfl"]


def get_lammps_command():
    """
    Return the Bash command to run Lammps.

    The look up orders:
        1. lmp_serial
        2. lammps
        3. ${LAMMPS_COMMAND}
        4. None

    """
    lmp = shutil.which("lmp_serial")
    if lmp:
        return lmp
    elif shutil.which("lammps"):
        return shutil.which("lammps")
    elif "LAMMPS_COMMAND" in os.environ:
        return os.environ["LAMMPS_COMMAND"]
    else:
        return None


LAMMPS_COMMAND = get_lammps_command()


@dataclass
class SetFL:
    """
    Representation of a Lammps eam/alloy file.
    """

    elements: List[str]
    rho: Dict[str, np.ndarray]
    phi: Dict[str, np.ndarray]
    embed: Dict[str, np.ndarray]
    dipole: Dict[str, np.ndarray]
    quadrupole: Dict[str, np.ndarray]
    nr: int
    dr: float
    nrho: int
    drho: float
    rcut: float
    atomic_masses: List[float]
    lattice_constants: List[float]
    lattice_types: List[str]

    __slots__ = ("elements", "rho", "phi", "embed", "dipole", "quadrupole",
                 "nr", "dr", "nrho", "drho", "rcut", "atomic_masses",
                 "lattice_constants", "lattice_types")


def _read_setfl(filename, is_adp=False) -> SetFL:
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
                # ADP dipole and quadrupole tabulated values are just raw values
                # but not u(r) * r.
                if inc == 0 or ab_cycle > 0:
                    adict[key][1, inc] = float(line)
                else:
                    adict[key][1, inc] = float(line) / adict[key][0, inc]
                if div + 1 == nab and inc + 1 == nr and is_adp:
                    ab_cycle += 1

        return SetFL(**{'elements': elements, 'rho': rho, 'phi': phi,
                        'embed': frho, 'dipole': dipole,
                        'quadrupole': quadrupole, 'nr': nr, 'dr': dr,
                        'nrho': nrho, 'drho': drho, 'rcut': rcut,
                        'atomic_masses': atomic_masses,
                        'lattice_constants': lattice_constants,
                        'lattice_types': lattice_types})


def read_eam_alloy_setfl(filename) -> SetFL:
    """
    Read a Lammps eam/alloy setfl file.
    """
    return _read_setfl(filename, is_adp=False)


def read_adp_setfl(filename) -> SetFL:
    """
    Read a Lammps adp setfl file.
    """
    return _read_setfl(filename, is_adp=True)


def _write_setfl_pairpots(nr, dr, eampots, pairpots, out, is_phi=True):
    """
    A helper function to write pairwise potentials.
    """
    workout = StringIO()

    def pairkey(a, b):
        """ The unique key of interaction AB. """
        _k = [a, b]
        _k.sort()
        return tuple(_k)

    class ZeroPair(object):
        """ A wrapper class. """

        @staticmethod
        def energy(_):
            """ Return a zero. """
            return 0.0

    zero_pair = ZeroPair()

    # Make a dictionary of available pair pots
    pairpotsdict = {}
    for pp in pairpots:
        pairpotsdict[pairkey(pp.speciesA, pp.speciesB)] = pp

    # Make list of required pots
    for i in range(len(eampots)):
        for j in range(i + 1):
            k = pairkey(eampots[i].species, eampots[j].species)
            pp = pairpotsdict.get(k, zero_pair)
            for k in range(nr):
                r = float(k) * dr
                if is_phi:
                    val = r * pp.energy(r)
                else:
                    val = pp.energy(r)
                print(u"% 20.16e" % val, file=workout)
    out.write(workout.getvalue().encode())


def write_adp_setfl(nrho, drho, nr, dr, eampots, pairpots, out=sys.stdout,
                    comments=("", "", ""), cutoff=None):
    """
    Write an ADP potential to a setfl file.
    """
    n = len(eampots)
    nab = n * (n + 1) // 2
    if len(pairpots) != nab * 3:
        raise ValueError(f"{nab * 3} pair potentials are required")

    writeSetFL(
        nrho, drho, nr, dr, eampots, pairpots[:nab], out, comments, cutoff)
    _write_setfl_pairpots(
        nr, dr, eampots, pairpots[nab * 1: nab * 2], out, False)
    _write_setfl_pairpots(
        nr, dr, eampots, pairpots[nab * 2: nab * 3], out, False)
