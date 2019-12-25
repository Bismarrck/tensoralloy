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
from typing import List, Dict, Union
from atsim.potentials import writeSetFL

from tensoralloy.utils import add_slots

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["LAMMPS_COMMAND", "SetFL", "MeamSpline",
           "read_adp_setfl", "read_eam_alloy_setfl", "write_adp_setfl",
           "read_tersoff_file",
           "read_meam_spline_file"]


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


@add_slots
@dataclass
class Spline:
    """
    Representation of a cubic spline function.
    """
    bc_start: float
    bc_end: float
    x: np.ndarray
    y: np.ndarray


@add_slots
@dataclass
class SetFL:
    """
    Representation of a Lammps eam/alloy file.
    """
    elements: List[str]
    rho: Dict[str, Spline]
    phi: Dict[str, Spline]
    embed: Dict[str, Spline]
    dipole: Dict[str, Spline]
    quadrupole: Dict[str, Spline]
    nr: int
    dr: float
    nrho: int
    drho: float
    rcut: float
    atomic_masses: List[float]
    lattice_constants: List[float]
    lattice_types: List[str]


def _init_spline_dict(nx: int, dx: float, bc_start=0.0, bc_end=0.0):
    return {
        "bc_start": bc_start,
        "bc_end": bc_end,
        "x": np.linspace(0.0, nx * dx, nx, endpoint=False),
        "y": np.zeros(nx),
    }


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
                    rho[el_i] = _init_spline_dict(nr, dr)
                    frho[el_i] = _init_spline_dict(nrho, drho)
                    for j in range(i, n_el):
                        el_j = elements[j]
                        key = f"{el_i}{el_j}"
                        phi[key] = _init_spline_dict(nr, dr)
                        if is_adp:
                            dipole[key] = _init_spline_dict(nr, dr)
                            quadrupole[key] = _init_spline_dict(nr, dr)
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
                    frho[curr_element]['y'][inc] = float(line)
                else:
                    rho[curr_element]['y'][inc - nrho] = float(line)
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
                    adict[key]['y'][inc] = float(line)
                else:
                    adict[key]['y'][inc] = float(line) / adict[key]['x'][inc]
                if div + 1 == nab and inc + 1 == nr and is_adp:
                    ab_cycle += 1

        def _dict_to_spline(group: Dict[str, dict]):
            for _key, _val in group.items():
                group[_key] = Spline(**_val)

        _dict_to_spline(rho)
        _dict_to_spline(phi)
        _dict_to_spline(frho)
        _dict_to_spline(dipole)
        _dict_to_spline(quadrupole)

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


@dataclass
class TersoffPotential:
    """
    The representation of a Lammps Tersoff potential file.
    """
    elements: List[str]
    params: Dict[str, Dict[str, float]]


# Parameter names of the standard Tersoff potential
TERSOFF_KEYS = ["m", "gamma", "lambda3", "c", "d", "costheta0", "n",
                "beta", "lambda2", "B", "R", "D", "lambda1", "A"]


def read_tersoff_file(filename: str) -> TersoffPotential:
    """
    Read Lammps Tersoff potential file.
    """

    params = {}
    elements = []
    stack = []
    kbody_term = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            splits = line.split()
            if len(splits) == 10:
                kbody_term = f"{splits[0]}{splits[1]}{splits[2]}"
                elements.extend(splits[:3])
                stack.extend(splits[3:])
            elif len(splits) == 7:
                stack.extend(splits)
                params[kbody_term] = {
                    key: float(stack[i]) for i, key in enumerate(TERSOFF_KEYS)}
                stack.clear()

    return TersoffPotential(sorted(list(set(elements))), params)


@add_slots
@dataclass
class MeamSpline:
    elements: List[str]
    rho: Dict[str, Spline]
    phi: Dict[str, Spline]
    embed: Dict[str, Spline]
    fs: Dict[str, Spline]
    gs: Dict[str, Spline]


def read_old_meam_spline_file(filename: str, element: str):
    """
    Read an old meam/spline potential file.
    """
    return _read_meam_spline_file(filename, element)


def read_meam_spline_file(filename: str):
    """
    Read a meam/spline potential file.
    """
    return _read_meam_spline_file(filename)


def _read_meam_spline_file(filename: str, element=None):
    """
    Read the Lammps MEAM/Spline potential file.
    """
    with open(filename) as fp:
        is_new_format = False
        stage = 0
        ncols = 0
        idx = 0
        elements = []
        frho = {}
        rho = {}
        phi = {}
        gs = {}
        fs = {}
        nspline = 0
        kbody_terms = []
        spline = None
        for line in fp:
            if line.startswith("#"):
                continue
            line = line.strip()
            if stage == 0:
                if line.startswith("meam/spline"):
                    is_new_format = True
                if not is_new_format:
                    if element is None:
                        raise ValueError("The 'element' must be specified for "
                                         "old meam/spline format!")
                    elements.append(element)
                    kbody_terms.append(f"{element}{element}")
                    nel = 1
                else:
                    splits = line.split()
                    nel = int(splits[1])
                    if nel != len(splits) - 2:
                        raise IOError(f"Line error: {line}")
                    elements.extend(splits[2:])
                    for i in range(nel):
                        for j in range(i, nel):
                            kbody_terms.append(f"{elements[i]}{elements[j]}")
                ncols = int((nel + 1) * nel / 2)
                stage = 1
            if stage == 1:
                if is_new_format and line == "spline3eq":
                    continue
                nknots = int(line)
                stage = 2
            elif stage == 2:
                splits = line.split()
                if len(splits) != 2:
                    raise IOError(f"Line error: {line}")
                bc_start = float(splits[0])
                bc_end = float(splits[1])
                spline = _init_spline_dict(nknots, 0.0, bc_start, bc_end)
                if is_new_format:
                    stage = 4
                else:
                    stage = 3
            elif stage == 3:
                # Skip this line directly.
                stage = 4
            elif stage == 4:
                values = [float(x) for x in line.split()]
                if len(values) == 3:
                    spline['x'][idx] = values[0]
                    spline['y'][idx] = values[1]
                    idx += 1
                if idx == nknots:
                    obj = Spline(**spline)
                    if nspline < ncols:
                        phi[kbody_terms[nspline]] = obj
                    elif ncols <= nspline < ncols + nel:
                        rho[elements[nspline - ncols]] = obj
                    elif ncols + nel <= nspline < ncols + nel * 2:
                        frho[elements[nspline - ncols - nel]] = obj
                    elif ncols + nel * 2 <= nspline < ncols + nel * 3:
                        fs[elements[nspline - ncols - nel * 2]] = obj
                    else:
                        gs[kbody_terms[nspline - ncols - nel * 3]] = obj
                    nspline += 1
                    if nspline == ncols * 2 + nel * 3:
                        break
                    else:
                        stage = 1
                        idx = 0
        return MeamSpline(elements, rho, phi, frho, fs, gs)
