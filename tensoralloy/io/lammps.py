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
from datetime import datetime
from ase.calculators.lammps.coordinatetransform import Prism
from ase.calculators.lammps.unitconvert import convert
from ase.io.lammpsdata import _write_masses
from ase.utils import writer
from ase import Atoms

from tensoralloy.utils import add_slots, get_elements_from_kbody_term

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["LAMMPS_COMMAND", "SetFL", "MeamSpline",
           "read_adp_setfl", "read_eam_alloy_setfl", "write_adp_setfl",
           "read_tersoff_file", "write_tersoff_file",
           "read_meam_spline_file", "read_old_meam_spline_file"]


def get_lammps_command():
    """
    Return the Bash command to run Lammps.

    The look up orders:
        1. lmp_serial
        2. lammps
        3. ${LAMMPS_COMMAND}
        4. None

    """
    if "LAMMPS_COMMAND" in os.environ:
        return os.environ["LAMMPS_COMMAND"]
    else:
        lmp = shutil.which("lmp_serial")
        if lmp:
            return lmp
        elif shutil.which("lammps"):
            return shutil.which("lammps")
        else:
            return ""


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
    natural_boundary: bool


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


def _init_spline_dict(nx: int, dx: float, bc_start=0.0, bc_end=0.0,
                      deriv_order=2):
    assert deriv_order == 1 or deriv_order == 2
    return {
        "bc_start": bc_start,
        "bc_end": bc_end,
        "x": np.linspace(0.0, nx * dx, nx, endpoint=False),
        "y": np.zeros(nx),
        "natural_boundary": False if deriv_order == 1 else True,
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


_tersoff_header = """# DATE: {date} CONTRIBUTOR: TensorAlloy 

# Tersoff parameters for various elements and mixtures
# multiple entries can be added to this file, LAMMPS reads the ones it needs
# these entries are in LAMMPS "metal" units:
#   A,B = eV; lambda1,lambda2,lambda3 = 1/Angstroms; R,D = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3, 
#   m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A
"""


def write_tersoff_file(filename: str, potential: TersoffPotential):
    """
    Write tersoff parameters to file.
    """
    first_row_keys = TERSOFF_KEYS[:7]
    second_row_keys = TERSOFF_KEYS[7:]
    short_fstr_keys = ["m", "gamma", "D", "R"]
    with open(filename, "w") as fp:
        fp.write(_tersoff_header.format(date=str(datetime.today())))
        for kbody_term, params in potential.params.items():
            a, b, c = get_elements_from_kbody_term(kbody_term)
            first_row = " ".join(
                [f"{params[key]:.1f}"
                 if key in short_fstr_keys else f"{params[key]}"
                 for key in first_row_keys])
            second_row = " ".join(
                [f"{params[key]:.1f}"
                 if key in short_fstr_keys else f"{params[key]}"
                 for key in second_row_keys])
            fp.write(f"{a:2s} {b:2s} {c:2s} {first_row}\n")
            fp.write(f"          {second_row}\n")


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
                if is_new_format:
                    continue
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
                spline = _init_spline_dict(
                    nknots, 0.0, bc_start, bc_end, deriv_order=1)
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


@writer
def write_lammps_data(
    fd,
    atoms: Atoms,
    *,
    specorder: list = None,
    force_skew: bool = False,
    prismobj: Prism = None,
    masses: bool = False,
    velocities: bool = False,
    units: str = "metal",
    atom_style: str = "atomic",
    type_labels: bool = False,
):
    """Write atomic structure data to a LAMMPS data file.

    Parameters
    ----------
    fd : file|str
        File to which the output will be written.
    atoms : Atoms
        Atoms to be written.
    specorder : list[str], optional
        Chemical symbols in the order of LAMMPS atom types, by default None
    force_skew : bool, optional
        Force to write the cell as a
        `triclinic <https://docs.lammps.org/Howto_triclinic.html>`__ box,
        by default False
    prismobj : Prism|None, optional
        Prism, by default None
    masses : bool, optional
        Whether the atomic masses are written or not, by default False
    velocities : bool, optional
        Whether the atomic velocities are written or not, by default False
    units : str, optional
        `LAMMPS units <https://docs.lammps.org/units.html>`__,
        by default "metal"
    atom_style : {"atomic", "charge", "full"}, optional
        `LAMMPS atom style <https://docs.lammps.org/atom_style.html>`__,
        by default "atomic".
    type_labels: bool, optional
        The new type label feature: 
        <https://docs.lammps.org/Howto_type_labels.html>

    """

    # FIXME: We should add a check here that the encoding of the file object
    #        is actually ascii once the 'encoding' attribute of IOFormat objects
    #        starts functioning in implementation (currently it doesn't do
    #         anything).

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                "Can only write one configuration to a lammps data file!"
            )
        atoms = atoms[0]

    fd.write("(written by ASE)\n\n")

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    fd.write(f"{n_atoms} atoms\n")

    if specorder is None:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictably according to the alphabetic order
        species = sorted(set(symbols))
    else:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        species = specorder
    n_atom_types = len(species)
    fd.write(f"{n_atom_types} atom types\n\n")

    if prismobj is None:
        p = Prism(atoms.get_cell())
    else:
        p = prismobj

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
                                        "ASE", units)

    fd.write(f"0.0 {xhi:23.17g}  xlo xhi\n")
    fd.write(f"0.0 {yhi:23.17g}  ylo yhi\n")
    fd.write(f"0.0 {zhi:23.17g}  zlo zhi\n")

    if force_skew or p.is_skewed():
        fd.write(f"{xy:23.17g} {xz:23.17g} {yz:23.17g}  xy xz yz\n")
    fd.write("\n")

    if masses:
        _write_masses(fd, atoms, species, units)
    
    if type_labels and specorder is not None:
        fd.write("Atom Type Labels\n")
        fd.write("\n")
        for i, spec in enumerate(specorder):
            fd.write(f"{i + 1} {spec}\n")
        fd.write("\n")

    # Write (unwrapped) atomic positions.  If wrapping of atoms back into the
    # cell along periodic directions is desired, this should be done manually
    # on the Atoms object itself beforehand.
    fd.write(f"Atoms # {atom_style}\n\n")
    pos = p.vector_to_lammps(atoms.get_positions(), wrap=False)

    if atom_style == 'atomic':
        if not type_labels:
            for i, r in enumerate(pos):
                # Convert position from ASE units to LAMMPS units
                r = convert(r, "distance", "ASE", units)
                s = species.index(symbols[i]) + 1
                fd.write(
                    "{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n".format(
                        *(i + 1, s) + tuple(r)
                    )
                )
        else:
            for i, r in enumerate(pos):
                # Convert position from ASE units to LAMMPS units
                r = convert(r, "distance", "ASE", units)
                fd.write(
                    "{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n".format(
                        *(i + 1, symbols[i]) + tuple(r)
                    )
                )

    elif atom_style == 'charge':
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write("{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n"
                     .format(*(i + 1, s, q) + tuple(r)))
    elif atom_style == 'full':
        charges = atoms.get_initial_charges()
        # The label 'mol-id' has apparenlty been introduced in read earlier,
        # but so far not implemented here. Wouldn't a 'underscored' label
        # be better, i.e. 'mol_id' or 'molecule_id'?
        if atoms.has('mol-id'):
            molecules = atoms.get_array('mol-id')
            if not np.issubdtype(molecules.dtype, np.integer):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " mol-id dtype must be subtype of np.integer, and"
                    " not {:s}.").format(str(molecules.dtype)))
            if (len(molecules) != len(atoms)) or (molecules.ndim != 1):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " each atom must have exactly one mol-id."))
        else:
            # Assigning each atom to a distinct molecule id would seem
            # preferableabove assigning all atoms to a single molecule
            # id per default, as done within ase <= v 3.19.1. I.e.,
            # molecules = np.arange(start=1, stop=len(atoms)+1,
            # step=1, dtype=int) However, according to LAMMPS default
            # behavior,
            molecules = np.zeros(len(atoms), dtype=int)
            # which is what happens if one creates new atoms within LAMMPS
            # without explicitly taking care of the molecule id.
            # Quote from docs at https://lammps.sandia.gov/doc/read_data.html:
            #    The molecule ID is a 2nd identifier attached to an atom.
            #    Normally, it is a number from 1 to N, identifying which
            #    molecule the atom belongs to. It can be 0 if it is a
            #    non-bonded atom or if you don't care to keep track of molecule
            #    assignments.

        for i, (m, q, r) in enumerate(zip(molecules, charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write("{0:>6} {1:>3} {2:>3} {3:>5} {4:23.17g} {5:23.17g} "
                     "{6:23.17g}\n".format(*(i + 1, m, s, q) + tuple(r)))
    else:
        raise NotImplementedError

    if velocities and atoms.get_velocities() is not None:
        fd.write("\n\nVelocities\n\n")
        vel = p.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            # Convert velocity from ASE units to LAMMPS units
            v = convert(v, "velocity", "ASE", units)
            fd.write(
                "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                    *(i + 1,) + tuple(v)
                )
            )

    fd.flush()
