#!coding=utf-8
"""
This module is used to compute elastic constants of a crystal structure.

Most of these codes are copied from the Python package [elastic].
https://elastic.readthedocs.io/en/stable/lib-usage.html
"""
from __future__ import print_function, absolute_import

import numpy as np
import spglib as spg
import re

from numpy import array
from ase import Atoms

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_lattice_type(cryst):
    """
    Find the symmetry of the crystal using spglib symmetry finder.

    Derive name of the space group and its number extracted from the result.
    Based on the group number identify also the lattice type and the Bravais
    lattice of the crystal. The lattice type numbers are
    (the numbering starts from 1):
    Triclinic (1), Monoclinic (2), Orthorombic (3),
    Tetragonal (4), Trigonal (5), Hexagonal (6), Cubic (7)

    Parameters
    ----------
    cryst: Atoms
        The target crystal structure.

    Returns
    -------
    result : tuple
        A tuple with four values:
            * lattice type number (1-7)
            * lattice name
            * space group name
            * space group number.

    """

    # Table of lattice types and correcponding group numbers dividing
    # the ranges. See get_lattice_type method for precise definition.
    lattice_types = [
        [3, "Triclinic"],
        [16, "Monoclinic"],
        [75, "Orthorombic"],
        [143, "Tetragonal"],
        [168, "Trigonal"],
        [195, "Hexagonal"],
        [231, "Cubic"]
    ]

    sg = spg.get_spacegroup(cryst)
    m = re.match(r'([A-Z].*\b)\s*\(([0-9]*)\)', sg)
    sg_name = m.group(1)
    sg_nr = int(m.group(2))

    for n, l in enumerate(lattice_types):
        if sg_nr < l[0]:
            bravais = l[1]
            lattype = n + 1
            break
    else:
        raise Exception("Failed to find the lattice type.")
    return lattype, bravais, sg_name, sg_nr


def regular(u):
    """
    Equation matrix generation for the regular (cubic) lattice.
    The order of constants is as follows:
        C_{11}, C_{12}, C_{44}

    Parameters
    ----------
    u : array_like
        The vector of deformations:
            [ `u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    Returns
    -------
    matrix : array_like
        Symmetry defined stress-strain equation matrix

    """
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array([[uxx, uyy + uzz, 0],
                  [uyy, uxx + uzz, 0],
                  [uzz, uxx + uyy, 0],
                  [0, 0, 2 * uyz],
                  [0, 0, 2 * uxz],
                  [0, 0, 2 * uxy]])


def tetragonal(u):
    """
    Equation matrix generation for the tetragonal lattice.
    The order of constants is as follows:
        C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{14}

    Parameters
    ----------
    u : array_like
        The vector of deformations:
            [ `u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    Returns
    -------
    matrix : array_like
        Symmetry defined stress-strain equation matrix

    """

    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array([[uxx, 0, uyy, uzz, 0, 0],
                  [uyy, 0, uxx, uzz, 0, 0],
                  [0, uzz, 0, uxx + uyy, 0, 0],
                  [0, 0, 0, 0, 0, 2 * uxy],
                  [0, 0, 0, 0, 2 * uxz, 0],
                  [0, 0, 0, 0, 2 * uyz, 0]])


def orthorombic(u):
    """
    Equation matrix generation for the orthorombic lattice.

    The order of constants is as follows:
        C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
        C_{44}, C_{55}, C_{66}

    Parameters
    ----------
    u : array_like
        The vector of deformations:
            [ `u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    Returns
    -------
    matrix : array_like
        Symmetry defined stress-strain equation matrix.

    """

    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [[uxx, 0, 0, uyy, uzz, 0, 0, 0, 0],
         [0, uyy, 0, uxx, 0, uzz, 0, 0, 0],
         [0, 0, uzz, 0, uxx, uyy, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy]])


def trigonal(u):
    """
    The matrix is constructed based on the approach from L & L using auxiliary
    coordinates: `xi = x + iy`, `eta = x - iy`.
    The components are calculated from free energy using formula introduced in
    `symmetry with appropriate coordinate changes`.

    The order of constants is as follows:
        C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{14}

    Parameters
    ----------
    u : array_like
        The vector of deformations:
            [ `u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    Returns
    -------
    matrix : array_like
        Symmetry defined stress-strain equation matrix.

    """
    # TODO: Not tested yet.
    # TODO: There is still some doubt about the :math:`C_{14}` constant.
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [[uxx, 0, uyy, uzz, 0, 2 * uxz],
         [uyy, 0, uxx, uzz, 0, -2 * uxz],
         [0, uzz, 0, uxx + uyy, 0, 0],
         [0, 0, 0, 0, 2 * uyz, -4 * uxy],
         [0, 0, 0, 0, 2 * uxz, 2 * (uxx - uyy)],
         [2 * uxy, 0, -2 * uxy, 0, 0, -4 * uyz]])


def hexagonal(u):
    """
    The matrix is constructed based on the approach from L & L using auxiliary
    coordinates: `xi = x + iy`, `eta = x - iy`.
    The components are calculated from free energy using formula introduced in
    `symmetry with appropriate coordinate changes`.

    The order of constants is as follows:
        C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{14}

    Parameters
    ----------
    u : array_like
        The vector of deformations:
            [ `u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    Returns
    -------
    matrix : array_like
        Symmetry defined stress-strain equation matrix.

    """
    # TODO: Still needs good verification
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [[uxx, 0, uyy, uzz, 0],
         [uyy, 0, uxx, uzz, 0],
         [0, uzz, 0, uxx + uyy, 0],
         [0, 0, 0, 0, 2 * uyz],
         [0, 0, 0, 0, 2 * uxz],
         [2 * uxy, 0, -2 * uxy, 0, 0]])


def monoclinic(u):
    """
    Monoclinic group,

    The order of constants is as follows:
        C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
        C_{44}, C_{55}, C_{66}, C_{16}, C_{26}, C_{36}, C_{45}

    Parameters
    ----------
    u : array_like
        The vector of deformations:
            [ `u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    Returns
    -------
    matrix : array_like
        Symmetry defined stress-strain equation matrix.

    """
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [[uxx, 0, 0, uyy, uzz, 0, 0, 0, 0, uxy, 0, 0, 0],
         [0, uyy, 0, uxx, 0, uzz, 0, 0, 0, 0, uxy, 0, 0],
         [0, 0, uzz, 0, uxx, uyy, 0, 0, 0, 0, 0, uxy, 0],
         [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0, 0, 0, 0, uxz],
         [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0, 0, 0, 0, uyz],
         [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy, uxx, uyy, uzz, 0]])


def triclinic(u):
    """
    Triclinic crystals.

    *Note*: This was never tested on the real case. Beware!
    The ordering of constants is:
       C_{11}, C_{22}, C_{33},
       C_{12}, C_{13}, C_{23},
       C_{44}, C_{55}, C_{66},
       C_{16}, C_{26}, C_{36}, C_{46}, C_{56},
       C_{14}, C_{15}, C_{25}, C_{45}

    Parameters
    ----------
    u : array_like
        The vector of deformations:
            [ `u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    Returns
    -------
    matrix : array_like
        Symmetry defined stress-strain equation matrix.

    """

    # Based on the monoclinic matrix and not tested on real case.
    # If you have test cases for this symmetry send them to the author.
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [[uxx, 0, 0, uyy, uzz, 0, 0, 0, 0, uxy, 0, 0, 0, 0, uyz, uxz, 0, 0],
         [0, uyy, 0, uxx, 0, uzz, 0, 0, 0, 0, uxy, 0, 0, 0, 0, 0, uxz, 0],
         [0, 0, uzz, 0, uxx, uyy, 0, 0, 0, 0, 0, uxy, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0, 0, 0, 0, uxy, 0, uxx, 0, 0, uxz],
         [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0, 0, 0, 0, 0, uxy, 0, uxx, uyy, uyz],
         [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy, uxx, uyy, uzz, uyz, uxz, 0, 0, 0, 0]])


def get_pressure(stress):
    """
    Return *external* isotropic (hydrostatic) pressure in ASE units.

    If the pressure is positive the system is under external pressure.
    This is a convenience function to convert output of get_stress function
    into external pressure.

    Parameters
    ----------
    stress : array_like
        The Voigt stress tensor.

    Returns
    -------
    pressure : float
        Theexternal hydrostatic pressure in ASE units.

    """
    return -np.mean(stress[0: 3])


def get_strain(cryst, refcell=None):
    """
    Calculate strain tensor in the Voight notation.

    Computes the strain tensor in the Voight notation as a conventional vector.
    The calculation is done with respect to the crystal geometry passed in
    `refcell` parameter.

    Parameters
    ----------
    cryst : Atoms
        The deformed structure.
    refcell : Atoms
        The reference, undeformed structure.

    Returns
    -------
    strain : array_like
        The strain tensor in the Voight notation.

    """
    if refcell is None:
        refcell = cryst
    du = cryst.get_cell() - refcell.get_cell()
    m = refcell.get_cell()
    m = np.linalg.inv(m)
    u = np.dot(m, du)
    u = (u + u.T) / 2
    return array([u[0, 0], u[1, 1], u[2, 2], u[2, 1], u[2, 0], u[1, 0]])


def get_cart_deformed_cell(base_cryst, axis=0, size=1):
    """
    Return the cell deformed along one of the cartesian directions.

    Creates new deformed structure. The deformation is based on the base
    structure and is performed along single axis. The axis is specified as
    follows:
        * 0, 1, 2 = x, y, z
        * sheers: 3, 4, 5 = yz, xz, xy.

    The size of the deformation is in percent and degrees, respectively.

    Parameters
    ----------
    base_cryst : Atoms
        The structure to deform.
    axis : int
        The direction of deformation.
    size : int or float
        The size of the deformation.

    Returns
    -------
    atoms : Atoms
        The new, deformed structure.

    """
    cryst = Atoms(base_cryst)
    uc = base_cryst.get_cell()
    s = size / 100.0
    L = np.diag(np.ones(3))
    if axis < 3:
        L[axis, axis] += s
    else:
        if axis == 3:
            L[1, 2] += s
        elif axis == 4:
            L[0, 2] += s
        else:
            L[0, 1] += s
    uc = np.dot(uc, L)
    cryst.set_cell(uc, scale_atoms=True)
    return cryst


def get_elementary_deformations(cryst, n=5, d=2):
    """
    Generate elementary deformations for elastic tensor calculation.
    The deformations are created based on the symmetry of the crystal and
    are limited to the non-equivalet axes of the crystal.

    Parameters
    ----------
    cryst : Atoms
        The base structure.
    n : int
        The number of deformations per non-equivalent axis.
    d : int or float
        The size of the maximum deformation in percent and degrees.

    Returns
    -------
    structures : List[Atoms]
        A list of deformed structures.

    """
    # Deformation look-up table
    # Perhaps the number of deformations for trigonal
    # system could be reduced to [0,3] but better safe then sorry
    deform = {
        "Cubic": [[0, 3], regular],
        "Hexagonal": [[0, 2, 3, 5], hexagonal],
        "Trigonal": [[0, 1, 2, 3, 4, 5], trigonal],
        "Tetragonal": [[0, 2, 3, 5], tetragonal],
        "Orthorombic": [[0, 1, 2, 3, 4, 5], orthorombic],
        "Monoclinic": [[0, 1, 2, 3, 4, 5], monoclinic],
        "Triclinic": [[0, 1, 2, 3, 4, 5], triclinic]
    }

    lattyp, brav, sg_name, sg_nr = get_lattice_type(cryst)

    # Decide which deformations should be used
    axis, symm = deform[brav]

    systems = []
    for a in axis:
        if a < 3:  # tetragonal deformation
            for dx in np.linspace(-d, d, n):
                systems.append(
                    get_cart_deformed_cell(cryst, axis=a, size=dx))
        elif a < 6:  # sheer deformation (skip the zero angle)
            for dx in np.linspace(d / 10.0, d, n):
                systems.append(
                    get_cart_deformed_cell(cryst, axis=a, size=dx))
    return systems


def get_elastic_tensor(cryst, systems):
    """
    Calculate elastic tensor of the crystal.

    The elastic tensor is calculated from the stress-strain relation and derived
    by fitting this relation to the set of linear equations build from the
    symmetry of the crystal and strains and stresses of the set of elementary
    deformations of the unit cell.

    It is assumed that the crystal is converged and optimized under intended
    pressure/stress. The geometry and stress on the cryst is taken as the
    reference point. No additional optimization will be run. Structures in cryst
    and systems list must have calculated stresses.

    The function returns tuple of `C_{ij}` elastic tensor, raw Birch
    coefficients `B_{ij}` and fitting results: residuals, solution rank,
    singular values returned by `numpy.linalg.lstsq`.

    Parameters
    ----------
    cryst : Atoms
        A `Atoms` object, the basic structure.
    systems: List[Atoms]
        A list of `Atoms` objects with calculated deformed structures.

    Returns
    -------
    elastic_constants : array_like
        The elastic constants.
    results : array_like
        Fitting results, including residuals, solution rank and singular values.

    """

    # Deformation look-up table
    # Perhaps the number of deformations for trigonal
    # system could be reduced to [0,3] but better safe then sorry
    deform = {
        "Cubic": [[0, 3], regular],
        "Hexagonal": [[0, 2, 3, 5], hexagonal],
        "Trigonal": [[0, 1, 2, 3, 4, 5], trigonal],
        "Tetragonal": [[0, 2, 3, 5], tetragonal],
        "Orthorombic": [[0, 1, 2, 3, 4, 5], orthorombic],
        "Monoclinic": [[0, 1, 2, 3, 4, 5], monoclinic],
        "Triclinic": [[0, 1, 2, 3, 4, 5], triclinic]
    }

    lattyp, brav, sg_name, sg_nr = get_lattice_type(cryst)
    # Decide which deformations should be used
    axis, symm = deform[brav]

    ul = []
    sl = []
    p = get_pressure(cryst.get_stress())
    for g in systems:
        ul.append(get_strain(g, refcell=cryst))
        # Remove the ambient pressure from the stress tensor
        sl.append(g.get_stress() - np.array([p, p, p, 0, 0, 0]))
    eqm = np.array([symm(u) for u in ul])
    eqm = np.reshape(eqm, (eqm.shape[0] * eqm.shape[1], eqm.shape[2]))
    slm = np.reshape(np.array(sl), (-1,))
    Bij = np.linalg.lstsq(eqm, slm, rcond=None)
    # Calculate elastic constants from Birch coeff.
    # TODO: Check the sign of the pressure array in the B <=> C relation
    if symm == orthorombic:
        Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p])
    elif symm == tetragonal:
        Cij = Bij[0] - np.array([-p, -p, p, p, -p, -p])
    elif symm == regular:
        Cij = Bij[0] - np.array([-p, p, -p])
    elif symm == trigonal:
        Cij = Bij[0] - np.array([-p, -p, p, p, -p, p])
    elif symm == hexagonal:
        Cij = Bij[0] - np.array([-p, -p, p, p, -p])
    elif symm == monoclinic:
        # TODO: verify this pressure array
        Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p])
    elif symm == triclinic:
        # TODO: verify this pressure array
        Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p,
                                 p, p, p, p, p, p, p, p, p])
    else:
        Cij = None
    return Cij, Bij
