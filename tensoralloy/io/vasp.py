"""
A modified implementation of `read_vasp_xml`.
"""
from __future__ import print_function, absolute_import

import numpy as np
import xml.etree.ElementTree as ET

from ase import Atoms
from ase.units import GPa

from tensoralloy import atoms_utils

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def __get_xml_parameter(par):
    """An auxiliary function that enables convenient extraction of
    parameter values from a vasprun.xml file with proper type
    handling.

    """

    def to_bool(b):
        """
        T: True, F: False
        """
        if b == 'T':
            return True
        else:
            return False

    to_type = {'int': int,
               'logical': to_bool,
               'string': str,
               'float': float}

    text = par.text
    if text is None:
        text = ''

    # Float parameters do not have a 'type' attrib
    var_type = to_type[par.attrib.get('type', 'float')]

    try:
        if par.tag == 'v':
            return list(map(var_type, text.split()))
        else:
            return var_type(text.strip())
    except ValueError:
        # Vasp can sometimes write "*****" due to overflow
        return None


def read_vasp_xml(filename='vasprun.xml',
                  index=-1,
                  finite_temperature=False):
    """Parse vasprun.xml file.

    Reads unit cell, atom positions, energies, forces, and constraints
    from vasprun.xml file

    Parameters
    ----------
    filename : str
        The vasprun xml.
    index : slice or int or List[str]
        Can be a slice, an int or a list of int.
    finite_temperature : bool
        A boolean. If True, the calculations will be treated as
        'finite temperature' calculations, The internal energy U or `energy
        without entropy` will be used as `energy`. Otherwise all calculations
        within this file will be considered as zero-temperature, hence
        `E(sigma->0)` will be used as `energy`.

    """


    from ase.constraints import FixAtoms, FixScaled
    from ase.calculators.singlepoint import (SinglePointDFTCalculator,
                                             SinglePointKPoint)
    from collections import OrderedDict

    tree = ET.iterparse(filename, events=['start', 'end'])

    atoms_init = None
    calculation = []
    ibz_kpts = None
    kpt_weights = None
    natoms = None
    species = None
    sigma = None
    parameters = OrderedDict()

    try:
        for event, elem in tree:

            if event == 'end':
                if elem.tag == 'kpoints':
                    for subelem in elem.iter(tag='generation'):
                        kpts_params = OrderedDict()
                        parameters['kpoints_generation'] = kpts_params
                        for par in subelem.iter():
                            if par.tag in ['v', 'i']:
                                parname = par.attrib['name'].lower()
                                kpts_params[parname] = __get_xml_parameter(par)

                    kpts = elem.findall("varray[@name='kpointlist']/v")
                    ibz_kpts = np.zeros((len(kpts), 3))

                    for i, kpt in enumerate(kpts):
                        ibz_kpts[i] = [float(val) for val in kpt.text.split()]

                    kpt_weights = elem.findall('varray[@name="weights"]/v')
                    kpt_weights = [float(val.text) for val in kpt_weights]

                elif elem.tag == 'parameters':
                    for par in elem.iter():
                        if par.tag in ['v', 'i']:
                            parname = par.attrib['name'].lower()
                            parameters[parname] = __get_xml_parameter(par)

                elif elem.tag == 'atominfo':
                    species = []

                    for entry in elem.find("array[@name='atoms']/set"):
                        species.append(entry[0].text.strip())

                    natoms = len(species)

                elif (elem.tag == 'structure' and
                      elem.attrib.get('name') == 'initialpos'):
                    cell_init = np.zeros((3, 3), dtype=float)

                    for i, v in enumerate(elem.find(
                            "crystal/varray[@name='basis']")):
                        cell_init[i] = np.array([
                            float(val) for val in v.text.split()])

                    scpos_init = np.zeros((natoms, 3), dtype=float)

                    for i, v in enumerate(elem.find(
                            "varray[@name='positions']")):
                        scpos_init[i] = np.array([
                            float(val) for val in v.text.split()])

                    constraints = []
                    fixed_indices = []

                    for i, entry in enumerate(elem.findall(
                            "varray[@name='selective']/v")):
                        flags = (np.array(entry.text.split() ==
                                          np.array(['F', 'F', 'F'])))
                        if flags.all():
                            fixed_indices.append(i)
                        elif flags.any():
                            constraints.append(FixScaled(cell_init, i, flags))

                    if fixed_indices:
                        constraints.append(FixAtoms(fixed_indices))

                    atoms_init = Atoms(species,
                                       cell=cell_init,
                                       scaled_positions=scpos_init,
                                       constraint=constraints,
                                       pbc=True)

                elif elem.tag == 'incar':
                    subelem = elem.find("i[@name='SIGMA']")
                    if subelem is not None:
                        sigma = float(subelem.text)

            elif event == 'start' and elem.tag == 'calculation':
                calculation.append(elem)

    except ET.ParseError as parse_error:
        if atoms_init is None:
            raise parse_error
        if calculation[-1].find('energy') is None:
            calculation = calculation[:-1]
        if not calculation:
            yield atoms_init

    if calculation:
        if isinstance(index, int):
            steps = [calculation[index]]
        elif isinstance(index, (list, tuple)):
            steps = [calculation[i] for i in index]
        else:
            steps = calculation[index]
    else:
        steps = []

    for step in steps:
        # Workaround for VASP bug, e_0_energy contains the wrong value
        # in calculation/energy, but calculation/scstep/energy does not
        # include classical VDW corrections. So, first calculate
        # e_0_energy - e_fr_energy from calculation/scstep/energy, then
        # apply that correction to e_fr_energy from calculation/energy.
        lastscf = step.findall('scstep/energy')[-1]
        dipoles = step.findall('scstep/dipole')
        if dipoles:
            lastdipole = dipoles[-1]
        else:
            lastdipole = None

        delta = (float(lastscf.find('i[@name="e_0_energy"]').text) -
                 float(lastscf.find('i[@name="e_fr_energy"]').text))

        eentropy = (float(lastscf.find('i[@name="e_fr_energy"]').text) -
                    float(lastscf.find('i[@name="e_wo_entrp"]').text))
        
        if sigma is None or np.abs(sigma) < 1e-6:
            eentropy = 0
        else:
            # electron entropy should be positive
            eentropy = np.abs(-eentropy / sigma)

        free_energy = float(step.find('energy/i[@name="e_fr_energy"]').text)
        energy = free_energy + delta

        try:
            kinetic_energy = float(step.find('energy/i[@name="kinetic"]').text)
        except Exception:
            kinetic_energy = None

        cell = np.zeros((3, 3), dtype=float)
        for i, vector in enumerate(step.find(
                'structure/crystal/varray[@name="basis"]')):
            cell[i] = np.array([float(val) for val in vector.text.split()])

        scpos = np.zeros((natoms, 3), dtype=float)
        for i, vector in enumerate(step.find(
                'structure/varray[@name="positions"]')):
            scpos[i] = np.array([float(val) for val in vector.text.split()])

        forces = None
        fblocks = step.find('varray[@name="forces"]')
        if fblocks is not None:
            forces = np.zeros((natoms, 3), dtype=float)
            for i, vector in enumerate(fblocks):
                forces[i] = np.array([float(val)
                                      for val in vector.text.split()])

        stress = None
        sblocks = step.find('varray[@name="stress"]')
        if sblocks is not None:
            stress = np.zeros((3, 3), dtype=float)
            for i, vector in enumerate(sblocks):
                stress[i] = np.array([float(val)
                                      for val in vector.text.split()])
            stress *= -0.1 * GPa
            stress = stress.reshape(9)[[0, 4, 8, 5, 2, 1]]

        dipole = None
        if lastdipole is not None:
            dblock = lastdipole.find('v[@name="dipole"]')
            if dblock is not None:
                dipole = np.array([float(val) for val in dblock.text.split()])

        dblock = step.find('dipole/v[@name="dipole"]')
        if dblock is not None:
            dipole = np.array([float(val) for val in dblock.text.split()])

        efermi = step.find('dos/i[@name="efermi"]')
        if efermi is not None:
            efermi = float(efermi.text)

        kpoints = []
        for ikpt in range(1, len(ibz_kpts) + 1):
            kblocks = step.findall(
                'eigenvalues/array/set/set/set[@comment="kpoint %d"]' % ikpt)
            if kblocks is not None:
                for spin, kpoint in enumerate(kblocks):
                    eigenvals = kpoint.findall('r')
                    eps_n = np.zeros(len(eigenvals))
                    f_n = np.zeros(len(eigenvals))
                    for j, val in enumerate(eigenvals):
                        val = val.text.split()
                        eps_n[j] = float(val[0])
                        f_n[j] = float(val[1])
                    if len(kblocks) == 1:
                        f_n *= 2
                    kpoints.append(SinglePointKPoint(kpt_weights[ikpt - 1],
                                                     spin, ikpt, eps_n, f_n))
        if len(kpoints) == 0:
            kpoints = None

        if finite_temperature:
            if sigma is None:
                raise ValueError("For finite temperature calculations "
                                 "`ISIGMA` should be non-zero")
            energy = free_energy + eentropy * sigma

        atoms = atoms_init.copy()
        atoms.set_cell(cell)
        atoms.set_scaled_positions(scpos)
        atoms.calc = SinglePointDFTCalculator(
            atoms, energy=energy, forces=forces,
            stress=stress, free_energy=free_energy,
            ibzkpts=ibz_kpts,
            efermi=efermi, dipole=dipole)
        atoms.calc.name = 'vasp'
        atoms.calc.kpts = kpoints
        atoms.calc.parameters = parameters

        if finite_temperature:
            if sigma is not None:
                atoms_utils.set_electron_temperature(atoms, sigma)
            if eentropy is not None:
                atoms_utils.set_electron_entropy(atoms, eentropy)
        if kinetic_energy is not None:
            atoms_utils.set_kinetic_energy(atoms, kinetic_energy)
        yield atoms
