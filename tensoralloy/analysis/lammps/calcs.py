# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""
This module provides basic LAMMPS calculator classes.

Original repo: https://github.com/materialsvirtuallab/mlearn

"""

import os
import abc
import io
import subprocess
import itertools

import six
import numpy as np
from monty.json import MSONable
from monty.tempfile import ScratchDir
from pymatgen.io.lammps.data import LammpsData
from pymatgen.core.structure import Structure, Lattice, Element

_sort_elements = lambda symbols: [e.symbol for e in
                                  sorted([Element(e) for e in symbols])]


class Potential(six.with_metaclass(abc.ABCMeta, MSONable)):
    """
    Abstract Base class for an Interatomic Potential.
    """

    @abc.abstractmethod
    def train(self, train_structures, energies, forces, stresses, **kwargs):
        """
        Train interatomic potentials with energies, forces and
        stresses corresponding to structure.

        Args:
            train_structures (list): List of Pymatgen Structure objects.
            energies (list): List of DFT-calculated total energies of each
                structure in structures list.
            forces (list): List of DFT-calculated (m, 3) forces of each
                structure with m atoms in structures list. m can be varied with
                each single structure case.
            stresses (list): List of DFT-calculated (6, ) virial stresses of
                each structure in structures list.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, test_structures, ref_energies, ref_forces, ref_stresses):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potentials.

        Args:
            test_structures (list): List of Pymatgen Structure Objects.
            ref_energies (list): List of DFT-calculated total energies of each
                structure in structures list.
            ref_forces (list): List of DFT-calculated (m, 3) forces of each
                structure with m atoms in structures list. m can be varied with
                each single structure case.
            ref_stresses (list): List of DFT-calculated (6, ) viriral stresses of
                each structure in structures list.

        Returns:
            DataFrame of original data and DataFrame of predicted data.
        """
        pass

    @abc.abstractmethod
    def predict(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): Pymatgen Structure object.

        Returns:
            energy, forces, stress
        """
        pass


def _pretty_input(lines):
    clean_lines = [l.strip('\n') for l in lines]
    commands = [l for l in clean_lines if len(l.strip()) > 0]
    keys = [c.split()[0] for c in commands
            if not c.split()[0].startswith('#')]
    width = max([len(k) for k in keys]) + 4
    prettify = lambda l: l.split()[0].ljust(width) + ' '.join(l.split()[1:]) \
        if not (len(l.split()) == 0 or l.strip().startswith('#')) else l
    new_lines = map(prettify, clean_lines)
    return '\n'.join(new_lines)


def _read_dump(file_name, dtype='float_'):
    with open(file_name) as f:
        lines = f.readlines()[9:]
    return np.loadtxt(io.StringIO(''.join(lines)), dtype=dtype)


class LMPStaticCalculator(six.with_metaclass(abc.ABCMeta, object)):
    """
    Abstract class to perform static structure property calculation
    using LAMMPS.

    """

    LMP_EXE = 'lmp_serial'
    LMP_PAR_EXE = 'lmp_mpi'
    _COMMON_CMDS = ['units metal',
                    'atom_style charge',
                    'box tilt large',
                    'read_data data.static',
                    'run 0']

    @abc.abstractmethod
    def _setup(self):
        """
        Setup a calculation, writing input files, etc.

        """
        return

    @abc.abstractmethod
    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return

    @abc.abstractmethod
    def _parse(self):
        """
        Parse results from dump files.

        """
        return

    def calculate(self, structures):
        """
        Perform the calculation on a series of structures.

        Args:
            structures [Structure]: Input structures in a list.

        Returns:
            List of computed data corresponding to each structure,
            varies with different subclasses.

        """
        for s in structures:
            assert self._sanity_check(s) is True, \
                'Incompatible structure found'
        ff_elements = None
        if hasattr(self, 'element_profile'):
            ff_elements = self.element_profile.keys()
        with ScratchDir('.'):
            input_file = self._setup()
            data = []
            for s in structures:
                ld = LammpsData.from_structure(s, ff_elements)
                ld.write_file('data.static')
                p = subprocess.Popen([self.LMP_EXE, '-in', input_file],
                                     stdout=subprocess.PIPE)
                stdout = p.communicate()[0]
                rc = p.returncode
                if rc != 0:
                    error_msg = 'LAMMPS exited with return code %d' % rc
                    msg = stdout.decode("utf-8").split('\n')[:-1]
                    try:
                        error_line = [i for i, m in enumerate(msg)
                                      if m.startswith('ERROR')][0]
                        error_msg += ', '.join([e for e in msg[error_line:]])
                    except Exception:
                        error_msg += msg[-1]
                    raise RuntimeError(error_msg)
                results = self._parse()
                data.append(results)
        return data


class EnergyForceStress(LMPStaticCalculator):
    """
    Calculate energy, forces and virial stress of structures.
    """

    def __init__(self, ff_settings):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for
                LAMMPS calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
        """
        self.ff_settings = ff_settings

    def _setup(self):
        template_dir = os.path.join(
            os.path.dirname(__file__), 'templates', 'efs')
        with open(os.path.join(template_dir, 'in.efs'), 'r') as f:
            input_template = f.read()

        input_file = 'in.efs'

        if hasattr(self.ff_settings, "write_param"):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings)))
        return input_file

    def _sanity_check(self, structure):
        return True

    def _parse(self):
        energy = float(np.loadtxt('energy.txt'))
        force = _read_dump('force.dump')
        stress = np.loadtxt('stress.txt')
        return energy, force, stress


class ElasticConstant(LMPStaticCalculator):
    """
    Elastic constant calculator.
    """
    _RESTART_CONFIG = {'internal': {'write_command': 'write_restart',
                                    'read_command': 'read_restart',
                                    'restart_file': 'restart.equil'},
                       'external': {'write_command': 'write_data',
                                    'read_command': 'read_data',
                                    'restart_file': 'data.static'}}

    def __init__(self, ff_settings, potential_type='external',
                 deformation_size=1e-6, jiggle=1e-5, lattice='bcc', alat=5.0,
                 maxiter=400, maxeval=1000):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            potential_type (str): 'internal' indicates the internal potentials
                installed in lammps, 'external' indicates the external potentials
                outside of lammps.
            deformation_size (float): Finite deformation size. Usually range from
                1e-2 to 1e-8, to confirm the results not depend on it.
            jiggle (float): The amount of random jiggle for atoms to
                prevent atoms from staying on saddle points.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
            maxiter (float): The maximum number of iteration. Default to 400.
            maxeval (float): The maximum number of evaluation. Default to 1000.
        """
        self.ff_settings = ff_settings
        self.write_command = self._RESTART_CONFIG[potential_type]['write_command']
        self.read_command = self._RESTART_CONFIG[potential_type]['read_command']
        self.restart_file = self._RESTART_CONFIG[potential_type]['restart_file']
        self.deformation_size = deformation_size
        self.jiggle = jiggle
        self.lattice = lattice
        self.alat = alat
        self.maxiter = maxiter
        self.maxeval = maxeval

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'elastic')

        with open(os.path.join(template_dir, 'in.elastic'), 'r') as f:
            input_template = f.read()
        with open(os.path.join(template_dir, 'init.template'), 'r') as f:
            init_template = f.read()
        with open(os.path.join(template_dir, 'potential.template'), 'r') as f:
            potential_template = f.read()
        with open(os.path.join(template_dir, 'displace.template'), 'r') as f:
            displace_template = f.read()

        input_file = 'in.elastic'

        if hasattr(self.ff_settings, "write_param"):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(write_restart=self.write_command,
                                          restart_file=self.restart_file))
        with open('init.mod', 'w') as f:
            f.write(init_template.format(deformation_size=self.deformation_size,
                                         jiggle=self.jiggle, maxiter=self.maxiter,
                                         maxeval=self.maxeval, lattice=self.lattice,
                                         alat=self.alat))
        with open('potential.mod', 'w') as f:
            f.write(potential_template.format(ff_settings='\n'.join(ff_settings)))
        with open('displace.mod', 'w') as f:
            f.write(displace_template.format(read_restart=self.read_command,
                                             restart_file=self.restart_file))
        return input_file

    def calculate(self, _):
        """
        Calculate the elastic constant given Potential class.
        """
        with ScratchDir('.'):
            input_file = self._setup()
            p = subprocess.Popen([self.LMP_EXE, '-in', input_file],
                                 stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = 'LAMMPS exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            result = self._parse()
        return result

    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return True

    def _parse(self):
        """
        Parse results from dump files.

        """
        C11, C12, C44, bulkmodulus = np.loadtxt('elastic.txt')
        return C11, C12, C44, bulkmodulus


class LatticeConstant(LMPStaticCalculator):
    """
    Lattice Constant Relaxation Calculator.
    """

    def __init__(self, ff_settings):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
        """
        self.ff_settings = ff_settings

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'latt')

        with open(os.path.join(template_dir, 'in.latt'), 'r') as f:
            input_template = f.read()

        input_file = 'in.latt'

        if hasattr(self.ff_settings, "write_param"):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings)))

        return input_file

    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return True

    def _parse(self):
        """
        Parse results from dump files.

        """
        a, b, c = np.loadtxt('lattice.txt')
        return a, b, c


class NudgedElasticBand(LMPStaticCalculator):
    """
    NudgedElasticBand migration energy calculator.
    """

    def __init__(self, ff_settings, specie, lattice, alat, num_replicas=7):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for
                LAMMPS calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
            num_replicas (int): Number of replicas to use.
        """
        self.ff_settings = ff_settings
        self.specie = specie
        self.lattice = lattice
        self.alat = alat
        self.num_replicas = num_replicas

    def get_unit_cell(self, specie, lattice, alat):
        """
        Get the unit cell from specie, lattice type and lattice constant.

        Args
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
        """
        if lattice == 'fcc':
            unit_cell = Structure.from_spacegroup(
                sg='Fm-3m',
                lattice=Lattice.cubic(alat),
                species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'bcc':
            unit_cell = Structure.from_spacegroup(
                sg='Im-3m',
                lattice=Lattice.cubic(alat),
                species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'diamond':
            unit_cell = Structure.from_spacegroup(
                sg='Fd-3m',
                lattice=Lattice.cubic(alat),
                species=[specie], coords=[[0, 0, 0]])
        else:
            raise ValueError("Lattice type is invalid.")

        return unit_cell

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__),
                                    'templates', 'neb')

        with open(os.path.join(template_dir, 'in.relax'), 'r') as f:
            relax_template = f.read()
        with open(os.path.join(template_dir, 'in.neb'), 'r') as f:
            neb_template = f.read()

        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=self.alat)
        lattice_calculator = LatticeConstant(ff_settings=self.ff_settings)
        lattice_calculator.LMP_EXE = self.LMP_EXE
        a, _, _ = lattice_calculator.calculate([unit_cell])[0]
        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=a)

        if self.lattice == 'fcc':
            start_idx, final_idx = 95, 49
            scale_factor = [3, 3, 3]
        elif self.lattice == 'bcc':
            start_idx, final_idx = 40, 14
            scale_factor = [3, 3, 3]
        elif self.lattice == 'diamond':
            start_idx, final_idx = 7, 15
            scale_factor = [2, 2, 2]
        else:
            raise ValueError("Lattice type is invalid.")

        super_cell = unit_cell * scale_factor
        super_cell_ld = LammpsData.from_structure(
            super_cell, atom_style='atomic')
        super_cell_ld.write_file('data.supercell')

        if hasattr(self.ff_settings, "write_param"):
            ff_settings = '\n'.join(self.ff_settings.write_param())
        else:
            ff_settings = '\n'.join(self.ff_settings)

        with open('in.relax', 'w') as f:
            f.write(relax_template.format(
                ff_settings=ff_settings,
                lattice=self.lattice, alat=a, specie=self.specie,
                del_id=start_idx + 1, relaxed_file='initial.relaxed'))

        p = subprocess.Popen([self.LMP_EXE, '-in', 'in.relax'],
                             stdout=subprocess.PIPE)
        stdout = p.communicate()[0]

        rc = p.returncode
        if rc != 0:
            error_msg = 'LAMMPS exited with return code %d' % rc
            msg = stdout.decode("utf-8").split('\n')[:-1]
            try:
                error_line = [i for i, m in enumerate(msg)
                              if m.startswith('ERROR')][0]
                error_msg += ', '.join([e for e in msg[error_line:]])
            except Exception:
                error_msg += msg[-1]
            raise RuntimeError(error_msg)

        with open('in.relax', 'w') as f:
            f.write(relax_template.format(
                ff_settings=ff_settings,
                lattice=self.lattice, alat=a, specie=self.specie,
                del_id=final_idx + 1, relaxed_file='final.relaxed'))

        p = subprocess.Popen([self.LMP_EXE, '-in', 'in.relax'],
                             stdout=subprocess.PIPE)
        stdout = p.communicate()[0]

        rc = p.returncode
        if rc != 0:
            error_msg = 'LAMMPS exited with return code %d' % rc
            msg = stdout.decode("utf-8").split('\n')[:-1]
            try:
                error_line = [i for i, m in enumerate(msg)
                              if m.startswith('ERROR')][0]
                error_msg += ', '.join([e for e in msg[error_line:]])
            except Exception:
                error_msg += msg[-1]
            raise RuntimeError(error_msg)

        final_relaxed_struct = LammpsData.from_file(
            'final.relaxed',
            atom_style='atomic').structure

        lines = ['{}'.format(final_relaxed_struct.num_sites)]

        for idx, site in enumerate(final_relaxed_struct):
            if idx == final_idx:
                idx = final_relaxed_struct.num_sites
            elif idx == start_idx:
                idx = final_idx
            else:
                idx = idx
            lines.append('{}  {:.3f}  {:.3f}  {:.3f}'.format(
                idx + 1, site.x, site.y, site.z))

        with open('data.final_replica', 'w') as f:
            f.write('\n'.join(lines))

        input_file = 'in.neb'

        with open(input_file, 'w') as f:
            f.write(neb_template.format(ff_settings=ff_settings,
                                        start_replica='initial.relaxed',
                                        final_replica='data.final_replica'))

        return input_file

    def calculate(self, _):
        """
        Calculate the NEB barrier given Potential class.
        """
        with ScratchDir('.'):
            input_file = self._setup()
            p = subprocess.Popen(
                [
                    'mpirun', '-n', str(self.num_replicas),
                    self.LMP_PAR_EXE, '-partition', f'{self.num_replicas}x1',
                    '-in', input_file
                ],
                stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = 'LAMMPS exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            result = self._parse()
        return result

    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return True

    def _parse(self):
        """
        Parse results from dump files.

        """
        with open('log.lammps') as f:
            lines = f.readlines()[-1:]
        migration_barrier = float(lines[0].split()[6])
        return migration_barrier


class DefectFormation(LMPStaticCalculator):
    """
    Defect formation energy calculator.
    """

    def __init__(self, ff_settings, specie, lattice, alat):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for
                LAMMPS calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
        """
        self.ff_settings = ff_settings
        self.specie = specie
        self.lattice = lattice
        self.alat = alat

    def get_unit_cell(self, specie, lattice, alat):
        """
        Get the unit cell from specie, lattice type and lattice constant.

        Args
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
        """
        if lattice == 'fcc':
            unit_cell = Structure.from_spacegroup(
                sg='Fm-3m',
                lattice=Lattice.cubic(alat),
                species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'bcc':
            unit_cell = Structure.from_spacegroup(
                sg='Im-3m',
                lattice=Lattice.cubic(alat),
                species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'diamond':
            unit_cell = Structure.from_spacegroup(
                sg='Fd-3m',
                lattice=Lattice.cubic(alat),
                species=[specie], coords=[[0, 0, 0]])
        else:
            raise ValueError("Lattice type is invalid.")

        return unit_cell

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__),
                                    'templates', 'defect')

        with open(os.path.join(template_dir, 'in.defect'), 'r') as f:
            defect_template = f.read()

        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=self.alat)
        lattice_calculator = LatticeConstant(ff_settings=self.ff_settings)
        lattice_calculator.LMP_EXE = self.LMP_EXE
        a, _, _ = lattice_calculator.calculate([unit_cell])[0]
        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=a)

        if self.lattice == 'fcc':
            idx, scale_factor = 95, [3, 3, 3]
        elif self.lattice == 'bcc':
            idx, scale_factor = 40, [3, 3, 3]
        elif self.lattice == 'diamond':
            idx, scale_factor = 7, [2, 2, 2]
        else:
            raise ValueError("Lattice type is invalid.")

        super_cell = unit_cell * scale_factor
        efs_calculator = EnergyForceStress(ff_settings=self.ff_settings)
        efs_calculator.LMP_EXE = self.LMP_EXE
        energy_per_atom = efs_calculator.calculate(
            [super_cell])[0][0] / len(super_cell)

        super_cell_ld = LammpsData.from_structure(super_cell, atom_style='atomic')
        super_cell_ld.write_file('data.supercell')

        input_file = 'in.defect'

        if hasattr(self.ff_settings, "write_param"):
            ff_settings = '\n'.join(self.ff_settings.write_param())
        else:
            ff_settings = '\n'.join(self.ff_settings)

        with open(input_file, 'w') as f:
            f.write(defect_template.format(
                ff_settings=ff_settings,
                lattice=self.lattice, alat=a, specie=self.specie,
                del_id=idx + 1, relaxed_file='data.relaxed'))

        return input_file, energy_per_atom, len(super_cell) - 1

    def calculate(self, _):
        """
        Calculate the vacancy formation given Potential class.
        """
        with ScratchDir('.'):
            input_file, energy_per_atom, num_atoms = self._setup()
            p = subprocess.Popen([self.LMP_EXE, '-in', input_file],
                                 stdout=subprocess.PIPE)
            stdout = p.communicate()[0]

            rc = p.returncode
            if rc != 0:
                error_msg = 'LAMMPS exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            defect_energy, _, _ = self._parse()
        defect_formation_energy = defect_energy - energy_per_atom * num_atoms

        return defect_formation_energy

    def _sanity_check(self, structure):
        return True

    def _parse(self):
        energy = float(np.loadtxt('energy.txt'))
        force = _read_dump('force.dump')
        stress = np.loadtxt('stress.txt')
        return energy, force, stress
