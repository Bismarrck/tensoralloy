#!coding=utf-8
"""
Built-in cyrstal data for constraints.
"""
from __future__ import print_function, absolute_import

import numpy as np
import toml

from dataclasses import dataclass
from os.path import join, realpath, dirname
from pathlib import Path
from typing import Union, List, Tuple
from ase import Atoms
from ase.build import bulk
from ase.io import read
from ase.units import kB

from tensoralloy.nn.constraint.voigt import voigt_to_ijkl
from tensoralloy.test_utils import data_dir
from tensoralloy import atoms_utils


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# **Materials Project**
# https://wiki.materialsproject.org/Elasticity_calculations
#
# Note that in this work, conventional unit cells, obtained using
# `pymatgen.symmetry.SpacegroupAnalyzer.get_conventional_standard_structure`
# are employed for all elastic constant calculations. In our experience, these
# cells typically yield more accurate and better converged elastic constants
# than primitive cells, at the cost of more computational time. We suspect this
# has to do with the fact that unit cells often exhibit higher symmetries and
# simpler Brillouin zones than primitive cells (an example is face centered
# cubic cells).
_identifier = "conventional_standard"


_hcp_Be = Atoms(
    symbols="Be2",
    cell=np.asarray([[ 2.26440844, 0.00000000, 0.00000000],
                     [-1.13220422, 1.96103450, 0.00000000],
                     [ 0.00000000, 0.00000000, 3.56733004]]),
    scaled_positions=[[0.66666667, 0.33333333, 0.75000000],
                      [0.33333333, 0.66666667, 0.25000000]],
    pbc=True
)


@dataclass(frozen=True)
class ElasticConstant:
    """
    Represents a specific c_{ijkl}.
    """

    ijkl: Union[Tuple[int, int, int, int], np.ndarray]
    value: float
    weight: float = 1.0

    def __eq__(self, other):
        if other.ijkl == self.ijkl and other.value == self.value:
            return True
        else:
            return False


@dataclass(frozen=True)
class Crystal:
    """
    A elastic constants container for a crystal.
    """

    name: str
    phase: str
    atoms: Atoms
    bulk_modulus: float
    elastic_constants: List[ElasticConstant]
    temperature: float = 0.0
    eentropy: float = 0.0

    fc2: np.ndarray = None
    supercell: Atoms = None

    def __post_init__(self):
        atoms_utils.set_electron_temperature(self.atoms, self.temperature)


built_in_crystals = {
    "Be": Crystal(name='Be',
                  phase='hcp',
                  bulk_modulus=117,
                  atoms=bulk('Be'),
                  elastic_constants=[ElasticConstant((0, 0, 0, 0), 294),
                                     ElasticConstant((2, 2, 2, 2), 357),
                                     ElasticConstant((1, 2, 1, 2), 162),
                                     ElasticConstant((0, 1, 0, 1), 133),
                                     ElasticConstant((0, 0, 1, 1), 27),
                                     ElasticConstant((0, 0, 2, 2), 14)]),
    "Be/dft": Crystal(name="Be",
                      phase="hcp",
                      bulk_modulus=120,
                      atoms=_hcp_Be,
                      elastic_constants=[ElasticConstant((0, 0, 0, 0), 322),
                                         ElasticConstant((2, 2, 2, 2), 378),
                                         ElasticConstant((1, 2, 1, 2), 162),
                                         ElasticConstant((0, 1, 0, 1), 151),
                                         ElasticConstant((0, 0, 1, 1), 21),
                                         ElasticConstant((0, 0, 2, 2), 8)]),
    "Al": Crystal(name="Al",
                  phase="fcc",
                  bulk_modulus=76,
                  atoms=bulk('Al', cubic=True, crystalstructure='fcc'),
                  elastic_constants=[ElasticConstant((0, 0, 0, 0), 104),
                                     ElasticConstant((0, 0, 1, 1), 73),
                                     ElasticConstant((1, 2, 1, 2), 32)]),
    "Al/bcc": Crystal(name='Al',
                      phase='bcc',
                      bulk_modulus=0,
                      atoms=read(join(data_dir(),
                                      'crystals',
                                      f'Al_bcc_{_identifier}.cif')),
                      elastic_constants=[ElasticConstant((0, 0, 0, 0), 36),
                                         ElasticConstant((0, 0, 1, 1), 86),
                                         ElasticConstant((1, 2, 1, 2), 42)]),
    "Ni": Crystal(name="Ni",
                  phase="fcc",
                  bulk_modulus=188,
                  atoms=bulk("Ni", cubic=True, crystalstructure='fcc'),
                  elastic_constants=[ElasticConstant((0, 0, 0, 0), 276),
                                     ElasticConstant((0, 0, 1, 1), 159),
                                     ElasticConstant((1, 2, 1, 2), 132)]),
    "Mo": Crystal(name="Mo",
                  phase="bcc",
                  bulk_modulus=259,
                  atoms=bulk("Mo", cubic=True, crystalstructure='bcc'),
                  elastic_constants=[ElasticConstant((0, 0, 0, 0), 472),
                                     ElasticConstant((0, 0, 1, 1), 158),
                                     ElasticConstant((1, 2, 1, 2), 106)]),
    "Mo/dft": Crystal(name="Mo/dft",
                      phase="bcc",
                      bulk_modulus=259,
                      atoms=bulk("Mo", cubic=True, a=3.168,
                                 crystalstructure='bcc'),
                      elastic_constants=[ElasticConstant((0, 0, 0, 0), 472),
                                         ElasticConstant((0, 0, 1, 1), 158),
                                         ElasticConstant((1, 2, 1, 2), 106)]),
    "Ni4Mo": Crystal(name="Ni4Mo",
                     phase="cubic",
                     bulk_modulus=0,
                     atoms=read(join(data_dir(),
                                     'crystals',
                                     f'Ni4Mo_mp-11507_{_identifier}.cif')),
                     elastic_constants=[ElasticConstant((0, 0, 0, 0), 300),
                                        ElasticConstant((0, 0, 1, 1), 186),
                                        ElasticConstant((1, 1, 2, 2), 166),
                                        ElasticConstant((1, 1, 1, 1), 313),
                                        ElasticConstant((2, 2, 2, 2), 313),
                                        ElasticConstant((1, 2, 1, 2), 106),
                                        ElasticConstant((0, 2, 0, 2), 130),
                                        ElasticConstant((0, 1, 0, 1), 130)]),
    "Ni3Mo": Crystal(name="Ni3Mo",
                     phase="cubic",
                     bulk_modulus=0,
                     atoms=read(join(data_dir(),
                                     'crystals',
                                     f'Ni3Mo_mp-11506_{_identifier}.cif')),
                     elastic_constants=[ElasticConstant((0, 0, 0, 0), 385),
                                        ElasticConstant((0, 0, 1, 1), 166),
                                        ElasticConstant((0, 0, 2, 2), 145),
                                        ElasticConstant((1, 1, 1, 1), 402),
                                        ElasticConstant((1, 1, 2, 2), 131),
                                        ElasticConstant((2, 2, 2, 2), 402),
                                        ElasticConstant((1, 2, 1, 2), 58),
                                        ElasticConstant((0, 2, 0, 2), 66),
                                        ElasticConstant((0, 1, 0, 1), 94)]),
}


def read_external_crystal(toml_file: Union[str, Path]) -> Crystal:
    """
    Read a `Crystal` from the external toml file.
    """
    with open(toml_file) as fp:
        key_value_pairs = dict(toml.load(fp))

        ase_format = key_value_pairs.pop('format')
        name = key_value_pairs.pop('name')
        phase = key_value_pairs.pop('phase')
        real_path = realpath(join(dirname(toml_file),
                                  key_value_pairs.pop('file')))
        bulk_modulus = key_value_pairs.pop('bulk_modulus', 0)
        eentropy = key_value_pairs.pop('eentropy', 0.0)

        val = key_value_pairs.pop('temperature', 0)
        if isinstance(val, str):
            if val.endswith('eV'):
                temperature = float(val[:-2].strip())
            elif val.endswith('k') or val.endswith('K'):
                temperature = float(val[:-1].strip()) * kB
            else:
                temperature = float(val)
        else:
            temperature = val

        if real_path is None:
            atoms = bulk(name)
        else:
            atoms = read(real_path, format=ase_format)

        supercell_path = realpath(join(dirname(toml_file),
                                  key_value_pairs.pop('supercell')))
        if supercell_path:
            supercell = read(supercell_path, format=ase_format)
        else:
            supercell = None
        fc2_npy = realpath(join(dirname(toml_file),
                           key_value_pairs.pop('fc2')))
        if fc2_npy:
            fc2 = np.load(fc2_npy)
        else:
            fc2 = None

        constants = []
        for key, value in key_value_pairs.items():
            assert len(key) == 3
            assert key[0] == 'c'
            vi = int(key[1])
            vj = int(key[2])
            ijkl = voigt_to_ijkl(vi, vj, is_py_index=False)

            if np.isscalar(value):
                weight = 1.0
                cijkl = float(value)
            elif isinstance(value, (tuple, list)):
                assert len(value) == 2
                cijkl, weight = float(value[0]), float(value[1])
            else:
                raise ValueError("The value of Cij should be a float or list")

            constants.append(ElasticConstant(ijkl, value=cijkl, weight=weight))

        atoms_utils.set_electron_temperature(atoms, temperature)
        return Crystal(name=name,
                       phase=phase,
                       bulk_modulus=bulk_modulus,
                       temperature=temperature,
                       atoms=atoms,
                       eentropy=eentropy,
                       elastic_constants=constants,
                       supercell=supercell,
                       fc2=fc2)


def get_crystal(crystal_or_name_or_file: Union[str, Crystal, Path]) -> Crystal:
    """
    Return a `Crystal` object.

    Parameters
    ----------
    crystal_or_name_or_file : Unit[str, Crystal, Path]
        Maybe a str or a `Crystal` object. If str, it can be either a filename
        or one of the names of the built-in crystals.

    Returns
    -------
    crystal : Crystal
        A `Crystal` object.

    """
    if isinstance(crystal_or_name_or_file, (str, Path)):
        crystal_or_name_or_file = str(crystal_or_name_or_file)
        if crystal_or_name_or_file.endswith('toml'):
            crystal = read_external_crystal(crystal_or_name_or_file)
        else:
            crystal = built_in_crystals[crystal_or_name_or_file]
    elif not isinstance(crystal_or_name_or_file, Crystal):
        raise ValueError(
            "`crystal` must be a str or a `Crystal` object!")
    else:
        crystal = crystal_or_name_or_file
    return crystal
