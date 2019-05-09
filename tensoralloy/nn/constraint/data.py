#!coding=utf-8
"""
Built-in cyrstal data for constraints.
"""
from __future__ import print_function, absolute_import

import numpy as np

from dataclasses import dataclass
from os.path import join
from typing import Union, List
from ase import Atoms
from ase.build import bulk
from ase.io import read

from tensoralloy.nn.constraint.elastic import _identifier
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@dataclass(frozen=True)
class ElasticConstant:
    """
    Represents a specific c_{ijkl}.
    """

    ijkl: Union[List[int], np.ndarray]
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
    elastic_constants: List[ElasticConstant]


built_in_crystals = {
    "Al": Crystal(name="Al",
                  phase="fcc",
                  atoms=bulk('Al', cubic=True, crystalstructure='fcc'),
                  elastic_constants=[ElasticConstant([0, 0, 0, 0], 104),
                                     ElasticConstant([0, 0, 1, 1], 73),
                                     ElasticConstant([1, 2, 1, 2], 32)]),
    "Al/bcc": Crystal(name='Al',
                      phase='bcc',
                      atoms=read(join(test_dir(),
                                      'crystals',
                                      f'Al_bcc_{_identifier}.cif')),
                      elastic_constants=[ElasticConstant([0, 0, 0, 0], 36),
                                         ElasticConstant([0, 0, 1, 1], 86),
                                         ElasticConstant([1, 2, 1, 2], 42)]),
    "Ni": Crystal(name="Ni",
                  phase="fcc",
                  atoms=bulk("Ni", cubic=True, crystalstructure='fcc'),
                  elastic_constants=[ElasticConstant([0, 0, 0, 0], 276),
                                     ElasticConstant([0, 0, 1, 1], 159),
                                     ElasticConstant([1, 2, 1, 2], 132)]),
    "Mo": Crystal(name="Mo",
                  phase="fcc",
                  atoms=bulk("Mo", cubic=True, crystalstructure='bcc'),
                  elastic_constants=[ElasticConstant([0, 0, 0, 0], 472),
                                     ElasticConstant([0, 0, 1, 1], 158),
                                     ElasticConstant([1, 2, 1, 2], 106)]),
    "Ni4Mo": Crystal(name="Ni4Mo",
                     phase="cubic",
                     atoms=read(join(test_dir(),
                                     'crystals',
                                     f'Ni4Mo_mp-11507_{_identifier}.cif')),
                     elastic_constants=[ElasticConstant([0, 0, 0, 0], 300),
                                        ElasticConstant([0, 0, 1, 1], 186),
                                        ElasticConstant([1, 1, 2, 2], 166),
                                        ElasticConstant([1, 1, 1, 1], 313),
                                        ElasticConstant([2, 2, 2, 2], 313),
                                        ElasticConstant([1, 2, 1, 2], 106),
                                        ElasticConstant([0, 2, 0, 2], 130),
                                        ElasticConstant([0, 1, 0, 1], 130)]),
    "Ni3Mo": Crystal(name="Ni3Mo",
                     phase="cubic",
                     atoms=read(join(test_dir(),
                                     'crystals',
                                     f'Ni3Mo_mp-11506_{_identifier}.cif')),
                     elastic_constants=[ElasticConstant([0, 0, 0, 0], 385),
                                        ElasticConstant([0, 0, 1, 1], 166),
                                        ElasticConstant([0, 0, 2, 2], 145),
                                        ElasticConstant([1, 1, 1, 1], 402),
                                        ElasticConstant([1, 1, 2, 2], 131),
                                        ElasticConstant([2, 2, 2, 2], 402),
                                        ElasticConstant([1, 2, 1, 2], 58),
                                        ElasticConstant([0, 2, 0, 2], 66),
                                        ElasticConstant([0, 1, 0, 1], 94)]),
}
