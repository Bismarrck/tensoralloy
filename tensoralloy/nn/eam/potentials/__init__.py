#!coding=utf-8
"""
Global available potentials.
"""
from __future__ import print_function, absolute_import

from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.nn.eam.potentials.potentials import EamFSPotential
from tensoralloy.nn.eam.potentials.sutton90 import AgSutton90
from tensoralloy.nn.eam.potentials.zjw04 import Zjw04, Zjw04xc, Zjw04uxc
from tensoralloy.nn.eam.potentials.zjw04 import Zjw04xcp
from tensoralloy.nn.eam.potentials.msah11 import AlFeMsah11
from tensoralloy.nn.eam.potentials.grimmes import RWGrimes
from tensoralloy.nn.eam.potentials.mishin import MishinH
from tensoralloy.nn.eam.potentials.agrawal import AgrawalBe

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


available_potentials = {
    'sutton90': AgSutton90,
    'zjw04': Zjw04,
    'zjw04xc': Zjw04xc,
    'zjw04uxc': Zjw04uxc,
    'zjw04xcp': Zjw04xcp,
    'msah11': AlFeMsah11,
    'grimes': RWGrimes,
    "mishinh": MishinH,
    "Be/1": AgrawalBe,
}
