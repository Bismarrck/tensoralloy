# coding=utf-8
from __future__ import print_function, absolute_import

from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential, EamFSPotential
from tensoralloy.nn.eam.potentials.sutton90 import AgSutton90
from tensoralloy.nn.eam.potentials.zjw04 import Zjw04, Zjw04xc, Zjw04uxc
from tensoralloy.nn.eam.potentials.msah11 import AlFeMsah11

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


available_potentials = {
    'sutton90': AgSutton90,
    'zjw04': Zjw04,
    'zjw04xc': Zjw04xc,
    'zjw04uxc': Zjw04uxc,
    'msah11': AlFeMsah11,
}
