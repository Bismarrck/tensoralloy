# coding=utf-8
from __future__ import print_function, absolute_import

from .potentials import EamAlloyPotential, EamFSPotential
from .sutton90 import AgSutton90
from .zjw04 import Zjw04
from .msah11 import AlFeMsah11

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


available_potentials = {
    'sutton90': AgSutton90,
    'zjw04': Zjw04,
    'msah11': AlFeMsah11,
}
