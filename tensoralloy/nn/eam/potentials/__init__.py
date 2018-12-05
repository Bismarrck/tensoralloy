# coding=utf-8
from __future__ import print_function, absolute_import

from .sutton90 import AgSutton90
from .zjw04 import AlCuZJW04

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


available_potentials = {
    'sutton90': AgSutton90(),
    'zjw04': AlCuZJW04()
}
