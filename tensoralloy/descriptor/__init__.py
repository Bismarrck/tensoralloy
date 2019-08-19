# coding=utf-8
"""
The module `tensoralloy.descriptor` is used to buid tensor-graph for computing
various descriptors directly from atomic positions.
"""
from __future__ import print_function, absolute_import

from tensoralloy.descriptor.behler import SymmetryFunction
from tensoralloy.descriptor.behler import BatchSymmetryFunction
from tensoralloy.descriptor.cutoff import cosine_cutoff

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'
