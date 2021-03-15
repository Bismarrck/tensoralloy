# coding=utf-8
"""
The module `tensoralloy.transformer` is used to convert `Atoms` objects to
features.
"""
from __future__ import print_function, absolute_import

from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.transformer.universal import BatchUniversalTransformer
from tensoralloy.transformer.kmc import KMCTransformer
from tensoralloy.transformer.kmc import KMCPreComputedTransformer
from tensoralloy.transformer.vap import VirtualAtomMap

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["VirtualAtomMap",
           "UniversalTransformer", "BatchUniversalTransformer",
           "KMCTransformer", "KMCPreComputedTransformer"]
