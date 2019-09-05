# coding=utf-8
"""
`tensoralloy.nn` module defines various NN models.
"""
from __future__ import print_function, absolute_import

from tensoralloy.nn.atomic.resnet import AtomicResNN
from tensoralloy.nn.atomic.atomic import AtomicNN
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.nn.eam.fs import EamFsNN
from tensoralloy.nn.eam.adp import AdpNN

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["AtomicResNN", "AtomicNN",
           "EamAlloyNN", "EamFsNN", "AdpNN"]
