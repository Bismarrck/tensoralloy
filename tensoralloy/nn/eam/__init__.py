# coding=utf-8
"""
This module defines `EamNN` and its variants.
"""
from __future__ import print_function, absolute_import

from tensoralloy.nn.eam.fs import EamFsNN
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.nn.eam.adp import AdpNN

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["EamFsNN", "EamAlloyNN", "AdpNN"]
