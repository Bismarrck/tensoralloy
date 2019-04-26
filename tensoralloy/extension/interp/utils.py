# -*- coding: utf-8 -*-
"""
The helper function for this package.
"""
from __future__ import print_function, absolute_import

import os
import sysconfig
import tensorflow as tf

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["load_op_library"]


def load_op_library(name):
    """
    Load a cpp libraray.
    """
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    dirname = os.path.dirname(os.path.abspath(__file__))
    libfile = os.path.join(dirname, name)
    if suffix is not None:
        libfile += suffix
    else:
        libfile += ".so"
    return tf.load_op_library(libfile)
