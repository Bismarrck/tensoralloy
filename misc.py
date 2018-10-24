# coding=utf-8
"""
This module defines miscellaneous functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
from unittest import SkipTest
from os.path import dirname, isdir, join
from os import makedirs

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# The random seed.
RANDOM_STATE = 611


class Defaults:
    """
    A dataclass storing default parameters.
    """
    eta = np.array([0.05, 4.0, 20.0, 80.0])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])

    n_etas = 4
    n_betas = 1
    n_gammas = 2
    n_zetas = 2

    seed = RANDOM_STATE


def skip(func):
    """
    A decorator for skipping tests.
    """
    def _():
        raise SkipTest("Test %s is skipped" % func.__name__)
    _.__name__ = func.__name__
    return _


def check_path(path):
    """
    Make sure the given path is accessible.
    """
    dst = dirname(path)
    if not isdir(dst):
        makedirs(dst)
    return path


def test_dir():
    """
    Return the directory of `test_files`.
    """
    return join(dirname(__file__), "test_files")
