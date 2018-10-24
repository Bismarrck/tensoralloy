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


class AttributeDict(dict):
    """
    A subclass of `dict` with attribute-style access.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class Defaults:
    """
    A dataclass storing default parameters.
    """
    rc = 6.0

    eta = np.array([0.05, 4.0, 20.0, 80.0])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])

    n_etas = 4
    n_betas = 1
    n_gammas = 2
    n_zetas = 2

    seed = RANDOM_STATE


def safe_select(a, b):
    """
    A helper function to return `a` if it is neither None nor empty.
    """
    if a is None or len(a) == 0:
        return b
    return a


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
