# coding=utf-8
"""
This module defines miscellaneous functions and vars.
"""
from __future__ import print_function, absolute_import

from collections import Counter

import numpy as np
from unittest import SkipTest
from os.path import dirname, isdir, join
from os import makedirs

from ase import Atoms
from ase.io import read

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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


# The random seed.
RANDOM_STATE = 611

# Pd3O2
Pd3O2 = Atoms(symbols='Pd3O2', pbc=np.array([True, True, False], dtype=bool),
              cell=np.array([[7.78, 0., 0.],
                             [0., 5.50129076, 0.],
                             [0., 0., 15.37532269]]),
              positions=np.array([[3.89, 0., 8.37532269],
                                  [0., 2.75064538, 8.37532269],
                                  [3.89, 2.75064538, 8.37532269],
                                  [5.835, 1.37532269, 8.5],
                                  [5.835, 7.12596807, 8.]]))

# A permutation of Pd3O2
Pd2O2Pd = Atoms(symbols='Pd2O2Pd',
                pbc=np.array([True, True, False], dtype=bool),
                cell=np.array([[7.78, 0., 0.],
                               [0., 5.50129076, 0.],
                               [0., 0., 15.37532269]]),
                positions=np.array([[3.89, 0., 8.37532269],
                                    [0., 2.75064538, 8.37532269],
                                    [5.835, 1.37532269, 8.5],
                                    [5.835, 7.12596807, 8.],
                                    [3.89, 2.75064538, 8.37532269]]))

qm7m = AttributeDict(
    max_occurs=Counter({'C': 5, 'H': 8, 'O': 2}),
    nij_max=198,
    nijk_max=1217,
    trajectory=read('test_files/qm7m/qm7m.xyz', index=':', format='xyz'),
)

for _atoms in qm7m.trajectory:
    # Setting the boundary cell is important because `neighbor_list` may give
    # totally different results.
    _atoms.set_cell(np.eye(3) * 20.0)


class Defaults:
    """
    A dataclass storing default parameters.
    """
    rc = 6.0
    k_max = 2

    eta = np.array([0.05, 4.0, 20.0, 80.0])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])

    n_etas = 4
    n_betas = 1
    n_gammas = 2
    n_zetas = 2

    seed = RANDOM_STATE

    variable_moving_average_decay = 0.999

    activation = 'leaky_relu'
    hidden_sizes = [64, 32]
    learning_rate = 0.01


def safe_select(a, b):
    """
    A helper function to return `a` if it is neither None nor empty.
    """
    if a is None:
        return b
    elif hasattr(a, '__len__'):
        if len(a) == 0:
            return b
    elif isinstance(a, str):
        if a == '':
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
    return join(dirname(__file__), "..", "test_files")


def brange(start, stop, batch_size):
    """
    Range from `start` to `stop` given a batch size and return the start and
    stop of each batch.

    Parameters
    ----------
    start : int
        The start number of a sequence.
    stop : int,
        The end number of a sequence.
    batch_size : int
        The size of each batch.

    Yields
    ------
    istart : int
        The start number of a batch.
    istop : int
        The end number of a batch.

    """
    istart = start
    while istart < stop:
        istop = min(istart + batch_size, stop)
        yield istart, istop
        istart = istop
