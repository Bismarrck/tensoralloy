"""
Utility functions for manipulating `ase.Atoms`.
"""
from __future__ import print_function, absolute_import

import numpy as np
from ase import Atoms
from typing import Any, Union

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def __get_property(atoms: Atoms, prop: str, default_value: Any):
    if prop in atoms.info:
        return atoms.info.get(prop)
    elif 'data' in atoms.info and prop in atoms.info['data']:
        return atoms.info['data'][prop]
    elif 'key_value_pairs' in atoms.info and \
            prop in atoms.info['key_value_pairs']:
        return atoms.info['key_value_pairs'][prop]
    else:
        return default_value


def __set_property(atoms: Atoms, prop: str, value: Any):
    atoms.info[prop] = value


def get_electron_temperature(atoms: Atoms) -> float:
    """
    Return the electron temperature (eV).
    """
    return __get_property(atoms, 'etemperature', 0.0)


def set_electron_temperature(atoms: Atoms, t: float):
    """
    Set the electron temperature (eV).
    """
    __set_property(atoms, 'etemperature', t)


def get_electron_entropy(atoms: Atoms) -> float:
    """
    Return the electron entropy (unitless).
    """
    return __get_property(atoms, 'eentropy', 0.0)


def set_electron_entropy(atoms: Atoms, eentropy: float):
    """
    Set the electron entropy.
    """
    __set_property(atoms, 'eentropy', eentropy)


def get_polar_tensor(atoms: Atoms) -> np.ndarray:
    """
    Return the polarizability tensor (xx, yy, zz, yz, xz, xy)
    """
    return __get_property(atoms, "polar", np.zeros(6))


def set_polar_tensor(atoms: Atoms, val):
    """
    Set the polarizability tensor.
    """
    if np.isscalar(val):
        val = np.ones(6) * val
    else:
        val = np.asarray(val)
        if len(val.shape) == 1:
            if len(val) != 6:
                raise ValueError("`polar` should be a rank-1 tensor with "
                                 "6 elements: xx yy zz yz xz xy")
        elif len(val.shape) == 2:
            if val.shape[0] != 3 or val.shape[1] != 3:
                raise ValueError("`polar` should be a 3x3 tensor")
            xx, yy, zz, yz, xz, xy = val[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
            val = np.asarray([xx, yy, zz, yz, xz, xy])
        else:
            raise ValueError("`polar` should be a scalar, vector or matrix")
    __set_property(atoms, "polar", val)
