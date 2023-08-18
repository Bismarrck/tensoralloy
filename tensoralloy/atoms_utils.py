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
