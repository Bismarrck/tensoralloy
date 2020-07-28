"""
Utility functions for manipulating `ase.Atoms`.
"""
from __future__ import print_function, absolute_import

from ase import Atoms
from typing import Any

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def __get_property(atoms: Atoms, prop: str, default_value: Any):
    if prop in atoms.info:
        return atoms.info.get(prop)
    else:
        # The dict `atoms.info` cannot be written to a sqlite3 database
        # direclty. `prop` will be saved in the `key_value_pairs`
        # (tensoralloy.io.read, line 113).
        key_value_pairs = atoms.info.get('key_value_pairs', {})
        return key_value_pairs.get(prop, default_value)


def __set_property(atoms: Atoms, prop: str, value: Any):
    atoms.info[prop] = value


def get_pulay_stress(atoms: Atoms) -> float:
    """
    Return the pulay stress (eV/Ang**3).
    """
    return __get_property(atoms, 'pulay_stress', 0.0)


def set_pulay_stress(atoms: Atoms, pulay: float):
    """
    Set the pulay stress.
    """
    __set_property(atoms, 'pulay_stress', pulay)


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
