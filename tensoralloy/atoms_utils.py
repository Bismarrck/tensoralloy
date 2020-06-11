"""
Utility functions for manipulating `ase.Atoms`.
"""
from __future__ import print_function, absolute_import

from ase import Atoms

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_pulay_stress(atoms: Atoms) -> float:
    """
    Return the pulay stress (eV/Ang**3).
    """
    if 'pulay_stress' in atoms.info:
        return atoms.info.get('pulay_stress')
    else:
        # The dict `atoms.info` cannot be written to a sqlite3 database
        # direclty. `pulay_stress` will be saved in the `key_value_pairs`
        # (tensoralloy.io.read, line 113).
        key_value_pairs = atoms.info.get('key_value_pairs', {})
        return key_value_pairs.get('pulay_stress', 0.0)


def set_pulay_stress(atoms: Atoms, pulay: float):
    """
    Set the pulay stress.
    """
    atoms.info['pulay_stress'] = pulay
