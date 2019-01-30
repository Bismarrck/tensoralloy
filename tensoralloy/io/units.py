# coding=utf-8
"""
This module defines the unit conversion function.
"""
from __future__ import print_function, absolute_import

import re
import ase.units

from typing import Dict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


_replace = {
    'eV': ase.units.eV,
    'Hartree': ase.units.Hartree,
    'kcal': ase.units.kcal,
    'mol': ase.units.mol,
    'Bohr': ase.units.Bohr,
    'Angstrom': ase.units.Angstrom,
    'GPa': ase.units.GPa,
    'kbar': 0.1 * ase.units.GPa,
}
_replace = {k: str(v) for k, v in _replace.items()}
_replace = dict((re.escape(k), v) for k, v in _replace.items())
_pattern = re.compile("|".join(_replace.keys()))


def get_conversion_units(units: Dict[str, str]) -> (float, float, float):
    """
    Return the conversion factors:
        * 'energy' should be converted to 'eV'
        * 'forces' should be converted to 'eV / Angstrom'
        * 'stress' should be converted to 'eV / Angstrom**3'

    """

    def _parse_comb(comb: str):
        if not comb:
            return 1.0
        return eval(
            _pattern.sub(lambda m: _replace[re.escape(m.group(0))], comb))

    to_eV = _parse_comb(units.get('energy', None))
    to_eV_Angstrom = _parse_comb(units.get('forces', None))
    to_eV_Ang3 = _parse_comb(units.get('stress', None))

    return to_eV, to_eV_Angstrom, to_eV_Ang3
