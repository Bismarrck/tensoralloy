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


def get_conversion_units(units: Dict[str, str]) -> (float, float, float):
    """
    Return the conversion factors:
        * 'energy' should be converted to 'eV'
        * 'forces' should be converted to 'eV / Angstrom'
        * 'stress' should be converted to 'eV / Angstrom**3'

    """
    _units = {
        'kbar': 0.1 * ase.units.GPa,
    }

    def _parse_unit(unit: str):
        if hasattr(ase.units, unit):
            return getattr(ase.units, unit)
        elif unit in _units:
            return _units[unit]
        else:
            try:
                return float(unit)
            except Exception:
                raise ValueError("Unknown unit: {}".format(unit))

    def _parse_comb(comb: str):
        if '/' in comb or '*' in comb:
            values = [_parse_unit(unit) for unit in re.split(r'[/*]', comb)]
            total = values[0]
            index = 1
            for i in range(len(comb)):
                if comb[i] == '/':
                    total /= values[index]
                    index += 1
                elif comb[i] == '*':
                    total *= values[index]
                    index += 1
                if index == len(values):
                    break
            return total
        else:
            return _parse_unit(comb)

    eV = ase.units.eV
    Angstrom = ase.units.Angstrom

    if 'energy' not in units:
        to_eV = 1.0
    else:
        to_eV = _parse_comb(units['energy']) / eV

    if 'forces' not in units:
        to_eV_Angstrom = 1.0
    else:
        to_eV_Angstrom = _parse_comb(units['forces']) / eV / Angstrom

    if 'stress' not in units:
        to_eV_Ang3 = 1.0
    else:
        to_eV_Ang3 = _parse_comb(units['stress']) / eV / Angstrom**3

    return to_eV, to_eV_Angstrom, to_eV_Ang3
