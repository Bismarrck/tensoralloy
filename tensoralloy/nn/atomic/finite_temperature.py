#!coding=utf-8
"""
A special module for modeling finite-temperature systems.
"""
from __future__ import print_function, absolute_import

from dataclasses import dataclass
from typing import List

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@dataclass
class FiniteTemperatureOptions:
    """
    Options for modeling finite-temperature systems.
    """
    algorithm: str = "off"
    activation: str = "softplus"
    layers: List[int] = (128, 128)
    biased_eentropy: bool = True
    biased_internal_energy: bool = False
    biased_free_energy: bool = False

    def __post_init__(self):
        if self.algorithm not in ("off", "zero", "semi", "full"):
            raise ValueError(f"Algorithm {self.algorithm} is unknown!")

    @property
    def on(self):
        """
        Return True if finite-temperature is enabled.
        """
        return self.algorithm != "off"
