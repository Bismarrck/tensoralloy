#!coding=utf-8
"""
Data classes for `tensoralloy.nn.atomic`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import Dict, List
from dataclasses import dataclass
from collections import Counter

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@dataclass(frozen=True)
class AtomicDescriptors:
    """
    The data structure returned by `AtomicNN._get_atomic_descriptors`.
    """
    descriptors: Dict[str, tf.Tensor]
    max_occurs: Counter


@dataclass
class FiniteTemperatureOptions:
    """
    Options for modeling finite-temperature systems.
    """
    algorithm: str = "full"
    activation: str = "softplus"
    layers: List[int] = (128, 128)
    biased_eentropy: bool = True
    biased_internal_energy: bool = False

    def __post_init__(self):
        if self.algorithm not in ("off", "zero", "semi", "full"):
            raise ValueError(f"Algorithm {self.algorithm} is unknown!")

    @property
    def on(self):
        """
        Return True if finite-temperature is enabled.
        """
        return self.algorithm != "off"
