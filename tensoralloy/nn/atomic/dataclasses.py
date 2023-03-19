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
    activation: str = "softplus"
    layers: List[int] = (128, 128)
    algo: str = "default"
