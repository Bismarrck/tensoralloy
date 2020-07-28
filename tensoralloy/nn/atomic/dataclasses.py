#!coding=utf-8
"""
Data classes for `tensoralloy.nn.atomic`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import Dict
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
