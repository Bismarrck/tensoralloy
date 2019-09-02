#!coding=utf-8
"""
The Modified Embedded-Atom Method (Lenosky Style) descriptor.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from typing import List

from tensoralloy.descriptor.base import AtomicDescriptor
from tensoralloy.utils import AttributeDict, get_elements_from_kbody_term
from tensoralloy.utils import GraphKeys

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class MEAM(AtomicDescriptor):
    """
    A tensorflow based implementation of Modified Embedded-Atom Method (MEAM).

    References
    ----------
    Modelling Simul. Mater. Sci. Eng. 8 (2000) 825–841.
    Computational Materials Science 124 (2016) 204–210.

    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, rc: float, elements: List[str], angular_rc_scale=1.0):
        """
        Initialization method.

        Parameters
        ----------
        rc : float
            The cutoff radius.
        elements : List[str]
            A list of str as the ordered elements.
        angular_rc_scale : float
            The scaling factor of `rc` for angular interactions.

        """
        super(MEAM, self).__init__(rc, elements, angular=True, periodic=True)

        self._angular_rc_scale = angular_rc_scale
        self._angular_rc = rc * angular_rc_scale
