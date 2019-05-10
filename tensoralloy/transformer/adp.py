# coding=utf-8
"""
This module defines the transformers for the ADP potential.
"""
from __future__ import print_function, absolute_import

from collections import Counter
from typing import List

from tensoralloy.transformer.eam import EAMTransformer, BatchEAMTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["ADPTransformer", "BatchADPTransformer"]


class ADPTransformer(EAMTransformer):
    """
    The feature transformer for the ADP potential.
    """

    def __init__(self, rc: float, elements: List[str]):
        """
        Initialization method.
        """
        super(ADPTransformer, self).__init__(rc=rc, elements=elements)
        self._graph_scope_name = "ADP"


class BatchADPTransformer(BatchEAMTransformer):
    """
    A batch implementation of feature transformer for the ADP potential.
    """

    def __init__(self, rc: float, max_occurs: Counter, nij_max: int,
                 nnl_max: int, batch_size=None, use_forces=True,
                 use_stress=False):
        """
        Initialization method.
        """
        super(BatchADPTransformer, self).__init__(
            rc=rc, max_occurs=max_occurs, nij_max=nij_max, nnl_max=nnl_max,
            batch_size=batch_size, use_forces=use_forces, use_stress=use_stress
        )
        self._graph_scope_name = "ADP"

    @property
    def descriptor(self):
        """
        Return the name of the descriptor.
        """
        return "adp"
