#!coding=utf-8
"""
This module defines the transformers for the DeePMD model.
"""
from __future__ import print_function, absolute_import

from collections import Counter
from typing import List

from tensoralloy.transformer.eam import EAMTransformer, BatchEAMTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["DeePMDTransformer", "BatchDeePMDTransformer"]


class DeePMDTransformer(EAMTransformer):
    """
    The feature transformer for the ADP potential.
    """

    def __init__(self, rc: float, elements: List[str]):
        """
        Initialization method.
        """
        super(DeePMDTransformer, self).__init__(rc=rc, elements=elements)
        self._graph_scope_name = "DeePMD"


class BatchDeePMDTransformer(BatchEAMTransformer):
    """
    A batch implementation of feature transformer for the ADP potential.
    """

    def __init__(self, rc: float, max_occurs: Counter, nij_max: int,
                 nnl_max: int, batch_size=None, use_forces=True,
                 use_stress=False):
        """
        Initialization method.
        """
        super(BatchDeePMDTransformer, self).__init__(
            rc=rc, max_occurs=max_occurs, nij_max=nij_max, nnl_max=nnl_max,
            batch_size=batch_size, use_forces=use_forces, use_stress=use_stress
        )
        self._graph_scope_name = "DeePMD"

    @property
    def descriptor(self):
        """
        Return the name of the descriptor.
        """
        return "deepmd"
