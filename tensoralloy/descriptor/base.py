# coding=utf-8
"""
This module defines the abstract class of all atomic descriptors.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from ase.data import chemical_symbols
from typing import List, Dict, Tuple

from tensoralloy.utils import get_kbody_terms
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AtomicDescriptor:
    """
    The base class for all kinds of atomic descriptors.
    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, rc, elements: List[str], angular=False, periodic=True):
        """
        Initialization method.
        """
        for element in elements:
            if element not in chemical_symbols:
                raise ValueError(f"{element} is not a valid chemical symbol!")

        all_kbody_terms, kbody_terms, elements = \
            get_kbody_terms(elements, angular=angular)

        self._rc = rc
        self._angular = angular
        self._all_kbody_terms = all_kbody_terms
        self._kbody_terms = kbody_terms
        self._elements = elements
        self._n_elements = len(elements)
        self._periodic = periodic

    @property
    def rc(self) -> float:
        """
        Return the cutoff radius.
        """
        return self._rc

    @property
    def elements(self) -> List[str]:
        """
        Return a list of str as the ordered unique elements.
        """
        return self._elements

    @property
    def n_elements(self) -> int:
        """
        Return the total number of unique elements.
        """
        return self._n_elements

    @property
    def periodic(self):
        """
        Return True if this can be applied to periodic structures. For
        non-periodic molecules some Ops can be ignored.
        """
        return self._periodic

    @property
    def max_occurs(self):
        """
        There is no restriction for the occurances of an element.
        """
        return {el: np.inf for el in self._elements}

    @property
    def angular(self):
        """
        Return True if angular interactions are included.
        """
        return self._angular

    @property
    def all_kbody_terms(self):
        """
        A list of str as the ordered k-body terms.
        """
        return self._all_kbody_terms

    @property
    def kbody_terms(self):
        """
        A dict of (element, kbody_terms) as the k-body terms for each type of
        elements.
        """
        return self._kbody_terms

    @staticmethod
    def get_pbc_displacements(shift, cell, dtype=tf.float64):
        """
        Return the periodic boundary shift displacements.

        Parameters
        ----------
        shift : tf.Tensor or array_like
            A `float64` or `float32` tensor of shape `[-1, 3]` as the cell shift
            vector.
        cell : tf.Tensor or array_like
            A `float64` or `float32` tensor of shape `[3, 3]` as the cell.
        dtype : DType
            The corresponding data type of `shift` and `cell`.

        Returns
        -------
        Dij : tf.Tensor
            A `float64` tensor of shape `[-1, 3]` as the periodic displacements
            vector.

        """
        return tf.matmul(shift, cell, name='displacements')

    def get_rij(self, R, cell, ilist, jlist, shift, name):
        """
        Return the interatomic distances array, `rij`, and the corresponding
        differences.

        Returns
        -------
        rij : tf.Tensor
            The interatomic distances.
        dij : tf.Tensor
            The differences of `Rj - Ri`.

        """
        with tf.name_scope(name):
            dtype = get_float_dtype()
            Ri = self.gather_fn(R, ilist, 'Ri')
            Rj = self.gather_fn(R, jlist, 'Rj')
            Dij = tf.subtract(Rj, Ri, name='Dij')
            if self._periodic:
                pbc = self.get_pbc_displacements(shift, cell, dtype=dtype)
                Dij = tf.add(Dij, pbc, name='pbc')
            # By adding `eps` to the reduced sum NaN can be eliminated.
            with tf.name_scope("safe_norm"):
                eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
                rij = tf.sqrt(tf.reduce_sum(
                    tf.square(Dij, name='Dij2'), axis=-1) + eps)
                return rij, Dij

    def get_g_shape(self, features: dict):
        """
        Return the shape of the descriptor matrix.
        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")

    def get_v2g_map(self, features: dict, prefix: str = None):
        """
        A wrapper function to get `v2g_map` or re-indexed `v2g_map`.
        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")

    def get_row_split_sizes(self, placeholders):
        """
        Return the sizes of the rowwise splitted subsets of `g`.
        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")

    @staticmethod
    def get_row_split_axis():
        """
        Return the axis to rowwise split `g`.
        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")

    def build_graph(self, features: dict):
        """
        Build the tensorflow graph for computing atomic descriptors.

        Returns
        -------
        ops : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of (element, (value, mask)) where `element` is a str, value
            is the tensor of descriptors of `element` and `mask` represents the
            mask of `value`.

        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")
