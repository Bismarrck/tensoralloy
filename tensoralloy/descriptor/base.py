# coding=utf-8
"""
This module defines the abstract class of all atomic descriptors.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from ase.data import chemical_symbols
from typing import List, Dict, Tuple

from tensoralloy.utils import get_kbody_terms, AttributeDict
from tensoralloy.dtypes import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AtomicDescriptor:
    """
    The base class for all kinds of atomic descriptors.
    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, rc, elements: List[str], k_max, periodic=True):
        """
        Initialization method.
        """
        for element in elements:
            if element not in chemical_symbols:
                raise ValueError(f"{element} is not a valid chemical symbol!")

        all_kbody_terms, kbody_terms, elements = \
            get_kbody_terms(elements, k_max)

        self._rc = rc
        self._k_max = k_max
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
    def k_max(self):
        """
        Return the maximum k for the many-body expansion scheme.
        """
        return self._k_max

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
    def _get_pbc_displacements(shift, cells, dtype=tf.float64):
        """
        Return the periodic boundary shift displacements.

        Parameters
        ----------
        shift : tf.Tensor or array_like
            A `float64` or `float32` tensor of shape `[-1, 3]` as the cell shift
            vector.
        cells : tf.Tensor or array_like
            A `float64` or `float32` tensor of shape `[3, 3]` as the cell.
        dtype : DType
            The corresponding data type of `shift` and `cells`.

        Returns
        -------
        Dij : tf.Tensor
            A `float64` tensor of shape `[-1, 3]` as the periodic displacements
            vector.

        """
        return tf.matmul(shift, cells, name='displacements')

    def _get_rij(self, R, cells, ilist, jlist, shift, name):
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
                pbc = self._get_pbc_displacements(shift, cells, dtype=dtype)
                Dij = tf.add(Dij, pbc, name='pbc')
            # By adding `eps` to the reduced sum NaN can be eliminated.
            with tf.name_scope("safe_norm"):
                eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
                rij = tf.sqrt(tf.reduce_sum(
                    tf.square(Dij, name='Dij2'), axis=-1) + eps)
                return rij, Dij

    def _get_g_shape(self, placeholders):
        """
        Return the shape of the descriptor matrix.
        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")

    def _get_v2g_map(self, placeholders, **kwargs):
        """
        A wrapper function to get `v2g_map` or re-indexed `v2g_map`.
        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")

    def _get_row_split_sizes(self, placeholders):
        """
        Return the sizes of the rowwise splitted subsets of `g`.
        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")

    @staticmethod
    def _get_row_split_axis():
        """
        Return the axis to rowwise split `g`.
        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")

    def build_graph(self, placeholders: AttributeDict):
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
