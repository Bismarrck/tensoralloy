# coding=utf-8
"""
This module defines the abstract class of all atomic descriptors.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import abc
from collections import Counter
from typing import List, Dict, Tuple

from tensoralloy.utils import get_kbody_terms
from tensoralloy.misc import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AtomicDescriptorInterface(abc.ABC):
    """
    The required interafces for all atomic descriptor classes.
    """

    @property
    @abc.abstractmethod
    def elements(self):
        """
        Return a list of str as the ordered unique elements.
        """
        pass

    @property
    @abc.abstractmethod
    def cutoff(self):
        """
        Return the cutoff radius.
        """
        pass

    @property
    @abc.abstractmethod
    def max_occurs(self) -> Counter:
        """
        Return the maximum occurance of each type of element.
        """
        pass

    @property
    @abc.abstractmethod
    def k_max(self) -> int:
        """
        Return the maximum k for the many-body expansion scheme.
        """
        pass

    @property
    @abc.abstractmethod
    def all_kbody_terms(self) -> List[str]:
        """
        A list of str as the ordered k-body terms.
        """
        pass

    @property
    @abc.abstractmethod
    def kbody_terms(self) -> Dict[str, List[str]]:
        """
        A dict of (element, kbody_terms) as the k-body terms for each type of
        elements.
        """
        pass


class AtomicDescriptor(AtomicDescriptorInterface):
    """
    The base class for all kinds of atomic descriptors.
    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, rc, elements: List[str], k_max, periodic=True):
        """
        Initialization method.
        """
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
    def cutoff(self) -> float:
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
    def _get_pbc_displacements(shift, cells):
        """
        Return the periodic boundary shift displacements.

        Parameters
        ----------
        shift : tf.Tensor
            A `float64` tensor of shape `[-1, 3]` as the cell shift vector.
        cells : tf.Tensor
            A `float64` tensor of shape `[3, 3]` as the cell.

        Returns
        -------
        Dij : tf.Tensor
            A `float64` tensor of shape `[-1, 3]` as the periodic displacements
            vector.

        """
        return tf.matmul(shift, cells, name='displacements')

    def _get_rij(self, R, cells, ilist, jlist, shift, name):
        """
        Return the subgraph to compute `rij`.
        """
        with tf.name_scope(name):
            Ri = self.gather_fn(R, ilist, 'Ri')
            Rj = self.gather_fn(R, jlist, 'Rj')
            Dij = tf.subtract(Rj, Ri, name='Dij')
            if self._periodic:
                pbc = self._get_pbc_displacements(shift, cells)
                Dij = tf.add(Dij, pbc, name='pbc')
            # By adding `eps` to the reduced sum NaN can be eliminated.
            with tf.name_scope("safe_norm"):
                eps = tf.constant(1e-14, dtype=tf.float64, name='eps')
                return tf.sqrt(tf.reduce_sum(
                    tf.square(Dij, name='Dij2'), axis=-1) + eps)

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
