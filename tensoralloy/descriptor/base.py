# coding=utf-8
"""
This module defines the abstract class of all atomic descriptors.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import List

from tensoralloy.misc import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AtomicDescriptor:
    """
    The base class for all kinds of atomic descriptors.
    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, rc, elements: List[str], periodic=True):
        """
        Initialization method.
        """
        self._rc = rc
        self._elements = sorted(elements)
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
        """
        raise NotImplementedError(
            "This method must be overridden by a subclass!")
