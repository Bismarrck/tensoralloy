# coding=utf-8
"""
This module implements the Embedded-atom method.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
from collections import Counter
from typing import List, Dict

from tensoralloy.misc import AttributeDict
from tensoralloy.descriptor.base import AtomicDescriptor
from tensoralloy.utils import get_kbody_terms, get_elements_from_kbody_term

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["EAM", "BatchEAM"]


class EAM(AtomicDescriptor):
    """
    A tensorflow based implementation of Embedded Atom Method (EAM).
    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, rc: float, elements: List[str]):
        """
        Initialization method.

        Parameters
        ----------
        rc : float
            The cutoff radius.
        elements : List[str]
            A list of str as the ordered elements.

        """
        all_kbody_terms, kbody_terms, elements = \
            get_kbody_terms(elements, k_max=2)
        kbody_index = {}
        for kbody_term in all_kbody_terms:
            center = get_elements_from_kbody_term(kbody_term)[0]
            kbody_index[kbody_term] = kbody_terms[center].index(kbody_term)

        super(EAM, self).__init__(rc=rc, elements=elements, periodic=True)

        self._k_max = 2
        self._kbody_terms = kbody_terms
        self._all_kbody_terms = all_kbody_terms
        self._max_n_terms = max(map(len, kbody_terms.values()))
        self._kbody_index = kbody_index

    @property
    def all_kbody_terms(self) -> List[str]:
        """
        A list of str as the ordered k-body terms.
        """
        return self._all_kbody_terms

    @property
    def kbody_terms(self) -> Dict[str, List[str]]:
        """
        A dict of (element, kbody_terms) as the k-body terms for each type of
        elements.
        """
        return self._kbody_terms

    def _get_g_shape(self, placeholders):
        """
        Return the shape of the descriptor matrix.
        """
        return [placeholders.n_atoms, self._max_n_terms, placeholders.nnl_max]

    def _get_v2g_map(self, placeholders, **kwargs):
        """
        A wrapper function to get `v2g_map` or re-indexed `v2g_map`.
        """
        return tf.identity(placeholders.v2g_map, name='v2g_map')

    def _get_row_split_sizes(self, placeholders):
        """
        Return the sizes of the rowwise splitted subsets of `g`.
        """
        return placeholders.row_splits

    @staticmethod
    def _get_row_split_axis():
        """
        Return the axis to rowwise split `g`.
        """
        return 0

    def _split_descriptors(self, g, placeholders) -> Dict[str, tf.Tensor]:
        """
        Split the descriptors into `N_element` subsets.
        """
        with tf.name_scope("Split"):
            row_split_sizes = self._get_row_split_sizes(placeholders)
            row_split_axis = self._get_row_split_axis()
            splits = tf.split(
                g, row_split_sizes, axis=row_split_axis, name='rows')[1:]
            return dict(zip(self._elements, splits))

    def build_graph(self, placeholders: AttributeDict, split=True):
        """
        Get the tensorflow based computation graph of the EAM model.
        """
        with tf.name_scope("EAM"):
            r = self._get_rij(placeholders.positions,
                              placeholders.cells,
                              placeholders.ilist,
                              placeholders.jlist,
                              placeholders.shift,
                              name='rij')
            shape = self._get_g_shape(placeholders)
            v2g_map = self._get_v2g_map(placeholders)
            g = tf.scatter_nd(v2g_map, r, shape, name='g')
            if split:
                return self._split_descriptors(g, placeholders)
            else:
                return g


class BatchEAM(EAM):
    """
    A special implementation of the EAM model for batch training and evaluation.
    """

    gather_fn = staticmethod(tf.batch_gather)

    def __init__(self, rc, max_occurs: Counter, elements: List[str],
                 nij_max: int, nnl_max: int, batch_size: int):
        """
        Initialization method.
        """
        super(BatchEAM, self).__init__(rc=rc, elements=elements)

        self._max_occurs = max_occurs
        self._max_n_atoms = sum(max_occurs.values()) + 1
        self._nij_max = nij_max
        self._nnl_max = nnl_max
        self._batch_size = batch_size

    @property
    def nij_max(self):
        """
        Return the maximum allowed length of the flatten neighbor list.
        """
        return self._nij_max

    @property
    def nnl_max(self):
        """
        Return the maximum number of neighbors of the same type for a single
        atom.
        """
        return self._nnl_max

    @staticmethod
    def _get_pbc_displacements(shift, cells):
        """
        Return the periodic boundary shift displacements.

        Parameters
        ----------
        shift : tf.Tensor
            A `float64` tensor of shape `[batch_size, ndim, 3]` as the cell
            shift vector where `ndim == nij_max` or `ndim == nijk_max`.
        cells : tf.Tensor
            A `float64` tensor of shape `[batch_size, 3, 3]` as the cells.

        Returns
        -------
        Dij : tf.Tensor
            A `float64` tensor of shape `[-1, 3]` as the periodic displacements
            vector.

        """
        with tf.name_scope("Einsum"):
            shift = tf.convert_to_tensor(shift, dtype=tf.float64, name='shift')
            cells = tf.convert_to_tensor(cells, dtype=tf.float64, name='cells')
            return tf.einsum('ijk,ikl->ijl', shift, cells, name='displacements')

    def _get_g_shape(self, _):
        """
        Return the shape of the descriptor matrix.
        """
        return [self._batch_size, self._max_n_atoms, self._max_n_terms,
                self._nnl_max]

    def _get_v2g_map_batch_indexing_matrix(self):
        """
        Return an `int32` matrix of shape `[batch_size, ndim, 3]` to rebuild the
        batch indexing of a `v2g_map`.
        """
        indexing_matrix = np.zeros((self._batch_size, self._nij_max, 4),
                                   dtype=np.int32)
        for i in range(self._batch_size):
            indexing_matrix[i] += [i, 0, 0, 0]
        return indexing_matrix

    def _get_row_split_sizes(self, placeholders):
        """
        Return the sizes of the rowwise splitted subsets of `g`.
        """
        row_splits = [1, ]
        for i, element in enumerate(self._elements):
            row_splits.append(self._max_occurs[element])
        return row_splits

    @staticmethod
    def _get_row_split_axis():
        """
        Return the axis to rowwise split `g`.
        """
        return 1

    def _get_v2g_map(self, placeholders, **kwargs):
        """
        Return the re-indexed `v2g_map` for batch training and evaluation.
        """
        indexing = self._get_v2g_map_batch_indexing_matrix()
        return tf.add(placeholders.v2g_map, indexing, name='v2g_map')
