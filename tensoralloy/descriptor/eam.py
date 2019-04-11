# coding=utf-8
"""
This module implements the Embedded-atom method.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf

from collections import Counter
from typing import List

from tensoralloy.descriptor.base import AtomicDescriptor
from tensoralloy.utils import get_elements_from_kbody_term, AttributeDict

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
        super(EAM, self).__init__(rc, elements, k_max=2, periodic=True)

        kbody_index = {}
        for kbody_term in self._all_kbody_terms:
            center = get_elements_from_kbody_term(kbody_term)[0]
            kbody_index[kbody_term] = \
                self._kbody_terms[center].index(kbody_term)

        self._max_n_terms = max(map(len, self._kbody_terms.values()))
        self._kbody_index = kbody_index

    def _get_g_shape(self, placeholders):
        """
        Return the shape of the descriptor matrix.
        """
        return [self._max_n_terms,
                placeholders.n_atoms_plus_virt,
                placeholders.nnl_max]

    def _get_v2g_map(self, placeholders, **kwargs):
        """
        A wrapper function to get `v2g_map` or re-indexed `v2g_map`.
        """
        splits = tf.split(placeholders.v2g_map, [-1, 1], axis=1)
        v2g_map = tf.identity(splits[0], name='v2g_map')
        v2g_mask = tf.identity(splits[1], name='v2g_mask')
        return v2g_map, v2g_mask

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
        return 1

    def _split_descriptors(self, placeholders, g, mask, *args):
        """
        Split the descriptors into `N_element` subsets.
        """
        with tf.name_scope("Split"):
            split_sizes = self._get_row_split_sizes(placeholders)
            axis = self._get_row_split_axis()

            # `axis` should increase by one for `g` because `g` is created by
            # `tf.concat((gr, gx, gy, gz), axis=0, name='g')`
            rows = tf.split(g, split_sizes, axis=axis + 1, name='rows')[1:]

            # Use the original axis
            masks = tf.split(mask, split_sizes, axis=axis, name='masks')[1:]
            return dict(zip(self._elements, zip(rows, masks)))

    def _check_keys(self, placeholders: AttributeDict):
        """
        Make sure `placeholders` contains enough keys.
        """
        assert 'positions' in placeholders
        assert 'cells' in placeholders
        assert 'volume' in placeholders
        assert 'n_atoms_plus_virt' in placeholders
        assert 'nnl_max' in placeholders
        assert 'row_splits' in placeholders
        assert 'ilist' in placeholders
        assert 'jlist' in placeholders
        assert 'shift' in placeholders
        assert 'v2g_map' in placeholders

    def build_graph(self, placeholders: AttributeDict):
        """
        Get the tensorflow based computation graph of the EAM model.

        Returns
        -------
        ops : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of {element: (descriptor, mask)}.

            * `descriptor`: [4, max_n_terms, n_atoms_plus_virt, nnl_max]
                Represents th
        
        """
        self._check_keys(placeholders)

        with tf.name_scope("EAM"):
            rr, dij = self._get_rij(placeholders.positions,
                                    placeholders.cells,
                                    placeholders.ilist,
                                    placeholders.jlist,
                                    placeholders.shift,
                                    name='rij')
            shape = self._get_g_shape(placeholders)
            v2g_map, v2g_mask = self._get_v2g_map(placeholders)

            dx = tf.identity(dij[..., 0], name='dijx')
            dy = tf.identity(dij[..., 1], name='dijy')
            dz = tf.identity(dij[..., 2], name='dijz')

            gr = tf.expand_dims(tf.scatter_nd(v2g_map, rr, shape), 0, name='gr')
            gx = tf.expand_dims(tf.scatter_nd(v2g_map, dx, shape), 0, name='gx')
            gy = tf.expand_dims(tf.scatter_nd(v2g_map, dy, shape), 0, name='gy')
            gz = tf.expand_dims(tf.scatter_nd(v2g_map, dz, shape), 0, name='gz')

            g = tf.concat((gr, gx, gy, gz), axis=0, name='g')

            v2g_mask = tf.squeeze(v2g_mask, axis=self._get_row_split_axis())
            mask = tf.scatter_nd(v2g_map, v2g_mask, shape)
            mask = tf.cast(mask, dtype=rr.dtype, name='mask')

            return self._split_descriptors(placeholders, g, mask)


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
        self._max_n_atoms = sum(max_occurs.values())
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
    def _get_pbc_displacements(shift, cells, dtype=tf.float64):
        """
        Return the periodic boundary shift displacements.

        Parameters
        ----------
        shift : tf.Tensor
            A `float64` or `float32` tensor of shape `[batch_size, ndim, 3]` as
            the cell shift vectors and `ndim == nij_max` or `ndim == nijk_max`.
        cells : tf.Tensor
            A `float64` or `float32` tensor of shape `[batch_size, 3, 3]` as the
            cell tensors.

        Returns
        -------
        Dij : tf.Tensor
            A `float64` tensor of shape `[-1, 3]` as the periodic displacements
            vector.

        """
        with tf.name_scope("Einsum"):
            shift = tf.convert_to_tensor(shift, dtype=dtype, name='shift')
            cells = tf.convert_to_tensor(cells, dtype=dtype, name='cells')
            return tf.einsum('ijk,ikl->ijl', shift, cells, name='displacements')

    def _get_g_shape(self, _):
        """
        Return the shape of the descriptor matrix.
        """
        return [self._batch_size, self._max_n_terms, self._max_n_atoms + 1,
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
        return 2

    def _get_v2g_map(self, placeholders, **kwargs):
        """
        Return the re-indexed `v2g_map` for batch training and evaluation.
        """
        splits = tf.split(placeholders.v2g_map, [-1, 1], axis=2)
        v2g_map = tf.identity(splits[0])
        v2g_mask = tf.identity(splits[1], name='v2g_mask')
        indexing = self._get_v2g_map_batch_indexing_matrix()
        return tf.add(v2g_map, indexing, name='v2g_map'), v2g_mask

    def _check_keys(self, placeholders: AttributeDict):
        """
        Make sure `placeholders` contains enough keys.
        """
        assert 'positions' in placeholders
        assert 'cells' in placeholders
        assert 'volume' in placeholders
        assert 'ilist' in placeholders
        assert 'jlist' in placeholders
        assert 'shift' in placeholders
        assert 'v2g_map' in placeholders
