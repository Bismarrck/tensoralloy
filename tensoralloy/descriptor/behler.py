# coding=utf-8
"""
This module defines the Symmetry Function descriptor.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
from collections import Counter
from typing import List, Dict
from sklearn.model_selection import ParameterGrid

from tensoralloy.descriptor.interface import AtomicDescriptor
from misc import Defaults, AttributeDict
from tensoralloy.descriptor.cutoff import cosine_cutoff
from utils import get_elements_from_kbody_term, get_kbody_terms

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def compute_dimension(kbody_terms: List[str], n_etas, n_betas, n_gammas,
                      n_zetas):
    """
    Compute the total dimension of the feature vector.

    Parameters
    ----------
    kbody_terms : List[str]
        A list of str as all k-body terms.
    n_etas : int
        The number of `eta` for radial functions.
    n_betas : int
        The number of `beta` for angular functions.
    n_gammas : int
        The number of `gamma` for angular functions.
    n_zetas : int
        The number of `zeta` for angular functions.

    Returns
    -------
    total_dim : int
        The total dimension of the feature vector.
    kbody_sizes : List[int]
        The size of each k-body term.

    """
    total_dim = 0
    kbody_sizes = []
    for kbody_term in kbody_terms:
        k = len(get_elements_from_kbody_term(kbody_term))
        if k == 2:
            n = n_etas
        else:
            n = n_gammas * n_betas * n_zetas
        total_dim += n
        kbody_sizes.append(n)
    return total_dim, kbody_sizes


class SymmetryFunction(AtomicDescriptor):
    """
    A tensorflow based implementation of Behler-Parinello's SymmetryFunction
    descriptor.
    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, rc, elements, eta=Defaults.eta, beta=Defaults.beta,
                 gamma=Defaults.gamma, zeta=Defaults.zeta, k_max=3,
                 periodic=True):
        """
        Initialization method.

        Parameters
        ----------
        rc : float
            The cutoff radius.
        elements : List[str]
            A list of str as the ordered elements.
        eta : array_like
            The `eta` for radial functions.
        beta : array_like
            The `beta` for angular functions.
        gamma : array_like
            The `beta` for angular functions.
        zeta : array_like
            The `beta` for angular functions.
        k_max : int
            The maximum k for the many-body expansion.
        periodic : bool
            If False, some Ops of the computation graph will be ignored and this
            can only proceed non-periodic molecules.

        """
        kbody_terms, mapping, elements = get_kbody_terms(elements, k_max=k_max)
        ndim, kbody_sizes = compute_dimension(kbody_terms, len(eta), len(beta),
                                              len(gamma), len(zeta))

        self._rc = rc
        self._k_max = k_max
        self._elements = elements
        self._n_elements = len(elements)
        self._periodic = periodic
        self._mapping = mapping
        self._kbody_terms = kbody_terms
        self._kbody_sizes = kbody_sizes
        self._ndim = ndim
        self._kbody_index = {key: kbody_terms.index(key) for key in kbody_terms}
        self._offsets = np.insert(np.cumsum(kbody_sizes), 0, 0)
        self._eta = np.asarray(eta)
        self._gamma = np.asarray(gamma)
        self._beta = np.asarray(beta)
        self._zeta = np.asarray(zeta)
        self._parameter_grid = ParameterGrid({'beta': self._beta,
                                              'gamma': self._gamma,
                                              'zeta': self._zeta})

    @property
    def cutoff(self):
        """
        Return the cutoff radius.
        """
        return self._rc

    @property
    def k_max(self):
        """
        Return the maximum k for the many-body expansion scheme.
        """
        return self._k_max

    @property
    def elements(self):
        """
        Return a list of str as the sorted unique elements.
        """
        return self._elements

    @property
    def periodic(self):
        """
        Return True if this can be applied to periodic structures.
        For non-periodic molecules some Ops can be ignored.
        """
        return self._periodic

    @property
    def ndim(self):
        """
        Return the total dimension of an atom descriptor vector.
        """
        return self._ndim

    @property
    def kbody_terms(self) -> List[str]:
        """
        A list of str as the ordered k-body terms.
        """
        return self._kbody_terms

    @property
    def kbody_sizes(self) -> List[int]:
        """
        Return a list of int as the sizes of the k-body terms.
        """
        return self._kbody_sizes

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

    @staticmethod
    def _get_v2g_map_delta(index):
        """
        Return the step delta for `v2g_map`.
        """
        return tf.constant([0, index], dtype=tf.int32, name='delta')

    def _get_g2_graph_for_eta(self, index, shape, r2c, fc_r, v2g_map):
        """
        Return the subgraph to compute G2 with the given `eta`.

        Parameters
        ----------
        index : int
            The index of the `eta` to use.
        shape : Sized
            The shape of the descriptor.
        r2c : tf.Tensor
            The `float64` tensor of `r**2 / rc**2`.
        fc_r : tf.Tensor
            The `float64` tensor of `cutoff(r)`.
        v2g_map : tf.Tensor
            The `int32` tensor as the mapping from 'v' to 'g'.

        Returns
        -------
        g : tf.Tensor
            A `float64` tensor with the input `shape` as the fingerprints
            contributed by the radial function G2 with the given `eta`.

        """
        with tf.name_scope(f"eta{index}"):
            eta = tf.constant(self._eta[index], dtype=tf.float64, name='eta')
            delta = self._get_v2g_map_delta(index)
            v_index = tf.exp(-tf.multiply(eta, r2c, 'eta_r2c')) * fc_r
            v2g_map_index = tf.add(v2g_map, delta, f'v2g_map_{index}')
            return tf.scatter_nd(v2g_map_index, v_index, shape, f'g{index}')

    def _get_g_shape(self, placeholders):
        """
        Return the shape of the descriptor matrix.
        """
        return [placeholders.n_atoms, self._ndim]

    def _get_v2g_map(self, placeholders, fn_name: str):
        assert fn_name in ('g2', 'g4')
        return tf.identity(placeholders[fn_name].v2g_map, name='v2g_map')

    def _get_g2_graph(self, placeholders: AttributeDict):
        """
        The implementation of Behler's G2 symmetry function.
        """
        with tf.name_scope("G2"):

            r = self._get_rij(placeholders.positions,
                              placeholders.cells,
                              placeholders.g2.ilist,
                              placeholders.g2.jlist,
                              placeholders.g2.shift,
                              name='rij')
            r2 = tf.square(r, name='r2')
            rc2 = tf.constant(self._rc**2, dtype=tf.float64, name='rc2')
            r2c = tf.div(r2, rc2, name='div')
            fc_r = cosine_cutoff(r, rc=self._rc, name='fc_r')

            with tf.name_scope("v2g_map"):
                v2g_map = self._get_v2g_map(placeholders, 'g2')

            with tf.name_scope("features"):
                shape = self._get_g_shape(placeholders)
                # TODO: maybe `tf.while` can be used here
                blocks = []
                for index in range(len(self._eta)):
                    blocks.append(self._get_g2_graph_for_eta(
                        index, shape, r2c, fc_r, v2g_map)
                    )
                return tf.add_n(blocks, name='g')

    @staticmethod
    def _extract(_params):
        """
        A helper function to get `beta`, `gamma` and `zeta`.
        """
        return [tf.constant(_params[key], dtype=tf.float64, name=key)
                for key in ('beta', 'gamma', 'zeta')]

    def _get_g4_graph_for_params(self, index, shape, theta, r2c, fc_r, v2g_map):
        """
        Return the subgraph to compute angular descriptors with the given
        parameters set.

        Parameters
        ----------
        index : int
            The index of the `eta` to use.
        shape : Sized
            The shape of the descriptor.
        theta : tf.Tensor
            The `float64` tensor of `cos(theta)`.
        r2c : tf.Tensor
            The `float64` tensor of `r**2 / rc**2`.
        fc_r : tf.Tensor
            The `float64` tensor of `cutoff(r)`.
        v2g_map : tf.Tensor
            The `int32` tensor as the mapping from 'v' to 'g'.

        Returns
        -------
        g : tf.Tensor
            A `float64` tensor with the input `shape` as the fingerprints
            contributed by the angular function G4 with the given parameters.

        """
        with tf.name_scope(f"grid{index}"):
            beta, gamma, zeta = self._extract(self._parameter_grid[index])
            delta = self._get_v2g_map_delta(index)
            c = (1.0 + gamma * theta) ** zeta * 2.0 ** (1.0 - zeta)
            v_index = tf.multiply(c * tf.exp(-beta * r2c), fc_r, f'v_{index}')
            v2g_map_index = tf.add(v2g_map, delta, name=f'v2g_map_{index}')
            return tf.scatter_nd(v2g_map_index, v_index, shape, f'g{index}')

    def _get_g4_graph(self, placeholders):
        """
        The implementation of Behler's angular symmetry function.
        """
        with tf.name_scope("G4"):

            rij = self._get_rij(placeholders.positions,
                                placeholders.cells,
                                placeholders.g4.ij.ilist,
                                placeholders.g4.ij.jlist,
                                placeholders.g4.shift.ij,
                                name='rij')
            rik = self._get_rij(placeholders.positions,
                                placeholders.cells,
                                placeholders.g4.ik.ilist,
                                placeholders.g4.ik.klist,
                                placeholders.g4.shift.ik,
                                name='rik')
            rjk = self._get_rij(placeholders.positions,
                                placeholders.cells,
                                placeholders.g4.jk.jlist,
                                placeholders.g4.jk.klist,
                                placeholders.g4.shift.jk,
                                name='rjk')

            rij2 = tf.square(rij, name='rij2')
            rik2 = tf.square(rik, name='rik2')
            rjk2 = tf.square(rjk, name='rjk2')
            rc2 = tf.constant(self._rc ** 2, dtype=tf.float64, name='rc2')
            r2 = tf.add_n([rij2, rik2, rjk2], name='r2')
            r2c = tf.div(r2, rc2, name='r2_rc2')

            with tf.name_scope("cosine"):
                upper = tf.subtract(rij2 + rik2, rjk2, name='upper')
                lower = tf.multiply(2 * rij, rik, name='lower')
                theta = tf.div(upper, lower, name='theta')

            with tf.name_scope("fc"):
                fc_rij = cosine_cutoff(rij, self._rc, name='fc_rij')
                fc_rik = cosine_cutoff(rik, self._rc, name='fc_rik')
                fc_rjk = cosine_cutoff(rjk, self._rc, name='fc_rjk')
                fc_r = tf.multiply(fc_rij, fc_rik * fc_rjk, 'fc_r')

            with tf.name_scope("v2g_map"):
                v2g_map = self._get_v2g_map(placeholders, 'g4')

            with tf.name_scope("features"):
                shape = self._get_g_shape(placeholders)
                blocks = []
                for index in range(len(self._parameter_grid)):
                    blocks.append(self._get_g4_graph_for_params(
                        index, shape, theta, r2c, fc_r, v2g_map))
                return tf.add_n(blocks, name='g')

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

    def _get_column_split_sizes(self):
        """
        Return the sizes of the column-wise splitted subsets of `g`.
        """
        column_splits = {}
        for i, element in enumerate(self._elements):
            column_splits[element] = [len(self._elements), i]
        return column_splits

    @staticmethod
    def _get_column_split_axis():
        """
        Return the axis to column-wise split `g`.
        """
        return 1

    def _split_descriptors(self, g, placeholders) -> Dict[str, tf.Tensor]:
        """
        Split the descriptors into `N_element` subsets.
        """
        with tf.name_scope("Split"):
            row_split_sizes = self._get_row_split_sizes(placeholders)
            row_split_axis = self._get_row_split_axis()
            column_split_sizes = self._get_column_split_sizes()
            column_split_axis = self._get_column_split_axis()
            splits = tf.split(
                g, row_split_sizes, axis=row_split_axis, name='rows')[1:]
            if len(self._elements) > 1:
                # Further split the element arrays to remove redundant zeros
                blocks = []
                for i in range(len(splits)):
                    element = self._elements[i]
                    size_splits, idx = column_split_sizes[element]
                    block = tf.split(splits[i],
                                     size_splits,
                                     axis=column_split_axis,
                                     name='{}_block'.format(element))[idx]
                    blocks.append(block)
            else:
                blocks = splits
            return dict(zip(self._elements, blocks))

    def build_graph(self, placeholders: AttributeDict, split=True):
        """
        Get the tensorflow based computation graph of the Symmetry Function.
        """
        with tf.name_scope("Behler"):
            g = self._get_g2_graph(placeholders)
            if self._k_max == 3:
                g += self._get_g4_graph(placeholders)
        if split:
            return self._split_descriptors(g, placeholders)
        else:
            return g


class BatchSymmetryFunction(SymmetryFunction):
    """
    A special implementation of Behler-Parinello's Symmetry Function for batch
    training and evaluations.
    """

    gather_fn = staticmethod(tf.batch_gather)

    def __init__(self, rc, max_occurs: Counter, elements: List[str],
                 nij_max: int, nijk_max: int, batch_size: int, eta=Defaults.eta,
                 beta=Defaults.beta, gamma=Defaults.gamma, zeta=Defaults.zeta,
                 k_max=3, periodic=True):
        """
        Initialization method.
        """
        super(BatchSymmetryFunction, self).__init__(
            rc=rc, elements=elements, eta=eta, beta=beta, gamma=gamma,
            zeta=zeta, k_max=k_max, periodic=periodic)

        self._max_occurs = max_occurs
        self._max_n_atoms = sum(max_occurs.values()) + 1
        self._nij_max = nij_max
        self._nijk_max = nijk_max
        self._batch_size = batch_size

    @property
    def nij_max(self):
        """
        Return the maximum allowed length of the flatten neighbor list.
        """
        return self._nij_max

    @property
    def nijk_max(self):
        """
        Return the maximum allowed length of the expanded Angle[i,j,k] list.
        """
        return self._nijk_max

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
        return [self._batch_size, self._max_n_atoms, self._ndim]

    def _get_v2g_map_batch_indexing_matrix(self, fn_name='g2'):
        """
        Return an `int32` matrix of shape `[batch_size, ndim, 3]` to rebuild the
        batch indexing of a `v2g_map`.
        """
        if fn_name == 'g2':
            ndim = self._nij_max
        else:
            ndim = self._nijk_max
        indexing_matrix = np.zeros((self._batch_size, ndim, 3), dtype=np.int32)
        for i in range(self._batch_size):
            indexing_matrix[i] += [i, 0, 0]
        return indexing_matrix

    @staticmethod
    def _get_v2g_map_delta(index):
        return tf.constant([0, 0, index], tf.int32, name='delta')

    def _get_v2g_map(self, placeholders, fn_name: str):
        """
        Return the Op to get `v2g_map`. In the batch implementation, `v2g_map`
        has a shape of `[batch_size, ndim, 3]` and the first axis represents the
        local batch indices.
        """
        indexing = self._get_v2g_map_batch_indexing_matrix(fn_name=fn_name)
        return tf.add(placeholders[fn_name].v2g_map, indexing, name='v2g_map')

    def _get_row_split_sizes(self, _):
        row_splits = [1, ]
        for i, element in enumerate(self._elements):
            row_splits.append(self._max_occurs[element])
        return row_splits

    @staticmethod
    def _get_row_split_axis():
        return 1

    @staticmethod
    def _get_column_split_axis():
        return 2

    def build_graph(self, placeholders: AttributeDict, split=True):
        """
        Build the computation graph.
        """
        if not self._batch_size:
            raise ValueError("Batch size must be set first.")
        return super(BatchSymmetryFunction, self).build_graph(
            placeholders, split=split)