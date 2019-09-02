# coding=utf-8
"""
This module defines the Symmetry Function descriptor.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import functools

from collections import Counter
from typing import List
from sklearn.model_selection import ParameterGrid

from tensoralloy.descriptor.base import AtomicDescriptor
from tensoralloy.descriptor.cutoff import cosine_cutoff, polynomial_cutoff
from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.utils import get_kbody_terms
from tensoralloy.utils import Defaults, GraphKeys
from tensoralloy.extension.grad_ops import safe_pow

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["SymmetryFunction", "BatchSymmetryFunction", "compute_dimension"]


def compute_dimension(all_kbody_terms: List[str], n_etas, n_omegas, n_betas,
                      n_gammas, n_zetas):
    """
    Compute the total dimension of the feature vector.

    Parameters
    ----------
    all_kbody_terms : List[str]
        A list of str as all k-body terms.
    n_etas : int
        The number of `eta` for radial functions.
    n_omegas : int
        The number of `omega` for radial functions.
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
    for kbody_term in all_kbody_terms:
        k = len(get_elements_from_kbody_term(kbody_term))
        if k == 2:
            n = n_etas * n_omegas
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
                 gamma=Defaults.gamma, zeta=Defaults.zeta, omega=Defaults.omega,
                 angular=True, trainable=False, periodic=True,
                 cutoff_function='cosine'):
        """
        Initialization method.
        """
        super(SymmetryFunction, self).__init__(rc, elements, angular=angular,
                                               periodic=periodic)

        all_kbody_terms, kbody_terms, elements = \
            get_kbody_terms(elements, angular=angular)
        ndim, kbody_sizes = compute_dimension(
            all_kbody_terms, len(eta), len(omega), len(beta), len(gamma),
            len(zeta))
        eta = np.asarray(eta)
        omega = np.asarray(omega)
        beta = np.asarray(beta)
        gamma = np.asarray(gamma)
        zeta = np.asarray(zeta)

        self._trainable = trainable
        self.kbody_sizes = kbody_sizes
        self.ndim = ndim
        self.kbody_index = {kbody_term: self._all_kbody_terms.index(kbody_term)
                            for kbody_term in self._all_kbody_terms}
        self.offsets = np.insert(np.cumsum(kbody_sizes), 0, 0)
        self.radial_indices_grid = ParameterGrid({
            'eta': np.arange(len(eta), dtype=int),
            'omega': np.arange(len(omega), dtype=int)})
        self.angular_indices_grid = ParameterGrid({
            'beta': np.arange(len(beta), dtype=int),
            'gamma': np.arange(len(gamma), dtype=int),
            'zeta': np.arange(len(zeta), dtype=int)})
        self.initial_values = {'eta': eta, 'omega': omega, 'gamma': gamma,
                               'beta': beta, 'zeta': zeta}

        self._cutoff_function = cutoff_function
        if cutoff_function == 'cosine':
            self._cutoff_fn = functools.partial(cosine_cutoff, rc=self._rc)
        elif cutoff_function == 'polynomial':
            self._cutoff_fn = functools.partial(polynomial_cutoff, rc=self._rc)
        else:
            raise ValueError(f"Unknown cutoff function: {cutoff_function}")

    @property
    def trainable(self):
        """
        Return if symmetry function parameters are trainable or not.
        """
        return self._trainable

    def _get_variable(self, name: str, index: int, dtype):
        """
        Return a shared variable.
        """
        collections = [tf.GraphKeys.MODEL_VARIABLES,
                       tf.GraphKeys.GLOBAL_VARIABLES,
                       GraphKeys.DESCRIPTOR_VARIABLES,
                       GraphKeys.EVAL_METRICS]
        if self._trainable:
            collections.append(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.variable_scope(f'{name}', reuse=tf.AUTO_REUSE):
            initializer = tf.constant_initializer(
                self.initial_values[name][index], dtype=dtype)
            variable = tf.get_variable(
                name=f'{index}',
                shape=(),
                dtype=dtype,
                initializer=initializer,
                trainable=self._trainable,
                collections=collections)
            tf.summary.scalar(f'{index}/summary', variable)
            return variable

    def get_v2g_map(self, features: dict, prefix: str):
        """
        Return the base v2g map.
        """
        return tf.identity(features[f"{prefix}.v2g_map"], name='v2g_map')

    @staticmethod
    def get_v2g_map_delta(tau):
        """
        Return the delta vector for the tau-th `v2g_map`.
        """
        return tf.constant([0, tau], dtype=tf.int32, name='delta')

    def get_g_shape(self, features: dict):
        """
        Return the shape of the descriptor matrix.
        """
        return [features['n_atoms_vap'], self.ndim]

    def _extract_g2_variables(self, indices, dtype=tf.float64):
        """
        A helper function to get `eta` and `omega`.
        """
        eta_index = indices['eta']
        omega_index = indices['omega']
        eta = self._get_variable('eta', index=eta_index, dtype=dtype)
        omega = self._get_variable('omega', index=omega_index, dtype=dtype)
        return eta, omega

    def get_g2_op_for_tau(self, shape, tau, r, rc2, fc_r, base_v2g_map):
        """
        Return the Op to compute G2(tau) using tau-th (`eta`, `omega`) pair.
        """
        with tf.name_scope(f"Grid{tau}"):
            indices = self.radial_indices_grid[tau]
            eta, omega = self._extract_g2_variables(indices, r.dtype)
            delta = self.get_v2g_map_delta(tau)
            r2c = tf.math.truediv(
                tf.square(tf.math.subtract(r, omega)), rc2, name='r2c')
            v = tf.exp(-tf.multiply(eta, r2c, 'eta_r2c')) * fc_r
            v2g_map_tau = tf.add(base_v2g_map, delta, f'v2g_map_{tau}')
            return tf.scatter_nd(v2g_map_tau, v, shape, f"g{tau}")

    def get_g2_op(self, features: dict):
        """
        The implementation of Behler's G2 symmetry function.
        """
        with tf.variable_scope("G2"):
            r = self.get_rij(features['positions'],
                             features['cell'],
                             features['g2.ilist'],
                             features['g2.jlist'],
                             features['g2.n1'],
                             name='rij')[0]
            rc2 = tf.constant(self._rc ** 2, dtype=r.dtype, name='rc2')
            fc_r = cosine_cutoff(r, rc=self._rc, name='fc_r')
            base_v2g_map = self.get_v2g_map(features, prefix='g2')
            shape = self.get_g_shape(features)
            values = []
            for tau in range(len(self.radial_indices_grid)):
                values.append(
                    self.get_g2_op_for_tau(
                        shape, tau, r, rc2, fc_r, base_v2g_map))
            return tf.add_n(values, name='g')

    def _extract_g4_variables(self, indices, dtype=tf.float64):
        """
        A helper function to get `beta`, `gamma` and `zeta`.
        """
        beta_index = indices['beta']
        gamma_index = indices['gamma']
        zeta_index = indices['zeta']
        beta = self._get_variable('beta', index=beta_index, dtype=dtype)
        gamma = self._get_variable('gamma', index=gamma_index, dtype=dtype)
        zeta = self._get_variable('zeta', index=zeta_index, dtype=dtype)
        return beta, gamma, zeta

    def get_g4_op_for_tau(self, shape, tau: int, cos_theta, r2c, fc_r,
                          base_v2g_map):
        """
        Return the Op to compute G4(tau) using tau-th (`beta`, `gamma`, `zeta`)
        combination.
        """
        with tf.name_scope(f"Grid{tau}"):
            indices = self.angular_indices_grid[tau]
            beta, gamma, zeta = self._extract_g4_variables(indices, r2c.dtype)
            delta = self.get_v2g_map_delta(tau)
            one = tf.constant(1.0, dtype=r2c.dtype, name='one')
            two = tf.constant(2.0, dtype=r2c.dtype, name='two')
            gt = tf.math.multiply(gamma, cos_theta, name='gt')
            gt1 = tf.add(gt, one, name='gt1')
            gt1z = safe_pow(gt1, zeta)
            z1 = tf.math.subtract(one, zeta, name='z1')
            z12 = safe_pow(two, z1)
            c = tf.math.multiply(gt1z, z12, name='c')
            v = tf.multiply(c * tf.exp(-beta * r2c), fc_r, f'v_{tau}')
            v2g_map_tau = tf.add(base_v2g_map, delta, name=f'v2g_map_{tau}')
            return tf.scatter_nd(v2g_map_tau, v, shape, f'g{tau}')

    def get_g4_op(self, features: dict):
        """
        The implementation of Behler's angular symmetry function.
        """
        with tf.variable_scope("G4"):
            rij = self.get_rij(features['positions'],
                               features['cell'],
                               features['g4.ilist'],
                               features['g4.jlist'],
                               features['g4.n1'],
                               name='rij')[0]
            rik = self.get_rij(features['positions'],
                               features['cell'],
                               features['g4.ilist'],
                               features['g4.klist'],
                               features['g4.n2'],
                               name='rik')[0]
            rjk = self.get_rij(features['positions'],
                               features['cell'],
                               features['g4.jlist'],
                               features['g4.klist'],
                               features['g4.n3'],
                               name='rjk')[0]
            rij2 = tf.square(rij, name='rij2')
            rik2 = tf.square(rik, name='rik2')
            rjk2 = tf.square(rjk, name='rjk2')
            rc2 = tf.constant(self._rc ** 2, dtype=rij.dtype, name='rc2')
            r2 = tf.add_n([rij2, rik2, rjk2], name='r2')
            r2c = tf.math.truediv(r2, rc2, name='r2_rc2')
            with tf.name_scope("CosTheta"):
                two = tf.convert_to_tensor(2.0, rij.dtype, name='two')
                upper = tf.math.subtract(rij2 + rik2, rjk2, name='upper')
                lower = tf.math.multiply(
                    tf.math.multiply(two, rij), rik, name='lower')
                cos_theta = tf.math.truediv(upper, lower, name='theta')
            with tf.name_scope("Cutoff"):
                fc_rij = cosine_cutoff(rij, rc=self._rc, name='fc_rij')
                fc_rik = cosine_cutoff(rik, rc=self._rc, name='fc_rik')
                fc_rjk = cosine_cutoff(rjk, rc=self._rc, name='fc_rjk')
                fc_r = tf.multiply(fc_rij, fc_rik * fc_rjk, 'fc_r')
            base_v2g_map = self.get_v2g_map(features, prefix='g4')
            shape = self.get_g_shape(features)
            values = []
            for tau in range(len(self.angular_indices_grid)):
                values.append(
                    self.get_g4_op_for_tau(
                        shape, tau, cos_theta, r2c, fc_r, base_v2g_map))
            return tf.add_n(values, name='g')

    def get_row_split_sizes(self, features: dict):
        """
        Return the sizes of the row-wise splitted subsets of `descriptors`.
        """
        return features['row_splits']

    @staticmethod
    def get_row_split_axis():
        """
        Return the axis to row-wise split `descriptors`.
        """
        return 0

    def get_column_split_sizes(self):
        """
        Return the sizes of the column-wise splitted subsets of `descriptors`.
        """
        column_splits = {}
        for i, element in enumerate(self._elements):
            column_splits[element] = [len(self._elements), i]
        return column_splits

    @staticmethod
    def get_column_split_axis():
        """
        Return the axis to column-wise split `g`.
        """
        return 1

    def split_descriptors(self, descriptors, features: dict):
        """
        Split the descriptors into `N_element` subsets.
        """
        with tf.name_scope("Split"):
            row_split_sizes = self.get_row_split_sizes(features)
            row_split_axis = self.get_row_split_axis()
            column_split_sizes = self.get_column_split_sizes()
            column_split_axis = self.get_column_split_axis()
            splits = tf.split(
                descriptors, row_split_sizes, axis=row_split_axis,
                name='rows')[1:]
            atom_masks = tf.split(
                features['atom_masks'], row_split_sizes, axis=row_split_axis,
                name='atom_masks')[1:]
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
            return dict(zip(self._elements, zip(blocks, atom_masks)))

    def build_graph(self, features: dict):
        """
        Get the tensorflow based computation graph of the Symmetry Function.
        """
        with tf.variable_scope("Behler"):
            descriptors = self.get_g2_op(features)
            if self.angular:
                descriptors += self.get_g4_op(features)
        return self.split_descriptors(descriptors, features)


class BatchSymmetryFunction(SymmetryFunction):
    """
    A special implementation of Behler-Parinello's Symmetry Function for batch
    training and evaluations.
    """

    gather_fn = staticmethod(tf.batch_gather)

    def __init__(self, rc, max_occurs: Counter, nij_max: int, nijk_max: int,
                 batch_size: int, eta=np.array([0.05, 4.0, 20.0, 80.0]),
                 omega=np.array([0.0]), beta=np.array([0.005, ]),
                 gamma=np.array([1.0, -1.0]), zeta=np.array([1.0, 4.0]),
                 angular=True, periodic=True, trainable=False,
                 cutoff_function="cosine"):
        """
        Initialization method.
        """
        elements = sorted(list(max_occurs.keys()))

        super(BatchSymmetryFunction, self).__init__(
            rc=rc, elements=elements, eta=eta, beta=beta, gamma=gamma,
            zeta=zeta, omega=omega, angular=angular, periodic=periodic,
            trainable=trainable, cutoff_function=cutoff_function)
        self._max_occurs = max_occurs
        self._max_n_atoms = sum(max_occurs.values())
        self._nij_max = nij_max
        self._nijk_max = nijk_max
        self._batch_size = batch_size

    @property
    def nij_max(self):
        """
        Return the maximum allowed `nij`.
        """
        return self._nij_max

    @property
    def nijk_max(self):
        """
        Return the maximum allowed `nijk`.
        """
        return self._nijk_max

    @property
    def max_occurs(self):
        """
        Return a dict. `max_occrs[el]` indicates the maximum apperances of
        element `el`.
        """
        return self._max_occurs

    @staticmethod
    def get_pbc_displacements(shift, cells, dtype=tf.float32):
        """
        Compute r^{GSL} and D^{GSL} in the training phase.
        """
        with tf.name_scope("Einsum"):
            shift = tf.convert_to_tensor(shift, dtype=dtype, name='shift')
            cells = tf.convert_to_tensor(cells, dtype=dtype, name='cells')
            return tf.einsum('ijk,ikl->ijl', shift, cells, name='displacements')

    def get_g_shape(self, _):
        """
        Return the shape of the descriptors.
        """
        n_atoms_vap = self._max_n_atoms + 1
        return [self._batch_size, n_atoms_vap, self.ndim]

    def get_v2g_map_batch_indexing_matrix(self, prefix='g2'):
        """
        Return an `int32` matrix of shape `[batch_size, ndim, 3]` to rebuild the
        batch indexing of a `v2g_map`.
        """
        if prefix == 'g2':
            ndim = self._nij_max
        else:
            ndim = self._nijk_max
        indexing_matrix = np.zeros((self._batch_size, ndim, 3), dtype=np.int32)
        for i in range(self._batch_size):
            indexing_matrix[i] += [i, 0, 0]
        return indexing_matrix

    @staticmethod
    def get_v2g_map_delta(tau):
        """
        Return the delta vector for the tau-th `v2g_map`.
        """
        return tf.constant([0, 0, tau], tf.int32, name='delta')

    def get_v2g_map(self, features: dict, prefix="g2"):
        """
        Return the Op to get `v2g_map`. In the batch implementation, `v2g_map`
        has a shape of `[batch_size, ndim, 3]` and the first axis represents the
        local batch indices.
        """
        indexing = self.get_v2g_map_batch_indexing_matrix(prefix=prefix)
        return tf.add(features[f"{prefix}.v2g_map"], indexing, name='v2g_map')

    def get_row_split_sizes(self, _):
        """
        Return the sizes of the row-wise splitted subsets of `descriptors`.
        """
        row_splits = [1, ]
        for i, element in enumerate(self._elements):
            row_splits.append(self._max_occurs[element])
        return row_splits

    @staticmethod
    def get_row_split_axis():
        """
        Return the axis to row-wise split `descriptors`.
        """
        return 1

    @staticmethod
    def get_column_split_axis():
        """
        Return the axis to column-wise split `g`.
        """
        return 2
