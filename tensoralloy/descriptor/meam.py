#!coding=utf-8
"""
The Modified Embedded-Atom Method (Lenosky Style) descriptor.

Notes
-----
The Baskes derived 1NN/2NN MEAM is not suitable for this project.

"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from typing import List

from tensoralloy.descriptor.base import AtomicDescriptor
from tensoralloy.descriptor.cutoff import meam_cutoff
from tensoralloy.utils import AttributeDict, get_elements_from_kbody_term
from tensoralloy.utils import GraphKeys
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class MEAM(AtomicDescriptor):
    """
    A tensorflow based implementation of Modified Embedded-Atom Method (MEAM).

    References
    ----------
    Modelling Simul. Mater. Sci. Eng. 8 (2000) 825–841.
    Computational Materials Science 124 (2016) 204–210.

    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, rc: float, elements: List[str], angular_rc_scale=1.0):
        """
        Initialization method.

        Parameters
        ----------
        rc : float
            The cutoff radius.
        elements : List[str]
            A list of str as the ordered elements.
        angular_rc_scale : float
            The scaling factor of `rc` for angular interactions.

        """
        super(MEAM, self).__init__(rc, elements, k_max=3, periodic=True)

        self._angular_rc_scale = angular_rc_scale
        self._angular_rc = rc * angular_rc_scale

        kbody_index = {}
        for kbody_term in self._all_kbody_terms:
            atom_types = get_elements_from_kbody_term(kbody_term)
            center = atom_types[0]
            kbody_index[kbody_term] = \
                self._kbody_terms[center].index(kbody_term)

        self._max_n_terms = max(map(len, self._kbody_terms.values()))
        self._kbody_index = kbody_index

        angular_kbody_terms = []
        for symboli in self._elements:
            for symbolj in self._elements:
                for symbolk in self._elements:
                    angular_kbody_terms.append(f'{symboli}{symbolj}{symbolk}')

        self._screen_dr = 1.0
        self._angular_kbody_terms = angular_kbody_terms
        self._initial_values = np.zeros((2, self._n_elements**3))

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

    def _split_descriptors(self, g, dx, dy, dz, sij, mask, placeholders):
        """
        Split the descriptors into `N_element` subsets.
        """
        with tf.name_scope("Split"):
            row_split_sizes = self._get_row_split_sizes(placeholders)
            row_split_axis = self._get_row_split_axis()
            g_rows = tf.split(
                g, row_split_sizes, axis=row_split_axis, name='g_rows')[1:]
            dx_rows = tf.split(
                dx, row_split_sizes, axis=row_split_axis, name='dx_rows')[1:]
            dy_rows = tf.split(
                dy, row_split_sizes, axis=row_split_axis, name='dy_rows')[1:]
            dz_rows = tf.split(
                dz, row_split_sizes, axis=row_split_axis, name='dz_rows')[1:]
            sij_rows = tf.split(
                sij, row_split_sizes, axis=row_split_axis, name='sij_rows')[1:]
            masks = tf.split(
                mask, row_split_sizes, axis=row_split_axis, name='masks')[1:]
            return dict(zip(self._elements,
                            zip(g_rows, dx_rows, dy_rows,
                                dz_rows, sij_rows, masks)))

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

    @staticmethod
    def _get_variable_collections():
        collections = [tf.GraphKeys.MODEL_VARIABLES,
                       tf.GraphKeys.GLOBAL_VARIABLES,
                       GraphKeys.DESCRIPTOR_VARIABLES,
                       GraphKeys.EVAL_METRICS,
                       tf.GraphKeys.TRAINABLE_VARIABLES]
        return collections

    def _get_screen_variable(self, angular_term: str, is_min: bool, dtype):
        """
        Return the screen variable C_ikj_min/max.
        """
        if is_min:
            c_type = 'min'
        else:
            c_type = 'max'
        with tf.variable_scope(f'Screen/{angular_term}', reuse=tf.AUTO_REUSE):
            index = self._angular_kbody_terms.index(angular_term)
            initializer = tf.constant_initializer(
                self._initial_values[is_min][index], dtype=dtype)
            variable = tf.get_variable(
                name=f'{c_type}',
                shape=(),
                dtype=dtype,
                initializer=initializer,
                trainable=True,
                collections=self._get_variable_collections())
            tf.summary.scalar(f'{c_type}/summary', variable)
            return variable

    def _get_screen_dr(self, dtype):
        """
        Return the screen variable `dr`.
        """
        with tf.variable_scope(f'Screen', reuse=tf.AUTO_REUSE):
            initializer = tf.constant_initializer(self._screen_dr, dtype=dtype)
            variable = tf.get_variable(
                name='dr',
                shape=(),
                dtype=dtype,
                initializer=initializer,
                trainable=True,
                collections=self._get_variable_collections())
            tf.summary.scalar('dr/summary', variable)
            return variable

    def _get_cikj(self, placeholders: AttributeDict):
        """
        Return the Op to compute Cikj (Equation A16d).

        Returns
        -------
        cikj : tf.Tensor
            A float tensor of shape `[nijk_max, ]`

        """
        with tf.name_scope("Cikj"):

            rij = self._get_rij(placeholders.positions,
                                placeholders.cells,
                                placeholders.gs.ij.ilist,
                                placeholders.gs.ij.jlist,
                                placeholders.gs.shift.ij,
                                name='rij')[0]
            rik = self._get_rij(placeholders.positions,
                                placeholders.cells,
                                placeholders.gs.ik.ilist,
                                placeholders.gs.ik.klist,
                                placeholders.gs.shift.ik,
                                name='rik')[0]
            rjk = self._get_rij(placeholders.positions,
                                placeholders.cells,
                                placeholders.gs.jk.jlist,
                                placeholders.gs.jk.klist,
                                placeholders.gs.shift.jk,
                                name='rjk')[0]

            dtype = get_float_dtype()

            one = tf.convert_to_tensor(1.0, dtype=dtype, name='one')
            two = tf.convert_to_tensor(2.0, dtype=dtype, name='two')
            eps = tf.convert_to_tensor(dtype.eps, dtype=dtype, name='eps')

            rij2 = tf.square(rij, 2, name='rij2')
            rik2 = tf.square(rik, 2, name='rik2')
            rjk2 = tf.square(rjk, 2, name='rjk2')
            rij4 = tf.square(rij, 4, name='rij4')
            upper = tf.add_n([rij2 * rik2, rij2 * rjk2, rij4], name='upper')
            lower = tf.add(rij4, tf.square(rik2 - rjk2, 2), name='lower')
            lower = tf.add(lower, eps, name='lower/eps')
            right = tf.math.truediv(upper, lower) * two

            return tf.add(one, right, name='cikj')

    def _get_sikj_shape(self, placeholders: AttributeDict):
        """
        Return the shape of `Sikj`
        """
        return [placeholders.nij_max,
                len(self._angular_kbody_terms),
                placeholders.ntri_max]

    def _get_sij(self, rij: tf.Tensor, placeholders: AttributeDict):
        """
        Return the Op to compute Sij.
        """
        with tf.name_scope("Sij"):

            # The shape of `Cikj` should be `[nijk_max, ]`
            cikj = self._get_cikj(placeholders)
            dtype = cikj.dtype

            # The shape of `Sikj` should be
            # `[n_angular_terms, nij_max, ntri_max]`
            shape = self._get_sikj_shape(placeholders)
            sikj = tf.scatter_nd(placeholders.gs.v2g_map, cikj, shape,
                                 name='Sikj/raw')

            with tf.name_scope("Ones"):
                base = tf.ones_like(sikj, dtype=dtype, name='base')
                ones = tf.ones_like(cikj, dtype=dtype, name='ones')
                mask = tf.scatter_nd(placeholders.gs.v2g_map, ones, shape,
                                     name='mask')
                delta = tf.subtract(base, mask, name='delta')

            sikj = tf.math.add(sikj, delta, name='Sikj')
            values = []

            for ijk_idx, angular_term in enumerate(self._angular_kbody_terms):
                with tf.name_scope(f"{angular_term}"):
                    c_min = self._get_screen_variable(angular_term, True, dtype)
                    c_max = self._get_screen_variable(angular_term, False, dtype)

                    # Equation A16c
                    upper = tf.math.subtract(sikj[ijk_idx], c_min, name='upper')
                    lower = tf.math.subtract(c_max, c_min, name='lower')
                    x = tf.math.truediv(upper, lower, name='x')
                    fsikj_idx = meam_cutoff(x, name=f'fsikj_{ijk_idx}')
                    prod_fsikj_idx = tf.reduce_prod(
                        fsikj_idx,
                        axis=1,
                        keepdims=True,
                        name=f'prod_fsikj_{ijk_idx}')
                    values.append(prod_fsikj_idx)

            merged = tf.concat(values, axis=1, name='merged')
            sij_raw = tf.reduce_prod(merged, axis=1, keepdims=False,
                                     name='sij_raw')

            with tf.name_scope("Damp"):
                rc = tf.convert_to_tensor(self._rc, dtype=dtype, name='rc')
                dr = self._get_screen_dr(dtype)
                x = tf.math.truediv(rc - rij, dr, name='x')
                fc = meam_cutoff(x, name='fc')

            return tf.math.multiply(fc, sij_raw, name='sij')

    def build_graph(self, placeholders: AttributeDict):
        """
        Get the tensorflow based computation graph of the EAM model.

        Returns
        -------
        ops : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict.

        """
        self._check_keys(placeholders)

        with tf.name_scope("EAM"):

            rij, dij = self._get_rij(placeholders.positions,
                                     placeholders.cells,
                                     placeholders.ilist,
                                     placeholders.jlist,
                                     placeholders.shift,
                                     name='rij')

            dijx = tf.identity(dij[..., 0], name='dijx')
            dijy = tf.identity(dij[..., 0], name='dijy')
            dijz = tf.identity(dij[..., 0], name='dijz')

            shape = self._get_g_shape(placeholders)
            v2g_map, v2g_mask = self._get_v2g_map(placeholders)

            g = tf.scatter_nd(v2g_map, rij, shape, name='g')
            dx = tf.scatter_nd(v2g_map, dijx, shape, name='dx')
            dy = tf.scatter_nd(v2g_map, dijy, shape, name='dy')
            dz = tf.scatter_nd(v2g_map, dijz, shape, name='dz')

            v2g_mask = tf.squeeze(v2g_mask, axis=self._get_row_split_axis())
            mask = tf.scatter_nd(v2g_map, v2g_mask, shape)
            mask = tf.cast(mask, dtype=rij.dtype, name='mask')

            sij = self._get_sij(rij, placeholders)

            return self._split_descriptors(g, dx, dy, dz, sij, mask,
                                           placeholders)
