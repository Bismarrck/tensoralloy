# coding=utf-8
"""
This module defines the feature transformers (flexible, fixed) for Behler's
Symmetry Function descriptor.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from collections import Counter
from typing import Dict, Tuple
from ase import Atoms
from ase.neighborlist import neighbor_list

from tensoralloy.descriptor import SymmetryFunction, BatchSymmetryFunction
from tensoralloy.dtypes import get_float_dtype
from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.transformer.base import bytes_feature
from tensoralloy.transformer.index_transformer import IndexTransformer
from tensoralloy.transformer.indexed_slices import G2IndexedSlices
from tensoralloy.transformer.indexed_slices import G4IndexedSlices
from tensoralloy.utils import get_elements_from_kbody_term, AttributeDict
from tensoralloy.utils import Defaults

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["SymmetryFunctionTransformer", "BatchSymmetryFunctionTransformer"]


class SymmetryFunctionTransformer(SymmetryFunction, DescriptorTransformer):
    """
    A tranformer for providing the feed dict to calculate symmetry function
    descriptors of an `Atoms` object.
    """

    def __init__(self, rc, elements, eta=Defaults.eta, beta=Defaults.beta,
                 gamma=Defaults.gamma, zeta=Defaults.zeta, angular=False,
                 periodic=True, trainable=False, cutoff_function='cosine'):
        """
        Initialization method.
        """
        SymmetryFunction.__init__(
            self, rc=rc, elements=elements, eta=eta, beta=beta, gamma=gamma,
            zeta=zeta, angular=angular, periodic=periodic, trainable=trainable,
            cutoff_function=cutoff_function)
        DescriptorTransformer.__init__(self)

    def get_graph(self):
        """
        Return the graph to compute symmetry function descriptors.
        """
        if not self._placeholders:
            self._initialize_placeholders()
        return self.build_graph(self.placeholders)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this transformer.
        """
        d = {'class': self.__class__.__name__, 'rc': self._rc,
             'elements': self._elements, 'angular': self._angular,
             'periodic': self._periodic, 'eta': self._eta.tolist(),
             'gamma': self._gamma.tolist(), 'zeta': self._zeta.tolist(),
             'beta': self._beta.tolist(), 'trainable': self._trainable,
             'cutoff_function': self._cutoff_function}
        return d

    @property
    def placeholders(self) -> AttributeDict:
        """
        Return a dict of placeholders.
        """
        return self._placeholders

    def _initialize_placeholders(self):
        """
        Initialize the placeholders.
        """
        graph = tf.get_default_graph()

        # Make sure the all placeholder ops are placed under the absolute path
        # of 'Placeholders/'. Placeholder ops can be recovered from graph
        # directly.
        with tf.name_scope("Placeholders/"):

            def _get_or_create(dtype, shape, name):
                try:
                    return graph.get_tensor_by_name(f'Placeholders/{name}:0')
                except KeyError:
                    return tf.placeholder(dtype, shape, name)
                except Exception as excp:
                    raise excp

            float_dtype = get_float_dtype()

            def _float(name):
                return _get_or_create(float_dtype, (), name)

            def _float_1d(name):
                return _get_or_create(float_dtype, (None, ), name)

            def _float_2d(d1, name, d0=None):
                return _get_or_create(float_dtype, (d0, d1), name)

            def _int(name):
                return _get_or_create(tf.int32, (), name)

            def _int_1d(name, d0=None):
                return _get_or_create(tf.int32, (d0, ), name)

            def _int_2d(d1, name, d0=None):
                return _get_or_create(tf.int32, (d0, d1), name)

            self._placeholders.positions = _float_2d(3, 'positions')
            self._placeholders.cells = _float_2d(d0=3, d1=3, name='cells')
            self._placeholders.n_atoms = _int('n_atoms')
            self._placeholders.volume = _float('volume')
            self._placeholders.mask = _float_1d('mask')
            self._placeholders.composition = _float_1d('composition')
            self._placeholders.row_splits = _int_1d(
                'row_splits', d0=self._n_elements + 1)
            self._placeholders.g2 = AttributeDict(
                ilist=_int_1d('g2.ilist'),
                jlist=_int_1d('g2.jlist'),
                shift=_float_2d(3, 'g2.shift'),
                v2g_map=_int_2d(2, 'g2.v2g_map')
            )

            if self._k_max == 3:
                self._placeholders.g4 = AttributeDict(
                    v2g_map=_int_2d(2, 'g4.v2g_map'),
                    ij=AttributeDict(ilist=_int_1d('g4.ij.ilist'),
                                     jlist=_int_1d('g4.ij.jlist')),
                    ik=AttributeDict(ilist=_int_1d('g4.ik.ilist'),
                                     klist=_int_1d('g4.ik.klist')),
                    jk=AttributeDict(jlist=_int_1d('g4.jk.jlist'),
                                     klist=_int_1d('g4.jk.klist')),
                    shift=AttributeDict(ij=_float_2d(3, 'g4.shift.ij'),
                                        ik=_float_2d(3, 'g4.shift.ik'),
                                        jk=_float_2d(3, 'g4.shift.jk'))
                )
        return self._placeholders

    def _get_g2_indexed_slices(self,
                               atoms: Atoms,
                               index_transformer: IndexTransformer):
        """
        Return the indexed slices for the symmetry function G2.
        """
        symbols = atoms.get_chemical_symbols()
        float_dtype = get_float_dtype()
        ilist, jlist, Slist = neighbor_list('ijS', atoms, self._rc)
        nij = len(ilist)
        v2g_map = np.zeros((nij, 2), dtype=np.int32)

        tlist = np.zeros(nij, dtype=np.int32)
        for i in range(nij):
            symboli = symbols[ilist[i]]
            symbolj = symbols[jlist[i]]
            tlist[i] = self._kbody_index['{}{}'.format(symboli, symbolj)]

        ilist = index_transformer.inplace_map_index(ilist + 1)
        jlist = index_transformer.inplace_map_index(jlist + 1)
        shift = np.asarray(Slist, dtype=float_dtype.as_numpy_dtype)
        v2g_map[:, 0] = ilist
        v2g_map[:, 1] = self._offsets[tlist]
        return G2IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               shift=shift)

    def _get_g4_indexed_slices(self,
                               atoms: Atoms,
                               g2: G2IndexedSlices,
                               transformer: IndexTransformer):
        """
        Return the indexed slices for the symmetry function G4.
        """
        symbols = atoms.get_chemical_symbols()
        float_dtype = get_float_dtype()
        indices = {}
        vectors = {}
        for i, atomi in enumerate(g2.ilist):
            if atomi not in indices:
                indices[atomi] = []
                vectors[atomi] = []
            indices[atomi].append(g2.jlist[i])
            vectors[atomi].append(g2.shift[i])

        nijk = 0
        for atomi, nl in indices.items():
            n = len(nl)
            nijk += (n - 1) * n // 2

        v2g_map = np.zeros((nijk, 2), dtype=np.int32)
        ij = np.zeros((nijk, 2), dtype=np.int32)
        ik = np.zeros((nijk, 2), dtype=np.int32)
        jk = np.zeros((nijk, 2), dtype=np.int32)
        ij_shift = np.zeros((nijk, 3), dtype=float_dtype.as_numpy_dtype)
        ik_shift = np.zeros((nijk, 3), dtype=float_dtype.as_numpy_dtype)
        jk_shift = np.zeros((nijk, 3), dtype=float_dtype.as_numpy_dtype)

        count = 0
        for atomi, nl in indices.items():
            num = len(nl)
            indexi = transformer.inplace_map_index(atomi, True, True)
            symboli = symbols[indexi]
            prefix = '{}'.format(symboli)
            for j in range(num):
                atomj = nl[j]
                indexj = transformer.inplace_map_index(atomj, True, True)
                symbolj = symbols[indexj]
                for k in range(j + 1, num):
                    atomk = nl[k]
                    indexk = transformer.inplace_map_index(atomk, True, True)
                    symbolk = symbols[indexk]
                    suffix = ''.join(sorted([symbolj, symbolk]))
                    kbody_term = '{}{}'.format(prefix, suffix)
                    ij[count] = atomi, atomj
                    ik[count] = atomi, atomk
                    jk[count] = atomj, atomk
                    ij_shift[count] = vectors[atomi][j]
                    ik_shift[count] = vectors[atomi][k]
                    jk_shift[count] = vectors[atomi][k] - vectors[atomi][j]
                    index = self._kbody_index[kbody_term]
                    v2g_map[count, 0] = atomi
                    v2g_map[count, 1] = self._offsets[index]
                    count += 1
        return G4IndexedSlices(v2g_map=v2g_map, ij=ij, ik=ik, jk=jk,
                               ij_shift=ij_shift, ik_shift=ik_shift,
                               jk_shift=jk_shift)

    def get_feed_dict(self, atoms: Atoms):
        """
        Return the feed dict.
        """
        feed_dict = {}

        if not self._placeholders:
            self._initialize_placeholders()
        placeholders = self._placeholders

        index_transformer = self.get_index_transformer(atoms)
        g2 = self._get_g2_indexed_slices(atoms, index_transformer)

        numpy_float_dtype = get_float_dtype().as_numpy_dtype

        positions = index_transformer.map_positions(atoms.positions)
        n_atoms = index_transformer.n_atoms
        cells = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        mask = index_transformer.mask
        splits = [1] + [index_transformer.max_occurs[e] for e in self._elements]
        composition = self._get_composition(atoms)

        feed_dict[placeholders.positions] = positions.astype(numpy_float_dtype)
        feed_dict[placeholders.n_atoms] = n_atoms
        feed_dict[placeholders.mask] = mask
        feed_dict[placeholders.cells] = cells.astype(numpy_float_dtype)
        feed_dict[placeholders.volume] = numpy_float_dtype(volume)
        feed_dict[placeholders.composition] = composition
        feed_dict[placeholders.row_splits] = splits
        feed_dict[placeholders.g2.v2g_map] = g2.v2g_map
        feed_dict[placeholders.g2.ilist] = g2.ilist
        feed_dict[placeholders.g2.jlist] = g2.jlist
        feed_dict[placeholders.g2.shift] = g2.shift

        if self._k_max == 3:
            g4 = self._get_g4_indexed_slices(atoms, g2, index_transformer)
            feed_dict[placeholders.g4.v2g_map] = g4.v2g_map
            feed_dict[placeholders.g4.ij.ilist] = g4.ij[:, 0]
            feed_dict[placeholders.g4.ij.jlist] = g4.ij[:, 1]
            feed_dict[placeholders.g4.ik.ilist] = g4.ik[:, 0]
            feed_dict[placeholders.g4.ik.klist] = g4.ik[:, 1]
            feed_dict[placeholders.g4.jk.jlist] = g4.jk[:, 0]
            feed_dict[placeholders.g4.jk.klist] = g4.jk[:, 1]
            feed_dict[placeholders.g4.shift.ij] = g4.ij_shift
            feed_dict[placeholders.g4.shift.ik] = g4.ik_shift
            feed_dict[placeholders.g4.shift.jk] = g4.jk_shift

        return feed_dict


class BatchSymmetryFunctionTransformer(BatchSymmetryFunction,
                                       BatchDescriptorTransformer):
    """
    A batch implementation of `SymmetryFunctionTransformer`.
    """

    def __init__(self, rc, max_occurs: Counter, nij_max: int, nijk_max: int,
                 batch_size=None, eta=Defaults.eta, beta=Defaults.beta,
                 gamma=Defaults.gamma, zeta=Defaults.zeta, angular=False,
                 periodic=True, trainable=False, cutoff_function='cosine',
                 use_forces=True, use_stress=False):
        """
        Initialization method.

        Notes
        -----
        `batch_size` is set to None by default because tranforming `Atoms` into
        indexed slices does not need this value. However, `batch_size` must be
        set before calling `build_graph()`.
        """
        if (not periodic) and use_stress:
            raise ValueError(
                'The stress tensor is not applicable to molecules.')

        elements = sorted(max_occurs.keys())

        BatchSymmetryFunction.__init__(
            self, rc=rc, max_occurs=max_occurs, elements=elements,
            nij_max=nij_max, nijk_max=nijk_max, batch_size=batch_size, eta=eta,
            beta=beta, gamma=gamma, zeta=zeta, angular=angular,
            trainable=trainable, periodic=periodic,
            cutoff_function=cutoff_function)

        BatchDescriptorTransformer.__init__(self, use_forces=use_forces,
                                            use_stress=use_stress)

    @property
    def batch_size(self):
        """
        Return the batch size.
        """
        return self._batch_size

    def as_descriptor_transformer(self):
        """
        Return the corresponding `SymmetryFunctionTransformer`.
        """
        return SymmetryFunctionTransformer(
            rc=self._rc, elements=self._elements, eta=self._eta,
            beta=self._beta, gamma=self._gamma, zeta=self._zeta,
            angular=self._angular, trainable=self._trainable,
            periodic=self._periodic, cutoff_function=self._cutoff_function)

    def get_g2_indexed_slices(self, atoms: Atoms):
        """
        Return the indexed slices for the radial function G2.
        """
        v2g_map = np.zeros((self._nij_max, 3), dtype=np.int32)
        tlist = np.zeros(self._nij_max, dtype=np.int32)
        float_dtype = get_float_dtype()

        symbols = atoms.get_chemical_symbols()
        transformer = self.get_index_transformer(atoms)

        ilist, jlist, Slist = neighbor_list('ijS', atoms, self._rc)
        if self._k_max == 1:
            cols = [i for i in range(len(ilist))
                    if symbols[ilist[i]] == symbols[jlist[i]]]
            ilist = ilist[cols]
            jlist = jlist[cols]
            Slist = Slist[cols]
        nij = len(ilist)

        tlist.fill(0)
        for i in range(nij):
            symboli = symbols[ilist[i]]
            symbolj = symbols[jlist[i]]
            tlist[i] = self._kbody_index['{}{}'.format(symboli, symbolj)]

        ilist = self._resize_to_nij_max(ilist, True)
        jlist = self._resize_to_nij_max(jlist, True)
        Slist = self._resize_to_nij_max(Slist, False)
        ilist = transformer.inplace_map_index(ilist)
        jlist = transformer.inplace_map_index(jlist)
        shift = np.asarray(Slist, dtype=float_dtype.as_numpy_dtype)
        v2g_map[:self._nij_max, 1] = ilist
        v2g_map[:self._nij_max, 2] = self._offsets[tlist]

        return G2IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               shift=shift)

    def get_g4_indexed_slices(self, atoms: Atoms, g2: G2IndexedSlices):
        """
        Return the indexed slices for the angular function G4.
        """
        if self._k_max < 3:
            return None

        numpy_float_dtype = get_float_dtype().as_numpy_dtype

        v2g_map = np.zeros((self._nijk_max, 3), dtype=np.int32)
        ij = np.zeros((self._nijk_max, 2), dtype=np.int32)
        ik = np.zeros((self._nijk_max, 2), dtype=np.int32)
        jk = np.zeros((self._nijk_max, 2), dtype=np.int32)
        ij_shift = np.zeros((self._nijk_max, 3), dtype=numpy_float_dtype)
        ik_shift = np.zeros((self._nijk_max, 3), dtype=numpy_float_dtype)
        jk_shift = np.zeros((self._nijk_max, 3), dtype=numpy_float_dtype)

        symbols = atoms.get_chemical_symbols()
        transformer = self.get_index_transformer(atoms)
        indices = {}
        vectors = {}
        for i, atomi in enumerate(g2.ilist):
            if atomi == 0:
                break
            if atomi not in indices:
                indices[atomi] = []
                vectors[atomi] = []
            indices[atomi].append(g2.jlist[i])
            vectors[atomi].append(g2.shift[i])

        count = 0
        for atomi, nl in indices.items():
            num = len(nl)
            indexi = transformer.inplace_map_index(atomi, True, True)
            symboli = symbols[indexi]
            prefix = '{}'.format(symboli)
            for j in range(num):
                atomj = nl[j]
                indexj = transformer.inplace_map_index(atomj, True, True)
                symbolj = symbols[indexj]
                for k in range(j + 1, num):
                    atomk = nl[k]
                    indexk = transformer.inplace_map_index(atomk, True, True)
                    symbolk = symbols[indexk]
                    suffix = ''.join(sorted([symbolj, symbolk]))
                    kbody_term = '{}{}'.format(prefix, suffix)
                    ij[count] = atomi, atomj
                    ik[count] = atomi, atomk
                    jk[count] = atomj, atomk
                    ij_shift[count] = vectors[atomi][j]
                    ik_shift[count] = vectors[atomi][k]
                    jk_shift[count] = vectors[atomi][k] - vectors[atomi][j]
                    index = self._kbody_index[kbody_term]
                    v2g_map[count, 1] = atomi
                    v2g_map[count, 2] = self._offsets[index]
                    count += 1
        return G4IndexedSlices(v2g_map=v2g_map, ij=ij, ik=ik, jk=jk,
                               ij_shift=ij_shift, ik_shift=ik_shift,
                               jk_shift=jk_shift)

    def get_indexed_slices(self, atoms: Atoms):
        """
        Return both the radial and angular indexed slices for the trajectory.
        """
        g2 = self.get_g2_indexed_slices(atoms)
        g4 = self.get_g4_indexed_slices(atoms, g2)
        return g2, g4

    @staticmethod
    def _encode_g4_indexed_slices(g4: G4IndexedSlices):
        """
        Encode the indexed slices of G4:
            * `v2g_map`, `ij`, `ik` and `jk` are merged into a single array
              with key 'a_indices'.
            * `ij_shift`, `ik_shift` and `jk_shift` are merged into another
              array with key 'a_shifts'.

        """
        indices = np.concatenate(
            (g4.v2g_map, g4.ij, g4.ik, g4.jk), axis=1).tostring()
        shifts = np.concatenate(
            (g4.ij_shift, g4.ik_shift, g4.jk_shift), axis=1).tostring()
        return {'g4.indices': bytes_feature(indices),
                'g4.shifts': bytes_feature(shifts)}

    def encode(self, atoms: Atoms):
        """
        Encode the `Atoms` object and return a `tf.train.Example`.
        """
        feature_list = self._encode_atoms(atoms)
        g2, g4 = self.get_indexed_slices(atoms)
        feature_list.update(self._encode_g2_indexed_slices(g2))

        if self.k_max == 3:
            feature_list.update(self._encode_g4_indexed_slices(g4))

        return tf.train.Example(
            features=tf.train.Features(feature=feature_list))

    def _decode_atoms(self, example: Dict[str, tf.Tensor]) -> AttributeDict:
        """
        Decode `Atoms` related properties.
        """
        decoded = AttributeDict()

        length = 3 * self._max_n_atoms
        float_dtype = get_float_dtype()

        positions = tf.decode_raw(example['positions'], float_dtype)
        positions.set_shape([length])
        decoded.positions = tf.reshape(
            positions, (self._max_n_atoms, 3), name='R')

        n_atoms = tf.identity(example['n_atoms'], name='n_atoms')
        decoded.n_atoms = n_atoms

        y_true = tf.decode_raw(example['y_true'], float_dtype)
        y_true.set_shape([1])
        decoded.y_true = tf.squeeze(y_true, name='y_true')

        cells = tf.decode_raw(example['cells'], float_dtype)
        cells.set_shape([9])
        decoded.cells = tf.reshape(cells, (3, 3), name='cells')

        volume = tf.decode_raw(example['volume'], float_dtype)
        volume.set_shape([1])
        decoded.volume = tf.squeeze(volume, name='volume')

        mask = tf.decode_raw(example['mask'], float_dtype)
        mask.set_shape([self._max_n_atoms, ])
        decoded.mask = mask

        composition = tf.decode_raw(example['composition'], float_dtype)
        composition.set_shape([self._n_elements, ])
        decoded.composition = composition

        if self._use_forces:
            f_true = tf.decode_raw(example['f_true'], float_dtype)
            # Ignore the forces of the virtual atom
            f_true.set_shape([length, ])
            decoded.f_true = tf.reshape(
                f_true, (self._max_n_atoms, 3), name='f_true')

        if self._use_stress:
            stress = tf.decode_raw(
                example['stress'], float_dtype, name='stress')
            stress.set_shape([6])
            decoded.stress = stress

            total_pressure = tf.decode_raw(
                example['total_pressure'], float_dtype, name='stress')
            total_pressure.set_shape([1])
            decoded.total_pressure = total_pressure

        return decoded

    def _decode_g2_indexed_slices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ilist, jlist and Slist for radial functions.
        """
        with tf.name_scope("G2"):
            indices = tf.decode_raw(example['g2.indices'], tf.int32)
            indices.set_shape([self._nij_max * 5])
            indices = tf.reshape(
                indices, [self._nij_max, 5], name='g2.indices')
            v2g_map, ilist, jlist = tf.split(
                indices, [3, 1, 1], axis=1, name='splits')
            ilist = tf.squeeze(ilist, axis=1, name='ilist')
            jlist = tf.squeeze(jlist, axis=1, name='jlist')

            shift = tf.decode_raw(example['g2.shifts'], get_float_dtype())
            shift.set_shape([self._nij_max * 3])
            shift = tf.reshape(shift, [self._nij_max, 3], name='shift')

            return G2IndexedSlices(v2g_map, ilist, jlist, shift)

    def _decode_g4_indexed_slices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ij, ik, jk, ijSlist, ikSlist and jkSlist for angular
        functions.
        """
        if self.k_max < 3:
            return None

        with tf.name_scope("G4"):
            with tf.name_scope("indices"):
                indices = tf.decode_raw(example['g4.indices'], tf.int32)
                indices.set_shape([self._nijk_max * 9])
                indices = tf.reshape(
                    indices, [self._nijk_max, 9], name='g4.indices')
                v2g_map, ij, ik, jk = \
                    tf.split(indices, [3, 2, 2, 2], axis=1, name='splits')

            with tf.name_scope("shifts"):
                shifts = tf.decode_raw(example['g4.shifts'], get_float_dtype())
                shifts.set_shape([self._nijk_max * 9])
                shifts = tf.reshape(
                    shifts, [self._nijk_max, 9], name='g4.shifts')
                ij_shift, ik_shift, jk_shift = \
                    tf.split(shifts, [3, 3, 3], axis=1, name='splits')

        return G4IndexedSlices(v2g_map, ij, ik, jk, ij_shift, ik_shift,
                               jk_shift)

    def _decode_example(self, example: Dict[str, tf.Tensor]):
        """
        Decode the parsed single example.
        """
        decoded = self._decode_atoms(example)
        g2 = self._decode_g2_indexed_slices(example)
        g4 = self._decode_g4_indexed_slices(example)

        decoded.rv2g = g2.v2g_map
        decoded.ilist = g2.ilist
        decoded.jlist = g2.jlist
        decoded.shift = g2.shift

        if g4 is not None:
            decoded.av2g = g4.v2g_map
            decoded.ij = g4.ij
            decoded.ik = g4.ik
            decoded.jk = g4.jk
            decoded.ij_shift = g4.ij_shift
            decoded.ik_shift = g4.ik_shift
            decoded.jk_shift = g4.jk_shift

        return decoded

    def decode_protobuf(self, example_proto: tf.Tensor) -> AttributeDict:
        """
        Decode the scalar string Tensor, which is a single serialized Example.
        See `_parse_single_example_raw` documentation for more details.
        """
        with tf.name_scope("decoding"):

            feature_list = {
                'positions': tf.FixedLenFeature([], tf.string),
                'n_atoms': tf.FixedLenFeature([], tf.int64),
                'cells': tf.FixedLenFeature([], tf.string),
                'volume': tf.FixedLenFeature([], tf.string),
                'y_true': tf.FixedLenFeature([], tf.string),
                'g2.indices': tf.FixedLenFeature([], tf.string),
                'g2.shifts': tf.FixedLenFeature([], tf.string),
                'mask': tf.FixedLenFeature([], tf.string),
                'composition': tf.FixedLenFeature([], tf.string),
            }
            if self._use_forces:
                feature_list['f_true'] = tf.FixedLenFeature([], tf.string)

            if self._use_stress:
                feature_list['stress'] = \
                    tf.FixedLenFeature([], tf.string)
                feature_list['total_pressure'] = \
                    tf.FixedLenFeature([], tf.string)

            if self._k_max == 3:
                feature_list.update({
                    'g4.indices': tf.FixedLenFeature([], tf.string),
                    'g4.shifts': tf.FixedLenFeature([], tf.string),
                })

            example = tf.parse_single_example(example_proto, feature_list)
            return self._decode_example(example)

    def get_descriptors(self, batch_features: AttributeDict):
        """
        Return the Op to compute symmetry function descriptors.

        This function is necessary because nested dicts are not supported by
        `tf.data.Dataset.batch`.

        Parameters
        ----------
        batch_features : AttributeDict
            A batch of raw properties provided by `tf.data.Dataset`. Each batch
            is produced by the function `decode_protobuf`.

            Here are default keys:

            * 'positions': float64 or float32, [batch_size, max_n_atoms + 1, 3]
            * 'cells': float64 or float32, [batch_size, 3, 3]
            * 'volume': float64 or float32, [batch_size, ]
            * 'n_atoms': int64, [batch_size, ]
            * 'y_true': float64 or float32, [batch_size, ]
            * 'f_true': float64 or float32, [batch_size, max_n_atoms + 1, 3]
            * 'composition': float64 or float32, [batch_size, n_elements]
            * 'mask': float64 or float32, [batch_size, max_n_atoms + 1]
            * 'ilist': int32, [batch_size, nij_max]
            * 'jlist': int32, [batch_size, nij_max]
            * 'shift': float64 or float32, [batch_size, nij_max, 3]
            * 'rv2g': int32, [batch_size, nij_max, 3]

            If `self.stress` is `True`, these following keys will be provided:

            * 'stress': float64 or float32, [batch_size, 6]
            * 'total_pressure': float64 or float32, [batch_size, ]

            If `self.angular` is `True`, these following keys will be provided:

            * 'ij': int32, [batch_size, nijk_max, 2]
            * 'ik': int32, [batch_size, nijk_max, 2]
            * 'jk': int32, [batch_size, nijk_max, 2]
            * 'ij_shift': float64 or float32, [batch_size, nijk_max, 3]
            * 'ik_shift': float64 or float32, [batch_size, nijk_max, 3]
            * 'jk_shift': float64 or float32, [batch_size, nijk_max, 3]
            * 'av2g': int32, [batch_size, nijk_max, 3]

        Returns
        -------
        descriptors : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of (element, (value, mask)) where `element` is a symbol,
            `value` is the Op to compute its atomic descriptors and `mask` is a
            `tf.no_op`.

        """
        self._infer_batch_size(batch_features)

        inputs = AttributeDict()
        inputs.g2 = AttributeDict(
            ilist=batch_features.ilist,
            jlist=batch_features.jlist,
            shift=batch_features.shift,
            v2g_map=batch_features.rv2g
        )
        inputs.positions = batch_features.positions
        inputs.cells = batch_features.cells
        inputs.volume = batch_features.volume

        if self._k_max == 3:
            inputs.g4 = AttributeDict(
                ij=AttributeDict(
                    ilist=batch_features.ij[..., 0],
                    jlist=batch_features.ij[..., 1]),
                ik=AttributeDict(
                    ilist=batch_features.ik[..., 0],
                    klist=batch_features.ik[..., 1]),
                jk=AttributeDict(
                    jlist=batch_features.jk[..., 0],
                    klist=batch_features.jk[..., 1]),
                shift=AttributeDict(
                    ij=batch_features.ij_shift,
                    ik=batch_features.ik_shift,
                    jk=batch_features.jk_shift,),
                v2g_map=batch_features.av2g,
            )

        return self.build_graph(inputs)

    def get_descriptor_normalization_weights(self, method):
        """
        Return the initial weights for the column-wise normalising the output
        descriptors.
        """
        weights = {}
        for element in self._elements:
            kbody_terms = self._kbody_terms[element]
            values = []
            for kbody_term in kbody_terms:
                if len(get_elements_from_kbody_term(kbody_term)) == 2:
                    values.extend(self._eta.tolist())
                else:
                    for p in self._indices_grid:
                        values.append(p['beta'])
            values = np.asarray(values, dtype=get_float_dtype().as_numpy_dtype)
            weights[element] = 0.25 / np.exp(-values * 0.25**2)
        return weights
