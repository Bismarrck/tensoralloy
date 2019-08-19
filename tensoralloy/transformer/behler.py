# coding=utf-8
"""
This module defines the feature transformers (flexible, fixed) for Behler's
Symmetry Function descriptor.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from collections import Counter
from typing import Dict
from ase import Atoms
from ase.neighborlist import neighbor_list

from tensoralloy.descriptor import SymmetryFunction, BatchSymmetryFunction
from tensoralloy.precision import get_float_dtype
from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.transformer.base import bytes_feature
from tensoralloy.transformer.vap import VirtualAtomMap
from tensoralloy.transformer.indexed_slices import G2IndexedSlices
from tensoralloy.transformer.indexed_slices import G4IndexedSlices
from tensoralloy.utils import Defaults, get_pulay_stress

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["SymmetryFunctionTransformer", "BatchSymmetryFunctionTransformer"]


class SymmetryFunctionTransformer(SymmetryFunction, DescriptorTransformer):
    """
    A tranformer for providing the feed dict to calculate symmetry function
    descriptors of an `Atoms` object.
    """

    def __init__(self, rc, elements, eta=Defaults.eta, omega=Defaults.omega,
                 beta=Defaults.beta, gamma=Defaults.gamma, zeta=Defaults.zeta,
                 angular=False, periodic=True):
        """
        Initialization method.
        """
        SymmetryFunction.__init__(
            self, rc=rc, elements=elements, eta=eta, omega=omega, beta=beta,
            gamma=gamma, zeta=zeta, angular=angular, periodic=periodic)
        DescriptorTransformer.__init__(self)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this transformer.
        """
        d = {'class': self.__class__.__name__, 'rc': self._rc,
             'elements': self._elements, 'angular': self._angular,
             'periodic': self.periodic,
             'eta': self.initial_values["eta"].tolist(),
             'omega': self.initial_values["omega"].tolist(),
             'gamma': self.initial_values["gamma"].tolist(),
             'zeta': self.initial_values["zeta"].tolist(),
             'beta': self.initial_values["beta"].tolist()}
        return d

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

            self._placeholders["positions"] = _float_2d(3, 'positions')
            self._placeholders["cells"] = _float_2d(d0=3, d1=3, name='cells')
            self._placeholders["n_atoms_vap"] = _int('n_atoms_vap')
            self._placeholders["volume"] = _float('volume')
            self._placeholders["atom_masks"] = _float_1d('atom_masks')
            self._placeholders["pulay_stress"] = _float('pulay_stress')
            self._placeholders["composition"] = _float_1d('composition')
            self._placeholders["row_splits"] = _int_1d(
                'row_splits', d0=self.n_elements + 1)
            self._placeholders["g2.ilist"] = _int_1d('g2.ilist')
            self._placeholders["g2.jlist"] = _int_1d('g2.jlist')
            self._placeholders["g2.n1"] = _float_2d(3, 'g2.n1')
            self._placeholders["g2.v2g_map"] = _int_2d(2, 'g2.v2g_map')

            if self._angular:
                self._placeholders["g4.v2g_map"] = _int_2d(2, 'g4.v2g_map')
                self._placeholders["g4.ilist"] = _int_1d('g4.ilist')
                self._placeholders["g4.jlist"] = _int_1d('g4.jlist')
                self._placeholders["g4.klist"] = _int_1d('g4.klist')
                self._placeholders["g4.n1"] = _float_2d(3, 'g4.n1')
                self._placeholders["g4.n2"] = _float_2d(3, 'g4.n2')
                self._placeholders["g4.n3"] = _float_2d(3, 'g4.n3')

        return self._placeholders

    def _get_g2_indexed_slices(self,
                               atoms: Atoms,
                               vap: VirtualAtomMap):
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
            tlist[i] = self.kbody_index['{}{}'.format(symboli, symbolj)]

        ilist = vap.inplace_map_index(ilist + 1)
        jlist = vap.inplace_map_index(jlist + 1)
        n1 = np.asarray(Slist, dtype=float_dtype.as_numpy_dtype)
        v2g_map[:, 0] = ilist
        v2g_map[:, 1] = self.offsets[tlist]
        return G2IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist, n1=n1)

    def _get_g4_indexed_slices(self,
                               atoms: Atoms,
                               g2: G2IndexedSlices,
                               vap: VirtualAtomMap):
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
            vectors[atomi].append(g2.n1[i])

        nijk = 0
        for atomi, nl in indices.items():
            n = len(nl)
            nijk += (n - 1) * n // 2

        v2g_map = np.zeros((nijk, 2), dtype=np.int32)
        ilist = np.zeros(nijk, dtype=np.int32)
        jlist = np.zeros(nijk, dtype=np.int32)
        klist = np.zeros(nijk, dtype=np.int32)
        n1 = np.zeros((nijk, 3), dtype=float_dtype.as_numpy_dtype)
        n2 = np.zeros((nijk, 3), dtype=float_dtype.as_numpy_dtype)
        n3 = np.zeros((nijk, 3), dtype=float_dtype.as_numpy_dtype)

        count = 0
        for atomi, nl in indices.items():
            num = len(nl)
            indexi = vap.inplace_map_index(atomi, True, True)
            symboli = symbols[indexi]
            prefix = '{}'.format(symboli)
            for j in range(num):
                atomj = nl[j]
                indexj = vap.inplace_map_index(atomj, True, True)
                symbolj = symbols[indexj]
                for k in range(j + 1, num):
                    atomk = nl[k]
                    indexk = vap.inplace_map_index(atomk, True, True)
                    symbolk = symbols[indexk]
                    suffix = ''.join(sorted([symbolj, symbolk]))
                    kbody_term = '{}{}'.format(prefix, suffix)
                    ilist[count] = atomi
                    jlist[count] = atomj
                    klist[count] = atomk
                    n1[count] = vectors[atomi][j]
                    n2[count] = vectors[atomi][k]
                    n3[count] = vectors[atomi][k] - vectors[atomi][j]
                    index = self.kbody_index[kbody_term]
                    v2g_map[count, 0] = atomi
                    v2g_map[count, 1] = self.offsets[index]
                    count += 1
        return G4IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               klist=klist, n1=n1, n2=n2, n3=n3)

    def _get_np_features(self, atoms: Atoms):
        """
        Return a dict of features (Numpy or Python objects).
        """
        feed_dict = dict()

        vap = self.get_vap_transformer(atoms)
        g2 = self._get_g2_indexed_slices(atoms, vap)

        np_dtype = get_float_dtype().as_numpy_dtype
        positions = vap.map_positions(atoms.positions)

        # `max_n_atoms` must be used because every element shall have at least
        # one feature row (though it could be all zeros, a dummy or virtual row)
        vap_natoms = vap.max_vap_natoms
        cells = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        atom_masks = vap.atom_masks
        splits = [1] + [vap.max_occurs[e] for e in self._elements]
        composition = self._get_composition(atoms)
        pulay_stress = get_pulay_stress(atoms)

        feed_dict["positions"] = positions.astype(np_dtype)
        feed_dict["n_atoms_vap"] = np.int32(vap_natoms)
        feed_dict["atom_masks"] = atom_masks.astype(np_dtype)
        feed_dict["cells"] = cells.array.astype(np_dtype)
        feed_dict["volume"] = np_dtype(volume)
        feed_dict["composition"] = composition
        feed_dict["pulay_stress"] = np_dtype(pulay_stress)
        feed_dict["row_splits"] = np.int32(splits)
        feed_dict["g2.v2g_map"] = g2.v2g_map
        feed_dict["g2.ilist"] = g2.ilist
        feed_dict["g2.jlist"] = g2.jlist
        feed_dict["g2.n1"] = g2.n1

        if self._angular:
            g4 = self._get_g4_indexed_slices(atoms, g2, vap)
            feed_dict["g4.v2g_map"] = g4.v2g_map
            feed_dict["g4.ilist"] = g4.ilist
            feed_dict["g4.jlist"] = g4.jlist
            feed_dict["g4.klist"] = g4.klist
            feed_dict["g4.n1"] = g4.n1
            feed_dict["g4.n2"] = g4.n2
            feed_dict["g4.n3"] = g4.n3

        return feed_dict

    def get_feed_dict(self, atoms: Atoms):
        """
        Return the feed dict.
        """
        feed_dict = {}
        if not self._placeholders:
            self._initialize_placeholders()
        placeholders = self._placeholders
        for key, value in self._get_np_features(atoms).items():
            feed_dict[placeholders[key]] = value
        return feed_dict

    def get_constant_features(self, atoms: Atoms):
        """
        Return a dict of constant feature tensors for the given `Atoms`.
        """
        feed_dict = {}
        with tf.name_scope("Constants"):
            for key, value in self._get_np_features(atoms).items():
                feed_dict[key] = tf.convert_to_tensor(value, name=key)
            return feed_dict


class BatchSymmetryFunctionTransformer(BatchSymmetryFunction,
                                       BatchDescriptorTransformer):
    """
    A batch implementation of `SymmetryFunctionTransformer`.
    """

    def __init__(self, rc, max_occurs: Counter, nij_max: int, nijk_max: int,
                 batch_size=None, eta=Defaults.eta, omega=Defaults.omega,
                 beta=Defaults.beta, gamma=Defaults.gamma, zeta=Defaults.zeta,
                 angular=False, periodic=True, use_forces=True,
                 use_stress=False):
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

        BatchSymmetryFunction.__init__(
            self, rc=rc, max_occurs=max_occurs,
            nij_max=nij_max, nijk_max=nijk_max, batch_size=batch_size, eta=eta,
            omega=omega, beta=beta, gamma=gamma, zeta=zeta, angular=angular,
            periodic=periodic)

        BatchDescriptorTransformer.__init__(self, use_forces=use_forces,
                                            use_stress=use_stress)

    def as_dict(self) -> Dict:
        """
        Return a JSON serializable dict representation of this transformer.
        """
        d = {'class': self.__class__.__name__, 'rc': self._rc,
             'max_occurs': self._max_occurs, 'nij_max': self._nij_max,
             'nijk_max': self._nijk_max, 'batch_size': self._batch_size,
             'angular': self._angular, 'periodic': self.periodic,
             'eta': self.initial_values["eta"].tolist(),
             'omega': self.initial_values["omega"].tolist(),
             'gamma': self.initial_values["gamma"].tolist(),
             'zeta': self.initial_values["zeta"].tolist(),
             'beta': self.initial_values["beta"].tolist(),
             'use_forces': self._use_forces, 'use_stress': self._use_stress}
        return d

    @property
    def descriptor(self):
        """
        Return the name of the descriptor.
        """
        return "behler"

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
            rc=self._rc, elements=self._elements,
            eta=self.initial_values['eta'],
            omega=self.initial_values['omega'],
            beta=self.initial_values['beta'],
            gamma=self.initial_values['gamma'],
            zeta=self.initial_values['zeta'], angular=self._angular,
            periodic=self.periodic)

    def get_g2_indexed_slices(self, atoms: Atoms):
        """
        Return the indexed slices for the radial function G2.
        """
        v2g_map = np.zeros((self._nij_max, 3), dtype=np.int32)
        tlist = np.zeros(self._nij_max, dtype=np.int32)
        float_dtype = get_float_dtype()

        symbols = atoms.get_chemical_symbols()
        vap = self.get_vap_transformer(atoms)

        ilist, jlist, Slist = neighbor_list('ijS', atoms, self._rc)
        nij = len(ilist)

        tlist.fill(0)
        for i in range(nij):
            symboli = symbols[ilist[i]]
            symbolj = symbols[jlist[i]]
            tlist[i] = self.kbody_index['{}{}'.format(symboli, symbolj)]

        ilist = self._resize_to_nij_max(ilist, True)
        jlist = self._resize_to_nij_max(jlist, True)
        Slist = self._resize_to_nij_max(Slist, False)
        ilist = vap.inplace_map_index(ilist)
        jlist = vap.inplace_map_index(jlist)
        n1 = np.asarray(Slist, dtype=float_dtype.as_numpy_dtype)
        v2g_map[:self._nij_max, 1] = ilist
        v2g_map[:self._nij_max, 2] = self.offsets[tlist]

        return G2IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               n1=n1)

    def get_g4_indexed_slices(self, atoms: Atoms, g2: G2IndexedSlices):
        """
        Return the indexed slices for the angular function G4.
        """
        if not self._angular:
            return None

        numpy_float_dtype = get_float_dtype().as_numpy_dtype

        v2g_map = np.zeros((self._nijk_max, 3), dtype=np.int32)
        ilist = np.zeros(self._nijk_max, dtype=np.int32)
        jlist = np.zeros(self._nijk_max, dtype=np.int32)
        klist = np.zeros(self._nijk_max, dtype=np.int32)
        n1 = np.zeros((self._nijk_max, 3), dtype=numpy_float_dtype)
        n2 = np.zeros((self._nijk_max, 3), dtype=numpy_float_dtype)
        n3 = np.zeros((self._nijk_max, 3), dtype=numpy_float_dtype)

        symbols = atoms.get_chemical_symbols()
        vap = self.get_vap_transformer(atoms)
        indices = {}
        vectors = {}
        for i, atomi in enumerate(g2.ilist):
            if atomi == 0:
                break
            if atomi not in indices:
                indices[atomi] = []
                vectors[atomi] = []
            indices[atomi].append(g2.jlist[i])
            vectors[atomi].append(g2.n1[i])

        count = 0
        for atomi, nl in indices.items():
            num = len(nl)
            indexi = vap.inplace_map_index(atomi, True, True)
            symboli = symbols[indexi]
            prefix = '{}'.format(symboli)
            for j in range(num):
                atomj = nl[j]
                indexj = vap.inplace_map_index(atomj, True, True)
                symbolj = symbols[indexj]
                for k in range(j + 1, num):
                    atomk = nl[k]
                    indexk = vap.inplace_map_index(atomk, True, True)
                    symbolk = symbols[indexk]
                    suffix = ''.join(sorted([symbolj, symbolk]))
                    kbody_term = '{}{}'.format(prefix, suffix)
                    ilist[count] = atomi
                    jlist[count] = atomj
                    klist[count] = atomk
                    n1[count] = vectors[atomi][j]
                    n2[count] = vectors[atomi][k]
                    n3[count] = vectors[atomi][k] - vectors[atomi][j]
                    index = self.kbody_index[kbody_term]
                    v2g_map[count, 1] = atomi
                    v2g_map[count, 2] = self.offsets[index]
                    count += 1
        return G4IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               klist=klist, n1=n1, n2=n2, n3=n3)

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
            (g4.v2g_map,
             g4.ilist[..., np.newaxis],
             g4.jlist[..., np.newaxis],
             g4.klist[..., np.newaxis]), axis=1).tostring()
        shifts = np.concatenate((g4.n1, g4.n2, g4.n3), axis=1).tostring()
        return {'g4.indices': bytes_feature(indices),
                'g4.shifts': bytes_feature(shifts)}

    def encode(self, atoms: Atoms):
        """
        Encode the `Atoms` object and return a `tf.train.Example`.
        """
        feature_list = self._encode_atoms(atoms)
        g2, g4 = self.get_indexed_slices(atoms)
        feature_list.update(self._encode_g2_indexed_slices(g2))

        if self._angular:
            feature_list.update(self._encode_g4_indexed_slices(g4))

        return tf.train.Example(
            features=tf.train.Features(feature=feature_list))

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

            n1 = tf.decode_raw(example['g2.shifts'], get_float_dtype())
            n1.set_shape([self._nij_max * 3])
            n1 = tf.reshape(n1, [self._nij_max, 3], name='shift')

            return G2IndexedSlices(v2g_map, ilist, jlist, n1)

    def _decode_g4_indexed_slices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ij, ik, jk, ijSlist, ikSlist and jkSlist for angular
        functions.
        """
        with tf.name_scope("G4"):
            with tf.name_scope("indices"):
                indices = tf.decode_raw(example['g4.indices'], tf.int32)
                indices.set_shape([self._nijk_max * 6])
                indices = tf.reshape(
                    indices, [self._nijk_max, 6], name='g4.indices')
                v2g_map, ilist, jlist, klist = \
                    tf.split(indices, [3, 1, 1, 1], axis=1, name='splits')
                ilist = tf.squeeze(ilist, axis=1, name='ilist')
                jlist = tf.squeeze(jlist, axis=1, name='jlist')
                klist = tf.squeeze(klist, axis=1, name='klist')

            with tf.name_scope("shifts"):
                shifts = tf.decode_raw(example['g4.shifts'], get_float_dtype())
                shifts.set_shape([self._nijk_max * 9])
                shifts = tf.reshape(
                    shifts, [self._nijk_max, 9], name='g4.shifts')
                n1, n2, n3 = \
                    tf.split(shifts, [3, 3, 3], axis=1, name='splits')

        return G4IndexedSlices(v2g_map, ilist, jlist, klist, n1, n2, n3)

    def _decode_example(self, example: Dict[str, tf.Tensor]) -> dict:
        """
        Decode the parsed single example.
        """
        decoded = self._decode_atoms(
            example,
            max_n_atoms=self._max_n_atoms,
            n_elements=self.n_elements,
            use_forces=self._use_forces,
            use_stress=self._use_stress)
        g2 = self._decode_g2_indexed_slices(example)
        g4 = self._decode_g4_indexed_slices(example)

        decoded["g2.v2g_map"] = g2.v2g_map
        decoded["g2.ilist"] = g2.ilist
        decoded["g2.jlist"] = g2.jlist
        decoded["g2.n1"] = g2.n1

        if g4 is not None:
            decoded["g4.v2g_map"] = g4.v2g_map
            decoded["g4.ilist"] = g4.ilist
            decoded["g4.jlist"] = g4.jlist
            decoded["g4.klist"] = g4.klist
            decoded["g4.n1"] = g4.n1
            decoded["g4.n2"] = g4.n2
            decoded["g4.n3"] = g4.n3

        return decoded

    def decode_protobuf(self, example_proto: tf.Tensor) -> dict:
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
                'pulay': tf.FixedLenFeature([], tf.string),
            }
            if self._use_forces:
                feature_list['f_true'] = tf.FixedLenFeature([], tf.string)

            if self._use_stress:
                feature_list['stress'] = \
                    tf.FixedLenFeature([], tf.string)

            if self._angular:
                feature_list.update({
                    'g4.indices': tf.FixedLenFeature([], tf.string),
                    'g4.shifts': tf.FixedLenFeature([], tf.string)})

            example = tf.parse_single_example(example_proto, feature_list)
            return self._decode_example(example)

    def get_descriptors(self, next_batch: dict):
        """
        Return the Op to compute symmetry function descriptors.
        """
        self._infer_batch_size(next_batch)
        return self.build_graph(next_batch)
