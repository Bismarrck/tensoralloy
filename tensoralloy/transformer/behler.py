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
from tensorflow_estimator import estimator as tf_estimator

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


def get_g2_map(atoms: Atoms,
               rc: float,
               interactions: dict,
               vap: VirtualAtomMap,
               offsets: np.ndarray,
               mode: tf_estimator.ModeKeys,
               nij_max: int = None,
               dtype=np.float32):
    """
    Build the base `v2g_map` for radial symmetry functions.
    """
    if mode == tf_estimator.ModeKeys.PREDICT:
        iaxis = 0
    else:
        iaxis = 1

    ilist, jlist, n1 = neighbor_list('ijS', atoms, rc)
    nij = len(ilist)
    if nij_max is None:
        nij_max = nij

    g2_map = np.zeros((nij_max, iaxis + 2), dtype=np.int32)
    g2_map.fill(0)
    tlist = np.zeros(nij_max, dtype=np.int32)
    symbols = atoms.get_chemical_symbols()
    tlist.fill(0)
    for i in range(nij):
        symboli = symbols[ilist[i]]
        symbolj = symbols[jlist[i]]
        tlist[i] = interactions['{}{}'.format(symboli, symbolj)]
    ilist = np.pad(ilist + 1, (0, nij_max - nij), 'constant')
    jlist = np.pad(jlist + 1, (0, nij_max - nij), 'constant')
    n1 = np.pad(n1, ((0, nij_max - nij), (0, 0)), 'constant')
    n1 = n1.astype(dtype)
    for count in range(len(ilist)):
        if ilist[count] == 0:
            break
        ilist[count] = vap.local_to_gsl_map[ilist[count]]
        jlist[count] = vap.local_to_gsl_map[jlist[count]]
    g2_map[:, iaxis + 0] = ilist
    g2_map[:, iaxis + 1] = offsets[tlist]
    ilist = ilist.astype(np.int32)
    jlist = jlist.astype(np.int32)
    return G2IndexedSlices(v2g_map=g2_map, ilist=ilist, jlist=jlist, n1=n1)


def get_g4_map(atoms: Atoms,
               g2: G2IndexedSlices,
               interactions: dict,
               offsets: np.ndarray,
               vap: VirtualAtomMap,
               mode: tf_estimator.ModeKeys,
               nijk_max: int = None):
    """
    Build the base `v2g_map` for angular symmetry functions.
    """
    if mode == tf_estimator.ModeKeys.PREDICT:
        iaxis = 0
    else:
        iaxis = 1

    indices = {}
    vectors = {}
    for i, atom_gsl_i in enumerate(g2.ilist):
        if atom_gsl_i == 0:
            break
        if atom_gsl_i not in indices:
            indices[atom_gsl_i] = []
            vectors[atom_gsl_i] = []
        indices[atom_gsl_i].append(g2.jlist[i])
        vectors[atom_gsl_i].append(g2.n1[i])

    if nijk_max is None:
        nijk = 0
        for atomi, nl in indices.items():
            n = len(nl)
            nijk += (n - 1) * n // 2
        nijk_max = nijk

    g4_map = np.zeros((nijk_max, iaxis + 2), dtype=np.int32)
    g4_map.fill(0)
    ilist = np.zeros(nijk_max, dtype=np.int32)
    jlist = np.zeros(nijk_max, dtype=np.int32)
    klist = np.zeros(nijk_max, dtype=np.int32)
    n1 = np.zeros((nijk_max, 3), dtype=g2.n1.dtype)
    n2 = np.zeros((nijk_max, 3), dtype=g2.n1.dtype)
    n3 = np.zeros((nijk_max, 3), dtype=g2.n1.dtype)
    symbols = atoms.get_chemical_symbols()

    count = 0
    for atom_gsl_i, nl in indices.items():
        atom_local_i = vap.gsl_to_local_map[atom_gsl_i]
        symboli = symbols[atom_local_i]
        prefix = '{}'.format(symboli)
        for j in range(len(nl)):
            atom_vap_j = nl[j]
            atom_local_j = vap.gsl_to_local_map[atom_vap_j]
            symbolj = symbols[atom_local_j]
            for k in range(j + 1, len(nl)):
                atom_vap_k = nl[k]
                atom_local_k = vap.gsl_to_local_map[atom_vap_k]
                symbolk = symbols[atom_local_k]
                interaction = '{}{}'.format(
                    prefix, ''.join(sorted([symbolj, symbolk])))
                ilist[count] = atom_gsl_i
                jlist[count] = atom_vap_j
                klist[count] = atom_vap_k
                n1[count] = vectors[atom_gsl_i][j]
                n2[count] = vectors[atom_gsl_i][k]
                n3[count] = vectors[atom_gsl_i][k] - vectors[atom_gsl_i][j]
                index = interactions[interaction]
                g4_map[count, iaxis + 0] = atom_gsl_i
                g4_map[count, iaxis + 1] = offsets[index]
                count += 1
    return G4IndexedSlices(g4_map, ilist, jlist, klist, n1, n2, n3)


class SymmetryFunctionTransformer(SymmetryFunction, DescriptorTransformer):
    """
    A tranformer for providing the feed dict to calculate symmetry function
    descriptors of an `Atoms` object.
    """

    def __init__(self, rc, elements, eta=Defaults.eta, omega=Defaults.omega,
                 beta=Defaults.beta, gamma=Defaults.gamma, zeta=Defaults.zeta,
                 angular=False, periodic=True, trainable=False,
                 cutoff_function="cosine"):
        """
        Initialization method.
        """
        SymmetryFunction.__init__(
            self, rc=rc, elements=elements, eta=eta, omega=omega, beta=beta,
            gamma=gamma, zeta=zeta, angular=angular, periodic=periodic,
            trainable=trainable, cutoff_function=cutoff_function)
        DescriptorTransformer.__init__(self)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this transformer.
        """
        d = {'class': self.__class__.__name__, 'rc': self._rc,
             'elements': self._elements, 'angular': self._angular,
             'periodic': self.periodic, 'trainable': self._trainable,
             'eta': self.initial_values["eta"].tolist(),
             'omega': self.initial_values["omega"].tolist(),
             'gamma': self.initial_values["gamma"].tolist(),
             'zeta': self.initial_values["zeta"].tolist(),
             'beta': self.initial_values["beta"].tolist(),
             'cutoff_function': self._cutoff_function}
        return d

    def _initialize_placeholders(self):
        """
        Initialize the placeholders.
        """
        # Make sure the all placeholder ops are placed under the absolute path
        # of 'Placeholders/'. Placeholder ops can be recovered from graph
        # directly.
        with tf.name_scope("Placeholders/"):

            dtype = get_float_dtype()

            self._placeholders["positions"] = self._create_float_2d(
                dtype=dtype, d0=None, d1=3, name='positions')
            self._placeholders["cell"] = self._create_float_2d(
                dtype=dtype, d0=3, d1=3, name='cell')
            self._placeholders["n_atoms_vap"] = self._create_int('n_atoms_vap')
            self._placeholders["volume"] = self._create_float(
                dtype=dtype, name='volume')
            self._placeholders["atom_masks"] = self._create_float_1d(
                dtype=dtype, name='atom_masks')
            self._placeholders["pulay_stress"] = self._create_float(
                dtype=dtype, name='pulay_stress')
            self._placeholders["compositions"] = self._create_float_1d(
                dtype=dtype, name='compositions')
            self._placeholders["row_splits"] = self._create_int_1d(
                name='row_splits', d0=self.n_elements + 1)
            self._placeholders["g2.ilist"] = self._create_int_1d('g2.ilist')
            self._placeholders["g2.jlist"] = self._create_int_1d('g2.jlist')
            self._placeholders["g2.n1"] = self._create_float_2d(
                dtype=dtype, d0=None, d1=3, name='g2.n1')
            self._placeholders["g2.v2g_map"] = self._create_int_2d(
                d0=None, d1=2, name='g2.v2g_map')

            if self._angular:
                self._placeholders["g4.v2g_map"] = self._create_int_2d(
                    d0=None, d1=2, name='g4.v2g_map')
                self._placeholders["g4.ilist"] = self._create_int_1d('g4.ilist')
                self._placeholders["g4.jlist"] = self._create_int_1d('g4.jlist')
                self._placeholders["g4.klist"] = self._create_int_1d('g4.klist')
                self._placeholders["g4.n1"] = self._create_float_2d(
                    dtype=dtype, d0=None, d1=3, name='g4.n1')
                self._placeholders["g4.n2"] = self._create_float_2d(
                    dtype=dtype, d0=None, d1=3, name='g4.n2')
                self._placeholders["g4.n3"] = self._create_float_2d(
                    dtype=dtype, d0=None, d1=3, name='g4.n3')

        return self._placeholders

    def _get_g2_indexed_slices(self,
                               atoms: Atoms,
                               vap: VirtualAtomMap):
        """
        Return the indexed slices for the symmetry function G2.
        """
        return get_g2_map(
            atoms,
            rc=self.rc,
            nij_max=None,
            interactions=self.kbody_index,
            vap=vap,
            offsets=self.offsets,
            mode=tf_estimator.ModeKeys.PREDICT,
            dtype=get_float_dtype().as_numpy_dtype)

    def _get_g4_indexed_slices(self,
                               atoms: Atoms,
                               g2: G2IndexedSlices,
                               vap: VirtualAtomMap):
        """
        Return the indexed slices for the symmetry function G4.
        """
        return get_g4_map(
            atoms=atoms,
            g2=g2,
            interactions=self.kbody_index,
            offsets=self.offsets,
            vap=vap,
            nijk_max=None,
            mode=tf_estimator.ModeKeys.PREDICT)

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
        cell = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        atom_masks = vap.atom_masks
        splits = [1] + [vap.max_occurs[e] for e in self._elements]
        compositions = self._get_compositions(atoms)
        pulay_stress = get_pulay_stress(atoms)

        feed_dict["positions"] = positions.astype(np_dtype)
        feed_dict["n_atoms_vap"] = np.int32(vap_natoms)
        feed_dict["atom_masks"] = atom_masks.astype(np_dtype)
        feed_dict["cell"] = cell.array.astype(np_dtype)
        feed_dict["volume"] = np_dtype(volume)
        feed_dict["compositions"] = compositions
        feed_dict["pulay_stress"] = np_dtype(pulay_stress)
        feed_dict["row_splits"] = np.int32(splits)
        feed_dict.update(g2.as_dict())

        if self._angular:
            g4 = self._get_g4_indexed_slices(atoms, g2, vap)
            feed_dict.update(g4.as_dict())

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
                 angular=False, periodic=True, trainable=False, use_forces=True,
                 use_stress=False, cutoff_function="cosine"):
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
            periodic=periodic, trainable=trainable,
            cutoff_function=cutoff_function)

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
             'trainable': self._trainable,
             'eta': self.initial_values["eta"].tolist(),
             'omega': self.initial_values["omega"].tolist(),
             'gamma': self.initial_values["gamma"].tolist(),
             'zeta': self.initial_values["zeta"].tolist(),
             'beta': self.initial_values["beta"].tolist(),
             'use_forces': self._use_forces, 'use_stress': self._use_stress,
             'cutoff_function': self._cutoff_function}
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
            periodic=self.periodic, trainable=self._trainable)

    def get_g2_indexed_slices(self, atoms: Atoms):
        """
        Return the indexed slices for the radial function G2.
        """
        return get_g2_map(
            atoms=atoms,
            rc=self._rc,
            nij_max=self.nij_max,
            interactions=self.kbody_index,
            vap=self.get_vap_transformer(atoms),
            offsets=self.offsets,
            mode=tf_estimator.ModeKeys.TRAIN,
            dtype=get_float_dtype().as_numpy_dtype)

    def get_g4_indexed_slices(self, atoms: Atoms, g2: G2IndexedSlices):
        """
        Return the indexed slices for the angular function G4.
        """
        return get_g4_map(
            atoms=atoms,
            g2=g2,
            interactions=self.kbody_index,
            offsets=self.offsets,
            vap=self.get_vap_transformer(atoms),
            mode=tf_estimator.ModeKeys.TRAIN,
            nijk_max=self.nijk_max)

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
        g2 = self.get_g2_indexed_slices(atoms)
        feature_list.update(self._encode_g2_indexed_slices(g2))

        if self._angular:
            g4 = self.get_g4_indexed_slices(atoms, g2)
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
            indices = tf.decode_raw(example['g4.indices'], tf.int32)
            indices.set_shape([self._nijk_max * 6])
            indices = tf.reshape(
                indices, [self._nijk_max, 6], name='g4.indices')
            v2g_map, ilist, jlist, klist = \
                tf.split(indices, [3, 1, 1, 1], axis=1, name='splits')
            ilist = tf.squeeze(ilist, axis=1, name='ilist')
            jlist = tf.squeeze(jlist, axis=1, name='jlist')
            klist = tf.squeeze(klist, axis=1, name='klist')

            shifts = tf.decode_raw(example['g4.shifts'], get_float_dtype())
            shifts.set_shape([self._nijk_max * 9])
            shifts = tf.reshape(
                shifts, [self._nijk_max, 9], name='g4.shifts')
            n1, n2, n3 = tf.split(shifts, [3, 3, 3], axis=1, name='splits')

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
        decoded.update(g2.as_dict())

        if self.angular:
            g4 = self._decode_g4_indexed_slices(example)
            decoded.update(g4.as_dict())

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
                'cell': tf.FixedLenFeature([], tf.string),
                'volume': tf.FixedLenFeature([], tf.string),
                'y_true': tf.FixedLenFeature([], tf.string),
                'g2.indices': tf.FixedLenFeature([], tf.string),
                'g2.shifts': tf.FixedLenFeature([], tf.string),
                'atom_masks': tf.FixedLenFeature([], tf.string),
                'compositions': tf.FixedLenFeature([], tf.string),
                'pulay': tf.FixedLenFeature([], tf.string),
            }
            if self._use_forces:
                feature_list['f_true'] = tf.FixedLenFeature([], tf.string)

            if self._use_stress:
                feature_list['stress'] = \
                    tf.FixedLenFeature([], tf.string)
                feature_list['total_pressure'] = \
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
