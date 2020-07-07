# coding=utf-8
"""
This module defines the fixed transformer for EAM.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from tensorflow_estimator import estimator as tf_estimator
from collections import Counter
from typing import Dict, Tuple, List
from ase import Atoms
from ase.neighborlist import neighbor_list

from tensoralloy.descriptor.eam import EAM, BatchEAM
from tensoralloy.atoms_utils import get_pulay_stress
from tensoralloy.precision import get_float_dtype
from tensoralloy.transformer.indexed_slices import G2IndexedSlices
from tensoralloy.transformer.vap import VirtualAtomMap
from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.transformer.base import DescriptorTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["BatchEAMTransformer", "EAMTransformer"]


def get_g2_map(atoms: Atoms,
               rc: float,
               interactions: dict,
               vap: VirtualAtomMap,
               mode: tf_estimator.ModeKeys,
               nij_max: int = None,
               dtype=np.float32):
    """
    Build the base `v2g_map`.
    """
    if mode == tf_estimator.ModeKeys.PREDICT:
        iaxis = 0
    else:
        iaxis = 1

    ilist, jlist, n1, rij, dij = neighbor_list('ijSdD', atoms, rc)
    nij = len(ilist)

    if nij_max is None:
        nij_max = nij

    g2_map = np.zeros((nij_max, iaxis + 4), dtype=np.int32)
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
    ilist = ilist.astype(np.int32)
    jlist = jlist.astype(np.int32)
    g2_map[:, iaxis + 0] = tlist
    g2_map[:, iaxis + 1] = ilist

    # The indices of center atoms
    counters = {}
    for index in range(nij):
        atomi = ilist[index]
        if atomi not in counters:
            counters[atomi] = Counter()

        # The indices of the pair atoms
        g2_map[index, iaxis + 2] = counters[atomi][tlist[index]]
        counters[atomi][tlist[index]] += 1

    # The mask
    g2_map[:, iaxis + 3] = ilist > 0

    g2 = G2IndexedSlices(v2g_map=g2_map, ilist=ilist, jlist=jlist, n1=n1,
                         rij=None)
    return {"g2": g2, "rij": rij, "dij": dij}


class EAMTransformer(EAM, DescriptorTransformer):
    """
    The feature transformer for the EAM potential.
    """

    def __init__(self, rc: float, elements: List[str], use_direct_rij=False):
        """
        Initialization method.
        """
        EAM.__init__(self, rc=rc, elements=elements,
                     use_direct_rij=use_direct_rij)
        DescriptorTransformer.__init__(self)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this transformer.
        """
        d = {'class': self.__class__.__name__,
             'rc': self._rc,
             'elements': self._elements,
             'use_direct_rij': self._use_direct_rij}
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
            self._placeholders["n_atoms_vap"] = self._create_int(
                name='n_atoms_plus_virt')
            self._placeholders["volume"] = self._create_float(
                dtype=dtype, name='volume')
            self._placeholders["atom_masks"] = self._create_float_1d(
                dtype=dtype, name='atom_masks')
            self._placeholders["nnl_max"] = self._create_int('nnl_max')
            self._placeholders["pulay_stress"] = self._create_float(
                dtype=dtype, name='pulay_stress')
            self._placeholders["row_splits"] = self._create_int_1d(
                'row_splits', d0=self._n_elements + 1)
            if not self._use_direct_rij:
                self._placeholders["g2.ilist"] = self._create_int_1d('g2.ilist')
                self._placeholders["g2.jlist"] = self._create_int_1d('g2.jlist')
                self._placeholders["g2.n1"] = self._create_float_2d(
                    dtype=dtype, d0=None, d1=3, name='g2.n1')
            else:
                self._placeholders["rij"] = self._create_float_1d(
                    dtype=dtype, name='rij')
                self._placeholders["dij"] = self._create_float_2d(
                    dtype=dtype, d1=3, name='dij')
            self._placeholders["g2.v2g_map"] = self._create_int_2d(
                d0=None, d1=4, name='g2.v2g_map')
            self._placeholders["is_constant"] = False

        return self._placeholders

    def _get_indexed_slices(self, atoms, vap: VirtualAtomMap):
        """
        Return the corresponding indexed slices.

        Parameters
        ----------
        atoms : Atoms
            The target `ase.Atoms` object.
        vap : VirtualAtomMap
            The corresponding virtual-atom map.

        Returns
        -------
        g2_dict : dict
            The indexed slices for the target `Atoms` object.

        """
        return get_g2_map(atoms=atoms,
                          rc=self._rc,
                          interactions=self._kbody_index,
                          vap=vap,
                          mode=tf_estimator.ModeKeys.PREDICT,
                          nij_max=None,
                          dtype=get_float_dtype().as_numpy_dtype)

    def get_np_feed_dict(self, atoms: Atoms):
        """
        Return a dict of features (Numpy or Python objects).
        """
        np_dtype = get_float_dtype().as_numpy_dtype

        vap = self.get_vap_transformer(atoms)
        g2_dict = self._get_indexed_slices(atoms, vap)
        g2 = g2_dict["g2"]
        nnl_max = g2.v2g_map[:, 2].max() + 1

        positions = vap.map_positions(atoms.positions)

        # `max_n_atoms` must be used because every element shall have at least
        # one feature row (though it could be all zeros, a dummy or virtual row)
        cell = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        atom_masks = vap.atom_masks.astype(np_dtype)
        pulay_stress = get_pulay_stress(atoms)
        splits = [1] + [vap.max_occurs[e] for e in self._elements]

        feed_dict = dict()

        feed_dict["positions"] = positions.astype(np_dtype)
        feed_dict["n_atoms_vap"] = np.int32(vap.max_vap_natoms)
        feed_dict["nnl_max"] = np.int32(nnl_max)
        feed_dict["atom_masks"] = atom_masks.astype(np_dtype)
        feed_dict["cell"] = cell.array.astype(np_dtype)
        feed_dict["volume"] = np_dtype(volume)
        feed_dict["pulay_stress"] = np_dtype(pulay_stress)
        feed_dict["row_splits"] = np.int32(splits)
        feed_dict.update(g2.as_dict())

        if self._use_direct_rij:
            feed_dict["rij"] = g2_dict["rij"].astype(np_dtype)
            feed_dict["dij"] = g2_dict["dij"].astype(np_dtype)

        return feed_dict

    def get_feed_dict(self, atoms: Atoms):
        """
        Return the feed dict.
        """
        feed_dict = {}

        if not self._placeholders:
            self._initialize_placeholders()
        placeholders = self._placeholders

        for key, value in self.get_np_feed_dict(atoms).items():
            if self._use_direct_rij \
                    and key in ("g2.ilist", "g2.jlist", "g2.n1"):
                continue
            feed_dict[placeholders[key]] = value

        return feed_dict

    def get_constant_features(self, atoms: Atoms):
        """
        Return a dict of constant feature tensors for the given `Atoms`.
        """
        feed_dict = dict()
        with tf.name_scope("Constants"):
            for key, val in self.get_np_feed_dict(atoms).items():
                feed_dict[key] = tf.convert_to_tensor(val, name=key)
        return feed_dict


class BatchEAMTransformer(BatchEAM, BatchDescriptorTransformer):
    """
    A batch implementation of feature transformer for the EAM potential.
    """

    def __init__(self, rc: float, max_occurs: Counter, nij_max: int,
                 nnl_max: int, batch_size=None, use_forces=True,
                 use_stress=False):
        """
        Initialization method.
        """
        elements = sorted(max_occurs.keys())

        BatchEAM.__init__(
            self, rc=rc, max_occurs=max_occurs, elements=elements,
            nij_max=nij_max, nnl_max=nnl_max, batch_size=batch_size)

        BatchDescriptorTransformer.__init__(self, use_forces=use_forces,
                                            use_stress=use_stress)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this transformer.
        """
        d = {'class': self.__class__.__name__,
             'rc': self._rc,
             'max_occurs': self._max_occurs,
             'nij_max': self._nij_max,
             'nnl_max': self._nnl_max,
             'batch_size': self._batch_size,
             'use_forces': self._use_forces,
             'use_stress': self._use_stress}
        return d

    @property
    def descriptor(self):
        """
        Return the name of the descriptor.
        """
        return "eam"

    @property
    def batch_size(self):
        """
        Return the batch size.
        """
        return self._batch_size

    @property
    def max_occurs(self):
        """
        Return the maximum occurances of the elements.
        """
        return self._max_occurs

    def as_descriptor_transformer(self):
        """
        This method is temporarily disabled as `EamTransformer` is neither
        implemented nor needed.
        """
        return EAMTransformer(self._rc, self._elements)

    def get_indexed_slices(self, atoms: Atoms):
        """
        Return the indexed slices.
        """
        return get_g2_map(atoms=atoms,
                          rc=self._rc,
                          interactions=self._kbody_index,
                          vap=self.get_vap_transformer(atoms),
                          mode=tf_estimator.ModeKeys.TRAIN,
                          nij_max=self._nij_max,
                          dtype=get_float_dtype().as_numpy_dtype)

    def encode(self, atoms: Atoms):
        """
        Encode the `Atoms` object and return a `tf.train.Example`.
        """
        feature_list = self._encode_atoms(atoms)
        g2 = self.get_indexed_slices(atoms)["g2"]
        feature_list.update(self._encode_g2_indexed_slices(g2))
        return tf.train.Example(
            features=tf.train.Features(feature=feature_list))

    def _decode_g2_indexed_slices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ilist, jlist and Slist for radial functions.
        """
        with tf.name_scope("G2"):
            indices = tf.decode_raw(example['g2.indices'], tf.int32)
            indices.set_shape([self._nij_max * 7])
            indices = tf.reshape(
                indices, [self._nij_max, 7], name='g2.indices')
            v2g_map, ilist, jlist = tf.split(
                indices, [5, 1, 1], axis=1, name='splits')
            ilist = tf.squeeze(ilist, axis=1, name='ilist')
            jlist = tf.squeeze(jlist, axis=1, name='jlist')

            shift = tf.decode_raw(example['g2.shifts'], get_float_dtype())
            shift.set_shape([self._nij_max * 3])
            shift = tf.reshape(shift, [self._nij_max, 3], name='shift')

            return G2IndexedSlices(v2g_map, ilist, jlist, shift, None)

    def _decode_example(self, example: Dict[str, tf.Tensor]):
        """
        Decode the parsed single example.
        """
        decoded = self._decode_atoms(
            example,
            max_n_atoms=self._max_n_atoms,
            use_forces=self._use_forces,
            use_stress=self._use_stress
        )
        g2 = self._decode_g2_indexed_slices(example)
        decoded.update(g2.as_dict())
        return decoded

    def decode_protobuf(self, example_proto: tf.Tensor):
        """
        Decode the scalar string Tensor, which is a single serialized Example.
        See `_parse_single_example_raw` documentation for more details.
        """
        with tf.name_scope("decoding"):

            feature_list = {
                'positions': tf.FixedLenFeature([], tf.string),
                'n_atoms_vap': tf.FixedLenFeature([], tf.int64),
                'cell': tf.FixedLenFeature([], tf.string),
                'volume': tf.FixedLenFeature([], tf.string),
                'energy': tf.FixedLenFeature([], tf.string),
                'free_energy': tf.FixedLenFeature([], tf.string),
                'g2.indices': tf.FixedLenFeature([], tf.string),
                'g2.shifts': tf.FixedLenFeature([], tf.string),
                'atom_masks': tf.FixedLenFeature([], tf.string),
                'pulay_stress': tf.FixedLenFeature([], tf.string),
                'etemperature': tf.FixedLenFeature([], tf.string),
                'eentropy': tf.FixedLenFeature([], tf.string)
            }
            if self._use_forces:
                feature_list['forces'] = tf.FixedLenFeature([], tf.string)

            if self._use_stress:
                feature_list['stress'] = \
                    tf.FixedLenFeature([], tf.string)
                feature_list['total_pressure'] = \
                    tf.FixedLenFeature([], tf.string)

            example = tf.io.parse_single_example(example_proto, feature_list)
            return self._decode_example(example)

    def get_descriptors(self, batch_features: dict):
        """
        Return the graph for calculating symmetry function descriptors for the
        given batch of examples.

        This function is necessary because nested dicts are not supported by
        `tf.data.Dataset.batch`.

        Parameters
        ----------
        batch_features : dict
            A batch of raw properties provided by `tf.data.Dataset`. Each batch
            is produced by the function `decode_protobuf`.

            Here are default keys:

            * 'positions': float64 or float32, [batch_size, max_n_atoms + 1, 3]
            * 'cell': float64 or float32, [batch_size, 3, 3]
            * 'volume': float64 or float32, [batch_size, ]
            * 'n_atoms_vap': int64, [batch_size, ]
            * 'atom_masks': float64, [batch_size, max_n_atoms + 1]
            * 'energy': float64, [batch_size, ]
            * 'free_energy': float64, [batch_size, ]
            * 'eentropy': float64, [batch_size, ]
            * 'etemperature': float64, [batch_size, ]
            * 'forces': float64, [batch_size, max_n_atoms + 1, 3]
            * 'g2.ilist': int32, [batch_size, nij_max]
            * 'g2.jlist': int32, [batch_size, nij_max]
            * 'g2.n1': float64, [batch_size, nij_max, 3]
            * 'g2.v2g_map': int32, [batch_size, nij_max, 5]

            If `self.stress` is `True`, these following keys will be provided:

            * 'stress': float64, [batch_size, 6]
            * 'total_pressure': float64, [batch_size, ]

        Returns
        -------
        descriptors : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of (element, (g, mask)) where `element` is the symbol of the
            element, `g` is the Op to compute atomic descriptors and `mask` is
            the Op to compute value masks.

        """
        self._infer_batch_size(batch_features)
        return self.build_graph(batch_features)
