# coding=utf-8
"""
This module defines the fixed transformer for EAM.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from collections import Counter
from typing import Dict, Tuple
from ase import Atoms
from ase.neighborlist import neighbor_list

from tensoralloy.descriptor.eam import BatchEAM
from tensoralloy.misc import AttributeDict
from tensoralloy.transformer.indexed_slices import G2IndexedSlices
from tensoralloy.transformer.base import BatchDescriptorTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["BatchEAMTransformer"]


class BatchEAMTransformer(BatchEAM, BatchDescriptorTransformer):
    """
    A batch implementation of feature tranformer for the EAM model.
    """

    def __init__(self, rc: float, max_occurs: Counter, nij_max: int,
                 nnl_max: int, batch_size=None, forces=True, stress=False):
        """
        Initialization method.
        """
        elements = sorted(max_occurs.keys())

        super(BatchEAMTransformer, self).__init__(
            rc=rc, max_occurs=max_occurs, elements=elements, nij_max=nij_max,
            nnl_max=nnl_max, batch_size=batch_size)
        self._index_transformers = {}
        self._forces = forces
        self._stress = stress

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

    @property
    def forces(self):
        """
        Return True if atomic forces can be calculated.
        """
        return self._forces

    @property
    def stress(self):
        """
        Return True if the stress tensor can be calculated.
        """
        return self._stress

    def as_runtime_transformer(self):
        """
        This method is temporarily disabled as `EamTransformer` is neither
        implemented nor needed.
        """
        raise NotImplementedError("`EamTransformer` is not implemented yet")

    def get_indexed_slices(self, atoms: Atoms):
        """
        Return the indexed slices.
        """
        v2g_map = np.zeros((self._nij_max, 5), dtype=np.int32)
        tlist = np.zeros(self._nij_max, dtype=np.int32)

        symbols = atoms.get_chemical_symbols()
        clf = self.get_index_transformer(atoms)

        ilist, jlist, Slist = neighbor_list('ijS', atoms, self._rc)
        nij = len(ilist)

        tlist.fill(0)
        for index in range(nij):
            symboli = symbols[ilist[index]]
            symbolj = symbols[jlist[index]]
            tlist[index] = self._kbody_index['{}{}'.format(symboli, symbolj)]

        ilist = self._resize_to_nij_max(ilist, True)
        jlist = self._resize_to_nij_max(jlist, True)
        Slist = self._resize_to_nij_max(Slist, False)
        ilist = clf.inplace_map_index(ilist)
        jlist = clf.inplace_map_index(jlist)
        shift = np.asarray(Slist, dtype=np.float64)

        v2g_map[:, 1] = tlist
        v2g_map[:, 2] = ilist
        counters = {}
        for index in range(nij):
            atomi = ilist[index]
            if atomi not in counters:
                counters[atomi] = Counter()
            v2g_map[index, 3] = counters[atomi][tlist[index]]
            counters[atomi][tlist[index]] += 1
        v2g_map[:, 4] = ilist > 0

        return G2IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               shift=shift)

    def _get_composition(self, atoms: Atoms) -> np.ndarray:
        """
        Return the composition of the `Atoms`.
        """
        composition = np.zeros(self._n_elements, dtype=np.float64)
        for element, count in Counter(atoms.get_chemical_symbols()).items():
            composition[self._elements.index(element)] = float(count)
        return composition

    def encode(self, atoms: Atoms):
        """
        Encode the `Atoms` object and return a `tf.train.Example`.
        """
        feature_list = self._encode_atoms(atoms)
        g2 = self.get_indexed_slices(atoms)
        feature_list.update(self._encode_g2_indexed_slices(g2))
        return tf.train.Example(
            features=tf.train.Features(feature=feature_list))

    def _decode_atoms(self, example: Dict[str, tf.Tensor]) -> AttributeDict:
        """
        Decode `Atoms` related properties.
        """
        decoded = AttributeDict()

        length = 3 * self._max_n_atoms

        positions = tf.decode_raw(example['positions'], tf.float64)
        positions.set_shape([length])
        decoded.positions = tf.reshape(
            positions, (self._max_n_atoms, 3), name='R')

        n_atoms = tf.identity(example['n_atoms'], name='n_atoms')
        decoded.n_atoms = n_atoms

        y_true = tf.decode_raw(example['y_true'], tf.float64)
        y_true.set_shape([1])
        decoded.y_true = tf.squeeze(y_true, name='y_true')

        cells = tf.decode_raw(example['cells'], tf.float64)
        cells.set_shape([9])
        decoded.cells = tf.reshape(cells, (3, 3), name='cells')

        volume = tf.decode_raw(example['volume'], tf.float64)
        volume.set_shape([1])
        decoded.volume = tf.squeeze(volume, name='volume')

        mask = tf.decode_raw(example['mask'], tf.float64)
        mask.set_shape([self._max_n_atoms, ])
        decoded.mask = mask

        composition = tf.decode_raw(example['composition'], tf.float64)
        composition.set_shape([self._n_elements, ])
        decoded.composition = composition

        if self._forces:
            f_true = tf.decode_raw(example['f_true'], tf.float64)
            # Ignore the forces of the virtual atom
            f_true.set_shape([length - 3])
            decoded.f_true = tf.reshape(
                f_true, (self._max_n_atoms - 1, 3), name='f_true')

        if self._stress:
            reduced_stress = tf.decode_raw(
                example['reduced_stress'], tf.float64, name='stress')
            reduced_stress.set_shape([6])
            decoded.reduced_stress = reduced_stress

            reduced_total_pressure = tf.decode_raw(
                example['reduced_total_pressure'], tf.float64, name='stress')
            reduced_total_pressure.set_shape([1])
            decoded.reduced_total_pressure = reduced_total_pressure

        return decoded

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

            shift = tf.decode_raw(example['g2.shifts'], tf.float64)
            shift.set_shape([self._nij_max * 3])
            shift = tf.reshape(shift, [self._nij_max, 3], name='shift')

            return G2IndexedSlices(v2g_map, ilist, jlist, shift)

    def _decode_example(self, example: Dict[str, tf.Tensor]):
        """
        Decode the parsed single example.
        """
        decoded = self._decode_atoms(example)
        g2 = self._decode_g2_indexed_slices(example)

        decoded.rv2g = g2.v2g_map
        decoded.ilist = g2.ilist
        decoded.jlist = g2.jlist
        decoded.shift = g2.shift

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
            if self._forces:
                feature_list['f_true'] = tf.FixedLenFeature([], tf.string)

            if self._stress:
                feature_list['reduced_stress'] = \
                    tf.FixedLenFeature([], tf.string)
                feature_list['reduced_total_pressure'] = \
                    tf.FixedLenFeature([], tf.string)

            example = tf.parse_single_example(example_proto, feature_list)
            return self._decode_example(example)

    def get_descriptor_ops_from_batch(self,
                                      batch: AttributeDict,
                                      batch_size: int):
        """
        Return the graph for calculating symmetry function descriptors for the
        given batch of examples.

        This function is necessary because nested dicts are not supported by
        `tf.data.Dataset.batch`.

        Parameters
        ----------
        batch : AttributeDict
            A of batch examples produced by `tf.data.Dataset`. Each example is
            produced by the function `decode_protobuf`.

            Here are default keys:

            * 'positions': float64, [batch_size, max_n_atoms, 3]
            * 'cells': float64, [batch_size, 3, 3]
            * 'volume': float64, [batch_size, ]
            * 'n_atoms': int64, [batch_size, ]
            * 'y_true': float64, [batch_size, ]
            * 'f_true': float64, [batch_size, max_n_atoms - 1, 3]
            * 'composition': float64, [batch_size, n_elements]
            * 'mask': float64, [batch_size, max_n_atoms]
            * 'ilist': int32, [batch_size, nij_max]
            * 'jlist': int32, [batch_size, nij_max]
            * 'shift': float64, [batch_size, nij_max, 3]
            * 'rv2g': int32, [batch_size, nij_max, 5]

            If `self.stress` is `True`, the following keys are provided:

            * 'reduced_stress': float64, [batch_size, 6]
            * 'total_pressure': float64, [batch_size, ]

        batch_size : int
            The size of the batch.

        Returns
        -------
        ops : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of (element, (g, mask)) where `element` is the symbol of the
            element, `g` is the Op to compute atomic descriptors and `mask` is
            the Op to compute value masks.

        """
        self._batch_size = batch_size

        inputs = AttributeDict(
            ilist=batch.ilist, jlist=batch.jlist,
            shift=batch.shift, v2g_map=batch.rv2g
        )
        inputs.positions = batch.positions
        inputs.cells = batch.cells
        inputs.volume = batch.volume

        return self.build_graph(inputs)

    def get_descriptor_normalization_weights(self, method):
        """
        Return the initial weights for the column-wise normalising the output
        descriptors.
        """
        raise NotImplementedError(
            "Descriptor normalization is disabled for EAM.")
