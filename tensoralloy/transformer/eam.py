# coding=utf-8
"""
This module defines the fixed transformer for EAM.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from collections import Counter
from typing import Dict, Tuple, List
from ase import Atoms
from ase.neighborlist import neighbor_list

from tensoralloy.descriptor.eam import EAM, BatchEAM
from tensoralloy.misc import AttributeDict
from tensoralloy.dtypes import get_float_dtype
from tensoralloy.transformer.indexed_slices import G2IndexedSlices
from tensoralloy.transformer.index_transformer import IndexTransformer
from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.transformer.base import DescriptorTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["BatchEAMTransformer", "EAMTransformer"]


class EAMTransformer(EAM, DescriptorTransformer):
    """
    The feature transformer for the EAM model.
    """

    def __init__(self, rc: float, elements: List[str]):
        """
        Initialization method.
        """
        EAM.__init__(self, rc=rc, elements=elements)
        DescriptorTransformer.__init__(self)

    def get_graph(self):
        """
        Return the graph to compute symmetry function descriptors.
        """
        if not self._placeholders:
            self._initialize_placeholders()
        return self.build_graph(self._placeholders)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this transformer.
        """
        d = {'class': self.__class__.__name__,
             'rc': self._rc,
             'elements': self._elements}
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
            self._placeholders.nnl_max = _int('nnl_max')
            self._placeholders.row_splits = _int_1d(
                'row_splits', d0=self._n_elements + 1)
            self._placeholders.ilist = _int_1d('ilist')
            self._placeholders.jlist = _int_1d('jlist')
            self._placeholders.shift = _float_2d(3, 'shift')
            self._placeholders.v2g_map = _int_2d(4, 'v2g_map')

        return self._placeholders

    def _get_indexed_slices(self, atoms, index_transformer: IndexTransformer):
        """
        Return the corresponding indexed slices.

        Parameters
        ----------
        atoms : Atoms
            The target `ase.Atoms` object.
        index_transformer : IndexTransformer
            The corresponding index transformer.

        Returns
        -------
        g2 : G2IndexedSlices
            The indexed slices for the target `Atoms` object.

        """
        symbols = atoms.get_chemical_symbols()
        ilist, jlist, Slist = neighbor_list('ijS', atoms, self._rc)
        nij = len(ilist)

        v2g_map = np.zeros((nij, 4), dtype=np.int32)

        tlist = np.zeros(nij, dtype=np.int32)
        for i in range(nij):
            symboli = symbols[ilist[i]]
            symbolj = symbols[jlist[i]]
            tlist[i] = self._kbody_index['{}{}'.format(symboli, symbolj)]

        ilist = index_transformer.inplace_map_index(ilist + 1)
        jlist = index_transformer.inplace_map_index(jlist + 1)
        shift = np.asarray(Slist, dtype=get_float_dtype().as_numpy_dtype)

        # The type of the (atomi, atomj) interaction.
        v2g_map[:, 0] = tlist

        # The indices of center atoms
        v2g_map[:, 1] = ilist
        counters = {}
        for index in range(nij):
            atomi = ilist[index]
            if atomi not in counters:
                counters[atomi] = Counter()

            # The indices of the pair atoms
            v2g_map[index, 2] = counters[atomi][tlist[index]]
            counters[atomi][tlist[index]] += 1

        # The mask
        v2g_map[:, 3] = ilist > 0

        return G2IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               shift=shift)

    def get_feed_dict(self, atoms: Atoms):
        """
        Return the feed dict.
        """
        feed_dict = {}

        if not self._placeholders:
            self._initialize_placeholders()
        placeholders = self._placeholders

        index_transformer = self.get_index_transformer(atoms)
        g2 = self._get_indexed_slices(atoms, index_transformer)
        nnl_max = g2.v2g_map[:, 2].max() + 1

        positions = index_transformer.map_positions(atoms.positions)
        n_atoms = index_transformer.n_atoms
        cells = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        mask = index_transformer.mask
        splits = [1] + [index_transformer.max_occurs[e] for e in self._elements]
        composition = self._get_composition(atoms)

        numpy_float_dtype = get_float_dtype().as_numpy_dtype

        feed_dict[placeholders.positions] = positions.astype(numpy_float_dtype)
        feed_dict[placeholders.n_atoms] = n_atoms
        feed_dict[placeholders.nnl_max] = nnl_max
        feed_dict[placeholders.mask] = mask.astype(numpy_float_dtype)
        feed_dict[placeholders.cells] = cells.astype(numpy_float_dtype)
        feed_dict[placeholders.volume] = numpy_float_dtype(volume)
        feed_dict[placeholders.composition] = composition
        feed_dict[placeholders.row_splits] = splits
        feed_dict[placeholders.v2g_map] = g2.v2g_map
        feed_dict[placeholders.ilist] = g2.ilist
        feed_dict[placeholders.jlist] = g2.jlist
        feed_dict[placeholders.shift] = g2.shift

        return feed_dict


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

        BatchEAM.__init__(
            self, rc=rc, max_occurs=max_occurs, elements=elements,
            nij_max=nij_max, nnl_max=nnl_max, batch_size=batch_size)

        BatchDescriptorTransformer.__init__(self, forces=forces, stress=stress)

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
        shift = np.asarray(Slist, dtype=get_float_dtype().as_numpy_dtype)

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
            reduced_stress = tf.decode_raw(
                example['stress'], float_dtype, name='stress')
            reduced_stress.set_shape([6])
            decoded.reduced_stress = reduced_stress

            reduced_total_pressure = tf.decode_raw(
                example['total_pressure'], float_dtype, name='stress')
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

            shift = tf.decode_raw(example['g2.shifts'], get_float_dtype())
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
            if self._use_forces:
                feature_list['f_true'] = tf.FixedLenFeature([], tf.string)

            if self._use_stress:
                feature_list['stress'] = \
                    tf.FixedLenFeature([], tf.string)
                feature_list['total_pressure'] = \
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

            * 'positions': float64 or float32, [batch_size, max_n_atoms + 1, 3]
            * 'cells': float64 or float32, [batch_size, 3, 3]
            * 'volume': float64 or float32, [batch_size, ]
            * 'n_atoms': int64, [batch_size, ]
            * 'y_true': float64, [batch_size, ]
            * 'f_true': float64, [batch_size, max_n_atoms + 1, 3]
            * 'composition': float64, [batch_size, n_elements]
            * 'mask': float64, [batch_size, max_n_atoms + 1]
            * 'ilist': int32, [batch_size, nij_max]
            * 'jlist': int32, [batch_size, nij_max]
            * 'shift': float64, [batch_size, nij_max, 3]
            * 'rv2g': int32, [batch_size, nij_max, 5]

            If `self.stress` is `True`, these following keys will be provided:

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
