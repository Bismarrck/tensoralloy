# coding=utf-8
"""
This module defines interfaces for feature transformers.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import abc
from collections import Counter
from typing import Dict
from ase import Atoms

from tensoralloy.descriptor import IndexTransformer, G2IndexedSlices
from tensoralloy.descriptor.base import AtomicDescriptorInterface
from tensoralloy.misc import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class DescriptorTransformer(AtomicDescriptorInterface):
    """
    This interface class defines the required methods for atomic descriptor
    transformers.
    """

    @property
    @abc.abstractmethod
    def placeholders(self) -> Dict[str, tf.Tensor]:
        """
        Return a dict of names and placeholders.
        """
        pass

    @abc.abstractmethod
    def get_feed_dict(self, atoms: Atoms):
        """
        Return a feed dict.
        """
        pass

    @abc.abstractmethod
    def get_graph(self):
        """
        Return the graph for computing atomic descriptors.
        """
        pass


def bytes_feature(value):
    """
    Convert the `value` to Protobuf bytes.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """
    Convert the `value` to Protobuf float32.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """
    Convert the `value` to Protobuf int64.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class BatchDescriptorTransformer(AtomicDescriptorInterface):
    """
    This interface class defines the required methods for atomic descriptor
    transformers.
    """

    def __init__(self):
        """
        Initialization method.
        """
        self._index_transformers = {}

    @property
    @abc.abstractmethod
    def forces(self):
        """
        Return True if atomic forces should be encoded and trained.
        """
        pass

    @property
    @abc.abstractmethod
    def stress(self):
        """
        Return True if the stress tensor should be encoded and trained.
        """
        pass

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        """
        Return the batch size.
        """
        pass

    @property
    @abc.abstractmethod
    def nij_max(self) -> int:
        """
        Return the maximum length of the neighbor list of any `Atoms` object.
        """
        pass

    @property
    @abc.abstractmethod
    def max_occurs(self) -> Counter:
        """
        Return the maximum occurance of each type of element.
        """
        pass

    def _get_composition(self, atoms: Atoms) -> np.ndarray:
        """
        Return the composition of the `Atoms`.
        """
        n_el = len(self.elements)
        composition = np.zeros(n_el, dtype=np.float64)
        for element, count in Counter(atoms.get_chemical_symbols()).items():
            composition[self.elements.index(element)] = float(count)
        return composition

    def get_index_transformer(self, atoms: Atoms):
        """
        Return the corresponding `IndexTransformer`.

        Parameters
        ----------
        atoms : Atoms
            An `Atoms` object.

        Returns
        -------
        clf : IndexTransformer
            The `IndexTransformer` for the given `Atoms` object.

        """
        # The mode 'reduce' is important here because chemical symbol lists of
        # ['C', 'H', 'O'] and ['C', 'O', 'H'] should be treated differently!
        formula = atoms.get_chemical_formula(mode='reduce')
        if formula not in self._index_transformers:
            self._index_transformers[formula] = IndexTransformer(
                self.max_occurs, atoms.get_chemical_symbols()
            )
        return self._index_transformers[formula]

    def _resize_to_nij_max(self, alist: np.ndarray, is_indices=True):
        """
        A helper function to resize the given array.
        """
        if np.ndim(alist) == 1:
            shape = [self.nij_max, ]
        else:
            shape = [self.nij_max, ] + list(alist.shape[1:])
        nlist = np.zeros(shape, dtype=np.int32)
        length = len(alist)
        nlist[:length] = alist
        if is_indices:
            nlist[:length] += 1
        return nlist

    @staticmethod
    def _encode_g2_indexed_slices(g2: G2IndexedSlices):
        """
        Encode the indexed slices of G2:
            * `v2g_map`, `ilist` and `jlist` are merged into a single array
              with key 'r_indices'.
            * `shift` will be encoded separately with key 'r_shifts'.

        """
        indices = np.concatenate((
            g2.v2g_map, g2.ilist[..., np.newaxis], g2.jlist[..., np.newaxis],
        ), axis=1).tostring()
        return {'g2.indices': bytes_feature(indices),
                'g2.shifts': bytes_feature(g2.shift.tostring())}

    def _encode_atoms(self, atoms: Atoms) -> dict:
        """
        Encode the basic properties of an `Atoms` object.
        """
        clf = self.get_index_transformer(atoms)
        positions = clf.map_array(atoms.positions)
        cells = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        y_true = atoms.get_total_energy()
        composition = self._get_composition(atoms)
        mask = clf.mask.astype(np.float64)
        feature_list = {
            'positions': bytes_feature(positions.tostring()),
            'cells': bytes_feature(cells.tostring()),
            'n_atoms': int64_feature(len(atoms)),
            'volume': bytes_feature(np.atleast_1d(volume).tostring()),
            'y_true': bytes_feature(np.atleast_1d(y_true).tostring()),
            'mask': bytes_feature(mask.tostring()),
            'composition': bytes_feature(composition.tostring()),
        }
        if self.forces:
            f_true = clf.map_array(atoms.get_forces())[1:]
            feature_list['f_true'] = bytes_feature(f_true.tostring())

        if self.stress:
            # Convert the unit of the stress tensor to 'eV' for simplification:
            # 1 eV/Angstrom**3 = 160.21766208 GPa
            # 1 GPa = 10 kbar
            # reduced_stress (eV) = stress * volume
            virial = atoms.get_stress(voigt=True) * volume
            total_pressure = virial[:3].mean()
            feature_list['reduced_stress'] = bytes_feature(virial.tostring())
            feature_list['reduced_total_pressure'] = bytes_feature(
                np.atleast_1d(total_pressure).tostring())
        return feature_list

    @abc.abstractmethod
    def encode(self, atoms: Atoms) -> tf.train.Example:
        """
        Encode the `Atoms` object to a tensorflow example.
        """
        pass

    @abc.abstractmethod
    def decode_protobuf(self, example_proto: tf.Tensor) -> AttributeDict:
        """
        Decode the scalar string Tensor, which is a single serialized Example.
        See `_parse_single_example_raw` documentation for more details.
        """
        pass

    @abc.abstractmethod
    def get_graph_from_batch(self, batch: AttributeDict, batch_size: int):
        """
        Return the tensorflow graph for computing atomic descriptors from an
        input batch.
        """
        pass

    @abc.abstractmethod
    def get_descriptor_normalization_weights(self, method):
        """
        Return the initial weights for column-wise normalising the output atomic
        descriptors.
        """
        pass
