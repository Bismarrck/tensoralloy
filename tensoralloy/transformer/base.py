# coding=utf-8
"""
This module defines interfaces for feature transformers.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import abc

from collections import Counter
from typing import Dict, Tuple, List
from ase import Atoms

from tensoralloy.descriptor.base import AtomicDescriptorInterface
from tensoralloy.misc import AttributeDict
from tensoralloy.transformer.indexed_slices import G2IndexedSlices


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class IndexTransformer:
    """
    If a dataset has different stoichiometries, a global ordered symbols list
    should be kept. This class is used to transform the local indices of the
    symbols of arbitrary `Atoms` to the global indexing system.
    """
    _ISTART = 1

    def __init__(self, max_occurs: Counter, symbols: List[str]):
        """
        Initialization method.
        """
        self._max_occurs = max_occurs
        self._symbols = symbols
        self._n_atoms = sum(max_occurs.values()) + IndexTransformer._ISTART

        istart = IndexTransformer._ISTART
        elements = sorted(max_occurs.keys())
        offsets = np.cumsum([max_occurs[e] for e in elements])[:-1]
        offsets = np.insert(offsets, 0, 0)
        delta = Counter()
        index_map = {}
        mask = np.zeros(self._n_atoms, dtype=bool)
        for i, symbol in enumerate(symbols):
            idx_old = i + istart
            idx_new = offsets[elements.index(symbol)] + delta[symbol] + istart
            index_map[idx_old] = idx_new
            delta[symbol] += 1
            mask[idx_new] = True
        reverse_map = {v: k for k, v in index_map.items()}
        index_map[0] = 0
        reverse_map[0] = 0
        self._mask = mask
        self._index_map = index_map
        self._reverse_map = reverse_map

    @property
    def n_atoms(self):
        """
        Return the number of atoms (including the virtual atom at index 0).
        """
        return self._n_atoms

    @property
    def max_occurs(self) -> Counter:
        """
        Return the maximum occurance for each type of element.
        """
        return self._max_occurs

    @property
    def symbols(self) -> List[str]:
        """
        Return a list of str as the ordered chemical symbols of the target
        stoichiometry.
        """
        return self._symbols

    @property
    def reference_symbols(self) -> List[str]:
        """
        Return a list of str as the ordered chemical symbols of the reference
        (global) stoichiometry.
        """
        return sorted(self._max_occurs.elements())

    @property
    def mask(self) -> np.ndarray:
        """
        Return a `bool` array.
        """
        return self._mask

    # FIXME: the index here should start from one. This may be confusing.
    def inplace_map_index(self, index_or_indices, reverse=False,
                          exclude_extra=False):
        """
        Do the in-place index transformation.

        Parameters
        ----------
        index_or_indices : int or List[int] or array_like
            An atom index or a list of indices. One must be aware that indices
            here start from one!
        reverse : bool, optional
            If True, the indices will be mapped to the local reference from the
            global reference.
        exclude_extra : bool
            Exclude the virtual atom when calculating the index.

        """
        if reverse:
            index_map = self._reverse_map
        else:
            index_map = self._index_map
        delta = int(exclude_extra)
        if not hasattr(index_or_indices, "__len__"):
            return index_map[index_or_indices] - delta
        else:
            for i in range(len(index_or_indices)):
                index_or_indices[i] = index_map[index_or_indices[i]] - delta
            return index_or_indices

    def map_array(self, params, reverse=False, fill=0):
        """
        Do the array transformation.

        Parameters
        ----------
        params : array_like
            A 2D or 3D array.
        reverse : bool, optional
            If True, the array will be mapped to the local reference from the
            global reference.
        fill : int or float
            The value to fill for the virtual atom.

        """
        params = np.asarray(params)
        rank = np.ndim(params)
        if rank == 2:
            params = params[np.newaxis, ...]
        if not reverse and params.shape[1] == len(self._symbols):
            params = np.insert(
                params, 0, np.asarray(fill, dtype=params.dtype), axis=1)

        indices = []
        istart = IndexTransformer._ISTART
        if reverse:
            for i in range(istart, istart + len(self._symbols)):
                indices.append(self._index_map[i])
        else:
            for i in range(self._n_atoms):
                indices.append(self._reverse_map.get(i, 0))
        output = params[:, indices]
        if rank == 2:
            output = np.squeeze(output, axis=0)
        return output

    def reverse_map_hessian(self, hessian: np.ndarray):
        """
        Transform the [Np, 3, Np, 3] Hessian matrix to [3N, 3N] where N is the
        number of atoms (excluding the virtual atom) and Np is number of atoms
        in the projected frame.
        """
        rank = np.ndim(hessian)
        if rank != 4 or hessian.shape[1] != 3 or hessian.shape[3] != 3:
            raise ValueError(
                "The input array should be a 4D matrix of shape [Np, 3, Np, 3]")

        indices = []
        istart = IndexTransformer._ISTART
        for i in range(istart, istart + len(self._symbols)):
            indices.append(self._index_map[i])

        n = len(self._symbols)
        h = np.zeros((n * 3, n * 3))

        for i in range(n):
            for alpha in range(3):
                for j in range(n):
                    for beta in range(3):
                        row = i * 3 + alpha
                        col = j * 3 + beta
                        h[row, col] = h[indices[i], alpha, indices[j], beta]

        return h


class DescriptorTransformer(AtomicDescriptorInterface):
    """
    This class represents atomic descriptor transformers for runtime prediction.
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

    @abc.abstractmethod
    def as_dict(self) -> Dict:
        """
        Return a JSON serializable dict representation of this transformer.
        """
        pass

    @abc.abstractmethod
    def get_index_transformer(self, atoms: Atoms) -> IndexTransformer:
        """
        Return the corresponding `IndexTransformer`.
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
    This class represents atomic descriptor transformers for batch training and
    evaluation.
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

    @abc.abstractmethod
    def as_runtime_transformer(self) -> DescriptorTransformer:
        """
        Return a corresponding `DescriptorTransformer`.
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
    def get_descriptor_ops_from_batch(self, batch: AttributeDict,
                                      batch_size: int):
        """
        Return the tensorflow graph for computing atomic descriptors from an
        input batch.

        Returns
        -------
        ops : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of (element, (value, mask)) where `element` is a symbol and
            `value` is the Op to compute atomic descriptors and `mask` is the Op
            to compute mask of `value`. `mask` may be a `tf.no_op`.

        """
        pass

    @abc.abstractmethod
    def get_descriptor_normalization_weights(self, method):
        """
        Return the initial weights for column-wise normalising the output atomic
        descriptors.
        """
        pass
