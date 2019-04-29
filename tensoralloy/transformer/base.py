# coding=utf-8
"""
This module defines interfaces for feature transformers.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import abc

from collections import Counter
from typing import Dict, List
from ase import Atoms
from ase.units import GPa

from tensoralloy.utils import AttributeDict, get_pulay_stress
from tensoralloy.dtypes import get_float_dtype
from tensoralloy.transformer.index_transformer import IndexTransformer
from tensoralloy.transformer.indexed_slices import G2IndexedSlices


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class BaseTransformer:
    """
    The base class for all transformers.
    """

    def __init__(self):
        """
        Initialization method.
        """
        self._index_transformers = {}

    @property
    @abc.abstractmethod
    def elements(self) -> List[str]:
        """
        A property that all subclasses should implement.

        Return a list of str as the ordered elements.
        """
        pass

    def _get_composition(self, atoms) -> np.ndarray:
        """
        Return a vector as the composition of the `Atoms`.
        """
        n_elements = len(self.elements)
        dtype = get_float_dtype().as_numpy_dtype
        composition = np.zeros(n_elements, dtype=dtype)
        for element, count in Counter(atoms.get_chemical_symbols()).items():
            composition[self.elements.index(element)] = float(count)
        return composition

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def get_descriptors(self, features: AttributeDict):
        """
        An abstract method. Return the Op to compute atomic descriptors from raw
        properties (positions, volume, v2g_map, ...).

        Parameters
        ----------
        features : AttributeDict
            A dict of raw property tensors.

        Returns
        -------
        descriptors : AttributeDict
            A dict of Ops to get atomic descriptors.

        """
        pass


class DescriptorTransformer(BaseTransformer):
    """
    This class represents atomic descriptor transformers for prediction.
    """

    def __init__(self):
        """
        Initialization method.
        """
        super(DescriptorTransformer, self).__init__()
        self._placeholders = AttributeDict()

    @abc.abstractmethod
    def get_feed_dict(self, atoms: Atoms):
        """
        Return a feed dict.
        """
        pass

    @abc.abstractmethod
    def get_constant_features(self, atoms: Atoms):
        """
        Return a dict of constant feature tensors for the given `Atoms`.
        """
        pass

    @abc.abstractmethod
    def as_dict(self) -> Dict:
        """
        Return a JSON serializable dict representation of this transformer.
        """
        pass

    def get_descriptors(self, features):
        """
        An abstract method. Return the Op to compute atomic descriptors from raw
        properties (positions, volume, v2g_map, ...).

        Parameters
        ----------
        features : AttributeDict
            A dict of raw property tensors.

        Returns
        -------
        descriptors : AttributeDict
            A dict of Ops to get atomic descriptors.

        """
        return getattr(self, "build_graph")(features)

    @abc.abstractmethod
    def _initialize_placeholders(self):
        """
        A helper function to initialize placeholders.
        """
        pass

    def get_placeholder_features(self):
        """
        Return the dict of placeholder features.
        """
        if not self._placeholders:
            self._initialize_placeholders()
        return self._placeholders

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
            symbols = atoms.get_chemical_symbols()
            max_occurs = Counter()
            counter = Counter(symbols)
            for element in self.elements:
                max_occurs[element] = max(1, counter[element])
            self._index_transformers[formula] = IndexTransformer(
                max_occurs, symbols
            )
        return self._index_transformers[formula]


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


def _get_confidence_vector(atoms: Atoms) -> np.ndarray:
    """
    Return a vector as the confidences of energy, forces and stress of this
    `Atoms`.
    """
    if 'data' in atoms.info:
        return atoms.info['data'].get("weights", np.ones(3))
    else:
        return atoms.info.get("weights", np.ones(3))


class BatchDescriptorTransformer(BaseTransformer):
    """
    This class represents atomic descriptor transformers for batch training and
    evaluation.
    """

    def __init__(self, use_forces=True, use_stress=False):
        """
        Initialization method.

        Parameters
        ----------
        use_forces : bool
            A boolean flag indicating whether atomic forces should be encoded.
        use_stress : bool
            A boolean flag indicating whether stress tensors should be encoded.

        """
        super(BatchDescriptorTransformer, self).__init__()

        self._use_forces = use_forces
        self._use_stress = use_stress

    @property
    def use_forces(self):
        """
        Return True if atomic forces should be encoded and trained.
        """
        return self._use_forces

    @property
    def use_stress(self):
        """
        Return True if the stress tensor should be encoded and trained.
        """
        return self._use_stress

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
    def as_descriptor_transformer(self) -> DescriptorTransformer:
        """
        Return a corresponding `DescriptorTransformer`.
        """
        pass

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
        np_dtype = get_float_dtype().as_numpy_dtype
        positions = clf.map_positions(atoms.positions).astype(np_dtype)
        cells = atoms.get_cell(complete=True).array.astype(np_dtype)
        volume = np.atleast_1d(atoms.get_volume()).astype(np_dtype)
        y_true = np.atleast_1d(atoms.get_total_energy()).astype(np_dtype)
        composition = self._get_composition(atoms)
        mask = clf.mask.astype(np_dtype)
        confidences = np.reshape(
            _get_confidence_vector(atoms).astype(np_dtype), (3, 1))
        pulay = np.atleast_1d(get_pulay_stress(atoms)).astype(np_dtype)

        feature_list = {
            'positions': bytes_feature(positions.tostring()),
            'cells': bytes_feature(cells.tostring()),
            'n_atoms': int64_feature(len(atoms)),
            'volume': bytes_feature(volume.tostring()),
            'y_true': bytes_feature(y_true.tostring()),
            'y_conf': bytes_feature(confidences[0].tostring()),
            'mask': bytes_feature(mask.tostring()),
            'composition': bytes_feature(composition.tostring()),
            'pulay': bytes_feature(pulay.tostring()),
        }
        if self.use_forces:
            f_true = clf.map_forces(atoms.get_forces()).astype(np_dtype)
            feature_list['f_true'] = bytes_feature(f_true.tostring())
            feature_list['f_conf'] = bytes_feature(confidences[1].tostring())

        if self.use_stress:
            # Convert the unit of the stress tensor to 'eV' for simplification:
            # 1 eV/Angstrom**3 = 160.21766208 GPa
            # 1 GPa = 10 kbar
            # reduced_stress (eV) = stress * volume
            virial = atoms.get_stress(voigt=True).astype(np_dtype)
            internal_pressure = np.atleast_1d(
                -virial[:3].mean() / GPa).astype(np_dtype)
            feature_list['stress'] = bytes_feature(virial.tostring())
            feature_list['total_pressure'] = bytes_feature(
                np.atleast_1d(internal_pressure).tostring())
            feature_list['s_conf'] = bytes_feature(confidences[2].tostring())
        return feature_list

    @abc.abstractmethod
    def encode(self, atoms: Atoms) -> tf.train.Example:
        """
        Encode the `Atoms` object to a tensorflow example.

        Parameters
        ----------
        atoms : Atoms
            The target `Atoms` object to encode.
        """
        pass

    @staticmethod
    def _decode_atoms(example: Dict[str, tf.Tensor],
                      max_n_atoms: int,
                      n_elements: int,
                      use_forces=True,
                      use_stress=False) -> AttributeDict:
        """
        Decode `Atoms` related properties.
        """
        decoded = AttributeDict()

        length = 3 * (max_n_atoms + 1)
        float_dtype = get_float_dtype()

        positions = tf.decode_raw(example['positions'], float_dtype)
        positions.set_shape([length])
        decoded.positions = tf.reshape(
            positions, (max_n_atoms + 1, 3), name='R')

        n_atoms = tf.identity(example['n_atoms'], name='n_atoms')
        decoded.n_atoms = n_atoms

        y_true = tf.decode_raw(example['y_true'], float_dtype)
        y_true.set_shape([1])
        decoded.y_true = tf.squeeze(y_true, name='y_true')

        y_conf = tf.decode_raw(example['y_conf'], float_dtype)
        y_conf.set_shape([1])
        decoded.y_conf = tf.squeeze(y_conf, name='y_conf')

        cells = tf.decode_raw(example['cells'], float_dtype)
        cells.set_shape([9])
        decoded.cells = tf.reshape(cells, (3, 3), name='cells')

        volume = tf.decode_raw(example['volume'], float_dtype)
        volume.set_shape([1])
        decoded.volume = tf.squeeze(volume, name='volume')

        mask = tf.decode_raw(example['mask'], float_dtype)
        mask.set_shape([max_n_atoms + 1, ])
        decoded.mask = mask

        composition = tf.decode_raw(example['composition'], float_dtype)
        composition.set_shape([n_elements, ])
        decoded.composition = composition

        pulay = tf.decode_raw(example['pulay'], float_dtype)
        pulay.set_shape([1])
        decoded.pulay_stress = tf.squeeze(pulay, name='pulay_stress')

        if use_forces:
            f_true = tf.decode_raw(example['f_true'], float_dtype)
            # Ignore the forces of the virtual atom
            f_true.set_shape([length, ])
            decoded.f_true = tf.reshape(
                f_true, (max_n_atoms + 1, 3), name='f_true')

            f_conf = tf.decode_raw(example['f_conf'], float_dtype)
            f_conf.set_shape([1])
            decoded.f_conf = tf.squeeze(f_conf, name='f_conf')

        if use_stress:
            stress = tf.decode_raw(
                example['stress'], float_dtype, name='stress')
            stress.set_shape([6])
            decoded.stress = stress

            total_pressure = tf.decode_raw(
                example['total_pressure'], float_dtype)
            total_pressure.set_shape([1])
            decoded.total_pressure = tf.squeeze(
                total_pressure, name='total_pressure')

            s_conf = tf.decode_raw(example['s_conf'], float_dtype)
            s_conf.set_shape([1])
            decoded.s_conf = tf.squeeze(s_conf, name='s_conf')

        return decoded

    @abc.abstractmethod
    def decode_protobuf(self, example_proto: tf.Tensor) -> AttributeDict:
        """
        Decode the scalar string Tensor, which is a single serialized Example.
        See `_parse_single_example_raw` documentation for more details.
        """
        pass

    def _infer_batch_size(self, batch_raw_properties: AttributeDict):
        """
        Infer `batch_size` from `batch_raw_properties`.
        """

        for name, tensor in batch_raw_properties.items():
            if isinstance(tensor, tf.Tensor):
                batch_size = tensor.shape[0].value
            elif isinstance(tensor, np.ndarray):
                batch_size = tensor.shape[0]
            else:
                raise ValueError(
                    f"batch_raw_properties.{name} is neither a `tf.Tensor` "
                    f"nor an `np.ndarray`")
            if batch_size is not None:
                break
        else:
            raise ValueError("`batch_size` cannot be inferred!")

        self._batch_size = batch_size

    @abc.abstractmethod
    def get_descriptors(self, batch_features: AttributeDict):
        """
        An abstract method. Return the Op to compute atomic descriptors from raw
        properties (positions, volume, v2g_map, ...).

        Parameters
        ----------
        batch_features : AttributeDict
            A dict of batched raw property tensors.

        Returns
        -------
        descriptors : AttributeDict
            A dict of Ops to get atomic descriptors.

        """
        pass
