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
from ase.calculators.singlepoint import SinglePointCalculator

from tensoralloy import atoms_utils
from tensoralloy.precision import get_float_dtype
from tensoralloy.transformer.vap import VirtualAtomMap
from tensoralloy.transformer.metadata import RadialMetadata


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
        self._vap_transformers = {}

    @property
    @abc.abstractmethod
    def rc(self):
        """
        Return the cutoff radius.
        """
        pass

    @property
    @abc.abstractmethod
    def angular(self):
        """
        Return if angular symmetry functions should be used or not.
        """
        pass

    @property
    @abc.abstractmethod
    def elements(self) -> List[str]:
        """
        A property that all subclasses should implement.

        Return a list of str as the ordered elements.
        """
        pass

    @abc.abstractmethod
    def as_dict(self) -> Dict:
        """
        Return a JSON serializable dict representation of this transformer.
        """
        pass

    @abc.abstractmethod
    def get_vap_transformer(self, atoms: Atoms):
        """
        Return the corresponding `VirtualAtomMap`.

        Parameters
        ----------
        atoms : Atoms
            An `Atoms` object.

        Returns
        -------
        vap : VirtualAtomMap
            The `VirtualAtomMap` for the given `Atoms` object.

        """
        pass

    @abc.abstractmethod
    def get_descriptors(self, features: dict):
        """
        An abstract method. Return the Op to compute atomic descriptors from raw
        properties (positions, volume, v2g_map, ...).

        Parameters
        ----------
        features : dict
            A dict of features.

        Returns
        -------
        descriptors : dict
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
        self._placeholders = dict()

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

    def get_descriptors(self, features):
        """
        An abstract method. Return the Op to compute atomic descriptors from raw
        properties (positions, volume, v2g_map, ...).

        Parameters
        ----------
        features : Dict
            A dict of tensors.

        Returns
        -------
        descriptors : Dict
            A dict of Ops to get atomic descriptors.

        """
        return getattr(self, "build_graph")(features)

    @staticmethod
    def _get_or_create_placeholder(dtype, shape, name):
        """
        Get a placeholder tensor from the graph or just create a new one.
        """
        try:
            return tf.get_default_graph().get_tensor_by_name(
                f'Placeholders/{name}:0')
        except KeyError:
            return tf.placeholder(dtype, shape, name)
        except Exception as excp:
            raise excp

    def _create_float(self, name, dtype):
        return self._get_or_create_placeholder(dtype, (), name)

    def _create_float_1d(self, name, dtype):
        return self._get_or_create_placeholder(dtype, (None,), name)

    def _create_float_2d(self, dtype, d1, name, d0=None):
        return self._get_or_create_placeholder(dtype, (d0, d1), name)

    def _create_int(self, name):
        return self._get_or_create_placeholder(tf.int32, (), name)

    def _create_int_1d(self, name, d0=None):
        return self._get_or_create_placeholder(tf.int32, (d0,), name)

    def _create_int_2d(self, d1, name, d0=None):
        return self._get_or_create_placeholder(tf.int32, (d0, d1), name)

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

    def get_vap_transformer(self, atoms: Atoms):
        """
        Return the corresponding `VirtualAtomMap`.

        Parameters
        ----------
        atoms : Atoms
            An `Atoms` object.

        Returns
        -------
        vap : VirtualAtomMap
            The `VirtualAtomMap` for the given `Atoms` object.

        """
        # The mode 'reduce' is important here because chemical symbol lists of
        # ['C', 'H', 'O'] and ['C', 'O', 'H'] should be treated differently!
        formula = atoms.get_chemical_formula(mode='reduce')
        if formula not in self._vap_transformers:
            symbols = atoms.get_chemical_symbols()
            max_occurs = Counter()
            counter = Counter(symbols)
            for element in self.elements:
                max_occurs[element] = max(1, counter[element])
            self._vap_transformers[formula] = VirtualAtomMap(
                max_occurs, symbols
            )
        return self._vap_transformers[formula]


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
    @abc.abstractmethod
    def descriptor(self):
        """
        Return the name of the descriptor.
        """
        pass

    @property
    def use_forces(self):
        """
        Return True if atomic forces should be encoded and trained.
        """
        return self._use_forces

    @use_forces.setter
    def use_forces(self, flag: bool):
        """
        The setter function of the property `use_stress`.
        """
        self._use_forces = flag

    @property
    def use_stress(self):
        """
        Return True if the stress tensor should be encoded and trained.
        """
        return self._use_stress

    @use_stress.setter
    def use_stress(self, flag: bool):
        """
        The setter function of the property `use_stress`.
        """
        self._use_stress = flag

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

    def get_vap_transformer(self, atoms: Atoms):
        """
        Return the corresponding `VirtualAtomMap`.

        Parameters
        ----------
        atoms : Atoms
            An `Atoms` object.

        Returns
        -------
        vap : VirtualAtomMap
            The `VirtualAtomMap` for the given `Atoms` object.

        """
        # The mode 'reduce' is important here because chemical symbol lists of
        # ['C', 'H', 'O'] and ['C', 'O', 'H'] should be treated differently!
        formula = atoms.get_chemical_formula(mode='reduce')
        if formula not in self._vap_transformers:
            self._vap_transformers[formula] = VirtualAtomMap(
                self.max_occurs, atoms.get_chemical_symbols()
            )
        return self._vap_transformers[formula]

    @staticmethod
    def _encode_radial_metadata(data: RadialMetadata):
        """
        Encode the radial metadata:
            * `v2g_map`, `ilist` and `jlist` are merged into a single array
              with key 'g2.indices'.
            * `n1` will be encoded separately with key 'g2.shifts'.

        """
        indices = np.concatenate((
            data.v2g_map,
            data.ilist[..., np.newaxis],
            data.jlist[..., np.newaxis]), axis=1)
        return {'g2.indices': bytes_feature(indices.tostring()),
                'g2.shifts': bytes_feature(data.n1.tostring())}

    def _encode_additional_properties(self, atoms) -> dict:
        return {}

    def _encode_atoms(self, atoms: Atoms) -> dict:
        """
        Encode the basic properties of an `Atoms` object.
        """
        if atoms.calc is None:
            # Add a dummy calculator
            calc = SinglePointCalculator(
                atoms, **{'energy': 0.0,
                          'forces': np.zeros((len(atoms), 3)),
                          'stress': np.zeros(6)})
            atoms.calc = calc

        vap = self.get_vap_transformer(atoms)
        np_dtype = get_float_dtype().as_numpy_dtype
        positions = vap.map_positions(atoms.positions).astype(np_dtype)
        cell = atoms.get_cell(complete=True).array.astype(np_dtype)
        volume = np.atleast_1d(atoms.get_volume()).astype(np_dtype)
        energy = np.atleast_1d(atoms.get_total_energy()).astype(np_dtype)
        mask = vap.atom_masks.astype(np_dtype)
        pulay_stress = np.atleast_1d(
            atoms_utils.get_pulay_stress(atoms)).astype(np_dtype)
        etemp = np.atleast_1d(
            atoms_utils.get_electron_temperature(atoms)).astype(np_dtype)
        eentropy = np.atleast_1d(
            atoms_utils.get_electron_entropy(atoms)).astype(np_dtype)
        free_energy = energy - etemp * eentropy

        feature_list = {
            'positions': bytes_feature(positions.tostring()),
            'cell': bytes_feature(cell.tostring()),
            'n_atoms_vap': int64_feature(len(atoms)),
            'volume': bytes_feature(volume.tostring()),
            'energy': bytes_feature(energy.tostring()),
            'free_energy': bytes_feature(free_energy.tostring()),
            'atom_masks': bytes_feature(mask.tostring()),
            'pulay_stress': bytes_feature(pulay_stress.tostring()),
            'eentropy': bytes_feature(eentropy.tostring()),
            'etemperature': bytes_feature(etemp.tostring())
        }
        feature_list.update(self._encode_additional_properties(atoms))

        if self.use_forces:
            forces = vap.map_forces(atoms.get_forces()).astype(np_dtype)
            feature_list['forces'] = bytes_feature(forces.tostring())

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
    def _decode_additional_properties(example: Dict[str, tf.Tensor]):
        return {}

    @staticmethod
    def _decode_atoms(example: Dict[str, tf.Tensor],
                      max_n_atoms: int,
                      use_forces=True,
                      use_stress=False) -> dict:
        """
        Decode `Atoms` related properties.
        """
        decoded = dict()

        length = 3 * (max_n_atoms + 1)
        float_dtype = get_float_dtype()

        positions = tf.decode_raw(example['positions'], float_dtype)
        positions.set_shape([length])
        decoded["positions"] = tf.reshape(
            positions, (max_n_atoms + 1, 3), name='R')

        n_atoms = tf.identity(example['n_atoms_vap'], name='n_atoms_vap')
        decoded["n_atoms_vap"] = n_atoms

        energy = tf.decode_raw(example['energy'], float_dtype)
        energy.set_shape([1])
        decoded["energy"] = tf.squeeze(energy, name='energy')

        cell = tf.decode_raw(example['cell'], float_dtype)
        cell.set_shape([9])
        decoded["cell"] = tf.reshape(cell, (3, 3), name='cell')

        volume = tf.decode_raw(example['volume'], float_dtype)
        volume.set_shape([1])
        decoded["volume"] = tf.squeeze(volume, name='volume')

        atom_masks = tf.decode_raw(example['atom_masks'], float_dtype)
        atom_masks.set_shape([max_n_atoms + 1, ])
        decoded["atom_masks"] = atom_masks

        pulay = tf.decode_raw(example['pulay_stress'], float_dtype)
        pulay.set_shape([1])
        decoded["pulay_stress"] = tf.squeeze(pulay, name='pulay_stress')

        etemp = tf.decode_raw(example['etemperature'], float_dtype)
        etemp.set_shape([1])
        decoded["etemperature"] = tf.squeeze(etemp, name='etemperature')

        eentropy = tf.decode_raw(example['eentropy'], float_dtype)
        eentropy.set_shape([1])
        decoded["eentropy"] = tf.squeeze(eentropy, name='eentropy')

        free_energy = tf.decode_raw(example['free_energy'], float_dtype)
        free_energy.set_shape([1])
        decoded["free_energy"] = tf.squeeze(free_energy, name='free_energy')

        if use_forces:
            forces = tf.decode_raw(example['forces'], float_dtype)
            # Ignore the forces of the virtual atom
            forces.set_shape([length, ])
            decoded["forces"] = tf.reshape(
                forces, (max_n_atoms + 1, 3), name='forces')

        if use_stress:
            stress = tf.decode_raw(
                example['stress'], float_dtype, name='stress')
            stress.set_shape([6])
            decoded["stress"] = stress

            total_pressure = tf.decode_raw(
                example['total_pressure'], float_dtype)
            total_pressure.set_shape([1])
            decoded["total_pressure"] = tf.squeeze(
                total_pressure, name='total_pressure')

        return decoded

    def get_decode_feature_list(self):
        feature_list = {
            'positions': tf.FixedLenFeature([], tf.string),
            'n_atoms_vap': tf.FixedLenFeature([], tf.int64),
            'cell': tf.FixedLenFeature([], tf.string),
            'volume': tf.FixedLenFeature([], tf.string),
            'energy': tf.FixedLenFeature([], tf.string),
            'free_energy': tf.FixedLenFeature([], tf.string),
            'atom_masks': tf.FixedLenFeature([], tf.string),
            'pulay_stress': tf.FixedLenFeature([], tf.string),
            'etemperature': tf.FixedLenFeature([], tf.string),
            'eentropy': tf.FixedLenFeature([], tf.string),
            'g2.indices': tf.FixedLenFeature([], tf.string),
            'g2.shifts': tf.FixedLenFeature([], tf.string)
        }
        return feature_list

    @abc.abstractmethod
    def decode_protobuf(self, example_proto: tf.Tensor) -> dict:
        """
        Decode the scalar string Tensor, which is a single serialized Example.
        See `_parse_single_example_raw` documentation for more details.
        """
        pass

    def _infer_batch_size(self, next_batch: dict):
        """
        Infer `batch_size` from `next_batch`.
        """

        for name, tensor in next_batch.items():
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
    def get_descriptors(self, next_batch: dict):
        """
        An abstract method. Return the Op to compute atomic descriptors from raw
        properties (positions, volume, v2g_map, ...).

        Parameters
        ----------
        next_batch : dict
            A dict of batched raw property tensors.

        Returns
        -------
        descriptors : dict
            A dict of Ops to get atomic descriptors.

        """
        pass
