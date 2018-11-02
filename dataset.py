# coding=utf-8
"""
This module defines the `Dataset` class for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import sys
import time
import glob
from tensorflow.train import Example, Features
from tensorflow.contrib.data import shuffle_and_repeat
from behler import SymmetryFunction
from behler import RadialIndexedSlices, AngularIndexedSlices
from ase import Atoms
from ase.db.sqlite import SQLite3Database
from file_io import find_neighbor_sizes, convert_k_max_to_key, convert_rc_to_key
from misc import check_path, Defaults, AttributeDict, brange, safe_select
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os.path import join, basename, splitext
from enum import Enum
from typing import List, Dict


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TrainableProperty(Enum):
    """
    A enumeration list declaring the trainable properties.
    """
    energy = 0
    forces = 1


def _bytes_feature(value):
    """
    Convert the `value` to Protobuf bytes.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """
    Convert the `value` to Protobuf float32.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


class RawSerializer:
    """
    This class is used to serialize training examples in raw format.
    """

    def __init__(self, k_max: int, natoms: int, nij_max: int, nijk_max: int,
                 trainable_properties: List[TrainableProperty]):
        """
        Initialization method.
        """
        self.k_max = k_max
        self.natoms = natoms
        self.nij_max = nij_max
        self.nijk_max = nijk_max
        if not trainable_properties:
            trainable_properties = [TrainableProperty.energy, ]
        self.trainable_properties = trainable_properties

    @staticmethod
    def _merge_and_encode_rslices(rslices: RadialIndexedSlices):
        """
        Merge `v2g_map`, `ilist`, `jlist` and `Slist` of radial functions into a
        single array and encode this array.
        """
        merged = np.concatenate((
            rslices.v2g_map,
            rslices.Slist,
            rslices.ilist[..., np.newaxis],
            rslices.jlist[..., np.newaxis],
        ), axis=2)
        return _bytes_feature(merged.tostring())

    @staticmethod
    def _merge_and_encode_aslices(aslices: AngularIndexedSlices):
        """
        Merge `v2g_map`, `ij`, `ik`, `jk`, `ijSlist`, `ikSlist` and `jkSlist` of
        angular functions into a single array and encode this array.
        """
        merged = np.concatenate((
            aslices.v2g_map,
            aslices.ijSlist,
            aslices.ikSlist,
            aslices.jkSlist,
            aslices.ij,
            aslices.ik,
            aslices.jk,
        ), axis=2)
        return _bytes_feature(merged.tostring())

    def encode(self, positions: np.ndarray, cell: np.ndarray,
               y_true: float, f_true: np.ndarray, mask: np.ndarray,
               rslices: RadialIndexedSlices, aslices: AngularIndexedSlices):
        """
        Encode the data and return a `tf.train.Example`.
        """
        feature_list = {
            'positions': _bytes_feature(positions.tostring()),
            'cell': _bytes_feature(cell.tostring()),
            'y_true': _bytes_feature(np.atleast_2d(y_true).tostring()),
            'rslices': self._merge_and_encode_rslices(rslices),
            'mask': _bytes_feature(mask.tostring()),
        }
        if TrainableProperty.forces in self.trainable_properties:
            feature_list.update({'f_true': _bytes_feature(f_true.tostring())})
        if self.k_max == 3:
            feature_list.update(
                {'aslices': self._merge_and_encode_aslices(aslices)})
        return Example(features=Features(feature=feature_list))

    def _decode_atoms(self, example: Dict[str, tf.Tensor]):
        """
        Decode `Atoms` related properties.
        """
        length = 3 * (self.natoms + 1)

        positions = tf.decode_raw(example['positions'], tf.float64)
        positions.set_shape([length])
        positions = tf.reshape(positions, (self.natoms + 1, 3), name='R')

        cell = tf.decode_raw(example['cell'], tf.float64)
        cell.set_shape([9])
        cell = tf.reshape(cell, (3, 3), name='cell')

        y_true = tf.decode_raw(example['y_true'], tf.float64)
        y_true.set_shape([1])
        y_true = tf.squeeze(y_true, name='y_true')

        mask = tf.decode_raw(example['mask'], tf.float64)
        mask.set_shape([self.natoms + 1, ])

        if TrainableProperty.forces in self.trainable_properties:
            f_true = tf.decode_raw(example['f_true'], tf.float64)
            f_true.set_shape([length])
            f_true = tf.reshape(f_true, (self.natoms + 1, 3), name='f_true')
        else:
            f_true = None

        return positions, cell, y_true, f_true, mask

    def _decode_rslices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ilist, jlist and Slist for radial functions.
        """
        with tf.name_scope("rslices"):
            rslices = tf.decode_raw(example['rslices'], tf.int32, name='merged')
            rslices.set_shape([self.nij_max * 8])
            rslices = tf.reshape(rslices, [self.nij_max, 8], name='rslices')
            v2g_map, Slist, ilist, jlist = \
                tf.split(rslices, [3, 3, 1, 1], axis=1, name='splits')
            ilist = tf.squeeze(ilist, axis=1, name='ilist')
            jlist = tf.squeeze(jlist, axis=1, name='jlist')
            return RadialIndexedSlices(v2g_map, ilist, jlist, Slist)

    def _decode_aslices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ij, ik, jk, ijSlist, ikSlist and jkSlist for angular
        functions.
        """
        if self.k_max < 3:
            return None

        with tf.name_scope("aslices"):
            aslices = tf.decode_raw(example['aslices'], tf.int32, name='merged')
            aslices.set_shape([self.nijk_max * 18])
            aslices = tf.reshape(aslices, [self.nijk_max, 18], name='rslices')
            v2g_map, ijSlist, ikSlist, jkSlist, ij, ik, jk = \
                tf.split(aslices, [3, 3, 3, 3, 2, 2, 2], axis=1, name='splits')
            return AngularIndexedSlices(v2g_map, ij, ik, jk, ijSlist, ikSlist,
                                        jkSlist)

    def decode_example(self, example: Dict[str, tf.Tensor]):
        """
        Decode the parsed single example.
        """
        positions, cell, y_true, f_true, mask = self._decode_atoms(example)
        rslices = self._decode_rslices(example)
        aslices = self._decode_aslices(example)

        decoded = AttributeDict(positions=positions, cell=cell, y_true=y_true,
                                mask=mask, rv2g=rslices.v2g_map,
                                ilist=rslices.ilist, jlist=rslices.jlist,
                                Slist=rslices.Slist)

        if f_true is not None:
            decoded.f_true = f_true

        if aslices is not None:
            decoded.av2g = aslices.v2g_map
            decoded.ij = aslices.ij
            decoded.ik = aslices.ik
            decoded.jk = aslices.jk
            decoded.ijS = aslices.ijSlist
            decoded.ikS = aslices.ikSlist
            decoded.jkS = aslices.jkSlist

        return decoded

    def decode_protobuf(self, example_proto: tf.Tensor) -> AttributeDict:
        """
        Decode the scalar string Tensor, which is a single serialized Example.
        See `_parse_single_example_raw` documentation for more details.
        """
        with tf.name_scope("decoding"):
            feature_list = {
                'positions': tf.FixedLenFeature([], tf.string),
                'cell': tf.FixedLenFeature([], tf.string),
                'y_true': tf.FixedLenFeature([], tf.string),
                'rslices': tf.FixedLenFeature([], tf.string),
                'mask': tf.FixedLenFeature([], tf.string)
            }
            if TrainableProperty.forces in self.trainable_properties:
                feature_list['f_true'] = tf.FixedLenFeature([], tf.string)
            if self.k_max == 3:
                feature_list.update({
                    'aslices': tf.FixedLenFeature([], tf.string)})
            example = tf.parse_single_example(example_proto, feature_list)
            return self.decode_example(example)


class Dataset:
    """
    This class is used to manipulate data examples for this project.
    """

    def __init__(self, database, name, k_max=3, rc=Defaults.rc, eta=None,
                 beta=None, gamma=None, zeta=None):
        """
        Initialization method.

        Parameters
        ----------
        database : SQLite3Database
            A `SQLite3Database` created by `file_io.read`.
        name : str
            The name of this dataset.
        rc : float
            The cutoff radius.
        eta : array_like
            A list of float as the `eta` for radial functions.
        beta : array_like
            A list of float as the `beta` for angular functions.
        gamma : array_like
            A list of float as the `gamma` for angular functions.
        zeta : array_like
            A list of float as the `zeta` for angular functions.

        """
        self._database = database
        self._name = name
        self._k_max = k_max
        self._rc = rc
        self._eta = safe_select(eta, Defaults.eta)
        self._beta = safe_select(beta, Defaults.beta)
        self._gamma = safe_select(gamma, Defaults.gamma)
        self._zeta = safe_select(zeta, Defaults.zeta)
        self._files = {}
        self._file_sizes = {}
        self._read_database()

    @property
    def database(self):
        """
        Return the sqlite database of this dataset.
        """
        return self._database

    @property
    def name(self):
        """
        Return the name of this dataset.
        """
        return self._name

    @property
    def trainable_properties(self) -> List[TrainableProperty]:
        """
        Return a list of `TrainableProperty`. Currently only energy and forces
        are supported.
        """
        return self._trainable_properties

    @property
    def k_max(self):
        """
        Return the maximum k for the many-body expansion scheme.
        """
        return self._k_max

    @property
    def cutoff_radius(self):
        """
        Return the cutoff radius.
        """
        return self._rc

    @property
    def max_occurs(self):
        """
        Return a dict of (element, max_occur) as the maximum occurance for each
        type of element.
        """
        return self._max_occurs

    @property
    def descriptor(self):
        """
        Return the descriptor instance. Currently only `SymmetryFunction` is
        implemented.
        """
        return self._symmetry_function

    def __len__(self) -> int:
        """
        Return the number of examples in this dataset.
        """
        return len(self._database)

    def _should_find_neighbor_sizes(self):
        """
        A helper function. Return True if `nij_max` and `nijk_max` for `k_max`
        and `rc` cannot be accessed.
        """
        k_max = convert_k_max_to_key(self._k_max)
        rc = convert_rc_to_key(self._rc)
        neighbors = self._database.metadata.get('neighbors', {})
        if k_max not in neighbors:
            return True
        elif rc not in neighbors[k_max]:
            return True
        else:
            return False

    def _get_nij_and_nijk(self):
        """
        A helper function to get `nij_max` and `nijk_max` from metadata of the
        database.
        """
        k_max = convert_k_max_to_key(self._k_max)
        rc = convert_rc_to_key(self._rc)
        details = self._database.metadata['neighbors'][k_max][rc]
        return details['nij_max'], details['nijk_max']

    def _read_database(self):
        """
        Read the metadata of the database and finalize the initialization.
        """
        if self._should_find_neighbor_sizes():
            find_neighbor_sizes(self._database, self._rc, k_max=self._k_max)

        trainable_properties = [TrainableProperty.energy]
        if self._database.metadata['extxyz']:
            trainable_properties += [TrainableProperty.forces]
        max_occurs = self._database.metadata['max_occurs']
        natoms = sum(max_occurs.values())
        nij_max, nijk_max = self._get_nij_and_nijk()
        sf = SymmetryFunction(self._rc, max_occurs, k_max=self._k_max,
                              nij_max=nij_max, nijk_max=nijk_max, eta=self._eta,
                              beta=self._beta, gamma=self._gamma,
                              zeta=self._zeta)
        self._symmetry_function = sf
        self._max_occurs = max_occurs
        self._nij_max = nij_max
        self._nijk_max = nijk_max
        self._trainable_properties = trainable_properties
        self._serializer = RawSerializer(self._k_max, natoms, nij_max, nijk_max,
                                         trainable_properties)

    def convert_atoms_to_example(self, atoms: Atoms) -> Example:
        """
        Convert an `Atoms` object to `tf.train.Example`.
        """
        transformer = self._symmetry_function.get_index_transformer(atoms)
        positions = transformer.gather(atoms.positions)
        cell = atoms.cell
        mask = transformer.mask.astype(np.float64)
        y_true = np.atleast_2d(atoms.get_total_energy())
        f_true = transformer.gather(atoms.get_forces())
        rslices, aslices = self._symmetry_function.get_indexed_slices([atoms])
        return self._serializer.encode(positions, cell, y_true, f_true, mask,
                                       rslices, aslices)

    def _write_subset(self, mode: tf.estimator.ModeKeys, filename: str,
                      indices: List[int], parallel=True, verbose=False):
        """
        Write a subset of this dataset to the given file.

        Parameters
        ----------
        mode : tf.estimator.ModeKeys
            The purpose of this subset.
        filename : str
            The file to write.
        indices : List[int]
            A list of int as the ids of the `Atoms` to use for this file.
        parallel : bool
            If True, a joblib based parallel scheme shall be used.
        verbose : bool
            If True, the progress shall be logged.

        """
        with tf.python_io.TFRecordWriter(filename) as writer:

            num_examples = len(indices)

            if parallel:
                batch_size = cpu_count() * 50
                n_cpus = cpu_count()
            else:
                batch_size = 1
                n_cpus = 1

            batch_size = min(num_examples, batch_size)
            n_cpus = min(num_examples, n_cpus)

            logstr = "\rProgress: {:7d} / {:7d} | Speed = {:6.1f}"

            if verbose:
                print("Start writing {} subset ...\n".format(str(mode)))

            tic = time.time()

            for istart, istop in brange(0, num_examples, batch_size):
                trajectory = []
                for atoms_id in indices[istart: istop]:
                    trajectory.append(self._database.get_atoms(id=atoms_id))
                examples = Parallel(n_jobs=n_cpus)(
                    delayed(self.convert_atoms_to_example)(atoms)
                    for atoms in trajectory
                )
                for example in examples:
                    writer.write(example.SerializeToString())
                    if verbose:
                        speed = istop / (time.time() - tic)
                        sys.stdout.write(
                            logstr.format(istop, num_examples, speed))

            if verbose:
                print("Done.\n")

    def _get_signature(self) -> str:
        """
        Return a str as the signature of this dataset.
        """
        return "k{:d}-rc{:.2f}".format(self._k_max, self._rc)

    def to_records(self, savedir, test_size=0.2, parallel=True, verbose=False):
        """
        Split the dataset into a training set and a testing set and write these
        subsets to tfrecords files.

        Parameters
        ----------
        savedir : str
            The directory to save the converted tfrecords files.
        test_size : float or int
            The proportion (float) or size (int) of the dataset to include in
            the test split.
        parallel : bool
            If True, a joblib based parallel scheme shall be used.
        verbose : bool
            If True, the progress shall be logged.

        """
        signature = self._get_signature()
        train, test = train_test_split(range(1, 1 + len(self)),
                                       random_state=Defaults.seed,
                                       test_size=test_size)
        self._write_subset(
            tf.estimator.ModeKeys.EVAL,
            check_path(join(savedir, '{}-test-{}-{}.tfrecords'.format(
                self._name, signature, len(test)))),
            test,
            parallel=parallel,
            verbose=verbose,
        )
        self._write_subset(
            tf.estimator.ModeKeys.TRAIN,
            check_path(join(savedir, '{}-train-{}-{}.tfrecords'.format(
                self._name, signature, len(train)))),
            train,
            parallel=parallel,
            verbose=verbose,
        )

        self.load_tfrecords(savedir)

    def decode_protobuf(self, example_proto):
        """
        Decode the protobuf into a tuple of tensors.

        Parameters
        ----------
        example_proto : tf.Tensor
            A scalar string Tensor, a single serialized Example. See
            `_parse_single_example_raw` documentation for more details.

        Returns
        -------
        example : DecodedExample
            A `DecodedExample` from a tfrecords file.

        """
        return self._serializer.decode_protobuf(example_proto)

    def load_tfrecords(self, savedir, idx=0) -> bool:
        """
        Load converted tfrecords files for this dataset.
        """
        signature = self._get_signature()

        def _get_file_size(filename):
            return int(splitext(basename(filename))[0].split('-')[4])

        def _load(key):
            try:
                files = list(glob.glob(
                    '{}/{}-{}-{}-*.tfrecords'.format(
                        savedir, self._name, key, signature)))
            except Exception:
                return None, 0
            if len(files) >= 1:
                return files[idx], _get_file_size(files[idx])
            else:
                return None, 0

        test_file, test_size = _load('test')
        train_file, train_size = _load('train')
        success = test_file and train_file

        self._files = {tf.estimator.ModeKeys.TRAIN: train_file,
                       tf.estimator.ModeKeys.EVAL: test_file}
        self._file_sizes = {tf.estimator.ModeKeys.TRAIN: train_size,
                            tf.estimator.ModeKeys.EVAL: test_size}

        return success

    def input_fn(self, mode: tf.estimator.ModeKeys, batch_size=25,
                 num_epochs=None, shuffle=False):
        """
        Return a Callable input function for `tf.estimator.Estimator`.
        """
        def _input_fn():
            with tf.name_scope("Dataset"):
                batch = self.next_batch(
                    mode, batch_size=batch_size, num_epochs=num_epochs,
                    shuffle=shuffle)
            splits = self.descriptor.get_descriptors_graph(batch, batch_size)
            features = AttributeDict(descriptors=splits,
                                     positions=batch.positions,
                                     mask=batch.mask)
            labels = AttributeDict(y=batch.y_true)
            if TrainableProperty.forces in self.trainable_properties:
                labels.update({'f': batch.f_true})
            return features, labels

        return _input_fn

    def next_batch(self, mode=tf.estimator.ModeKeys.TRAIN, batch_size=25,
                   num_epochs=None, shuffle=False):
        """
        Return batch inputs of this dataset.

        Parameters
        ----------
        mode : ModeKeys
            A `ModeKeys` selecting between the training and validation data.
        batch_size : int
            A `int` as the number of examples per batch.
        num_epochs : int
            A `int` as the maximum number of epochs to run.
        shuffle : bool
            A `bool` indicating whether the batches shall be shuffled or not.

        Returns
        -------
        next_batch : DataGroup
            A tuple of Tensors.

        """
        with tf.device('/cpu:0'):

            # Get the tfrecords file
            tfrecords_file = self._files[mode]

            # Initialize a basic dataset
            dataset = tf.data.TFRecordDataset([tfrecords_file])
            dataset = dataset.map(self.decode_protobuf,
                                  num_parallel_calls=cpu_count())

            # Shuffle it if needed
            if shuffle:
                size = self._file_sizes[mode]
                min_queue_examples = int(size * 0.4) + 10 * batch_size
                dataset = dataset.apply(
                    shuffle_and_repeat(min_queue_examples, count=num_epochs,
                                       seed=Defaults.seed))
            else:
                # Repeat the dataset
                dataset = dataset.repeat(count=num_epochs)

            # Setup the batch
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=batch_size)

            # Return the iterator
            return dataset.make_one_shot_iterator().get_next()
