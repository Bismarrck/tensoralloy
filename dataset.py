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
from tensorflow.contrib.learn.python import ModeKeys
from tensorflow.train import Example, Features
from behler import NeighborIndexBuilder
from behler import RadialIndexedSlices, AngularIndexedSlices
from behler import get_kbody_terms, compute_dimension
from ase import Atoms
from ase.db.sqlite import SQLite3Database
from file_io import find_neighbor_sizes
from misc import check_path, Defaults, AttributeDict
from sklearn.model_selection import train_test_split
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

    def __init__(self, k_max: int, natoms: int, nij_max: int, n_etas: int,
                 nijk_max: int, trainable_properties: List[TrainableProperty]):
        """
        Initialization method.
        """
        self.k_max = k_max
        self.natoms = natoms
        self.nij_max = nij_max
        self.n_etas = n_etas
        self.nijk_max = nijk_max
        if not trainable_properties:
            trainable_properties = [TrainableProperty.energy, ]
        self.trainable_properties = trainable_properties

    def encode(self, positions: np.ndarray, cell: np.ndarray, y_true: float,
               f_true: np.ndarray, rslices: RadialIndexedSlices,
               aslices: AngularIndexedSlices):
        """
        Encode the data and return a `tf.train.Example`.
        """
        feature_list = {
            'positions': _bytes_feature(positions.tostring()),
            'cell': _bytes_feature(cell.tostring()),
            'y_true': _float_feature(y_true),
            'r_v2g': _bytes_feature(rslices.v2g_map.tostring()),
            'r_ilist': _bytes_feature(rslices.ilist.tostring()),
            'r_jlist': _bytes_feature(rslices.jlist.tostring()),
            'r_Slist': _bytes_feature(rslices.Slist.tostring()),
        }
        if TrainableProperty.forces in self.trainable_properties:
            feature_list.update({'f_true': _bytes_feature(f_true.tostring())})
        if self.k_max == 3:
            feature_list.update({
                'a_v2g': _bytes_feature(aslices.v2g_map.tostring()),
                'a_ij': _bytes_feature(aslices.ik.tostring()),
                'a_ik': _bytes_feature(aslices.ij.tostring()),
                'a_jk': _bytes_feature(aslices.jk.tostring()),
                'a_ijSlist': _bytes_feature(aslices.ijSlist.tostring()),
                'a_ikSlist': _bytes_feature(aslices.ikSlist.tostring()),
                'a_jkSlist': _bytes_feature(aslices.jkSlist.tostring()),
            })
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

        y_true = tf.cast(example['y_true'], tf.float32, name='y_true')

        if TrainableProperty.forces in self.trainable_properties:
            f_true = tf.decode_raw(example['f_true'], tf.float64)
            f_true.set_shape([length])
            f_true = tf.reshape(f_true, (self.natoms + 1, 3), name='f_true')
        else:
            f_true = None

        return positions, cell, y_true, f_true

    def _decode_rslices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ilist, jlist and Slist for radial functions.
        """
        length = self.nij_max * self.n_etas

        v2g_map = tf.decode_raw(example['r_v2g'], tf.int32)
        v2g_map.set_shape(length * 3)
        v2g_map = tf.reshape(v2g_map, (length, 3), name='r_v2g')

        ilist = tf.decode_raw(example['r_ilist'], tf.int32, name='r_ilist')
        ilist.set_shape([self.nij_max])

        jlist = tf.decode_raw(example['r_jlist'], tf.int32, name='r_jlist')
        jlist.set_shape([self.nij_max])

        Slist = tf.decode_raw(example['r_Slist'], tf.int32)
        Slist.set_shape([self.nij_max * 3])
        Slist = tf.reshape(Slist, (self.nij_max, 3), name='Slist')

        return RadialIndexedSlices(v2g_map, ilist, jlist, Slist)

    def _decode_aslices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ij, ik, jk, ijSlist, ikSlist and jkSlist for angular
        functions.
        """
        if TrainableProperty.forces not in self.trainable_properties:
            return None

        length = self.nijk_max * 3

        v2g_map = tf.decode_raw(example['a_v2g'], tf.int32)
        v2g_map.set_shape(length)
        v2g_map = tf.reshape(v2g_map, (self.nijk_max, 3), name='a_v2g')

        ij = tf.decode_raw(example['a_ij'], tf.int32)
        ij.set_shape([self.nijk_max])

        ik = tf.decode_raw(example['a_ik'], tf.int32)
        ik.set_shape([self.nijk_max])

        jk = tf.decode_raw(example['a_jk'], tf.int32)
        jk.set_shape([self.nijk_max])

        ijSlist = tf.decode_raw(example['a_ijSlist'], tf.int32)
        ijSlist.set_shape([length])
        ijSlist = tf.reshape(v2g_map, (self.nijk_max, 3), name='a_ijSlist')

        ikSlist = tf.decode_raw(example['a_ikSlist'], tf.int32)
        ikSlist.set_shape([length])
        ikSlist = tf.reshape(v2g_map, (self.nijk_max, 3), name='a_ikSlist')

        jkSlist = tf.decode_raw(example['a_jkSlist'], tf.int32)
        jkSlist.set_shape([length])
        jkSlist = tf.reshape(v2g_map, (self.nijk_max, 3), name='a_jkSlist')

        return AngularIndexedSlices(v2g_map, ij, ik, jk, ijSlist, ikSlist,
                                    jkSlist)

    def decode_example(self, example: Dict[str, tf.Tensor]):
        """
        Decode the parsed single example.
        """
        positions, cell, y_true, f_true = self._decode_atoms(example)
        rslices = self._decode_rslices(example)
        aslices = self._decode_aslices(example)

        decoded = AttributeDict(positions=positions, cell=cell, y_true=y_true,
                                rv2g=rslices.v2g_map, ilist=rslices.ilist,
                                jlist=rslices.jlist, Slist=rslices.Slist)

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
        feature_list = {
            'positions': tf.FixedLenFeature([], tf.string),
            'cell': tf.FixedLenFeature([], tf.string),
            'y_true': tf.FixedLenFeature([], tf.float32),
            'r_v2g': tf.FixedLenFeature([], tf.string),
            'r_ilist': tf.FixedLenFeature([], tf.string),
            'r_jlist': tf.FixedLenFeature([], tf.string),
            'r_Slist': tf.FixedLenFeature([], tf.string),
        }
        if TrainableProperty.forces in self.trainable_properties:
            feature_list['f_true'] = tf.FixedLenFeature([], tf.string)
        if self.k_max == 3:
            feature_list.update({
                'a_v2g': tf.FixedLenFeature([], tf.string),
                'a_ij': tf.FixedLenFeature([], tf.string),
                'a_ik': tf.FixedLenFeature([], tf.string),
                'a_jk': tf.FixedLenFeature([], tf.string),
                'a_ijSlist': tf.FixedLenFeature([], tf.string),
                'a_ikSlist': tf.FixedLenFeature([], tf.string),
                'a_jkSlist': tf.FixedLenFeature([], tf.string),
            })
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
        def _select(a, b):
            """ A helper function to select `a` if it's valided. """
            if a is None or len(a) == 0:
                return b
            return a

        self._database = database
        self._name = name
        self._k_max = k_max
        self._rc = rc
        self._eta = _select(eta, Defaults.eta)
        self._beta = _select(beta, Defaults.beta)
        self._gamma = _select(gamma, Defaults.gamma)
        self._zeta = _select(zeta, Defaults.zeta)
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

    def __len__(self) -> int:
        """ Return the number of examples in this dataset. """
        return len(self._database)

    def _read_database(self):
        """
        Read the metadata of the database and finalize the initialization.
        """
        find_neighbor_sizes(self._database, self._rc, k_max=self._k_max)

        metadata = self._database.metadata
        trainable_properties = [TrainableProperty.energy]
        if metadata['extxyz']:
            trainable_properties += [TrainableProperty.forces]
        max_occurs = metadata['max_occurs']
        nij_max = metadata['nij_max']
        nijk_max = metadata['nijk_max']
        kbody_terms, mapping, elements = get_kbody_terms(
            max_occurs.keys(), k_max=self._k_max)
        total_size, kbody_sizes = compute_dimension(
            kbody_terms, len(self._eta), len(self._beta), len(self._gamma),
            len(self._zeta))
        natoms = sum(max_occurs.values())
        nl = NeighborIndexBuilder(self._rc, kbody_terms, kbody_sizes,
                                  max_occurs, len(self._eta), self._k_max,
                                  nij_max, nijk_max)
        self._nl = nl
        self._max_occurs = max_occurs
        self._nij_max = nij_max
        self._nijk_max = nijk_max
        self._kbody_terms = kbody_terms
        self._kbody_sizes = kbody_sizes
        self._total_size = total_size
        self._elements = elements
        self._mapping = mapping
        self._trainable_properties = trainable_properties
        self._serializer = RawSerializer(self._k_max, natoms, nij_max,
                                         len(self._eta), nijk_max,
                                         trainable_properties)

    def convert_atoms_to_example(self, atoms: Atoms) -> Example:
        """
        Convert an `Atoms` object to `tf.train.Example`.
        """
        transformer = self._nl.get_index_transformer(atoms)
        positions = transformer.gather(atoms.positions)
        cell = atoms.cell
        y_true = np.atleast_2d(atoms.get_total_energy())
        f_true = transformer.gather(atoms.get_forces())
        rslices, aslices = self._nl.get_indexed_slices([atoms])
        return self._serializer.encode(positions, cell, y_true, f_true, rslices,
                                       aslices)

    def _write_subset(self, mode: ModeKeys, filename: str, indices: List[int],
                      verbose=False):
        """
        Write a subset of this dataset to the given file.

        Parameters
        ----------
        mode : ModeKeys
            The purpose of this subset.
        filename : str
            The file to write.
        indices : List[int]
            A list of int as the ids of the `Atoms` to use for this file.
        verbose : bool
            If True, the progress shall be logged.

        """
        with tf.python_io.TFRecordWriter(filename) as writer:

            tic = time.time()
            num_examples = len(indices)
            logstr = "\rProgress: {:7d} / {:7d} | Speed = {:6.1f}"

            if verbose:
                sys.stdout.write("Writing {} subset ...\n".format(str(mode)))

            for i, atoms_id in enumerate(indices):
                atoms = self._database.get_atoms(id=atoms_id)
                example = self.convert_atoms_to_example(atoms)
                writer.write(example.SerializeToString())
                if verbose:
                    speed = (i + 1) / (time.time() - tic)
                    sys.stdout.write(logstr.format(i + 1, num_examples, speed))

            if verbose:
                sys.stdout.write("Done.\n")

    def to_records(self, savedir, test_size=0.2, verbose=False):
        """
        Split the dataset into a training set and a testing set and write these
        subsets to tfrecords files.

        Parameters
        ----------
        savedir : str
            The directory to save the converted tfrecords files.
        test_size : float
            The proportion of the dataset to include in the test split.
        verbose : bool
            If True, the progress shall be logged.

        """
        test_size = min(max(test_size, 0.0), 1.0)
        train, test = train_test_split(range(1, 1 + len(self)),
                                       random_state=Defaults.seed,
                                       test_size=test_size)
        self._write_subset(
            ModeKeys.EVAL,
            check_path(join(savedir, '{}-test-{}.tfrecords'.format(
                self._name, len(test)))),
            test,
            verbose=verbose,
        )
        self._write_subset(
            ModeKeys.TRAIN,
            check_path(join(savedir, '{}-train-{}.tfrecords'.format(
                self._name, len(train)))),
            train,
            verbose=verbose,
        )

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

    def load_tfrecords(self, savedir, idx=0) -> Dict[ModeKeys, str]:
        """
        Load converted tfrecords files for this dataset.
        """

        def _get_file_size(filename):
            return int(splitext(basename(filename))[0].split('-')[2])

        def _load(key):
            files = list(glob.glob(
                '{}/{}-{}-?.tfrecords'.format(savedir, self._name, key)))
            if len(files) >= 1:
                return files[idx], _get_file_size(files[idx])
            else:
                return None, 0

        test_file, test_size = _load('test')
        train_file, train_size = _load('train')

        self._files = {ModeKeys.TRAIN: train_file, ModeKeys.EVAL: test_file}
        self._file_sizes = {
            ModeKeys.TRAIN: train_size, ModeKeys.EVAL: test_size}

        return self._files

    def next_batch(self, mode=ModeKeys.TRAIN, batch_size=25, num_epochs=None,
                   shuffle=False):
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
            dataset = dataset.map(self.decode_protobuf)

            # Repeat the dataset
            dataset = dataset.repeat(count=num_epochs)

            # Shuffle it if needed
            if shuffle:
                size = self._file_sizes[mode]
                min_queue_examples = int(size * 0.4) + 10 * batch_size
                dataset = dataset.shuffle(buffer_size=min_queue_examples,
                                          seed=Defaults.seed)

            # Setup the batch
            dataset = dataset.batch(batch_size)

            # Return the iterator
            return dataset.make_one_shot_iterator().get_next()
