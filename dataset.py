# coding=utf-8
"""
This module defines the `Dataset` class for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import sys
import time
import glob
from tensorflow.contrib.data import shuffle_and_repeat
from ase.db.sqlite import SQLite3Database
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os.path import join, basename, splitext
from typing import List, Dict

from transformer import BatchSymmetryFunctionTransformer
from descriptor import BatchDescriptorTransformer
from file_io import find_neighbor_size_limits, compute_atomic_static_energy
from file_io import convert_rc_to_key, convert_k_max_to_key
from misc import check_path, Defaults, AttributeDict, brange, safe_select


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class Dataset:
    """
    This class is used to manipulate data examples for this project.
    """

    def __init__(self, database, name, k_max=3, rc=Defaults.rc, eta=None,
                 beta=None, gamma=None, zeta=None, serial=False):
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
        serial : bool
            If True, all parallel routines will be disabled.

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
        self._serial = serial
        self._forces = False
        self._stress = False
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
    def forces(self) -> bool:
        """
        Return True if the atomic forces are provided.
        """
        return self._forces

    @property
    def stress(self) -> bool:
        """
        Return True if the stress tensors are provided.
        """
        return self._stress

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
    def transformer(self) -> BatchDescriptorTransformer:
        """
        Return the assigned batch version of an atomic descriptor transformer.
        """
        return self._transformer

    @property
    def train_size(self):
        """
        Return the size of the training dataset.
        """
        return self._file_sizes.get(tf.estimator.ModeKeys.TRAIN, 0)

    @property
    def test_size(self):
        """
        Return the size of the evaluation dataset.
        """
        return self._file_sizes.get(tf.estimator.ModeKeys.EVAL, 0)

    @property
    def atomic_static_energy(self) -> Dict[str, float]:
        """
        Return a list of `float` as the static energy for each type of element.
        """
        return self._atomic_static_energy

    @property
    def serial(self):
        """
        Return True if this dataset is in serial mode.
        """
        return self._serial

    def __len__(self) -> int:
        """
        Return the number of examples in this dataset.
        """
        return len(self._database)

    def _should_compute_atomic_static_energy(self):
        """
        A helper function. Return True if `y_static` cannot be accessed.
        """
        return len(self._database.metadata.get('atomic_static_energy', {})) == 0

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
        if self._serial:
            n_jobs = 1
        else:
            n_jobs = -1

        if self._should_find_neighbor_sizes():
            find_neighbor_size_limits(self._database, self._rc, n_jobs=n_jobs,
                                      k_max=self._k_max, verbose=True)

        periodic = self._database.metadata['periodic']
        forces = self._database.metadata['forces']
        stress = self._database.metadata['stress']

        max_occurs = self._database.metadata['max_occurs']
        nij_max, nijk_max = self._get_nij_and_nijk()
        sf = BatchSymmetryFunctionTransformer(
            rc=self._rc,  max_occurs=max_occurs, k_max=self._k_max,
            nij_max=nij_max, nijk_max=nijk_max, eta=self._eta, beta=self._beta,
            gamma=self._gamma, zeta=self._zeta, periodic=periodic,
            stress=stress, forces=forces)

        if self._should_compute_atomic_static_energy():
            compute_atomic_static_energy(self._database, sf.elements, True)

        self._transformer = sf
        self._forces = forces
        self._stress = stress
        self._max_occurs = max_occurs
        self._nij_max = nij_max
        self._nijk_max = nijk_max
        self._atomic_static_energy = \
            self._database.metadata['atomic_static_energy']

    def _write_subset(self, mode: tf.estimator.ModeKeys, filename: str,
                      indices: List[int], verbose=False):
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
        verbose : bool
            If True, the progress shall be logged.

        """
        with tf.python_io.TFRecordWriter(filename) as writer:

            num_examples = len(indices)

            if not self._serial:
                batch_size = cpu_count() * 50
                n_cpus = cpu_count()
            else:
                batch_size = 1
                n_cpus = 1

            batch_size = min(num_examples, batch_size)
            n_cpus = min(num_examples, n_cpus)

            logstr = "\rProgress: {:7d} / {:7d} | Speed = {:6.1f}"

            if verbose:
                print("Start writing {} subset ...".format(str(mode)))

            tic = time.time()

            for istart, istop in brange(0, num_examples, batch_size):
                trajectory = []
                for atoms_id in indices[istart: istop]:
                    trajectory.append(self._database.get_atoms(id=atoms_id))
                examples = Parallel(n_jobs=n_cpus)(
                    delayed(self._transformer.encode)(atoms)
                    for atoms in trajectory
                )
                for example in examples:
                    writer.write(example.SerializeToString())
                    if verbose:
                        speed = istop / (time.time() - tic)
                        sys.stdout.write(
                            logstr.format(istop, num_examples, speed))
            if verbose:
                print("")
                print("Done.")
                print("")

    def _get_signature(self) -> str:
        """
        Return a str as the signature of this dataset.
        """
        return "k{:d}-rc{:.2f}".format(self._k_max, self._rc)

    def to_records(self, savedir, test_size=0.2, verbose=False):
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
            verbose=verbose,
        )
        self._write_subset(
            tf.estimator.ModeKeys.TRAIN,
            check_path(join(savedir, '{}-train-{}-{}.tfrecords'.format(
                self._name, signature, len(train)))),
            train,
            verbose=verbose,
        )

        self.load_tfrecords(savedir)

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
            splits = self._transformer.get_graph_from_batch(batch, batch_size)
            features = AttributeDict(descriptors=splits,
                                     positions=batch.positions,
                                     n_atoms=batch.n_atoms,
                                     cells=batch.cells,
                                     composition=batch.composition,
                                     mask=batch.mask)
            labels = AttributeDict(energy=batch.y_true)
            if self._forces:
                labels['forces'] = batch.f_true
            if self._stress:
                labels['stress'] = batch.stress
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
            dataset = dataset.map(self._transformer.decode_protobuf,
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
