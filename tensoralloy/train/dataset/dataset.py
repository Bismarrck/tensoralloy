# coding=utf-8
"""
This module defines the data container for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import glob
import sys
import time
import os

from multiprocessing import cpu_count
from os.path import join, splitext, basename, exists, dirname
from typing import Dict, List, Tuple, Union
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.utils import Defaults, check_path, ModeKeys
from tensoralloy.train.dataset.utils import should_be_serial
from tensoralloy.train.dataset.utils import brange
from tensoralloy.io.sqlite import CoreDatabase
from tensoralloy.io.db import connect
from tensoralloy.precision import get_float_dtype, precision_scope
from tensoralloy.precision import get_float_precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["Dataset"]


class Dataset:
    """
    This class is used to manipulate data examples for this project.
    """

    default_input_keys = ['energy', 'eentropy', 'free_energy']

    def __init__(self,
                 database: Union[CoreDatabase, str],
                 name: str,
                 transformer: BatchUniversalTransformer,
                 serial=False):
        """
        Initialization method.

        Parameters
        ----------
        database : CoreDatabase or str
            A `SQLite3Database` created by `file_io.read`.
        name : str
            The name of this dataset.
        transformer : BatchDescriptorTransformer
            A batch descriptor transformer.
        serial : bool
            If True, all parallel routines will be disabled.

        """
        if isinstance(database, str):
            self._database = connect(database)
        elif isinstance(database, CoreDatabase):
            self._database = database
        else:
            raise ValueError("`database` must be a SQLite3Database or a str!")
        self._name = name
        self._transformer = transformer
        self._transformer.use_forces = database.has_forces
        self._transformer.use_stress = database.has_stress
        self._files = {}
        self._file_sizes = {}

        if should_be_serial():
            self._serial = False
            tf.logging.info(
                'Warning: glibc < 2.17, set `Dataset` to serial mode.')
        else:
            self._serial = serial

        self._check()

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
    def has_forces(self) -> bool:
        """
        Return True if the atomic forces are provided. The unit of the atomic
        forces is 'eV / Angstrom'.
        """
        return self._database.has_forces

    @property
    def has_stress(self) -> bool:
        """
        Return True if the stress tensors are provided. The unit of the stress
        tensors is 'eV/Angstrom'.
        """
        return self._database.has_stress

    @property
    def max_occurs(self):
        """
        Return a dict of (element, max_occur) as the maximum occurance for each
        type of element.
        """
        return self._database.max_occurs

    @property
    def transformer(self) -> BatchUniversalTransformer:
        """
        Return the attached universal transformer.
        """
        return self._transformer

    @property
    def train_size(self):
        """
        Return the size of the training dataset.
        """
        return self._file_sizes.get(ModeKeys.TRAIN, 0)

    @property
    def test_size(self):
        """
        Return the size of the evaluation dataset.
        """
        return self._file_sizes.get(ModeKeys.EVAL, 0)

    @property
    def atomic_static_energy(self) -> Dict[str, float]:
        """
        Return a list of `float` as the static energy for each type of element.
        """
        return self._database.get_atomic_static_energy(allow_calculation=False)

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

    def _check(self):
        """
        Necessary internal check calls.
        """
        self._database.get_atomic_static_energy(allow_calculation=True)

    def _write_subset(self, mode: ModeKeys, filename: str,
                      indices: List[int], verbose=False):
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
        write_dir = dirname(filename)
        if not exists(write_dir):
            os.makedirs(write_dir)

        with tf.io.TFRecordWriter(filename) as writer:

            num_examples = len(indices)

            if not self._serial:
                n_cpus = max(1, cpu_count() - 1)
                batch_size = n_cpus * 10
            else:
                batch_size = 1
                n_cpus = 1

            batch_size = min(num_examples, batch_size)
            n_cpus = min(num_examples, n_cpus)
            precision = get_float_precision()

            logstr = "\rProgress: {:7d} / {:7d} | Speed = {:6.1f}"
            if verbose:
                print("Start writing {} subset ...".format(str(mode)))

            tic = time.time()

            if not self._serial:

                def pipeline(_atoms, _precision):
                    """
                    The parallel pipeline function.

                    If we use `BatchDescriptorTransformer.encode` direcly as the
                    pipeline function of `of process-based `joblib.Parallel`,
                    the global floating-point precision of the main process can
                    not be accessed by child processes. So here we must set
                    `precision` on every child process.
                    """
                    with precision_scope(_precision):
                        return self._transformer.encode(_atoms)

                for istart, istop in brange(0, num_examples, batch_size):
                    trajectory = []
                    for atoms_id in indices[istart: istop]:
                        trajectory.append(
                            self._database.get_atoms(
                                id=atoms_id,
                                add_additional_information=True))
                    examples = Parallel(n_jobs=n_cpus)(
                        delayed(pipeline)(atoms, precision)
                        for atoms in trajectory
                    )
                    for example in examples:
                        writer.write(example.SerializeToString())
                        if verbose:
                            speed = istop / (time.time() - tic)
                            sys.stdout.write(
                                logstr.format(istop, num_examples, speed))
            else:
                for index, atoms_id in enumerate(indices):
                    atoms = self._database.get_atoms(
                        id=atoms_id,
                        add_additional_information=True)
                    with precision_scope(precision):
                        example = self._transformer.encode(atoms)
                    writer.write(example.SerializeToString())
                    if (index + 1) % 10 == 0 and verbose:
                        speed = (index + 1) / (time.time() - tic)
                        sys.stdout.write(
                            logstr.format(index + 1, num_examples, speed))

            if verbose:
                print("")
                print(f"Done: {filename}")
                print("")

    def _get_signature(self) -> str:
        """
        Return a str as the signature of this dataset.
        """
        if self._transformer.angular:
            k_max = 3
        else:
            k_max = 2
        if hasattr(self._transformer, "rcut"):
            rc = self._transformer.rcut
        else:
            rc = self._transformer.rc
        sig = "k{:d}-rc{:.2f}".format(k_max, rc)
        dtype = get_float_dtype()
        if dtype == tf.float32:
            sig += "-fp32"
        else:
            sig += "-fp64"
        return sig

    def to_records(self, savedir, test_size=0.2, seed=None, write='all',
                   verbose=False):
        """
        Split the dataset into a training set and a testing set and write these
        subsets to tfrecords files.

        Parameters
        ----------
        savedir : str
            The directory to save the converted tfrecords files.
        test_size : float or int or List[int] or Tuple[int]
            The proportion (float) or size (int) or indices (List[int]) of the
            dataset to include in the test split.
        seed : int
            The random seed.
        write : str
            Specify the subsets to write, options: 'all', 'train' or 'eval'.
        verbose : bool
            If True, the progress shall be logged.

        """
        signature = self._get_signature()

        if seed is None:
            seed = Defaults.seed

        assert write in ('all', 'eval', 'train')

        if isinstance(test_size, (list, tuple)):
            test = list(test_size)
            train = [x for x in range(1, 1 + len(self)) if x not in test]
        else:
            train, test = train_test_split(range(1, 1 + len(self)),
                                           random_state=seed,
                                           test_size=test_size)
        assert min(test) >= 1
        assert min(train) >= 1

        if write == 'all' or write == 'eval':
            self._write_subset(
                ModeKeys.EVAL,
                check_path(join(savedir, '{}-test-{}-{}.{}.tfrecords'.format(
                    self._name,
                    signature,
                    len(test),
                    self._transformer.descriptor,))),
                test,
                verbose=verbose,
            )

        if write == 'all' or write == 'train':
            self._write_subset(
                ModeKeys.TRAIN,
                check_path(join(savedir, '{}-train-{}-{}.{}.tfrecords'.format(
                    self._name,
                    signature,
                    len(train),
                    self._transformer.descriptor,))),
                train,
                verbose=verbose,
            )

        self.load_tfrecords(savedir)

    def load_tfrecords(self, savedir: str, idx=0, test_size=None) -> bool:
        """
        Load tfrecords files for this dataset.
        """
        signature = self._get_signature()
        total_size = len(self._database)

        if isinstance(test_size, float):
            test_size = int(test_size * total_size)

        def _get_file_size(filename):
            suffix = str(splitext(basename(filename))[0].split('-')[5])
            return int(suffix.split('.')[0])

        def _load(is_test=False):
            if is_test:
                key = 'test'
                target_size = test_size
            else:
                key = 'train'
                if test_size is None:
                    target_size = None
                else:
                    target_size = total_size - test_size
            try:
                files = list(glob.glob(
                    '{}/{}-{}-{}-*.{}.tfrecords'.format(
                        savedir, self._name, key, signature,
                        self._transformer.descriptor)))
            except Exception:
                return None, 0
            if len(files) >= 1:
                if target_size is None:
                    return files[idx], _get_file_size(files[idx])
                else:
                    for _idx in range(len(files)):
                        _size = _get_file_size(files[_idx])
                        if _size == target_size:
                            return files[_idx], _size
                    else:
                        tf.logging.info(
                            f"The {key} tfrecords of size {target_size} cannot "
                            f"be accessed in {savedir}")
                        return None, 0
            else:
                return None, 0

        test_file, test_size = _load(is_test=True)
        train_file, train_size = _load(is_test=False)
        success = (test_file is not None) and (train_file is not None)

        self._files = {ModeKeys.TRAIN: train_file,
                       ModeKeys.EVAL: test_file}
        self._file_sizes = {ModeKeys.TRAIN: train_size,
                            ModeKeys.EVAL: test_size}

        return success

    def input_fn(self, mode: ModeKeys, batch_size=25,
                 num_epochs=None, shuffle=False):
        """
        Return a Callable input function for `tf_estimator.Estimator`.

        Parameters
        ----------
        mode : ModeKeys
            A `ModeKeys` selecting between the training and validation data.
        batch_size : int
            The total batch size for training to be divided by the number of
            shards.
        num_epochs : Union[int, None]
            The maximum number of epochs to run.
        shuffle : bool
            Shuffle the input data if True.

        """
        if ModeKeys.for_prediction(mode):
            raise ValueError("Prediction mode does not need an `input_fn`.")

        def _input_fn() -> Tuple[dict, dict]:
            """
            The input function for `tf_estimator.Estimator`.

            Returns
            -------
            features : dict
                A dict of input features.
            labels : dict
                A dict of labels.

            """
            with tf.name_scope("Dataset"):
                features = self.next_batch(
                    mode, batch_size=batch_size, num_epochs=num_epochs,
                    shuffle=shuffle)
                for tensor in features.values():
                    shape = tensor.shape.as_list()
                    if shape[0] is None:
                        tensor.set_shape([batch_size] + shape[1:])

                labels = {}
                for key in self.default_input_keys:
                    labels[key] = features.pop(key)

                if self._database.has_forces:
                    labels['forces'] = features.pop('forces')
                if self._database.has_stress:
                    labels['stress'] = features.pop('stress')

                return features, labels

        return _input_fn

    def next_batch(self, mode=ModeKeys.TRAIN, batch_size=25,
                   num_epochs=None, shuffle=False):
        """
        Return batch inputs of this dataset.

        Parameters
        ----------
        mode : ModeKeys
            A `ModeKeys` selecting between the training and validation data.
        batch_size : int
            A `int` as the number of examples per batch.
        num_epochs : Union[int, None]
            A `int` as the maximum number of epochs to run.
        shuffle : bool
            A `bool` indicating whether the batches shall be shuffled or not.

        Returns
        -------
        next_batch : dict
            A dict of tensors.

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
                dataset = dataset.shuffle(min_queue_examples,
                                          seed=Defaults.seed,
                                          reshuffle_each_iteration=True)
                dataset = dataset.repeat(num_epochs)

            else:
                # Repeat the dataset
                dataset = dataset.repeat(count=num_epochs)

            # Setup the batch
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=1)

            # Return the iterator
            return dataset.make_one_shot_iterator().get_next()
