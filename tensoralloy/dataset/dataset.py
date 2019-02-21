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
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tensorflow.contrib.data import shuffle_and_repeat

from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.transformer import BatchEAMTransformer
from tensoralloy.transformer import BatchSymmetryFunctionTransformer
from tensoralloy.utils import AttributeDict, Defaults, safe_select, check_path
from tensoralloy.io.neighbor import convert_k_max_to_key, convert_rc_to_key
from tensoralloy.io.neighbor import find_neighbor_size_limits
from tensoralloy.dataset.utils import compute_atomic_static_energy
from tensoralloy.dataset.utils import should_be_serial
from tensoralloy.dataset.utils import brange
from tensoralloy.dtypes import get_float_dtype, set_float_precision
from tensoralloy.dtypes import get_float_precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["Dataset"]


class Dataset:
    """
    This class is used to manipulate data examples for this project.
    """

    def __init__(self, database, name, descriptor='behler', rc=Defaults.rc,
                 serial=False, **kwargs):
        """
        Initialization method.

        Parameters
        ----------
        database : SQLite3Database
            A `SQLite3Database` created by `file_io.read`.
        name : str
            The name of this dataset.
        descriptor : str
            The name of the descriptor transformer. Defaults to 'behler'. 'eam'
            is another valid option.
        rc : float
            The cutoff radius.
        serial : bool
            If True, all parallel routines will be disabled.
        kwargs :
            Key-value args for initializing the `BatchDescriptorTransformer` for
            the `descriptor`.

        """
        self._database = database
        self._name = name
        self._descriptor = descriptor
        self._rc = rc
        self._files = {}
        self._file_sizes = {}
        self._use_forces = False
        self._use_stress = False

        if should_be_serial():
            self._serial = False
            tf.logging.info(
                'Warning: glibc < 2.17, set `Dataset` to serial mode.')
        else:
            self._serial = serial

        self._read_database(**kwargs)

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
    def use_forces(self) -> bool:
        """
        Return True if the atomic forces are provided. The unit of the atomic
        forces is 'eV / Angstrom'.
        """
        return self._use_forces

    @property
    def use_stress(self) -> bool:
        """
        Return True if the stress tensors are provided. The unit of the stress
        tensors is 'eV/Angstrom'.
        """
        return self._use_stress

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
    def descriptor(self) -> str:
        """
        Return the name of the descriptor.
        """
        return self._descriptor

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

    def _should_find_neighbor_sizes(self, k_max):
        """
        A helper function. Return True if `nij_max` and `nijk_max` for `k_max`
        and `rc` cannot be accessed.
        """
        k_max = convert_k_max_to_key(k_max)
        rc = convert_rc_to_key(self._rc)
        neighbors = self._database.metadata.get('neighbors', {})
        if k_max not in neighbors:
            return True
        elif rc not in neighbors[k_max]:
            return True

        details = neighbors[k_max][rc]
        if 'nij_max' not in details:
            return True
        elif 'nijk_max' not in details:
            return True
        elif 'nnl_max' not in details:
            return True

        return False

    def _get_neighbor_sizes(self, k_max):
        """
        A helper function to get `nij_max`, `nijk_max` and `nnl_max`.
        """
        k_max = convert_k_max_to_key(k_max)
        rc = convert_rc_to_key(self._rc)
        details = self._database.metadata['neighbors'][k_max][rc]
        return details['nij_max'], details['nijk_max'], details['nnl_max']

    def _read_database(self, **kwargs):
        """
        Read the metadata of the database and finalize the initialization.
        """
        if self._serial:
            n_jobs = 1
        else:
            n_jobs = -1

        k_max = 2

        if self._descriptor == 'behler' and kwargs.get('angular', False):
            k_max = 3

        if self._should_find_neighbor_sizes(k_max):
            find_neighbor_size_limits(self._database, self._rc, n_jobs=n_jobs,
                                      k_max=k_max, verbose=True)

        periodic = self._database.metadata['periodic']
        forces = self._database.metadata['forces']
        stress = self._database.metadata['stress']
        max_occurs = self._database.metadata['max_occurs']

        nij_max, nijk_max, nnl_max = self._get_neighbor_sizes(k_max)

        if self._descriptor == 'behler':

            eta = safe_select(kwargs.get('eta', None), Defaults.eta)
            beta = safe_select(kwargs.get('beta', None), Defaults.beta)
            gamma = safe_select(kwargs.get('gamma', None), Defaults.gamma)
            zeta = safe_select(kwargs.get('zeta', None), Defaults.zeta)
            cutoff_function = safe_select(kwargs.get('cutoff', None),
                                          Defaults.cutoff_function)
            trainable = safe_select(kwargs.get('trainable', None), False)

            transformer = BatchSymmetryFunctionTransformer(
                rc=self._rc,  max_occurs=max_occurs, angular=(k_max == 3),
                nij_max=nij_max, nijk_max=nijk_max, eta=eta, beta=beta,
                gamma=gamma, zeta=zeta, trainable=trainable, periodic=periodic,
                use_stress=stress, use_forces=forces,
                cutoff_function=cutoff_function)
        else:
            transformer = BatchEAMTransformer(
                rc=self._rc, max_occurs=max_occurs, nij_max=nij_max,
                nnl_max=nnl_max, use_forces=forces, use_stress=stress)

        if self._should_compute_atomic_static_energy():
            compute_atomic_static_energy(
                self._database, transformer.elements, True)

        self._transformer = transformer
        self._use_forces = forces
        self._use_stress = stress
        self._max_occurs = max_occurs
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
        write_dir = dirname(filename)
        if not exists(write_dir):
            os.makedirs(write_dir)

        with tf.python_io.TFRecordWriter(filename) as writer:

            num_examples = len(indices)

            if not self._serial:
                n_cpus = min(cpu_count(), 16)
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
                    set_float_precision(_precision)
                    return self._transformer.encode(_atoms)

                for istart, istop in brange(0, num_examples, batch_size):
                    trajectory = []
                    for atoms_id in indices[istart: istop]:
                        trajectory.append(self._database.get_atoms(id=atoms_id))
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
                    atoms = self._database.get_atoms(id=atoms_id)
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
        k_max = self._transformer.k_max
        sig = "k{:d}-rc{:.2f}".format(k_max, self._rc)
        dtype = get_float_dtype()
        if dtype == tf.float32:
            sig += "-fp32"
        else:
            sig += "-fp64"
        return sig

    def to_records(self, savedir, test_size=0.2, verbose=False):
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
        verbose : bool
            If True, the progress shall be logged.

        """
        signature = self._get_signature()

        if isinstance(test_size, (list, tuple)):
            test = list(test_size)
            train = [x for x in range(1, 1 + len(self)) if x not in test]
        else:
            train, test = train_test_split(range(1, 1 + len(self)),
                                           random_state=Defaults.seed,
                                           test_size=test_size)
        assert min(test) >= 1
        assert min(train) >= 1

        self._write_subset(
            tf.estimator.ModeKeys.EVAL,
            check_path(join(savedir, '{}-test-{}-{}.{}.tfrecords'.format(
                self._name, signature, len(test), self._descriptor,))),
            test,
            verbose=verbose,
        )
        self._write_subset(
            tf.estimator.ModeKeys.TRAIN,
            check_path(join(savedir, '{}-train-{}-{}.{}.tfrecords'.format(
                self._name, signature, len(train), self._descriptor,))),
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
            suffix = str(splitext(basename(filename))[0].split('-')[5])
            return int(suffix.split('.')[0])

        def _load(key):
            try:
                files = list(glob.glob(
                    '{}/{}-{}-{}-*.{}.tfrecords'.format(
                        savedir, self._name, key, signature, self._descriptor)))
            except Exception:
                return None, 0
            if len(files) >= 1:
                return files[idx], _get_file_size(files[idx])
            else:
                return None, 0

        test_file, test_size = _load('test')
        train_file, train_size = _load('train')
        success = bool(test_file) and bool(train_file)

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
        if mode == tf.estimator.ModeKeys.PREDICT:
            raise ValueError("The PREDICT does not need an `input_fn`.")

        def _input_fn() -> Tuple[AttributeDict, AttributeDict]:
            """
            The input function for `tf.estimator.Estimator`.

            Returns
            -------
            features : AttributeDict
                A dict of input features.
            labels : AttributeDict
                A dict of labels.

            """
            with tf.name_scope("Dataset"):
                batch = self.next_batch(
                    mode, batch_size=batch_size, num_epochs=num_epochs,
                    shuffle=shuffle)
                for batch_of_tensors in batch.values():
                    shape = batch_of_tensors.shape.as_list()
                    if shape[0] is None:
                        batch_of_tensors.set_shape([batch_size] + shape[1:])
            labels = AttributeDict(energy=batch.pop('y_true'))
            if self._use_forces:
                labels['forces'] = batch.pop('f_true')
            if self._use_stress:
                labels['stress'] = batch.pop('stress')
                labels['total_pressure'] = batch.pop('total_pressure')
            return batch, labels

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
        next_batch : AttributeDict
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
                dataset = dataset.apply(
                    shuffle_and_repeat(min_queue_examples, count=num_epochs,
                                       seed=Defaults.seed))
            else:
                # Repeat the dataset
                dataset = dataset.repeat(count=num_epochs)

            # Setup the batch
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=1)

            # Return the iterator
            return dataset.make_one_shot_iterator().get_next()
