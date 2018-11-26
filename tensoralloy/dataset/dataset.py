from __future__ import print_function, absolute_import

import tensorflow as tf
import glob
import sys
import time
from multiprocessing import cpu_count
from os.path import join, splitext, basename
from typing import Dict, List
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tensorflow.contrib.data import shuffle_and_repeat

from descriptor import BatchDescriptorTransformer
from eam import BatchEAMTransformer
from misc import Defaults, safe_select, brange, check_path, AttributeDict
from tensoralloy.io.neighbor import convert_k_max_to_key, convert_rc_to_key
from tensoralloy.io.neighbor import find_neighbor_size_limits
from tensoralloy.dataset.utils import compute_atomic_static_energy
from transformer import BatchSymmetryFunctionTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class Dataset:
    """
    This class is used to manipulate data examples for this project.
    """

    def __init__(self, database, name, descriptor='behler', k_max=3,
                 rc=Defaults.rc, serial=False, **kwargs):
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
        kwargs : dict
            Key-value args for initializing the `BatchDescriptorTransformer` for
            the `descriptor`.

        """
        assert descriptor in ('behler', 'eam')
        if descriptor == 'eam' and k_max != 2:
            raise ValueError("EAM requires `k_max = 2`.")

        self._database = database
        self._name = name
        self._descriptor = descriptor
        self._k_max = k_max
        self._rc = rc
        self._files = {}
        self._file_sizes = {}
        self._serial = serial
        self._forces = False
        self._stress = False
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
    def forces(self) -> bool:
        """
        Return True if the atomic forces are provided. The unit of the atomic
        forces is 'eV / Angstrom'.
        """
        return self._forces

    @property
    def stress(self) -> bool:
        """
        Return True if the stress tensors are provided. The unit of the stress
        tensors is 'eV/Angstrom'.
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

        details = neighbors[k_max][rc]
        if 'nij_max' not in details:
            return True
        elif 'nijk_max' not in details:
            return True
        elif 'nnl_max' not in details:
            return True

        return False

    def _get_neighbor_sizes(self):
        """
        A helper function to get `nij_max`, `nijk_max` and `nnl_max`.
        """
        k_max = convert_k_max_to_key(self._k_max)
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

        if self._should_find_neighbor_sizes():
            find_neighbor_size_limits(self._database, self._rc, n_jobs=n_jobs,
                                      k_max=self._k_max, verbose=True)

        periodic = self._database.metadata['periodic']
        forces = self._database.metadata['forces']
        stress = self._database.metadata['stress']
        max_occurs = self._database.metadata['max_occurs']

        nij_max, nijk_max, nnl_max = self._get_neighbor_sizes()

        if self._descriptor == 'behler':

            eta = safe_select(kwargs.get('eta', None), Defaults.eta)
            beta = safe_select(kwargs.get('beta', None), Defaults.beta)
            gamma = safe_select(kwargs.get('gamma', None), Defaults.gamma)
            zeta = safe_select(kwargs.get('zeta', None), Defaults.zeta)

            transformer = BatchSymmetryFunctionTransformer(
                rc=self._rc,  max_occurs=max_occurs, k_max=self._k_max,
                nij_max=nij_max, nijk_max=nijk_max, eta=eta, beta=beta,
                gamma=gamma, zeta=zeta, periodic=periodic, stress=stress,
                forces=forces)
        else:
            transformer = BatchEAMTransformer(
                rc=self._rc, max_occurs=max_occurs, nij_max=nij_max,
                nnl_max=nnl_max, forces=forces, stress=stress)

        if self._should_compute_atomic_static_energy():
            compute_atomic_static_energy(
                self._database, transformer.elements, True)

        self._transformer = transformer
        self._forces = forces
        self._stress = stress
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
            suffix = str(splitext(basename(filename))[0].split('-')[4])
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
                                     mask=batch.mask,
                                     volume=batch.volume)
            labels = AttributeDict(energy=batch.y_true)
            if self._forces:
                labels['forces'] = batch.f_true
            if self._stress:
                labels['reduced_stress'] = batch.reduced_stress
                labels['reduced_total_pressure'] = batch.reduced_total_pressure
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