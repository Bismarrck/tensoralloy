# coding=utf-8
"""
This module defines the `Dataset` class for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import sys
import time
from tensorflow.train import Example, Features
from behler import NeighborIndexBuilder
from behler import get_kbody_terms, compute_dimension
from ase import Atoms
from ase.db.sqlite import SQLite3Database
from file_io import find_neighbor_sizes
from misc import RANDOM_STATE, check_path
from sklearn.model_selection import train_test_split
from os.path import join
from typing import List


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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


class Dataset:
    """
    This class is used to manipulate data examples for this project.
    """

    def __init__(self, database, name, k_max=3, rc=6.5, eta=None, beta=None,
                 gamma=None, zeta=None):
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
        eta : List[float]
            A list of float as the `eta` for radial functions.
        beta : List[float]
            A list of float as the `beta` for angular functions.
        gamma : List[float]
            A list of float as the `gamma` for angular functions.
        zeta : List[float]
            A list of float as the `zeta` for angular functions.

        """
        self._database = database
        self._name = name
        self._k_max = k_max
        self._rc = rc
        self._eta = eta
        self._beta = beta
        self._gamma = gamma
        self._zeta = zeta
        self._setup()

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

    def _setup(self):
        """
        Post-initialization.
        """
        find_neighbor_sizes(self._database, self._rc, k_max=self._k_max)

        metadata = self._database.metadata
        max_occurs = metadata['max_occurs']
        nij_max = metadata['nij_max']
        nijk_max = metadata['nijk_max']
        kbody_terms, mapping, elements = get_kbody_terms(
            max_occurs.keys(), k_max=self._k_max)
        total_size, kbody_sizes = compute_dimension(
            kbody_terms, len(self._eta), len(self._beta), len(self._gamma),
            len(self._zeta))
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

    def convert_atoms_to_example(self, atoms: Atoms) -> Example:
        """
        Convert an `Atoms` object to `tf.train.Example`.
        """
        transformer = self._nl.get_index_transformer(atoms)
        positions = transformer.gather(atoms.positions).tostring()
        cell = atoms.cell.reshape((1, 3, 3)).tostring()
        y_true = np.atleast_2d(atoms.get_total_energy()).tostring()
        f_true = transformer.gather(atoms.get_forces()).tostring()
        rslices, aslices = self._nl.get_indexed_slices([atoms])
        feature = {
            'positions': _bytes_feature(positions),
            'cell': _bytes_feature(cell),
            'y_true': _bytes_feature(y_true),
            'f_true': _bytes_feature(f_true),
            'r_v2g': _bytes_feature(rslices.v2g_map.tostring()),
            'r_ilist': _bytes_feature(rslices.ilist.tostring()),
            'r_jlist': _bytes_feature(rslices.jlist.tostring()),
            'r_Slist': _bytes_feature(rslices.Slist.tostring()),
        }
        if self._k_max == 3:
            feature.update({
                'a_v2g': _bytes_feature(aslices.v2g_map.tostring()),
                'a_ij': _bytes_feature(aslices.ik.tostring()),
                'a_ik': _bytes_feature(aslices.ij.tostring()),
                'a_jk': _bytes_feature(aslices.jk.tostring()),
                'a_ijSlist': _bytes_feature(aslices.ijSlist.tostring()),
                'a_ikSlist': _bytes_feature(aslices.ikSlist.tostring()),
                'a_jkSlist': _bytes_feature(aslices.jkSlist.tostring()),
            })
        return Example(features=Features(feature=feature))

    def _write_split(self, filename: str, indices: List[int], verbose=False):
        """
        Write a split of this dataset to the given file.

        Parameters
        ----------
        filename : str
            The file to write.
        indices : List[int]
            A list of int as the ids of the `Atoms` to use for this file.
        verbose : bool


        """
        with tf.python_io.TFRecordWriter(filename) as writer:

            tic = time.time()
            num_examples = len(indices)
            logstr = "\rProgress: {:7d} / {:7d} | Speed = {:6.1f}"

            for i, atoms_id in enumerate(indices):
                atoms = self._database.get_atoms(id=atoms_id)
                example = self.convert_atoms_to_example(atoms)
                writer.write(example.SerializeToString())
                if verbose:
                    speed = (i + 1) / (time.time() - tic)
                    sys.stdout.write(logstr.format(i + 1, num_examples, speed))

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


        """
        test_size = min(max(test_size, 0.0), 1.0)
        train, test = train_test_split(range(1, 1 + len(self)),
                                       random_state=RANDOM_STATE,
                                       test_size=test_size)
        self._write_split(
            check_path(join(savedir, '{}-test.tfrecords'.format(self._name))),
            train,
            verbose=verbose,
        )
        self._write_split(
            check_path(join(savedir, '{}-train.tfrecords'.format(self._name))),
            train,
            verbose=verbose,
        )
