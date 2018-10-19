# coding=utf-8
"""
This module defines the `Dataset` class for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from behler import IndexTransformer, build_radial_v2g_map, build_angular_v2g_map
from behler import get_kbody_terms, compute_dimension
from ase import Atoms
from ase.db.sqlite import SQLite3Database
from file_io import find_neighbor_sizes
from typing import List


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class Dataset:
    """

    """

    def __init__(self, database, name, rc=6.5, eta=None, beta=None, gamma=None,
                 zeta=None):
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

    def _setup(self):
        """
        Post-initialization.
        """
        find_neighbor_sizes(self._database, self._rc)
        metadata = self._database.metadata
        max_occurs = metadata['max_occurs']

        self._expanded_elements = sorted(max_occurs.values())
        self._max_occurs = max_occurs
        self._nij_max = metadata['nij_max']
        self._nijk_max = metadata['nijk_max']
        self._index_transformers = {}
        self._kbody_terms = get_kbody_terms(max_occurs.keys())

    def get_index_transformer(self, atoms: Atoms) -> IndexTransformer:
        """
        Return the
        """
        formula = atoms.get_chemical_formula()
        symbols = atoms.get_chemical_symbols()
        if formula not in self._index_transformers:
            clf = IndexTransformer(self._max_occurs, symbols)
            self._index_transformers[formula] = clf
        return self._index_transformers[formula]

    def convert_atoms_to_records(self, atoms: Atoms):
        """

        """
        transformer = self.get_index_transformer(atoms)
        positions = transformer.map(atoms.positions)
        y_true = atoms.get_total_energy()
        f_true = transformer.map(atoms.get_forces())

        rmap = build_radial_v2g_map([Atoms], rc=self._rc, n_etas=len(self._eta),
                                    nij_max=self._nij_max, kbody_terms=self._kbody_terms)
