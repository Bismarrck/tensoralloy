# coding=utf-8
"""
This module defines the descriptor transformers.
"""
from __future__ import print_function, absolute_import

from collections.__init__ import Counter
from typing import List, Dict

import numpy as np
import tensorflow as tf
from ase import Atoms
from ase.neighborlist import neighbor_list

from behler import SymmetryFunction, BatchSymmetryFunction
from behler import get_elements_from_kbody_term
from behler import IndexTransformer, G2IndexedSlices, G4IndexedSlices
from descriptor import DescriptorTransformer, BatchDescriptorTransformer
from misc import Defaults, AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class SymmetryFunctionTransformer(SymmetryFunction, DescriptorTransformer):
    """
    A tranformer for providing the feed dict to calculate symmetry function
    descriptors of an `Atoms` object.
    """

    def __init__(self, rc, elements, eta=Defaults.eta, beta=Defaults.beta,
                 gamma=Defaults.gamma, zeta=Defaults.zeta, k_max=3,
                 periodic=True):
        """
        Initialization method.
        """
        super(SymmetryFunctionTransformer, self).__init__(
            rc=rc, elements=elements, eta=eta, beta=beta, gamma=gamma,
            zeta=zeta, k_max=k_max, periodic=periodic)
        self._index_transformers = {}
        self._placeholders = AttributeDict()

    def get_graph(self, **kwargs):
        """
        Return the graph to compute symmetry function descriptors.
        """
        if not self._placeholders:
            self._initialize_placeholders()
        return self.build_graph(self.placeholders, **kwargs)

    @property
    def placeholders(self) -> AttributeDict:
        """
        Return a dict of placeholders.
        """
        return self._placeholders

    def _initialize_placeholders(self):
        """
        Initialize the placeholders.
        """
        with tf.name_scope("Placeholders"):

            def _double_1d(name):
                return tf.placeholder(tf.float64, (None, ), name)

            def _double_2d(ndim, name):
                return tf.placeholder(tf.float64, (None, ndim), name)

            def _int(name):
                return tf.placeholder(tf.int32, (), name)

            def _int_1d(name, length=None):
                return tf.placeholder(tf.int32, (length, ), name)

            def _int_2d(ndim, name):
                return tf.placeholder(tf.int32, (None, ndim), name)

            self._placeholders.positions = _double_2d(3, 'positions')
            self._placeholders.n_atoms = _int('n_atoms')
            self._placeholders.mask = _double_1d('mask')
            self._placeholders.composition = _double_1d('composition')
            self._placeholders.row_splits = _int_1d(
                'row_splits', length=self._n_elements + 1)
            self._placeholders.g2 = AttributeDict(
                ilist=_int_1d('g2.ilist'),
                jlist=_int_1d('g2.jlist'),
                shift=_double_2d(3, 'g2.shift'),
                v2g_map=_int_2d(2, 'g2.v2g_map')
            )

            if self._k_max == 3:
                self._placeholders.g4 = AttributeDict(
                    v2g_map=_int_2d(2, 'g2.v2g_map'),
                    ij=AttributeDict(ilist=_int_1d('g4.ij.ilist'),
                                     jlist=_int_1d('g4.ij.jlist')),
                    ik=AttributeDict(ilist=_int_1d('g4.ik.ilist'),
                                     klist=_int_1d('g4.ik.klist')),
                    jk=AttributeDict(jlist=_int_1d('g4.jk.jlist'),
                                     klist=_int_1d('g4.jk.klist')),
                    shift=AttributeDict(ij=_double_2d(3, 'g4.shift.ij'),
                                        ik=_double_2d(3, 'g4.shift.ik'),
                                        jk=_double_2d(3, 'g4.shift.jk'))
                )
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
            for element in self._elements:
                max_occurs[element] = max(1, counter[element])
            self._index_transformers[formula] = IndexTransformer(
                max_occurs, symbols
            )
        return self._index_transformers[formula]

    def _get_g2_indexed_slices(self, atoms, index_transformer: IndexTransformer):
        """
        Return the indexed slices for the symmetry function G2.
        """
        symbols = atoms.get_chemical_symbols()
        ilist, jlist, Slist = neighbor_list('ijS', atoms, self._rc)
        if self._k_max == 1:
            cols = [i for i in range(len(ilist))
                    if symbols[ilist[i]] == symbols[jlist[i]]]
            ilist = ilist[cols]
            jlist = jlist[cols]
            Slist = Slist[cols]
        nij = len(ilist)
        v2g_map = np.zeros((nij, 2), dtype=np.int32)

        tlist = np.zeros(nij, dtype=np.int32)
        for i in range(nij):
            symboli = symbols[ilist[i]]
            symbolj = symbols[jlist[i]]
            tlist[i] = self._kbody_index['{}{}'.format(symboli, symbolj)]

        ilist = index_transformer.map(ilist + 1)
        jlist = index_transformer.map(jlist + 1)
        shift = Slist @ atoms.cell
        v2g_map[:, 0] = ilist
        v2g_map[:, 1] = self._offsets[tlist]
        return G2IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               shift=shift)

    def _get_g4_indexed_slices(self, atoms: Atoms, g2: G2IndexedSlices,
                               index_transformer: IndexTransformer):
        """
        Return the indexed slices for the symmetry function G4.
        """
        symbols = atoms.get_chemical_symbols()
        indices = {}
        vectors = {}
        for i, atomi in enumerate(g2.ilist):
            if atomi not in indices:
                indices[atomi] = []
                vectors[atomi] = []
            indices[atomi].append(g2.jlist[i])
            vectors[atomi].append(g2.shift[i])

        nijk = 0
        for atomi, nl in indices.items():
            n = len(nl)
            nijk += (n - 1) * n // 2

        v2g_map = np.zeros((nijk, 2), dtype=np.int32)
        ij = np.zeros((nijk, 2), dtype=np.int32)
        ik = np.zeros((nijk, 2), dtype=np.int32)
        jk = np.zeros((nijk, 2), dtype=np.int32)
        ij_shift = np.zeros((nijk, 3), dtype=np.float64)
        ik_shift = np.zeros((nijk, 3), dtype=np.float64)
        jk_shift = np.zeros((nijk, 3), dtype=np.float64)

        count = 0
        for atomi, nl in indices.items():
            num = len(nl)
            symboli = symbols[index_transformer.map(atomi, True, True)]
            prefix = '{}'.format(symboli)
            for j in range(num):
                atomj = nl[j]
                symbolj = symbols[index_transformer.map(atomj, True, True)]
                for k in range(j + 1, num):
                    atomk = nl[k]
                    symbolk = symbols[index_transformer.map(atomk, True, True)]
                    suffix = ''.join(sorted([symbolj, symbolk]))
                    term = '{}{}'.format(prefix, suffix)
                    ij[count] = atomi, atomj
                    ik[count] = atomi, atomk
                    jk[count] = atomj, atomk
                    ij_shift[count] = vectors[atomi][j]
                    ik_shift[count] = vectors[atomi][k]
                    jk_shift[count] = vectors[atomi][k] - vectors[atomi][j]
                    index = self._kbody_index[term]
                    v2g_map[count, 0] = atomi
                    v2g_map[count, 1] = self._offsets[index]
                    count += 1
        return G4IndexedSlices(v2g_map=v2g_map, ij=ij, ik=ik, jk=jk,
                               ij_shift=ij_shift, ik_shift=ik_shift,
                               jk_shift=jk_shift)

    def _get_composition(self, atoms: Atoms) -> np.ndarray:
        """
        Return the composition of the `Atoms`.
        """
        composition = np.zeros(self._n_elements, dtype=np.float64)
        for element, count in Counter(atoms.get_chemical_symbols()).items():
            composition[self._elements.index(element)] = float(count)
        return composition

    def get_feed_dict(self, atoms: Atoms):
        """
        Return the feed dict.
        """
        feed_dict = {}

        if not self._placeholders:
            self._initialize_placeholders()
        placeholders = self._placeholders

        index_transformer = self.get_index_transformer(atoms)
        g2 = self._get_g2_indexed_slices(atoms, index_transformer)

        positions = index_transformer.gather(atoms.positions)
        n_atoms = index_transformer.n_atoms
        mask = index_transformer.mask
        splits = [1] + [index_transformer.max_occurs[e] for e in self._elements]
        composition = self._get_composition(atoms)

        feed_dict[placeholders.positions] = positions
        feed_dict[placeholders.n_atoms] = n_atoms
        feed_dict[placeholders.mask] = mask
        feed_dict[placeholders.composition] = composition
        feed_dict[placeholders.row_splits] = splits
        feed_dict[placeholders.g2.v2g_map] = g2.v2g_map
        feed_dict[placeholders.g2.ilist] = g2.ilist
        feed_dict[placeholders.g2.jlist] = g2.jlist
        feed_dict[placeholders.g2.shift] = g2.shift

        if self._k_max == 3:
            g4 = self._get_g4_indexed_slices(atoms, g2, index_transformer)
            feed_dict[placeholders.g4.v2g_map] = g4.v2g_map
            feed_dict[placeholders.g4.ij.ilist] = g4.ij[:, 0]
            feed_dict[placeholders.g4.ij.jlist] = g4.ij[:, 1]
            feed_dict[placeholders.g4.ik.ilist] = g4.ik[:, 0]
            feed_dict[placeholders.g4.ik.klist] = g4.ik[:, 1]
            feed_dict[placeholders.g4.jk.jlist] = g4.jk[:, 0]
            feed_dict[placeholders.g4.jk.klist] = g4.jk[:, 1]
            feed_dict[placeholders.g4.shift.ij] = g4.ij_shift
            feed_dict[placeholders.g4.shift.ik] = g4.ik_shift
            feed_dict[placeholders.g4.shift.jk] = g4.jk_shift

        return feed_dict


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


class BatchSymmetryFunctionTransformer(BatchSymmetryFunction,
                                       BatchDescriptorTransformer):
    """
    A batch implementation of `SymmetryFunctionTransformer`.
    """

    def __init__(self, rc, max_occurs: Counter, elements: List[str],
                 nij_max: int, nijk_max: int, batch_size=None, eta=Defaults.eta,
                 beta=Defaults.beta, gamma=Defaults.gamma, zeta=Defaults.zeta,
                 k_max=3, periodic=True):
        """
        Initialization method.

        Notes
        -----
        `batch_size` is set to None by default because tranforming `Atoms` into
        indexed slices does not need this value. However, `batch_size` must be
        set before calling `build_graph()`.
        """
        super(BatchSymmetryFunctionTransformer, self).__init__(
            rc=rc, max_occurs=max_occurs, elements=elements, nij_max=nij_max,
            nijk_max=nijk_max, batch_size=batch_size, eta=eta, beta=beta,
            gamma=gamma, zeta=zeta, k_max=k_max, periodic=periodic)
        self._index_transformers = {}

    @property
    def batch_size(self):
        """
        Return the batch size.
        """
        return self._batch_size

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
                self._max_occurs, atoms.get_chemical_symbols()
            )
        return self._index_transformers[formula]

    def _resize_to_nij_max(self, alist: np.ndarray, is_indices=True):
        """
        A helper function to resize the given array.
        """
        if np.ndim(alist) == 1:
            shape = [self._nij_max, ]
        else:
            shape = [self._nij_max, ] + list(alist.shape[1:])
        nlist = np.zeros(shape, dtype=np.int32)
        length = len(alist)
        nlist[:length] = alist
        if is_indices:
            nlist[:length] += 1
        return nlist

    def get_g2_indexed_slices(self, atoms: Atoms):
        """
        Return the indexed slices for the radial function G2.
        """
        v2g_map = np.zeros((self._nij_max, 3), dtype=np.int32)
        tlist = np.zeros(self._nij_max, dtype=np.int32)

        symbols = atoms.get_chemical_symbols()
        transformer = self.get_index_transformer(atoms)

        ilist, jlist, Slist = neighbor_list('ijS', atoms, self._rc)
        if self._k_max == 1:
            cols = [i for i in range(len(ilist))
                    if symbols[ilist[i]] == symbols[jlist[i]]]
            ilist = ilist[cols]
            jlist = jlist[cols]
            Slist = Slist[cols]
        nij = len(ilist)

        tlist.fill(0)
        for i in range(nij):
            symboli = symbols[ilist[i]]
            symbolj = symbols[jlist[i]]
            tlist[i] = self._kbody_index['{}{}'.format(symboli, symbolj)]

        ilist = self._resize_to_nij_max(ilist, True)
        jlist = self._resize_to_nij_max(jlist, True)
        Slist = self._resize_to_nij_max(Slist, False)
        ilist = transformer.map(ilist)
        jlist = transformer.map(jlist)
        shift = Slist @ atoms.cell
        v2g_map[:self._nij_max, 1] = ilist
        v2g_map[:self._nij_max, 2] = self._offsets[tlist]

        return G2IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               shift=shift)

    def get_g4_indexed_slices(self, atoms: Atoms, g2: G2IndexedSlices):
        """
        Return the indexed slices for the angular function G4.
        """
        if self._k_max < 3:
            return None

        v2g_map = np.zeros((self._nijk_max, 3), dtype=np.int32)
        ij = np.zeros((self._nijk_max, 2), dtype=np.int32)
        ik = np.zeros((self._nijk_max, 2), dtype=np.int32)
        jk = np.zeros((self._nijk_max, 2), dtype=np.int32)
        ij_shift = np.zeros((self._nijk_max, 3), dtype=np.float64)
        ik_shift = np.zeros((self._nijk_max, 3), dtype=np.float64)
        jk_shift = np.zeros((self._nijk_max, 3), dtype=np.float64)

        symbols = atoms.get_chemical_symbols()
        transformer = self.get_index_transformer(atoms)
        indices = {}
        vectors = {}
        for i, atomi in enumerate(g2.ilist):
            if atomi == 0:
                break
            if atomi not in indices:
                indices[atomi] = []
                vectors[atomi] = []
            indices[atomi].append(g2.jlist[i])
            vectors[atomi].append(g2.shift[i])

        count = 0
        for atomi, nl in indices.items():
            num = len(nl)
            symboli = symbols[transformer.map(atomi, True, True)]
            prefix = '{}'.format(symboli)
            for j in range(num):
                atomj = nl[j]
                symbolj = symbols[transformer.map(atomj, True, True)]
                for k in range(j + 1, num):
                    atomk = nl[k]
                    symbolk = symbols[transformer.map(atomk, True, True)]
                    suffix = ''.join(sorted([symbolj, symbolk]))
                    term = '{}{}'.format(prefix, suffix)
                    ij[count] = atomi, atomj
                    ik[count] = atomi, atomk
                    jk[count] = atomj, atomk
                    ij_shift[count] = vectors[atomi][j]
                    ik_shift[count] = vectors[atomi][k]
                    jk_shift[count] = vectors[atomi][k] - vectors[atomi][j]
                    index = self._kbody_index[term]
                    v2g_map[count, 1] = atomi
                    v2g_map[count, 2] = self._offsets[index]
                    count += 1
        return G4IndexedSlices(v2g_map=v2g_map, ij=ij, ik=ik, jk=jk,
                               ij_shift=ij_shift, ik_shift=ik_shift,
                               jk_shift=jk_shift)

    def get_indexed_slices(self, atoms: Atoms):
        """
        Return both the radial and angular indexed slices for the trajectory.
        """
        g2 = self.get_g2_indexed_slices(atoms)
        g4 = self.get_g4_indexed_slices(atoms, g2)
        return g2, g4

    def get_initial_weights_for_normalizers(self) -> Dict[str, np.ndarray]:
        """
        Return the initial weights for the `arctan` input normalizers.
        """
        weights = {}
        for element in self._elements:
            kbody_terms = self._mapping[element]
            values = []
            for kbody_term in kbody_terms:
                if len(get_elements_from_kbody_term(kbody_term)) == 2:
                    values.extend(self._eta.tolist())
                else:
                    for p in self._parameter_grid:
                        values.append(p['beta'])
            weights[element] = np.exp(-np.asarray(values) / 20.0)
        return weights

    @staticmethod
    def _encode_g2_indexed_slices(g2: G2IndexedSlices):
        """
        Encode the indexed slices of G2:
            * `v2g_map`, `ilist` and `jlist` are merged into a single array
              with key 'r_indices'.
            * `shift` will be encoded separately with key 'r_shifts'.

        """
        merged = np.concatenate((
            g2.v2g_map, g2.ilist[..., np.newaxis], g2.jlist[..., np.newaxis],
        ), axis=2).tostring()
        return {'g2.indices': _bytes_feature(merged),
                'g2.shifts': _bytes_feature(g2.shift.tostring())}

    @staticmethod
    def _encode_g4_indexed_slices(g4: G4IndexedSlices):
        """
        Encode the indexed slices of G4:
            * `v2g_map`, `ij`, `ik` and `jk` are merged into a single array
              with key 'a_indices'.
            * `ij_shift`, `ik_shift` and `jk_shift` are merged into another
              array with key 'a_shifts'.

        """
        merged = np.concatenate(
            (g4.v2g_map, g4.ij, g4.ik, g4.jk), axis=2).tostring()
        shifts = np.concatenate(
            (g4.ij_shift, g4.ik_shift, g4.jk_shift), axis=2).tostring()
        return {'g4.indices': _bytes_feature(merged),
                'g4.shifts': _bytes_feature(shifts)}

    def _get_composition(self, atoms: Atoms) -> np.ndarray:
        """
        Return the composition of the `Atoms`.
        """
        composition = np.zeros(self._n_elements, dtype=np.float64)
        for element, count in Counter(atoms.get_chemical_symbols()).items():
            composition[self._elements.index(element)] = float(count)
        return composition

    def encode(self, atoms: Atoms):
        """
        Encode the `Atoms` object and return a `tf.train.Example`.
        """
        clf = self.get_index_transformer(atoms)
        positions = clf.gather(atoms.positions)
        y_true = atoms.get_total_energy()
        f_true = atoms.get_forces()
        composition = self._get_composition(atoms)
        g2, g4 = self.get_indexed_slices(atoms)
        feature_list = {
            'positions': _bytes_feature(positions.tostring()),
            'y_true': _bytes_feature(np.atleast_2d(y_true).tostring()),
            'mask': _bytes_feature(clf.mask.tostring()),
            'composition': _bytes_feature(composition.tostring()),
            'f_true': _bytes_feature(f_true.tostring()),
        }
        feature_list.update(self._encode_g2_indexed_slices(g2))
        if self.k_max == 3:
            feature_list.update(self._encode_g4_indexed_slices(g4))
        return tf.train.Example(
            features=tf.train.Features(feature=feature_list))

    def _decode_atoms(self, example: Dict[str, tf.Tensor]):
        """
        Decode `Atoms` related properties.
        """
        length = 3 * self._max_n_atoms

        positions = tf.decode_raw(example['positions'], tf.float64)
        positions.set_shape([length])
        positions = tf.reshape(positions, (self._max_n_atoms, 3), name='R')

        y_true = tf.decode_raw(example['y_true'], tf.float64)
        y_true.set_shape([1])
        y_true = tf.squeeze(y_true, name='y_true')

        mask = tf.decode_raw(example['mask'], tf.float64)
        mask.set_shape([self._max_n_atoms, ])

        composition = tf.decode_raw(example['composition'], tf.float64)
        composition.set_shape([self._n_elements, ])

        f_true = tf.decode_raw(example['f_true'], tf.float64)
        f_true.set_shape([length])
        f_true = tf.reshape(f_true, (self._max_n_atoms, 3), name='f_true')

        return positions, y_true, f_true, mask, composition

    def _decode_g2_indexed_slices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ilist, jlist and Slist for radial functions.
        """
        with tf.name_scope("G2"):
            indices = tf.decode_raw(example['g2.indices'], tf.int32)
            indices.set_shape([self._nij_max * 5])
            indices = tf.reshape(
                indices, [self._nij_max, 5], name='g2.indices')
            v2g_map, ilist, jlist = tf.split(
                indices, [3, 1, 1], axis=1, name='splits')
            ilist = tf.squeeze(ilist, axis=1, name='ilist')
            jlist = tf.squeeze(jlist, axis=1, name='jlist')

            shift = tf.decode_raw(example['g2.shifts'], tf.float64)
            shift.set_shape([self._nij_max * 3])
            shift = tf.reshape(shift, [self._nij_max, 3], name='shift')

            return G2IndexedSlices(v2g_map, ilist, jlist, shift)

    def _decode_g4_indexed_slices(self, example: Dict[str, tf.Tensor]):
        """
        Decode v2g_map, ij, ik, jk, ijSlist, ikSlist and jkSlist for angular
        functions.
        """
        if self.k_max < 3:
            return None

        with tf.name_scope("G4"):
            with tf.name_scope("indices"):
                indices = tf.decode_raw(example['g4.indices'], tf.int32)
                indices.set_shape([self.nijk_max * 9])
                indices = tf.reshape(
                    indices, [self._nijk_max, 9], name='g4.indices')
                v2g_map, ij, ik, jk = \
                    tf.split(indices, [3, 2, 2, 2], axis=1, name='splits')

            with tf.name_scope("shifts"):
                shifts = tf.decode_raw(example['g4.shifts'], tf.float64)
                shifts.set_shape([self.nijk_max * 9])
                shifts = tf.reshape(
                    shifts, [self._nijk_max, 9], name='g4.shifts')
                ij_shift, ik_shift, jk_shift = \
                    tf.split(shifts, [3, 3, 3], axis=1, name='splits')

        return G4IndexedSlices(v2g_map, ij, ik, jk, ij_shift, ik_shift,
                               jk_shift)

    def _decode_example(self, example: Dict[str, tf.Tensor]):
        """
        Decode the parsed single example.
        """
        positions, y_true, f_true, mask, composition = \
            self._decode_atoms(example)
        g2 = self._decode_g2_indexed_slices(example)
        g4 = self._decode_g4_indexed_slices(example)

        decoded = AttributeDict(
            positions=positions, y_true=y_true, f_true=f_true, mask=mask,
            composition=composition, rv2g=g2.v2g_map, ilist=g2.ilist,
            jlist=g2.jlist, shift=g2.shift)

        if g4 is not None:
            decoded.av2g = g4.v2g_map
            decoded.ij = g4.ij
            decoded.ik = g4.ik
            decoded.jk = g4.jk
            decoded.ij_shift = g4.ij_shift
            decoded.ik_shift = g4.ik_shift
            decoded.jk_shift = g4.jk_shift

        return decoded

    def decode_protobuf(self, example_proto: tf.Tensor) -> AttributeDict:
        """
        Decode the scalar string Tensor, which is a single serialized Example.
        See `_parse_single_example_raw` documentation for more details.
        """
        with tf.name_scope("decoding"):
            feature_list = {
                'positions': tf.FixedLenFeature([], tf.string),
                'y_true': tf.FixedLenFeature([], tf.string),
                'f_true': tf.FixedLenFeature([], tf.string),
                'g2.indices': tf.FixedLenFeature([], tf.string),
                'g2.shifts': tf.FixedLenFeature([], tf.string),
                'mask': tf.FixedLenFeature([], tf.string),
                'composition': tf.FixedLenFeature([], tf.string),
            }
            if self._k_max == 3:
                feature_list.update({
                    'g4.indices': tf.FixedLenFeature([], tf.string),
                    'g4.shifts': tf.FixedLenFeature([], tf.string),
                })
            example = tf.parse_single_example(example_proto, feature_list)
            return self._decode_example(example)

    def get_graph_from_batch(self, batch: AttributeDict):
        """
        Return the graph for calculating symmetry function descriptors for the
        given batch of examples.

        This function is necessary because nested dicts are not supported by
        `tf.data.Dataset.batch`.

        Parameters
        ----------
        batch : AttributeDict
            A of batch examples produced by `tf.data.Dataset`. Each example is
            produced by the function `decode_protobuf`.

            Here are default keys:

            * 'positions': float64, [batch_size, max_n_atoms, 3]
            * 'y_true': float64, [batch_size, ]
            * 'f_true': float64, [batch_size, 3]
            * 'composition': float64, [batch_size, n_elements]
            * 'mask': float64, [batch_size, max_n_atoms]
            * 'ilist': int32, [batch_size, nij_max]
            * 'jlist': int32, [batch_size, nij_max]
            * 'shift': float64, [batch_size, nij_max, 3]
            * 'rv2g': int32, [batch_size, nij_max, 3]

            These keys will only be valid if G4 functions are used:

            * 'ij': int32, [batch_size, nijk_max, 2]
            * 'ik': int32, [batch_size, nijk_max, 2]
            * 'jk': int32, [batch_size, nijk_max, 2]
            * 'ij_shift': float64, [batch_size, nijk_max, 3]
            * 'ik_shift': float64, [batch_size, nijk_max, 3]
            * 'jk_shift': float64, [batch_size, nijk_max, 3]
            * 'av2g': int32, [batch_size, nijk_max, 3]

        Returns
        -------
        g : tf.Tensor
            The tensor of the computed symmetry function descriptors for the
            given batch of examples.

        """
        if not self._batch_size:
            batch_size = batch.positions.shape[0]
            if not batch_size:
                raise ValueError('Batch size cannot be inferred')
            self._batch_size = batch_size

        inputs = AttributeDict()
        inputs.g2 = AttributeDict(
            ilist=batch.ilist, jlist=batch.jlist,
            shift=batch.shift, v2g_map=batch.rv2g
        )
        inputs.positions = batch.positions

        if self._k_max == 3:
            inputs.g4 = AttributeDict(
                ij=AttributeDict(
                    ilist=batch.ij[..., 0], jlist=batch.ij[..., 1]),
                ik=AttributeDict(
                    ilist=batch.ik[..., 0], klist=batch.ik[..., 1]),
                jk=AttributeDict(
                    jlist=batch.jk[..., 0], klist=batch.jk[..., 1]),
                shift=AttributeDict(
                    ij=batch.ij_shift, ik=batch.ik_shift, jk=batch.jk_shift,),
                v2g_map=batch.av2g,
            )
        return self.build_graph(inputs)
