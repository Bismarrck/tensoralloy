# coding=utf-8
"""
This module defines the descriptor transformers.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
from ase import Atoms
from ase.neighborlist import neighbor_list
from collections.__init__ import Counter
from typing import Dict

from tensoralloy.descriptor.behler import SymmetryFunction, BatchSymmetryFunction
from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.descriptor.indexed_slices import G2IndexedSlices, G4IndexedSlices, IndexTransformer
from tensoralloy.transformer.interface import DescriptorTransformer, BatchDescriptorTransformer
from tensoralloy.misc import Defaults, AttributeDict

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

            def _double(name):
                return tf.placeholder(tf.float64, (), name)

            def _double_1d(name):
                return tf.placeholder(tf.float64, (None, ), name)

            def _double_2d(d1, name, d0=None):
                return tf.placeholder(tf.float64, (d0, d1), name)

            def _int(name):
                return tf.placeholder(tf.int32, (), name)

            def _int_1d(name, d0=None):
                return tf.placeholder(tf.int32, (d0, ), name)

            def _int_2d(d1, name, d0=None):
                return tf.placeholder(tf.int32, (d0, d1), name)

            self._placeholders.positions = _double_2d(3, 'positions')
            self._placeholders.cells = _double_2d(d0=3, d1=3, name='cells')
            self._placeholders.n_atoms = _int('n_atoms')
            self._placeholders.volume = _double('volume')
            self._placeholders.mask = _double_1d('mask')
            self._placeholders.composition = _double_1d('composition')
            self._placeholders.row_splits = _int_1d(
                'row_splits', d0=self._n_elements + 1)
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
        shift = np.asarray(Slist, dtype=np.float64)
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
        cells = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        mask = index_transformer.mask
        splits = [1] + [index_transformer.max_occurs[e] for e in self._elements]
        composition = self._get_composition(atoms)

        feed_dict[placeholders.positions] = positions
        feed_dict[placeholders.n_atoms] = n_atoms
        feed_dict[placeholders.mask] = mask
        feed_dict[placeholders.cells] = cells
        feed_dict[placeholders.volume] = volume
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


def _int64_feature(value):
    """
    Convert the `value` to Protobuf int64.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class BatchSymmetryFunctionTransformer(BatchSymmetryFunction,
                                       BatchDescriptorTransformer):
    """
    A batch implementation of `SymmetryFunctionTransformer`.
    """

    def __init__(self, rc, max_occurs: Counter, nij_max: int, nijk_max: int,
                 batch_size=None, eta=Defaults.eta, beta=Defaults.beta,
                 gamma=Defaults.gamma, zeta=Defaults.zeta, k_max=3,
                 periodic=True, forces=True, stress=False):
        """
        Initialization method.

        Notes
        -----
        `batch_size` is set to None by default because tranforming `Atoms` into
        indexed slices does not need this value. However, `batch_size` must be
        set before calling `build_graph()`.
        """
        if (not periodic) and stress:
            raise ValueError(
                'The stress tensor is not applicable to molecules.')

        elements = sorted(max_occurs.keys())

        super(BatchSymmetryFunctionTransformer, self).__init__(
            rc=rc, max_occurs=max_occurs, elements=elements, nij_max=nij_max,
            nijk_max=nijk_max, batch_size=batch_size, eta=eta, beta=beta,
            gamma=gamma, zeta=zeta, k_max=k_max, periodic=periodic)

        self._index_transformers = {}
        self._forces = forces
        self._stress = stress

    @property
    def batch_size(self):
        """
        Return the batch size.
        """
        return self._batch_size

    @property
    def max_occurs(self):
        """
        Return the maximum occurances of the elements.
        """
        return self._max_occurs

    @property
    def forces(self):
        """
        Return True if atomic forces can be calculated.
        """
        return self._forces

    @property
    def stress(self):
        """
        Return True if the stress tensor can be calculated.
        """
        return self._stress

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
        shift = np.asarray(Slist, dtype=np.float64)
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

    @staticmethod
    def _encode_g2_indexed_slices(g2: G2IndexedSlices):
        """
        Encode the indexed slices of G2:
            * `v2g_map`, `ilist` and `jlist` are merged into a single array
              with key 'r_indices'.
            * `shift` will be encoded separately with key 'r_shifts'.

        """
        indices = np.concatenate((
            g2.v2g_map, g2.ilist[..., np.newaxis], g2.jlist[..., np.newaxis],
        ), axis=1).tostring()
        return {'g2.indices': _bytes_feature(indices),
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
        indices = np.concatenate(
            (g4.v2g_map, g4.ij, g4.ik, g4.jk), axis=1).tostring()
        shifts = np.concatenate(
            (g4.ij_shift, g4.ik_shift, g4.jk_shift), axis=1).tostring()
        return {'g4.indices': _bytes_feature(indices),
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
        cells = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        y_true = atoms.get_total_energy()
        composition = self._get_composition(atoms)
        mask = clf.mask.astype(np.float64)
        g2, g4 = self.get_indexed_slices(atoms)
        feature_list = {
            'positions': _bytes_feature(positions.tostring()),
            'cells': _bytes_feature(cells.tostring()),
            'n_atoms': _int64_feature(len(atoms)),
            'volume': _bytes_feature(np.atleast_1d(volume).tostring()),
            'y_true': _bytes_feature(np.atleast_1d(y_true).tostring()),
            'mask': _bytes_feature(mask.tostring()),
            'composition': _bytes_feature(composition.tostring()),
        }
        if self._forces:
            f_true = clf.gather(atoms.get_forces())[1:]
            feature_list['f_true'] = _bytes_feature(f_true.tostring())

        if self._stress:
            # Convert the unit of the stress tensor to 'eV' for simplification:
            # 1 eV/Angstrom**3 = 160.21766208 GPa
            # 1 GPa = 10 kbar
            # reduced_stress (eV) = stress * volume
            virial = atoms.get_stress(voigt=True) * volume
            total_pressure = virial[:3].mean()
            feature_list['reduced_stress'] = _bytes_feature(virial.tostring())
            feature_list['reduced_total_pressure'] = _bytes_feature(
                np.atleast_1d(total_pressure).tostring())

        feature_list.update(self._encode_g2_indexed_slices(g2))

        if self.k_max == 3:
            feature_list.update(self._encode_g4_indexed_slices(g4))

        return tf.train.Example(
            features=tf.train.Features(feature=feature_list))

    def _decode_atoms(self, example: Dict[str, tf.Tensor]) -> AttributeDict:
        """
        Decode `Atoms` related properties.
        """
        decoded = AttributeDict()

        length = 3 * self._max_n_atoms

        positions = tf.decode_raw(example['positions'], tf.float64)
        positions.set_shape([length])
        decoded.positions = tf.reshape(
            positions, (self._max_n_atoms, 3), name='R')

        n_atoms = tf.identity(example['n_atoms'], name='n_atoms')
        decoded.n_atoms = n_atoms

        y_true = tf.decode_raw(example['y_true'], tf.float64)
        y_true.set_shape([1])
        decoded.y_true = tf.squeeze(y_true, name='y_true')

        cells = tf.decode_raw(example['cells'], tf.float64)
        cells.set_shape([9])
        decoded.cells = tf.reshape(cells, (3, 3), name='cells')

        volume = tf.decode_raw(example['volume'], tf.float64)
        volume.set_shape([1])
        decoded.volume = tf.squeeze(volume, name='volume')

        mask = tf.decode_raw(example['mask'], tf.float64)
        mask.set_shape([self._max_n_atoms, ])
        decoded.mask = mask

        composition = tf.decode_raw(example['composition'], tf.float64)
        composition.set_shape([self._n_elements, ])
        decoded.composition = composition

        if self._forces:
            f_true = tf.decode_raw(example['f_true'], tf.float64)
            # Ignore the forces of the virtual atom
            f_true.set_shape([length - 3])
            decoded.f_true = tf.reshape(
                f_true, (self._max_n_atoms - 1, 3), name='f_true')

        if self._stress:
            reduced_stress = tf.decode_raw(
                example['reduced_stress'], tf.float64, name='stress')
            reduced_stress.set_shape([6])
            decoded.reduced_stress = reduced_stress

            reduced_total_pressure = tf.decode_raw(
                example['reduced_total_pressure'], tf.float64, name='stress')
            reduced_total_pressure.set_shape([1])
            decoded.reduced_total_pressure = reduced_total_pressure

        return decoded

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
                indices.set_shape([self._nijk_max * 9])
                indices = tf.reshape(
                    indices, [self._nijk_max, 9], name='g4.indices')
                v2g_map, ij, ik, jk = \
                    tf.split(indices, [3, 2, 2, 2], axis=1, name='splits')

            with tf.name_scope("shifts"):
                shifts = tf.decode_raw(example['g4.shifts'], tf.float64)
                shifts.set_shape([self._nijk_max * 9])
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
        decoded = self._decode_atoms(example)
        g2 = self._decode_g2_indexed_slices(example)
        g4 = self._decode_g4_indexed_slices(example)

        decoded.rv2g = g2.v2g_map
        decoded.ilist = g2.ilist
        decoded.jlist = g2.jlist
        decoded.shift = g2.shift

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
                'n_atoms': tf.FixedLenFeature([], tf.int64),
                'cells': tf.FixedLenFeature([], tf.string),
                'volume': tf.FixedLenFeature([], tf.string),
                'y_true': tf.FixedLenFeature([], tf.string),
                'g2.indices': tf.FixedLenFeature([], tf.string),
                'g2.shifts': tf.FixedLenFeature([], tf.string),
                'mask': tf.FixedLenFeature([], tf.string),
                'composition': tf.FixedLenFeature([], tf.string),
            }
            if self._forces:
                feature_list['f_true'] = tf.FixedLenFeature([], tf.string)

            if self._stress:
                feature_list['reduced_stress'] = \
                    tf.FixedLenFeature([], tf.string)
                feature_list['reduced_total_pressure'] = \
                    tf.FixedLenFeature([], tf.string)

            if self._k_max == 3:
                feature_list.update({
                    'g4.indices': tf.FixedLenFeature([], tf.string),
                    'g4.shifts': tf.FixedLenFeature([], tf.string),
                })

            example = tf.parse_single_example(example_proto, feature_list)
            return self._decode_example(example)

    def get_graph_from_batch(self, batch: AttributeDict, batch_size: int):
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
            * 'cells': float64, [batch_size, 3, 3]
            * 'volume': float64, [batch_size, ]
            * 'n_atoms': int64, [batch_size, ]
            * 'y_true': float64, [batch_size, ]
            * 'f_true': float64, [batch_size, max_n_atoms - 1, 3]
            * 'composition': float64, [batch_size, n_elements]
            * 'mask': float64, [batch_size, max_n_atoms]
            * 'ilist': int32, [batch_size, nij_max]
            * 'jlist': int32, [batch_size, nij_max]
            * 'shift': float64, [batch_size, nij_max, 3]
            * 'rv2g': int32, [batch_size, nij_max, 3]

            If `self.stress` is `True`, the following keys are provided:

            * 'reduced_stress': float64, [batch_size, 6]
            * 'total_pressure': float64, [batch_size, ]

            These keys will only be valid if G4 functions are used:

            * 'ij': int32, [batch_size, nijk_max, 2]
            * 'ik': int32, [batch_size, nijk_max, 2]
            * 'jk': int32, [batch_size, nijk_max, 2]
            * 'ij_shift': float64, [batch_size, nijk_max, 3]
            * 'ik_shift': float64, [batch_size, nijk_max, 3]
            * 'jk_shift': float64, [batch_size, nijk_max, 3]
            * 'av2g': int32, [batch_size, nijk_max, 3]

        batch_size : int
            The size of the batch.

        Returns
        -------
        g : tf.Tensor
            The tensor of the computed symmetry function descriptors for the
            given batch of examples.

        """
        self._batch_size = batch_size

        inputs = AttributeDict()
        inputs.g2 = AttributeDict(
            ilist=batch.ilist, jlist=batch.jlist,
            shift=batch.shift, v2g_map=batch.rv2g
        )
        inputs.positions = batch.positions
        inputs.cells = batch.cells
        inputs.volume = batch.volume

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

    def get_descriptor_normalization_weights(self, method):
        """
        Return the initial weights for the column-wise normalising the output
        descriptors.
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
            values = np.asarray(values)
            weights[element] = 0.25 / np.exp(-values * 0.25**2)
        return weights
