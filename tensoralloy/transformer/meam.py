#!coding=utf-8
"""
Descriptor transformers of MEAM.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from ase import Atoms
from ase.neighborlist import neighbor_list
from typing import List, Dict
from collections import Counter

from tensoralloy.descriptor.meam import MEAM
from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer import IndexTransformer
from tensoralloy.transformer.indexed_slices import G2IndexedSlices
from tensoralloy.transformer.indexed_slices import G4IndexedSlices
from tensoralloy.precision import get_float_dtype
from tensoralloy.utils import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class MeamTransformer(MEAM, DescriptorTransformer):
    """
    The feature transformer for the EAM model.
    """

    def __init__(self, rc: float, elements: List[str], angular_rc_scale=1.0):
        """
        Initialization method.
        """
        MEAM.__init__(self,
                      rc=rc,
                      elements=elements,
                      angular_rc_scale=angular_rc_scale)
        DescriptorTransformer.__init__(self)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this transformer.
        """
        d = {'class': self.__class__.__name__,
             'rc': self._rc,
             'elements': self._elements,
             'angular_rc_scale': self._angular_rc_scale}
        return d

    def _initialize_placeholders(self):
        """
        Initialize the placeholders.
        """
        graph = tf.get_default_graph()

        # Make sure the all placeholder ops are placed under the absolute path
        # of 'Placeholders/'. Placeholder ops can be recovered from graph
        # directly.
        with tf.name_scope("Placeholders/"):

            def _get_or_create(dtype, shape, name):
                try:
                    return graph.get_tensor_by_name(f'Placeholders/{name}:0')
                except KeyError:
                    return tf.placeholder(dtype, shape, name)
                except Exception as excp:
                    raise excp

            float_dtype = get_float_dtype()

            def _float(name):
                return _get_or_create(float_dtype, (), name)

            def _float_1d(name):
                return _get_or_create(float_dtype, (None, ), name)

            def _float_2d(d1, name, d0=None):
                return _get_or_create(float_dtype, (d0, d1), name)

            def _int(name):
                return _get_or_create(tf.int32, (), name)

            def _int_1d(name, d0=None):
                return _get_or_create(tf.int32, (d0, ), name)

            def _int_2d(d1, name, d0=None):
                return _get_or_create(tf.int32, (d0, d1), name)

            self._placeholders.positions = _float_2d(3, 'positions')
            self._placeholders.cells = _float_2d(d0=3, d1=3, name='cells')
            self._placeholders.n_atoms_plus_virt = _int('n_atoms_plus_virt')
            self._placeholders.volume = _float('volume')
            self._placeholders.mask = _float_1d('mask')
            self._placeholders.composition = _float_1d('composition')
            self._placeholders.nnl_max = _int('nnl_max')
            self._placeholders.row_splits = _int_1d(
                'row_splits', d0=self._n_elements + 1)
            self._placeholders.ilist = _int_1d('ilist')
            self._placeholders.jlist = _int_1d('jlist')
            self._placeholders.shift = _float_2d(3, 'shift')
            self._placeholders.v2g_map = _int_2d(4, 'v2g_map')

            self._placeholders.gs = AttributeDict(
                v2g_map=_int_2d(3, 'gs.v2g_map'),
                ij=AttributeDict(ilist=_int_1d('gs.ij.ilist'),
                                 jlist=_int_1d('gs.ij.jlist')),
                ik=AttributeDict(ilist=_int_1d('gs.ik.ilist'),
                                 klist=_int_1d('gs.ik.klist')),
                jk=AttributeDict(jlist=_int_1d('gs.jk.jlist'),
                                 klist=_int_1d('gs.jk.klist')),
                shift=AttributeDict(ij=_float_2d(3, 'gs.shift.ij'),
                                    ik=_float_2d(3, 'gs.shift.ik'),
                                    jk=_float_2d(3, 'gs.shift.jk'))
            )

        return self._placeholders

    @staticmethod
    def _get_g2_indexed_slices_with_rc(atoms: Atoms,
                                       rc: float,
                                       kbody_index: Dict[str, int],
                                       index_transformer: IndexTransformer):
        """
        Return the corresponding indexed slices given a cutoff radius.

        Parameters
        ----------
        atoms : Atoms
            The target `ase.Atoms` object.
        rc : float
            The cutoff radius.
        index_transformer : IndexTransformer
            The corresponding index transformer.

        Returns
        -------
        g2 : G2IndexedSlices
            The indexed slices for the target `Atoms` object.

        """
        symbols = atoms.get_chemical_symbols()
        ilist, jlist, Slist = neighbor_list('ijS', atoms, cutoff=rc)
        nij = len(ilist)

        v2g_map = np.zeros((nij, 4), dtype=np.int32)

        tlist = np.zeros(nij, dtype=np.int32)
        for i in range(nij):
            symboli = symbols[ilist[i]]
            symbolj = symbols[jlist[i]]
            tlist[i] = kbody_index['{}{}'.format(symboli, symbolj)]

        ilist = index_transformer.inplace_map_index(ilist + 1)
        jlist = index_transformer.inplace_map_index(jlist + 1)
        shift = np.asarray(Slist, dtype=get_float_dtype().as_numpy_dtype)

        # The type of the (atomi, atomj) interaction.
        v2g_map[:, 0] = tlist

        # The indices of center atoms
        v2g_map[:, 1] = ilist
        counters = {}
        for index in range(nij):
            atomi = ilist[index]
            if atomi not in counters:
                counters[atomi] = Counter()

            # The indices of the pair atoms
            v2g_map[index, 2] = counters[atomi][tlist[index]]
            counters[atomi][tlist[index]] += 1

        # The mask
        v2g_map[:, 3] = ilist > 0

        return G2IndexedSlices(v2g_map=v2g_map, ilist=ilist, jlist=jlist,
                               shift=shift)

    def _get_g2_indexed_slices(self,
                               atoms: Atoms,
                               transformer: IndexTransformer):
        """
        Return the indexed slices for pairwise interactions.
        """
        return self._get_g2_indexed_slices_with_rc(
            atoms, self._rc, self._kbody_index, transformer)

    def _get_g4_indexed_slices(self,
                               atoms: Atoms,
                               g2: G2IndexedSlices,
                               transformer: IndexTransformer):
        """
        Return the indexed slices for mapping Cikj to Sij.
        """
        symbols = atoms.get_chemical_symbols()
        dtype = get_float_dtype()
        np_dtype = dtype.as_numpy_dtype

        if self._angular_rc_scale != 1.0:
            g2 = self._get_g2_indexed_slices_with_rc(
                atoms, self._angular_rc, self._kbody_index, transformer)

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
        ij_shift = np.zeros((nijk, 3), dtype=np_dtype)
        ik_shift = np.zeros((nijk, 3), dtype=np_dtype)
        jk_shift = np.zeros((nijk, 3), dtype=np_dtype)

        count = 0
        for atomi, nl in indices.items():
            num = len(nl)
            indexi = transformer.inplace_map_index(atomi, True, True)
            symboli = symbols[indexi]
            prefix = '{}'.format(symboli)
            for j in range(num):
                atomj = nl[j]
                indexj = transformer.inplace_map_index(atomj, True, True)
                symbolj = symbols[indexj]
                for k in range(j + 1, num):
                    atomk = nl[k]
                    indexk = transformer.inplace_map_index(atomk, True, True)
                    symbolk = symbols[indexk]
                    suffix = ''.join(sorted([symbolj, symbolk]))
                    kbody_term = '{}{}'.format(prefix, suffix)
                    ij[count] = atomi, atomj
                    ik[count] = atomi, atomk
                    jk[count] = atomj, atomk
                    ij_shift[count] = vectors[atomi][j]
                    ik_shift[count] = vectors[atomi][k]
                    jk_shift[count] = vectors[atomi][k] - vectors[atomi][j]
                    index = self._kbody_index[kbody_term]
                    v2g_map[count, 0] = atomi
                    v2g_map[count, 1] = self._offsets[index]
                    count += 1
        return G4IndexedSlices(v2g_map=v2g_map, ij=ij, ik=ik, jk=jk,
                               ij_shift=ij_shift, ik_shift=ik_shift,
                               jk_shift=jk_shift)

        indices = {}
        vectors = {}
        j_prim_list = {}
        for i, atomi in enumerate(g2.ilist):
            if atomi not in j_prim_list:
                j_prim_list[atomi] = []
                vectors[atomi] = []
                indices[atomi] = []
            indices[atomi].append(i)
            j_prim_list[atomi].append(g2.jlist[i])
            vectors[atomi].append(g2.shift[i])
        nij = len(g2.ilist)

        nijk = 0
        for atomi, nl in j_prim_list.items():
            n = len(nl)
            nijk += (n - 1) * n // 2

        g4_map = np.zeros((nijk, 3), dtype=np.int32)

        ij = np.zeros((nijk, 2), dtype=np.int32)
        ik = np.zeros((nijk, 2), dtype=np.int32)
        jk = np.zeros((nijk, 2), dtype=np.int32)
        ij_shift = np.zeros((nijk, 3), dtype=np_dtype)
        ik_shift = np.zeros((nijk, 3), dtype=np_dtype)
        jk_shift = np.zeros((nijk, 3), dtype=np_dtype)

        ijk_idx = 0
        counter = np.zeros((nij, self._n_elements ** 3), dtype=int)

        for atom_prim_i, pair_indices in indices.items():

            num = len(pair_indices)
            indexi = transformer.inplace_map_index(atom_prim_i, True, True)
            symboli = symbols[indexi]

            for j in range(num):

                atomj = g2.jlist[pair_indices[j]]
                indexj = transformer.inplace_map_index(atomj, True, True)
                symbolj = symbols[indexj]

                for k in range(j + 1, num):
                    atomk = g2.jlist[pair_indices[k]]

                    indexk = transformer.inplace_map_index(atomj, True, True)
                    symbolk = symbols[indexk]
                    kbody_term = f'{symboli}{symbolj}{symbolk}'
                    index = self._angular_kbody_terms.index(kbody_term)

                    ij[ijk_idx] = atom_prim_i, atomj
                    ik[ijk_idx] = atom_prim_i, atomk
                    jk[ijk_idx] = atomj, atomk
                    ij_shift[ijk_idx] = vectors[atom_prim_i][j]
                    ik_shift[ijk_idx] = vectors[atom_prim_i][k]
                    jk_shift[ijk_idx] = vectors[atom_prim_i][k] - vectors[atom_prim_i][j]

                    g4_map[ijk_idx, 0] = pair_indices[j]
                    g4_map[ijk_idx, 1] = index
                    g4_map[ijk_idx, 2] = counter[pair_indices[j], index]
                    counter[pair_indices[j], index] += 1

                    ijk_idx += 1

        return G4IndexedSlices(v2g_map=g4_map, ij=ij, ik=ik, jk=jk,
                               ij_shift=ij_shift, ik_shift=ik_shift,
                               jk_shift=jk_shift)

    def _get_np_features(self, atoms: Atoms):
        """
        Return a dict of features (Numpy or Python objects).
        """
        np_dtype = get_float_dtype().as_numpy_dtype

        index_transformer = self.get_index_transformer(atoms)
        g2 = self._get_g2_indexed_slices(atoms, index_transformer)
        gs = self._get_sij_indexed_slices(atoms, g2, index_transformer)

        nij_max = len(g2.ilist)
        nnl_max = g2.v2g_map[:, 2].max() + 1
        ntri_max = gs.v2g_map[:, 2].max() + 1

        positions = index_transformer.map_positions(atoms.positions)

        # `max_n_atoms` must be used because every element shall have at least
        # one feature row (though it could be all zeros, a dummy or virtual row)
        n_atoms = index_transformer.max_n_atoms
        cells = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        mask = index_transformer.mask
        splits = [1] + [index_transformer.max_occurs[e] for e in self._elements]
        compositions = self._get_compositions(atoms)

        feed_dict = AttributeDict()

        feed_dict.positions = positions.astype(np_dtype)
        feed_dict.n_atoms_plus_virt = np.int32(n_atoms + 1)
        feed_dict.nnl_max = np.int32(nnl_max)
        feed_dict.nij_max = np.int32(nij_max)
        feed_dict.ntri_max = np.int32(ntri_max)
        feed_dict.mask = mask.astype(np_dtype)
        feed_dict.cells = cells.array.astype(np_dtype)
        feed_dict.volume = np_dtype(volume)
        feed_dict.composition = composition.astype(np_dtype)
        feed_dict.row_splits = np.int32(splits)
        feed_dict.v2g_map = g2.v2g_map
        feed_dict.ilist = g2.ilist
        feed_dict.jlist = g2.jlist
        feed_dict.shift = g2.shift
        feed_dict.gs = AttributeDict(
            v2g_map=gs.v2g_map,
            ij=AttributeDict(ilist=gs.ij[:, 0], jlist=gs.ij[:, 1]),
            ik=AttributeDict(ilist=gs.ik[:, 0], klist=gs.ik[:, 1]),
            jk=AttributeDict(jlist=gs.jk[:, 0], klist=gs.jk[:, 1]),
            shift=AttributeDict(ij=gs.ij_shift, ik=gs.ik_shift, jk=gs.jk_shift)
        )

        return feed_dict

    def get_feed_dict(self, atoms: Atoms):
        """
        Return the feed dict.
        """
        feed_dict = {}

        if not self._placeholders:
            self._initialize_placeholders()
        placeholders = self._placeholders

        for key, val in self._get_np_features(atoms).items():
            if key == 'gs':
                feed_dict[placeholders.gs.ij.ilist] = val['ij']['ilist']
                feed_dict[placeholders.gs.ij.jlist] = val['ij']['jlist']
                feed_dict[placeholders.gs.ik.ilist] = val['ik']['ilist']
                feed_dict[placeholders.gs.ik.klist] = val['ik']['klist']
                feed_dict[placeholders.gs.jk.jlist] = val['jk']['jlist']
                feed_dict[placeholders.gs.jk.klist] = val['jk']['klist']
                feed_dict[placeholders.gs.shift.ij] = val['shift']['ij']
                feed_dict[placeholders.gs.shift.ik] = val['shift']['ik']
                feed_dict[placeholders.gs.shift.jk] = val['shift']['jk']
                feed_dict[placeholders.gs.v2g_map] = val['v2g_map']
            else:
                feed_dict[placeholders[key]] = val

        return feed_dict

    def get_constant_features(self, atoms: Atoms):
        """
        Return a dict of constant feature tensors for the given `Atoms`.
        """
        feed_dict = AttributeDict()
        with tf.name_scope("Constants"):
            for key, val in self._get_np_features(atoms).items():
                if key == 'gs':
                    feed_dict['gs'] = AttributeDict(
                        v2g_map=tf.convert_to_tensor(
                            val['v2g_map'], name='gs.v2g_map'))
                    feed_dict['gs']['ij'] = AttributeDict(
                        ilist=tf.convert_to_tensor(
                            val['ij']['ilist'], name='gs.ij.ilist'),
                        jlist=tf.convert_to_tensor(
                            val['ij']['jlist'], name='gs.ij.jlist'))
                    feed_dict['gs']['ik'] = AttributeDict(
                        ilist=tf.convert_to_tensor(
                            val['ik']['ilist'], name='gs.ik.ilist'),
                        klist=tf.convert_to_tensor(
                            val['ik']['klist'], name='gs.ik.klist'))
                    feed_dict['gs']['jk'] = AttributeDict(
                        jlist=tf.convert_to_tensor(
                            val['jk']['jlist'], name='gs.jk.jlist'),
                        klist=tf.convert_to_tensor(
                            val['jk']['klist'], name='gs.jk.klist'))
                    feed_dict['gs']['shift'] = AttributeDict(
                        ij=tf.convert_to_tensor(
                            val['shift']['ij'], name='gs.shift.ij'),
                        ik=tf.convert_to_tensor(
                            val['shift']['ik'], name='gs.shift.ik'),
                        jk=tf.convert_to_tensor(
                            val['shift']['jk'], name='gs.shift.jk'),
                    )
                else:
                    feed_dict[key] = tf.convert_to_tensor(val, name=key)
        return feed_dict
