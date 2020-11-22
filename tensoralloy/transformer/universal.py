#!coding=utf-8
"""
The universal transformer.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import chemical_symbols
from collections import Counter
from typing import List, Dict
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.utils import get_elements_from_kbody_term, get_kbody_terms
from tensoralloy.utils import szudzik_pairing
from tensoralloy import atoms_utils
from tensoralloy.transformer.metadata import RadialMetadata, AngularMetadata
from tensoralloy.transformer.vap import VirtualAtomMap
from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.transformer.base import bytes_feature
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _get_iaxis(mode: tf_estimator.ModeKeys):
    if mode == tf_estimator.ModeKeys.PREDICT:
        return 0
    else:
        return 1


def get_ijn_id(i, j, nx, ny, nz):
    """
    Return the unique id of (i, j, nx, ny, nz)
    """
    ij_id = szudzik_pairing(int(i), int(j))
    n_id = szudzik_pairing(int(nx), int(ny), int(nz))
    return ij_id, n_id


def get_radial_metadata(atoms: Atoms,
                        rc: float,
                        interactions: Dict[str, int],
                        vap: VirtualAtomMap,
                        mode: tf_estimator.ModeKeys,
                        nij_max: int = None,
                        dtype=np.float32) -> (RadialMetadata, dict):
    """
    Build the base `v2g_map`.

    # TODO: improve the memory layout of interatomic distances
    """
    ilist, jlist, n1, rij, dij = neighbor_list('ijSdD', atoms, rc)
    nij = len(ilist)
    if nij_max is None:
        nij_max = nij

    iaxis = _get_iaxis(mode)
    g2_map = np.zeros((nij_max, iaxis + 5), dtype=np.int32)
    g2_map.fill(0)
    tlist = np.zeros(nij_max, dtype=np.int32)
    symbols = atoms.get_chemical_symbols()
    tlist.fill(0)
    for i in range(nij):
        symboli = symbols[ilist[i]]
        symbolj = symbols[jlist[i]]
        tlist[i] = interactions['{}{}'.format(symboli, symbolj)]
    ilist = np.pad(ilist + 1, (0, nij_max - nij), 'constant')
    jlist = np.pad(jlist + 1, (0, nij_max - nij), 'constant')
    n1 = np.pad(n1, ((0, nij_max - nij), (0, 0)), 'constant')
    n1 = n1.astype(dtype)
    for count in range(len(ilist)):
        if ilist[count] == 0:
            break
        ilist[count] = vap.local_to_gsl_map[ilist[count]]
        jlist[count] = vap.local_to_gsl_map[jlist[count]]
    ilist = ilist.astype(np.int32)
    jlist = jlist.astype(np.int32)
    g2_map[:, iaxis + 0] = tlist
    g2_map[:, iaxis + 1] = ilist

    # The indices of center atoms
    counters = {}
    ijn_id_map = {}
    for index in range(nij):
        atomi = ilist[index]
        if atomi not in counters:
            counters[atomi] = Counter()

        # The indices of the pair atoms
        inc = counters[atomi][tlist[index]]
        g2_map[index, iaxis + 2] = inc
        g2_map[index, iaxis + 3] = 0
        counters[atomi][tlist[index]] += 1

        # The `ijn_id` map
        ijn_id = get_ijn_id(ilist[index], jlist[index], *n1[index])
        ijn_id_map[ijn_id] = inc

        # The mask
        g2_map[index, iaxis + 4] = ilist[index] > 0

    rij = np.concatenate(
        (rij.astype(dtype).reshape((-1, 1)), dij.astype(dtype)), axis=1).T
    radial_metadata = RadialMetadata(
        v2g_map=g2_map, ilist=ilist, jlist=jlist, n1=n1, rij=rij)
    return radial_metadata, ijn_id_map


def get_angular_metadata(atoms: Atoms,
                         rc: float,
                         radial_interactions: Dict[str, int],
                         angular_interactions: Dict[str, int],
                         vap: VirtualAtomMap,
                         mode: tf_estimator.ModeKeys,
                         radial_metadata: RadialMetadata = None,
                         ijn_id_map: Dict[int, int] = None,
                         nijk_max: int = None,
                         angular_symmetricity=True,
                         dtype=np.float32) -> AngularMetadata:
    """
    Build the base `v2g_map`.
    """
    if radial_metadata is None:
        radial_metadata, ijn_id_map = get_radial_metadata(
            atoms=atoms,
            rc=rc,
            interactions=radial_interactions,
            vap=vap,
            mode=mode,
            dtype=dtype)
    indices = {}
    vectors = {}
    ijdists = {}
    for i, atom_vap_i in enumerate(radial_metadata.ilist):
        if atom_vap_i == 0:
            break
        if atom_vap_i not in indices:
            indices[atom_vap_i] = []
            vectors[atom_vap_i] = []
            ijdists[atom_vap_i] = []
        indices[atom_vap_i].append(radial_metadata.jlist[i])
        vectors[atom_vap_i].append(radial_metadata.n1[i])
        ijdists[atom_vap_i].append(radial_metadata.rij[:, i])

    if nijk_max is None:
        nijk = 0
        for atomi, nl in indices.items():
            n = len(nl)
            # nijk += (n - 1) * n // 2
            if angular_symmetricity:
                nijk += (n - 1) * n // 2
            else:
                nijk += (n - 1) * n
        nijk_max = nijk

    iaxis = _get_iaxis(mode)
    g4_map = np.zeros((nijk_max, iaxis + 5), dtype=np.int32)
    g4_map.fill(0)
    ilist = np.zeros(nijk_max, dtype=np.int32)
    jlist = np.zeros(nijk_max, dtype=np.int32)
    klist = np.zeros(nijk_max, dtype=np.int32)
    n1 = np.zeros((nijk_max, 3), dtype=radial_metadata.n1.dtype)
    n2 = np.zeros((nijk_max, 3), dtype=radial_metadata.n1.dtype)
    n3 = np.zeros((nijk_max, 3), dtype=radial_metadata.n1.dtype)
    rijk = np.zeros((12, nijk_max), dtype=dtype)
    symbols = atoms.get_chemical_symbols()
    counters = {}

    count = 0
    for atom_vap_i, nl in indices.items():
        atom_local_i = vap.gsl_to_local_map[atom_vap_i]
        symboli = symbols[atom_local_i]
        for j in range(len(nl)):
            atom_vap_j = nl[j]
            atom_local_j = vap.gsl_to_local_map[atom_vap_j]
            symbolj = symbols[atom_local_j]
            if angular_symmetricity:
                kstart = j + 1
            else:
                kstart = 0
            for k in range(kstart, len(nl)):
                atom_vap_k = nl[k]
                atom_local_k = vap.gsl_to_local_map[atom_vap_k]
                symbolk = symbols[atom_local_k]
                if angular_symmetricity:
                    interaction = \
                        f"{symboli}{''.join(sorted([symbolj, symbolk]))}"
                    if symbolj < symbolk:
                        ijn_id = get_ijn_id(
                            atom_vap_i, atom_vap_j, *vectors[atom_vap_i][j])
                    else:
                        ijn_id = get_ijn_id(
                            atom_vap_i, atom_vap_k, *vectors[atom_vap_i][k])
                else:
                    ijn_id = get_ijn_id(
                        atom_vap_i, atom_vap_j, *vectors[atom_vap_i][j])
                    ikn_id = get_ijn_id(
                        atom_vap_i, atom_vap_k, *vectors[atom_vap_i][k])
                    if ijn_id == ikn_id:
                        continue
                    interaction = f"{symboli}{symbolj}{symbolk}"
                ilist[count] = atom_vap_i
                jlist[count] = atom_vap_j
                klist[count] = atom_vap_k
                n1[count] = vectors[atom_vap_i][j]
                n2[count] = vectors[atom_vap_i][k]
                n3[count] = vectors[atom_vap_i][k] - vectors[atom_vap_i][j]
                rijk[0: 4, count] = ijdists[atom_vap_i][j]
                rijk[4: 8, count] = ijdists[atom_vap_i][k]
                djk = rijk[5: 8, count] - rijk[1: 4, count]
                rijk[8, count] = np.linalg.norm(djk)
                rijk[9: 12, count] = djk
                index = angular_interactions[interaction]
                if atom_vap_i not in counters:
                    counters[atom_vap_i] = {}
                if index not in counters[atom_vap_i]:
                    counters[atom_vap_i][index] = Counter()
                if ijn_id not in counters[atom_vap_i][index]:
                    counters[atom_vap_i][index][ijn_id] = 0
                g4_map[count, iaxis + 0] = index
                g4_map[count, iaxis + 1] = atom_vap_i
                g4_map[count, iaxis + 2] = ijn_id_map[ijn_id]
                g4_map[count, iaxis + 3] = counters[atom_vap_i][index][ijn_id]
                g4_map[count, iaxis + 4] = 1
                counters[atom_vap_i][index][ijn_id] += 1
                count += 1
    return AngularMetadata(g4_map, ilist, jlist, klist, n1, n2, n3, rijk=rijk)


class UniversalTransformer(DescriptorTransformer):
    """
    The universal transformer for all models.
    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, elements: List[str], rcut, acut=None, angular=False,
                 periodic=True, symmetric=True, use_computed_dists=True):
        """
        The initialization method

        Parameters
        ----------
        elements : List[str]
            A list of str as the ordered unique elements.
        rcut : float
            The cutoff radius for radial interactions.
        acut : float
            The cutoff radius for angular interactions. Defaults to None.
        angular : bool
            A boolean indicating whether angular interactions shall be included
            or not. Defaults to False.
        periodic : bool
            A boolean. If True, the periodic interatomic distance equation will
            be used.
        symmetric : bool
            A boolean. If True, symmetricity will be considered in building
            angular interactions.
        use_computed_dists : bool
            A boolean.

            If True, interatomic distance related variables (`rij`, `dij`)
            should be computed.

            If False, these should be provided by the input `features`. At this
            time, dE/dh and dE/dR can not be evaluated directly since the
            positions tensor and the cell tensor will not be include in the
            computation graph. For exporting Lammps/MPI-compatible models, this
            should be False.

        """
        DescriptorTransformer.__init__(self)

        for element in elements:
            if element not in chemical_symbols:
                raise ValueError(f"{element} is not a valid chemical symbol!")

        if angular and acut is None:
            acut = rcut

        all_kbody_terms, kbody_terms_for_element, elements = \
            get_kbody_terms(elements, angular=angular, symmetric=symmetric)

        angular_kbody_terms = []
        radial_kbody_terms = []
        for kbody_term in all_kbody_terms:
            if len(get_elements_from_kbody_term(kbody_term)) == 3:
                angular_kbody_terms.append(kbody_term)
            else:
                radial_kbody_terms.append(kbody_term)

        max_nr_terms = 0
        max_na_terms = 0
        for element in elements:
            max_na_terms = max(
                max_na_terms,
                len([x for x in kbody_terms_for_element[element]
                     if len(get_elements_from_kbody_term(x)) == 3]))
            max_nr_terms = max(
                max_nr_terms,
                len([x for x in kbody_terms_for_element[element]
                     if len(get_elements_from_kbody_term(x)) == 2]))

        self._all_kbody_terms = all_kbody_terms
        self._kbody_terms_for_element = kbody_terms_for_element
        self._max_na_terms = max_na_terms
        self._max_nr_terms = max_nr_terms
        self._rcut = rcut
        self._acut = acut
        self._elements = elements
        self._n_elements = len(elements)
        self._periodic = periodic
        self._angular = angular
        self._symmetric = symmetric
        self._use_computed_dists = use_computed_dists

    def as_dict(self) -> Dict:
        """
        Return a JSON serializable dict representation of this transformer.
        """
        return {'class': self.__class__.__name__, 'elements': self._elements,
                'rcut': self._rcut, 'acut': self._acut,
                'angular': self._angular, 'periodic': self._periodic,
                'symmetric': self._symmetric,
                'use_computed_dists': self._use_computed_dists}

    @property
    def descriptor(self):
        """
        Return the descriptor name. This property will be removed soon.
        """
        return "universal"

    @property
    def rc(self):
        """
        Return the cutoff radius. This property will be removed soon.
        """
        return self._rcut

    @property
    def rcut(self) -> float:
        """
        Return the cutoff radius for radial interactions.
        """
        return self._rcut

    @property
    def acut(self) -> float:
        """
        Return the cutoff radius for angular interactions.
        """
        return self._acut

    @property
    def elements(self) -> List[str]:
        """
        Return a list of str as the ordered unique elements.
        """
        return self._elements

    @property
    def n_elements(self) -> int:
        """
        Return the total number of unique elements.
        """
        return self._n_elements

    @property
    def periodic(self):
        """
        Return True if this can be applied to periodic structures. For
        non-periodic molecules some Ops can be ignored.
        """
        return self._periodic

    @property
    def max_occurs(self):
        """
        There is no restriction for the occurances of an element.
        """
        return {el: np.inf for el in self._elements}

    @property
    def angular(self):
        """
        Return True if angular interactions are included.
        """
        return self._angular

    @property
    def use_computed_dists(self):
        """
        Return True if interatomic distances shall be computed or given
        directly.
        """
        return self._use_computed_dists

    @use_computed_dists.setter
    def use_computed_dists(self, flag: bool):
        self._use_computed_dists = flag

    @property
    def all_kbody_terms(self):
        """
        A list of str as the ordered k-body terms.
        """
        return self._all_kbody_terms

    @property
    def kbody_terms_for_element(self) -> Dict[str, List[str]]:
        """
        A dict of (element, kbody_terms) as the k-body terms for each type of
        elements.
        """
        return self._kbody_terms_for_element

    @staticmethod
    def get_pbc_displacements(shift, cell, dtype=tf.float64):
        """
        Return the periodic boundary shift displacements.

        Parameters
        ----------
        shift : tf.Tensor or array_like
            A `float64` or `float32` tensor of shape `[-1, 3]` as the cell shift
            vector.
        cell : tf.Tensor or array_like
            A `float64` or `float32` tensor of shape `[3, 3]` as the cell.
        dtype : DType
            The corresponding data type of `shift` and `cell`.

        Returns
        -------
        Dij : tf.Tensor
            A `float64` tensor of shape `[-1, 3]` as the periodic displacements
            vector.

        """
        return tf.matmul(shift, cell, name='displacements')

    def calculate_rij(self, R, cell, ilist, jlist, shift, name):
        """
        Return the interatomic distances array, `rij`, and the corresponding
        differences.

        Returns
        -------
        rij : tf.Tensor
            The interatomic distances.
        dij : tf.Tensor
            The differences of `Rj - Ri`.

        """
        with tf.name_scope(name):
            dtype = get_float_dtype()
            Ri = self.gather_fn(R, ilist, 'Ri')
            Rj = self.gather_fn(R, jlist, 'Rj')
            Dij = tf.subtract(Rj, Ri, name='Dij')
            if self._periodic:
                pbc = self.get_pbc_displacements(shift, cell, dtype=dtype)
                Dij = tf.add(Dij, pbc, name='pbc')
            # By adding `eps` to the reduced sum NaN can be eliminated.
            with tf.name_scope("safe_norm"):
                eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
                rij = tf.sqrt(tf.reduce_sum(
                    tf.square(Dij, name='Dij2'), axis=-1) + eps)
                return rij, Dij

    def get_g_shape(self, features: dict, angular=False):
        """
        Return the shape of the descriptor matrix.
        """
        if angular:
            return [np.int32(self._max_na_terms),
                    features["n_atoms_vap"],
                    features["nnl_max"],
                    features["ij2k_max"]]
        else:
            return [np.int32(self._max_nr_terms),
                    features["n_atoms_vap"],
                    features["nnl_max"],
                    1]

    def get_v2g_map(self, features: dict, angular=False):
        """
        A wrapper function to get `v2g_map` and `v2g_masks`.
        """
        if angular:
            key = "g4.v2g_map"
        else:
            key = "g2.v2g_map"
        splits = tf.split(features[key], [-1, 1], axis=1)
        v2g_map = tf.identity(splits[0], name='v2g_map')
        v2g_masks = tf.identity(splits[1], name='v2g_masks')
        return v2g_map, v2g_masks

    def get_row_split_sizes(self, features):
        """
        Return the sizes of the rowwise splitted subsets of `g`.
        """
        return features["row_splits"]

    @staticmethod
    def get_row_split_axis():
        """
        Return the axis to rowwise split the universal descriptors.
        """
        return 1

    def _split_descriptors(self,
                           features,
                           universal_descriptors,
                           universal_descriptor_masks):
        """
        Split the universal descriptors into `N_element` subsets.
        """
        with tf.name_scope("Split"):
            split_sizes = self.get_row_split_sizes(features)
            axis = self.get_row_split_axis()

            # `axis` should increase by one for `g` because `g` is created by
            # `tf.concat((gr, gx, gy, gz), axis=0, name='g')`
            universal_descriptors = tf.split(
                universal_descriptors,
                split_sizes,
                axis=axis + 1,
                name='udesriptors')[1:]

            # Use the original axis
            # `dist_masks` refers to the masks for the universal interatomic
            # distances.
            universal_descriptor_masks = tf.split(
                universal_descriptor_masks,
                split_sizes,
                axis=axis,
                name='udescriptor_masks')[1:]

            return dict(zip(self._elements,
                            zip(universal_descriptors,
                                universal_descriptor_masks)))

    def _check_keys(self, features: dict):
        """
        Make sure `placeholders` contains enough keys.
        """
        assert 'positions' in features
        assert 'cell' in features
        assert 'volume' in features
        assert 'n_atoms_vap' in features
        assert 'nnl_max' in features
        assert 'row_splits' in features
        assert 'g2.v2g_map' in features

        if self._use_computed_dists:
            assert 'g2.ilist' in features
            assert 'g2.jlist' in features
            assert 'g2.n1' in features
        else:
            assert 'g2.rij' in features

        if self._angular:
            assert 'ij2k_max' in features
            assert 'g4.v2g_map' in features

            if self._use_computed_dists:
                assert 'g4.ilist' in features
                assert 'g4.jlist' in features
                assert 'g4.klist' in features
                assert 'g4.n1' in features
                assert 'g4.n2' in features
                assert 'g4.n3' in features
            else:
                assert 'g4.rijk' in features

    def build_radial_graph(self, features: dict):
        """
        Build the computation graph for calculating radial descriptors.
        """
        with tf.name_scope("Radial"):
            shape = self.get_g_shape(features, angular=False)
            v2g_map, v2g_masks = self.get_v2g_map(features)

            if self._use_computed_dists:
                rij, dij = self.calculate_rij(features["positions"],
                                              features["cell"],
                                              features["g2.ilist"],
                                              features["g2.jlist"],
                                              features["g2.n1"],
                                              name='rij')
                ijx = tf.identity(dij[..., 0], name='dijx')
                ijy = tf.identity(dij[..., 1], name='dijy')
                ijz = tf.identity(dij[..., 2], name='dijz')
            else:
                rij = tf.identity(features['g2.rij'][0], name='g2.rij')
                ijx = tf.identity(features['g2.rij'][1], name='dijx')
                ijy = tf.identity(features['g2.rij'][2], name='dijy')
                ijz = tf.identity(features['g2.rij'][3], name='dijz')

            gr = tf.expand_dims(tf.scatter_nd(v2g_map, rij, shape), 0, 'gr')
            gx = tf.expand_dims(tf.scatter_nd(v2g_map, ijx, shape), 0, 'gx')
            gy = tf.expand_dims(tf.scatter_nd(v2g_map, ijy, shape), 0, 'gy')
            gz = tf.expand_dims(tf.scatter_nd(v2g_map, ijz, shape), 0, 'gz')

            universal_descriptors = tf.concat(
                (gr, gx, gy, gz), axis=0, name='udescriptors')

            v2g_masks = tf.squeeze(v2g_masks, axis=self.get_row_split_axis())
            masks = tf.scatter_nd(v2g_map, v2g_masks, shape)
            universal_descriptor_masks = tf.cast(
                masks, dtype=rij.dtype, name='udescriptor_masks')
            return self._split_descriptors(
                features, universal_descriptors, universal_descriptor_masks)

    def build_angular_graph(self, features: dict):
        """
        Build the computation graph for calculating angular descriptors.
        """
        with tf.name_scope("Angular"):
            if self._use_computed_dists:
                rij, dij = self.calculate_rij(features['positions'],
                                              features['cell'],
                                              features['g4.ilist'],
                                              features['g4.jlist'],
                                              features['g4.n1'],
                                              name='rij')
                rik, dik = self.calculate_rij(features['positions'],
                                              features['cell'],
                                              features['g4.ilist'],
                                              features['g4.klist'],
                                              features['g4.n2'],
                                              name='rik')
                rjk, djk = self.calculate_rij(features['positions'],
                                              features['cell'],
                                              features['g4.jlist'],
                                              features['g4.klist'],
                                              features['g4.n3'],
                                              name='rjk')
                ijx = tf.identity(dij[..., 0], name='dijx')
                ijy = tf.identity(dij[..., 1], name='dijy')
                ijz = tf.identity(dij[..., 2], name='dijz')
                ikx = tf.identity(dik[..., 0], name='dikx')
                iky = tf.identity(dik[..., 1], name='diky')
                ikz = tf.identity(dik[..., 2], name='dikz')
                jkx = tf.identity(djk[..., 0], name='djkx')
                jky = tf.identity(djk[..., 1], name='djky')
                jkz = tf.identity(djk[..., 2], name='djkz')
            else:
                rij = tf.identity(features['g4.rijk'][0], name='g4.rij')
                ijx = tf.identity(features['g4.rijk'][1], name='dijx')
                ijy = tf.identity(features['g4.rijk'][2], name='dijy')
                ijz = tf.identity(features['g4.rijk'][3], name='dijz')
                rik = tf.identity(features['g4.rijk'][4], name='g4.rik')
                ikx = tf.identity(features['g4.rijk'][5], name='dikx')
                iky = tf.identity(features['g4.rijk'][6], name='diky')
                ikz = tf.identity(features['g4.rijk'][7], name='dikz')
                rjk = tf.identity(features['g4.rijk'][8], name='g4.dik')
                jkx = tf.identity(features['g4.rijk'][9], name='djkx')
                jky = tf.identity(features['g4.rijk'][10], name='djky')
                jkz = tf.identity(features['g4.rijk'][11], name='djkz')

            shape = self.get_g_shape(features, angular=True)
            v2g_map, v2g_masks = self.get_v2g_map(features, angular=True)

            grij = tf.expand_dims(tf.scatter_nd(v2g_map, rij, shape), 0, 'grij')
            gijx = tf.expand_dims(tf.scatter_nd(v2g_map, ijx, shape), 0, 'gijx')
            gijy = tf.expand_dims(tf.scatter_nd(v2g_map, ijy, shape), 0, 'gijy')
            gijz = tf.expand_dims(tf.scatter_nd(v2g_map, ijz, shape), 0, 'gijz')
            grik = tf.expand_dims(tf.scatter_nd(v2g_map, rik, shape), 0, 'grik')
            gikx = tf.expand_dims(tf.scatter_nd(v2g_map, ikx, shape), 0, 'gikx')
            giky = tf.expand_dims(tf.scatter_nd(v2g_map, iky, shape), 0, 'giky')
            gikz = tf.expand_dims(tf.scatter_nd(v2g_map, ikz, shape), 0, 'gikz')
            grjk = tf.expand_dims(tf.scatter_nd(v2g_map, rjk, shape), 0, 'grjk')
            gjkx = tf.expand_dims(tf.scatter_nd(v2g_map, jkx, shape), 0, 'gjkx')
            gjky = tf.expand_dims(tf.scatter_nd(v2g_map, jky, shape), 0, 'gjky')
            gjkz = tf.expand_dims(tf.scatter_nd(v2g_map, jkz, shape), 0, 'gjkz')
            universal_descriptors = tf.concat(
                (grij, gijx, gijy, gijz,
                 grik, gikx, giky, gikz,
                 grjk, gjkx, gjky, gjkz), axis=0, name='udescriptors')

            v2g_masks = tf.squeeze(v2g_masks, axis=self.get_row_split_axis())
            masks = tf.scatter_nd(v2g_map, v2g_masks, shape)
            universal_descriptor_masks = tf.cast(
                masks, dtype=rij.dtype, name='udescriptor_masks')
            return self._split_descriptors(
                features, universal_descriptors, universal_descriptor_masks)

    def _get_atom_masks(self, features: dict):
        # `atom_masks` indicates whether the corresponding atom is real or
        # virtual.
        split_sizes = self.get_row_split_sizes(features)
        axis = self.get_row_split_axis()
        atom_masks = tf.split(
            features['atom_masks'],
            split_sizes,
            axis=axis - 1,
            name='atom_masks')[1:]
        return dict(zip(self._elements, atom_masks))

    def build_graph(self, features: dict):
        """
        Build the graph for computing universal descriptors.

        Returns
        -------
        ops : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of {element: (descriptor, mask)}.
        """
        self._check_keys(features)

        with tf.name_scope(f"Transformer"):
            g2 = self.build_radial_graph(features)
            if self._angular:
                g4 = self.build_angular_graph(features)
            else:
                g4 = None
            atom_masks = self._get_atom_masks(features)
            return {"radial": g2, "angular": g4, "atom_masks": atom_masks}

    def _initialize_placeholders(self):
        """
        Initialize placeholder tensors.
        """
        with tf.name_scope("Placeholders/"):
            dtype = get_float_dtype()

            self._placeholders["positions"] = self._create_float_2d(
                dtype=dtype, d0=None, d1=3, name='positions')
            self._placeholders["cell"] = self._create_float_2d(
                dtype=dtype, d0=3, d1=3, name='cell')
            self._placeholders["n_atoms_vap"] = self._create_int('n_atoms_vap')
            self._placeholders["nnl_max"] = self._create_int('nnl_max')
            self._placeholders["volume"] = self._create_float(
                dtype=dtype, name='volume')
            self._placeholders["atom_masks"] = self._create_float_1d(
                dtype=dtype, name='atom_masks')
            self._placeholders["pulay_stress"] = self._create_float(
                dtype=dtype, name='pulay_stress')
            self._placeholders["etemperature"] = self._create_float(
                dtype=dtype, name='etemperature')
            self._placeholders["row_splits"] = self._create_int_1d(
                name='row_splits', d0=self.n_elements + 1)

            if self._use_computed_dists:
                self._placeholders["g2.ilist"] = self._create_int_1d('g2.ilist')
                self._placeholders["g2.jlist"] = self._create_int_1d('g2.jlist')
                self._placeholders["g2.n1"] = self._create_float_2d(
                    dtype=dtype, d0=None, d1=3, name='g2.n1')
            else:
                self._placeholders["g2.rij"] = self._create_float_2d(
                    d0=4, d1=None, dtype=dtype, name="g2.rij")
            self._placeholders["g2.v2g_map"] = self._create_int_2d(
                d0=None, d1=5, name='g2.v2g_map')

            if self._angular:
                self._placeholders["ij2k_max"] = self._create_int('ij2k_max')
                self._placeholders["g4.v2g_map"] = self._create_int_2d(
                    d0=None, d1=5, name='g4.v2g_map')

                if self._use_computed_dists:
                    self._placeholders["g4.ilist"] = \
                        self._create_int_1d('g4.ilist')
                    self._placeholders["g4.jlist"] = \
                        self._create_int_1d('g4.jlist')
                    self._placeholders["g4.klist"] = \
                        self._create_int_1d('g4.klist')
                    self._placeholders["g4.n1"] = self._create_float_2d(
                        dtype=dtype, d0=None, d1=3, name='g4.n1')
                    self._placeholders["g4.n2"] = self._create_float_2d(
                        dtype=dtype, d0=None, d1=3, name='g4.n2')
                    self._placeholders["g4.n3"] = self._create_float_2d(
                        dtype=dtype, d0=None, d1=3, name='g4.n3')
                else:
                    self._placeholders["g4.rijk"] = self._create_float_2d(
                        d0=12, d1=None, dtype=dtype, name="g4.rijk")

        return self._placeholders

    def get_metadata(self, atoms, vap: VirtualAtomMap):
        """
        Return the corresponding indexed slices.

        Parameters
        ----------
        atoms : Atoms
            The target `ase.Atoms` object.
        vap : VirtualAtomMap
            The corresponding virtual-atom map.

        Returns
        -------
        radial_metadata : RadialMetadata
            The radial metadata for the target `Atoms` object.
        angular_metadata : AngularMetadata
            The angular metadata for the target `Atoms` object.

        """

        radial_interactions = {}
        angular_interactions = {}
        n = len(self._elements)

        for element in self._elements:
            kbody_terms = self._kbody_terms_for_element[element]
            for i, kbody_term in enumerate(kbody_terms):
                if len(get_elements_from_kbody_term(kbody_term)) == 2:
                    radial_interactions[kbody_term] = i
                else:
                    angular_interactions[kbody_term] = i - n

        dtype = get_float_dtype().as_numpy_dtype
        radial_metadata, ijn_id_map = get_radial_metadata(
            atoms=atoms,
            rc=self._rcut,
            interactions=radial_interactions,
            vap=vap,
            mode=tf_estimator.ModeKeys.PREDICT,
            nij_max=None,
            dtype=dtype)
        if self._angular:
            if np.round(self._acut - self._rcut, 2) == 0.0:
                ref_radial_metadata = radial_metadata
                ref_ijn_id_map = ijn_id_map
            else:
                ref_radial_metadata = None
                ref_ijn_id_map = None
            angular_metadata = get_angular_metadata(
                atoms=atoms,
                rc=self._acut,
                radial_metadata=ref_radial_metadata,
                ijn_id_map=ref_ijn_id_map,
                radial_interactions=radial_interactions,
                angular_interactions=angular_interactions,
                vap=vap,
                angular_symmetricity=self._symmetric,
                mode=tf_estimator.ModeKeys.PREDICT,
                nijk_max=None,
                dtype=dtype)
        else:
            angular_metadata = None
        return radial_metadata, angular_metadata

    def get_np_feed_dict(self, atoms: Atoms):
        """
        Return a dict of features (Numpy or Python objects).
        """
        np_dtype = get_float_dtype().as_numpy_dtype

        vap = self.get_vap_transformer(atoms)
        radial_metadata, angular_metadata = self.get_metadata(atoms, vap)

        positions = vap.map_positions(atoms.positions)

        # `max_n_atoms` must be used because every element shall have at least
        # one feature row (though it could be all zeros, a dummy or virtual row)
        cell = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        atom_masks = vap.atom_masks.astype(np_dtype)
        pulay_stress = atoms_utils.get_pulay_stress(atoms)
        etemp = atoms_utils.get_electron_temperature(atoms)
        splits = [1] + [vap.max_occurs[e] for e in self._elements]

        feed_dict = dict()

        feed_dict["positions"] = positions.astype(np_dtype)
        feed_dict["n_atoms_vap"] = np.int32(vap.max_vap_natoms)
        feed_dict["nnl_max"] = np.int32(radial_metadata.v2g_map[:, 2].max() + 1)
        feed_dict["atom_masks"] = atom_masks.astype(np_dtype)
        feed_dict["cell"] = cell.array.astype(np_dtype)
        feed_dict["volume"] = np_dtype(volume)
        feed_dict["pulay_stress"] = np_dtype(pulay_stress)
        feed_dict["etemperature"] = np_dtype(etemp)
        feed_dict["row_splits"] = np.int32(splits)
        feed_dict.update(
            radial_metadata.as_dict(
                use_computed_dists=self._use_computed_dists))

        if self._angular:
            feed_dict['ij2k_max'] = np.int32(
                angular_metadata.v2g_map[:, 3].max() + 1)
            feed_dict.update(
                angular_metadata.as_dict(
                    use_computed_dists=self._use_computed_dists))

        return feed_dict

    def get_feed_dict(self, atoms: Atoms):
        """
        Return the feed dict.
        """
        feed_dict = {}

        if not self._placeholders:
            self._initialize_placeholders()
        placeholders = self._placeholders

        for key, value in self.get_np_feed_dict(atoms).items():
            feed_dict[placeholders[key]] = value

        return feed_dict

    def get_constant_features(self, atoms: Atoms):
        """
        Return a dict of constant feature tensors for the given `Atoms`.
        """
        feed_dict = dict()
        with tf.name_scope("Constants"):
            for key, val in self.get_np_feed_dict(atoms).items():
                feed_dict[key] = tf.convert_to_tensor(val, name=key)
        return feed_dict


class BatchUniversalTransformer(UniversalTransformer,
                                BatchDescriptorTransformer):
    """
    The universal transformer for mini-batch training.
    """

    gather_fn = staticmethod(tf.batch_gather)

    def __init__(self, max_occurs: Counter, rcut, acut=None, angular=False,
                 periodic=True, symmetric=True, nij_max=None, nijk_max=None,
                 nnl_max=None, ij2k_max=None, batch_size=None, use_forces=True,
                 use_stress=False):
        """
        Initialization method.
        """
        elements = sorted(max_occurs.keys())

        UniversalTransformer.__init__(
            self, elements=elements, rcut=rcut, acut=acut, angular=angular,
            periodic=periodic, symmetric=symmetric, use_computed_dists=True)

        BatchDescriptorTransformer.__init__(self, use_forces=use_forces,
                                            use_stress=use_stress)

        self._nij_max = nij_max
        self._nijk_max = nijk_max
        self._nnl_max = nnl_max
        self._ij2k_max = ij2k_max
        self._batch_size = batch_size
        self._max_occurs = max_occurs.copy()
        self._max_n_atoms = sum(max_occurs.values())

        radial_interactions = {}
        angular_interactions = {}
        for element in self._elements:
            kbody_terms = self._kbody_terms_for_element[element]
            for i, kbody_term in enumerate(kbody_terms):
                if len(get_elements_from_kbody_term(kbody_term)) == 2:
                    radial_interactions[kbody_term] = i
                else:
                    angular_interactions[kbody_term] = i - len(self._elements)

        self._radial_interactions = radial_interactions
        self._angular_interactions = angular_interactions

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this transformer.
        """
        d = {'class': self.__class__.__name__,
             'max_occurs': self._max_occurs,
             'rcut': self._rcut,
             'acut': self._acut,
             'angular': self._angular,
             'nij_max': self._nij_max,
             'nijk_max': self._nijk_max,
             'nnl_max': self._nnl_max,
             'ij2k_max': self._ij2k_max,
             'batch_size': self._batch_size,
             'use_forces': self._use_forces,
             'use_stress': self._use_stress}
        return d

    @property
    def descriptor(self):
        """
        Return the name of the descriptor.
        """
        return "universal"

    @property
    def batch_size(self):
        """
        Return the batch size.
        """
        return self._batch_size

    @property
    def rc(self):
        """
        Return the cutoff radius for radial interactions.
        """
        return self._rcut

    @property
    def nij_max(self):
        """
        Return the corresponding `nij_max`.
        """
        return self._nij_max

    @property
    def nijk_max(self):
        """
        Return the corresponding `nijk_max`.
        """
        return self._nijk_max

    @property
    def nnl_max(self):
        """
        Return the corresponding `nnl_max`.
        """
        return self._nnl_max

    @property
    def ij2k_max(self):
        """
        Return the corresponding `ij2k_max`.
        """
        return self._ij2k_max

    @property
    def max_occurs(self):
        """
        Maximum
        """
        return self._max_occurs

    def as_descriptor_transformer(self) -> UniversalTransformer:
        """
        Return the corresponding `UniversalTransformer`.
        """
        return UniversalTransformer(elements=sorted(self._max_occurs.keys()),
                                    rcut=self._rcut, acut=self._acut,
                                    angular=self._angular,
                                    periodic=self._periodic,
                                    symmetric=self._symmetric)

    def get_metadata(self, atoms, vap: VirtualAtomMap):
        """
        Return metadata for the `atoms`.
        """
        dtype = get_float_dtype().as_numpy_dtype
        radial_metadata, ijn_id_map = get_radial_metadata(
            atoms=atoms,
            rc=self._rcut,
            interactions=self._radial_interactions,
            vap=vap,
            mode=tf_estimator.ModeKeys.TRAIN,
            nij_max=self._nij_max,
            dtype=dtype)
        if self._angular:
            if np.round(self._acut - self._rcut, 2) == 0.0:
                ref_radial_metadata = radial_metadata
                ref_ijn_id_map = ijn_id_map
            else:
                ref_radial_metadata = None
                ref_ijn_id_map = None
            angular_metadata = get_angular_metadata(
                atoms=atoms,
                rc=self._acut,
                radial_metadata=ref_radial_metadata,
                ijn_id_map=ref_ijn_id_map,
                radial_interactions=self._radial_interactions,
                angular_interactions=self._angular_interactions,
                vap=vap,
                angular_symmetricity=self._symmetric,
                mode=tf_estimator.ModeKeys.TRAIN,
                nijk_max=self._nijk_max,
                dtype=dtype)
        else:
            angular_metadata = None
        return radial_metadata, angular_metadata

    @staticmethod
    def get_pbc_displacements(shift, cell, dtype=tf.float64):
        """
        Return the periodic boundary shift displacements.

        Parameters
        ----------
        shift : tf.Tensor
            A `float64` or `float32` tensor of shape `[batch_size, ndim, 3]` as
            the cell shift vectors and `ndim == nij_max` or `ndim == nijk_max`.
        cell : tf.Tensor
            A `float64` or `float32` tensor of shape `[batch_size, 3, 3]` as the
            cell tensors.
        dtype : DType
            The corresponding data type of `shift` and `cell`.

        Returns
        -------
        Dij : tf.Tensor
            A `float64` tensor of shape `[-1, 3]` as the periodic displacements
            vector.

        """
        with tf.name_scope("Einsum"):
            shift = tf.convert_to_tensor(shift, dtype=dtype, name='shift')
            cell = tf.convert_to_tensor(cell, dtype=dtype, name='cell')
            return tf.einsum('ijk,ikl->ijl', shift, cell, name='displacements')

    def get_g_shape(self, features: dict, angular=False):
        """
        Return the shape of the descriptor matrix.
        """
        if angular:
            return [self._batch_size,
                    self._max_na_terms,
                    self._max_n_atoms + 1,
                    self._nnl_max,
                    self._ij2k_max]
        else:
            return [self._batch_size,
                    self._max_nr_terms,
                    self._max_n_atoms + 1,
                    self._nnl_max,
                    1]

    @staticmethod
    def get_row_split_axis():
        """
        Return the axis to split raw universal descriptors into elementary
        subsets.
        """
        return 2

    def get_row_split_sizes(self, _):
        """
        Return the sizes of elementary subsets.
        """
        row_splits = [1, ]
        for i, element in enumerate(self._elements):
            row_splits.append(self._max_occurs[element])
        return row_splits

    def _get_v2g_map_batch_indexing_matrix(self, angular=False):
        """
        Return an `int32` matrix of shape `[batch_size, ndim, D]` to rebuild the
        batch indexing of a `v2g_map`.
        """
        if angular:
            ndim = self._nijk_max
        else:
            ndim = self._nij_max
        indexing_matrix = np.zeros((self._batch_size, ndim, 5), dtype=np.int32)
        for i in range(self._batch_size):
            indexing_matrix[i] += [i, 0, 0, 0, 0]
        return indexing_matrix

    def get_v2g_map(self, features: dict, angular=False):
        """
                A wrapper function to get `v2g_map` and `v2g_masks`.
                """
        if angular:
            key = "g4.v2g_map"
        else:
            key = "g2.v2g_map"
        splits = tf.split(features[key], [-1, 1], axis=2)
        v2g_map = tf.identity(splits[0])
        v2g_masks = tf.identity(splits[1], name='v2g_masks')
        indexing = self._get_v2g_map_batch_indexing_matrix(angular=angular)
        v2g_map = tf.add(v2g_map, indexing, name='v2g_map')
        return v2g_map, v2g_masks

    @staticmethod
    def _encode_angular_metadata(data: AngularMetadata):
        """
        Encode the angular metadata:
            * `v2g_map`, `ilist`, `jlist` and `klist` are merged into a single
              array with key 'g4.indices'.
            * `n1`, `n2` and `n3` are merged into another array with key
              'g4.shifts'.

        """
        indices = np.concatenate(
            (data.v2g_map,
             data.ilist[..., np.newaxis],
             data.jlist[..., np.newaxis],
             data.klist[..., np.newaxis]), axis=1).tostring()
        shifts = np.concatenate((data.n1, data.n2, data.n3), axis=1).tostring()
        return {'g4.indices': bytes_feature(indices),
                'g4.shifts': bytes_feature(shifts)}

    def get_vap_transformer(self, atoms: Atoms):
        """
        Return the corresponding `VirtualAtomMap`.

        Parameters
        ----------
        atoms : Atoms
            An `Atoms` object.

        Returns
        -------
        vap : VirtualAtomMap
            The `VirtualAtomMap` for the given `Atoms` object.

        """
        # The mode 'reduce' is important here because chemical symbol lists of
        # ['C', 'H', 'O'] and ['C', 'O', 'H'] should be treated differently!
        formula = atoms.get_chemical_formula(mode='reduce')
        if formula not in self._vap_transformers:
            self._vap_transformers[formula] = VirtualAtomMap(
                self._max_occurs, atoms.get_chemical_symbols()
            )
        return self._vap_transformers[formula]

    def encode(self, atoms: Atoms):
        """
        Encode the `Atoms` object and return a `tf.train.Example`.
        """
        feature_list = self._encode_atoms(atoms)
        vap = self.get_vap_transformer(atoms)
        radial_metadata, angular_metadata = self.get_metadata(atoms, vap=vap)
        feature_list.update(self._encode_radial_metadata(radial_metadata))
        if isinstance(angular_metadata, AngularMetadata):
            feature_list.update(self._encode_angular_metadata(angular_metadata))
        return tf.train.Example(
            features=tf.train.Features(feature=feature_list))

    def _decode_radial_metadata(self, example: Dict[str, tf.Tensor]):
        """
        Decode the radial metadata.
        """
        with tf.name_scope("Metadata/Radial"):
            indices = tf.decode_raw(example['g2.indices'], tf.int32)
            indices.set_shape([self._nij_max * 8])
            indices = tf.reshape(
                indices, [self._nij_max, 8], name='g2.indices')
            v2g_map, ilist, jlist = tf.split(
                indices, [6, 1, 1], axis=1, name='splits')
            ilist = tf.squeeze(ilist, axis=1, name='ilist')
            jlist = tf.squeeze(jlist, axis=1, name='jlist')

            shift = tf.decode_raw(example['g2.shifts'], get_float_dtype())
            shift.set_shape([self._nij_max * 3])
            shift = tf.reshape(shift, [self._nij_max, 3], name='shift')

            return RadialMetadata(v2g_map, ilist, jlist, shift, None)

    def _decode_angular_metadata(self, example: Dict[str, tf.Tensor]):
        """
        Decode the angular metadata.
        """
        with tf.name_scope("Metadata/Angular"):
            indices = tf.decode_raw(example['g4.indices'], tf.int32)
            indices.set_shape([self._nijk_max * 9])
            indices = tf.reshape(
                indices, [self._nijk_max, 9], name='g4.indices')
            v2g_map, ilist, jlist, klist = \
                tf.split(indices, [6, 1, 1, 1], axis=1, name='splits')
            ilist = tf.squeeze(ilist, axis=1, name='ilist')
            jlist = tf.squeeze(jlist, axis=1, name='jlist')
            klist = tf.squeeze(klist, axis=1, name='klist')

            shifts = tf.decode_raw(example['g4.shifts'], get_float_dtype())
            shifts.set_shape([self._nijk_max * 9])
            shifts = tf.reshape(
                shifts, [self._nijk_max, 9], name='g4.shifts')
            n1, n2, n3 = tf.split(shifts, [3, 3, 3], axis=1, name='splits')

        return AngularMetadata(v2g_map, ilist, jlist, klist, n1, n2, n3, None)

    def _decode_example(self, example: Dict[str, tf.Tensor]):
        """
        Decode the parsed single example.
        """
        decoded = self._decode_atoms(
            example,
            max_n_atoms=self._max_n_atoms,
            use_forces=self._use_forces,
            use_stress=self._use_stress
        )
        decoded.update(self._decode_additional_properties(example))

        radial_metadata = self._decode_radial_metadata(example)
        decoded.update(radial_metadata.as_dict())
        if self._angular:
            angular_metadata = self._decode_angular_metadata(example)
            decoded.update(angular_metadata.as_dict())
        return decoded

    def decode_protobuf(self, example_proto: tf.Tensor):
        """
        Decode the scalar string Tensor, which is a single serialized Example.
        See `_parse_single_example_raw` documentation for more details.
        """
        with tf.name_scope("decoding"):

            feature_list = self.get_decode_feature_list()

            if self._use_forces:
                feature_list['forces'] = tf.FixedLenFeature([], tf.string)

            if self._use_stress:
                feature_list['stress'] = \
                    tf.FixedLenFeature([], tf.string)
                feature_list['total_pressure'] = \
                    tf.FixedLenFeature([], tf.string)

            if self._angular:
                feature_list.update({
                    'g4.indices': tf.io.FixedLenFeature([], tf.string),
                    'g4.shifts': tf.io.FixedLenFeature([], tf.string)})

            example = tf.io.parse_single_example(example_proto, feature_list)
            return self._decode_example(example)

    def _check_keys(self, features: dict):
        """
        Make sure `placeholders` contains enough keys.
        """
        assert 'positions' in features
        assert 'cell' in features
        assert 'volume' in features
        assert 'n_atoms_vap' in features
        assert 'g2.ilist' in features
        assert 'g2.jlist' in features
        assert 'g2.n1' in features
        assert 'g2.v2g_map' in features

        if self._angular:
            assert 'g4.ilist' in features
            assert 'g4.jlist' in features
            assert 'g4.klist' in features
            assert 'g4.n1' in features
            assert 'g4.n2' in features
            assert 'g4.n3' in features
            assert 'g4.v2g_map' in features

    def get_descriptors(self, batch_features: dict):
        """
        Return the graph for calculating symmetry function descriptors for the
        given batch of examples.

        This function is necessary because nested dicts are not supported by
        `tf.data.Dataset.batch`.

        Parameters
        ----------
        batch_features : dict
            A batch of raw properties provided by `tf.data.Dataset`. Each batch
            is produced by the function `decode_protobuf`.

            Here are default keys:

            * 'positions': float64 or float32, [batch_size, max_n_atoms + 1, 3]
            * 'cell': float64 or float32, [batch_size, 3, 3]
            * 'volume': float64 or float32, [batch_size, ]
            * 'n_atoms': int64, [batch_size, ]
            * 'energy': float64, [batch_size, ]
            * 'free_energy': float64, [batch_size, ]
            * 'eentropy': float64, [batch_size, ]
            * 'etemperature': float64, [batch_size, ]
            * 'forces': float64, [batch_size, max_n_atoms + 1, 3]
            * 'atom_masks': float64, [batch_size, max_n_atoms + 1]
            * 'g2.ilist': int32, [batch_size, nij_max]
            * 'g2.jlist': int32, [batch_size, nij_max]
            * 'g2.n1': float64, [batch_size, nij_max, 3]
            * 'g2.v2g_map': int32, [batch_size, nij_max, 5]

            If `self.stress` is `True`, these following keys will be provided:

            * 'stress': float64, [batch_size, 6]
            * 'total_pressure': float64, [batch_size, ]

        Returns
        -------
        descriptors : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of (element, (g, mask)) where `element` is the symbol of the
            element, `g` is the Op to compute atomic descriptors and `mask` is
            the Op to compute value masks.

        """
        self._infer_batch_size(batch_features)
        return self.build_graph(batch_features)
