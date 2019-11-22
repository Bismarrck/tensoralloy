#!coding=utf-8
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
from tensoralloy.utils import get_pulay_stress
from tensoralloy.transformer.indexed_slices import G2IndexedSlices
from tensoralloy.transformer.indexed_slices import G4IndexedSlices
from tensoralloy.transformer.vap import VirtualAtomMap
from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _get_iaxis(mode: tf_estimator.ModeKeys):
    if mode == tf_estimator.ModeKeys.PREDICT:
        return 0
    else:
        return 1


def get_g2_map(atoms: Atoms,
               rc: float,
               interactions: Dict[str, int],
               vap: VirtualAtomMap,
               mode: tf_estimator.ModeKeys,
               nij_max: int = None,
               dtype=np.float32):
    """
    Build the base `v2g_map`.
    """
    ilist, jlist, n1 = neighbor_list('ijS', atoms, rc)
    nij = len(ilist)
    if nij_max is None:
        nij_max = nij

    iaxis = _get_iaxis(mode)
    g2_map = np.zeros((nij_max, iaxis + 4), dtype=np.int32)
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
    for index in range(nij):
        atomi = ilist[index]
        if atomi not in counters:
            counters[atomi] = Counter()

        # The indices of the pair atoms
        g2_map[index, iaxis + 2] = counters[atomi][tlist[index]]
        counters[atomi][tlist[index]] += 1

    # The mask
    g2_map[:, iaxis + 3] = ilist > 0

    return G2IndexedSlices(v2g_map=g2_map, ilist=ilist, jlist=jlist, n1=n1)


def get_g4_map(atoms: Atoms,
               rc: float,
               radial_interactions: Dict[str, int],
               angular_interactions: Dict[str, int],
               vap: VirtualAtomMap,
               mode: tf_estimator.ModeKeys,
               g2: G2IndexedSlices = None,
               nijk_max: int = None,
               dtype=np.float32):
    """
    Build the base `v2g_map`.
    """
    if g2 is None:
        g2 = get_g2_map(atoms=atoms,
                        rc=rc,
                        interactions=radial_interactions,
                        vap=vap,
                        mode=mode,
                        dtype=dtype)
    indices = {}
    vectors = {}
    for i, atom_vap_i in enumerate(g2.ilist):
        if atom_vap_i == 0:
            break
        if atom_vap_i not in indices:
            indices[atom_vap_i] = []
            vectors[atom_vap_i] = []
        indices[atom_vap_i].append(g2.jlist[i])
        vectors[atom_vap_i].append(g2.n1[i])

    if nijk_max is None:
        nijk = 0
        for atomi, nl in indices.items():
            n = len(nl)
            nijk += (n - 1) * n // 2
        nijk_max = nijk

    iaxis = _get_iaxis(mode)
    g4_map = np.zeros((nijk_max, iaxis + 4), dtype=np.int32)
    g4_map.fill(0)
    ilist = np.zeros(nijk_max, dtype=np.int32)
    jlist = np.zeros(nijk_max, dtype=np.int32)
    klist = np.zeros(nijk_max, dtype=np.int32)
    n1 = np.zeros((nijk_max, 3), dtype=g2.n1.dtype)
    n2 = np.zeros((nijk_max, 3), dtype=g2.n1.dtype)
    n3 = np.zeros((nijk_max, 3), dtype=g2.n1.dtype)
    symbols = atoms.get_chemical_symbols()
    counters = {}

    count = 0
    for atom_vap_i, nl in indices.items():
        atom_local_i = vap.gsl_to_local_map[atom_vap_i]
        symboli = symbols[atom_local_i]
        prefix = '{}'.format(symboli)
        for j in range(len(nl)):
            atom_vap_j = nl[j]
            atom_local_j = vap.gsl_to_local_map[atom_vap_j]
            symbolj = symbols[atom_local_j]
            for k in range(j + 1, len(nl)):
                atom_vap_k = nl[k]
                atom_local_k = vap.gsl_to_local_map[atom_vap_k]
                symbolk = symbols[atom_local_k]
                interaction = '{}{}'.format(
                    prefix, ''.join(sorted([symbolj, symbolk])))
                ilist[count] = atom_vap_i
                jlist[count] = atom_vap_j
                klist[count] = atom_vap_k
                n1[count] = vectors[atom_vap_i][j]
                n2[count] = vectors[atom_vap_i][k]
                n3[count] = vectors[atom_vap_i][k] - vectors[atom_vap_i][j]
                index = angular_interactions[interaction]
                if atom_vap_i not in counters:
                    counters[atom_vap_i] = Counter()
                    if index not in counters[atom_vap_i]:
                        counters[atom_vap_i][index] = 0
                g4_map[count, iaxis + 0] = index
                g4_map[count, iaxis + 1] = atom_vap_i
                g4_map[count, iaxis + 2] = counters[atom_vap_i][index]
                g4_map[count, iaxis + 3] = 1
                counters[atom_vap_i][index] += 1
                count += 1
    return G4IndexedSlices(g4_map, ilist, jlist, klist, n1, n2, n3)


class UniversalTransformer(DescriptorTransformer):
    """
    The universal transformer for all models.
    """

    gather_fn = staticmethod(tf.gather)

    def __init__(self, elements: List[str], rcut, acut=None, angular=False,
                 periodic=True):
        """
        The initialization metho.d

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

        """
        DescriptorTransformer.__init__(self)

        for element in elements:
            if element not in chemical_symbols:
                raise ValueError(f"{element} is not a valid chemical symbol!")

        if angular and acut is None:
            acut = rcut

        all_kbody_terms, kbody_terms_for_element, elements = \
            get_kbody_terms(elements, angular=angular)

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

    def as_dict(self) -> Dict:
        """
        Return a JSON serializable dict representation of this transformer.
        """
        return {'class': self.__class__.__name__, 'rcut': self._rcut,
                'acut': self._acut, 'angular': self._angular,
                'periodic': self._periodic}

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

    def get_rij(self, R, cell, ilist, jlist, shift, name):
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
            key = "nnl_a_max"
            ndim = self._max_na_terms
        else:
            key = "nnl_max"
            ndim = self._max_nr_terms
        return [np.int32(ndim), features["n_atoms_vap"], features[key]]

    def get_v2g_map(self, features: dict, angular=False):
        """
        A wrapper function to get `v2g_map`.
        """
        if angular:
            key = "g4.v2g_map"
        else:
            key = "g2.v2g_map"
        splits = tf.split(features[key], [-1, 1], axis=1)
        v2g_map = tf.identity(splits[0], name='v2g_map')
        v2g_mask = tf.identity(splits[1], name='v2g_mask')
        return v2g_map, v2g_mask

    def get_row_split_sizes(self, features):
        """
        Return the sizes of the rowwise splitted subsets of `g`.
        """
        return features["row_splits"]

    @staticmethod
    def get_row_split_axis():
        """
        Return the axis to rowwise split `g`.
        """
        return 1

    def _split_descriptors(self, features, g, mask):
        """
        Split the descriptors into `N_element` subsets.
        """
        with tf.name_scope("Split"):
            split_sizes = self.get_row_split_sizes(features)
            axis = self.get_row_split_axis()

            # `axis` should increase by one for `g` because `g` is created by
            # `tf.concat((gr, gx, gy, gz), axis=0, name='g')`
            rows = tf.split(g, split_sizes, axis=axis + 1, name='rows')[1:]

            # Use the original axis
            masks = tf.split(mask, split_sizes, axis=axis, name='masks')[1:]
            return dict(zip(self._elements, zip(rows, masks)))

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
        assert 'g2.ilist' in features
        assert 'g2.jlist' in features
        assert 'g2.n1' in features
        assert 'g2.v2g_map' in features

        if self._angular:
            assert 'nnl_a_max' in features
            assert 'g4.ilist' in features
            assert 'g4.jlist' in features
            assert 'g4.klist' in features
            assert 'g4.n1' in features
            assert 'g4.n2' in features
            assert 'g4.n3' in features
            assert 'g4.v2g_map' in features

    def build_radial_graph(self, features: dict):
        with tf.name_scope("Radial"):
            rr, dij = self.get_rij(features["positions"],
                                   features["cell"],
                                   features["g2.ilist"],
                                   features["g2.jlist"],
                                   features["g2.n1"],
                                   name='rij')
            shape = self.get_g_shape(features, angular=False)
            v2g_map, v2g_mask = self.get_v2g_map(features, angular=False)

            dx = tf.identity(dij[..., 0], name='dijx')
            dy = tf.identity(dij[..., 1], name='dijy')
            dz = tf.identity(dij[..., 2], name='dijz')

            gr = tf.expand_dims(tf.scatter_nd(v2g_map, rr, shape), 0, name='gr')
            gx = tf.expand_dims(tf.scatter_nd(v2g_map, dx, shape), 0, name='gx')
            gy = tf.expand_dims(tf.scatter_nd(v2g_map, dy, shape), 0, name='gy')
            gz = tf.expand_dims(tf.scatter_nd(v2g_map, dz, shape), 0, name='gz')

            g = tf.concat((gr, gx, gy, gz), axis=0, name='g')

            v2g_mask = tf.squeeze(v2g_mask, axis=self.get_row_split_axis())
            mask = tf.scatter_nd(v2g_map, v2g_mask, shape)
            mask = tf.cast(mask, dtype=rr.dtype, name='mask')
            return self._split_descriptors(features, g, mask)

    def build_angular_graph(self, features: dict):
        with tf.name_scope("Angular"):
            rij, dij = self.get_rij(features['positions'],
                                    features['cell'],
                                    features['g4.ilist'],
                                    features['g4.jlist'],
                                    features['g4.n1'],
                                    name='rij')
            rik, dik = self.get_rij(features['positions'],
                                    features['cell'],
                                    features['g4.ilist'],
                                    features['g4.klist'],
                                    features['g4.n2'],
                                    name='rik')
            rjk, djk = self.get_rij(features['positions'],
                                    features['cell'],
                                    features['g4.jlist'],
                                    features['g4.klist'],
                                    features['g4.n3'],
                                    name='rjk')
            shape = self.get_g_shape(features, angular=True)
            v2g_map, v2g_mask = self.get_v2g_map(features, angular=True)

            ijx = tf.identity(dij[..., 0], name='dijx')
            ijy = tf.identity(dij[..., 1], name='dijy')
            ijz = tf.identity(dij[..., 2], name='dijz')
            ikx = tf.identity(dik[..., 0], name='dikx')
            iky = tf.identity(dik[..., 1], name='diky')
            ikz = tf.identity(dik[..., 2], name='dikz')
            jkx = tf.identity(djk[..., 0], name='djkx')
            jky = tf.identity(djk[..., 1], name='djky')
            jkz = tf.identity(djk[..., 2], name='djkz')

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
            g = tf.concat((grij, gijx, gijy, gijz,
                           grik, gikx, giky, gikz,
                           grjk, gjkx, gjky, gjkz), axis=0, name='g')

            v2g_mask = tf.squeeze(v2g_mask, axis=self.get_row_split_axis())
            mask = tf.scatter_nd(v2g_map, v2g_mask, shape)
            mask = tf.cast(mask, dtype=rij.dtype, name='mask')
            return self._split_descriptors(features, g, mask)

    def build_graph(self, features: dict):
        """
        Get the tensorflow based computation graph of the EAM model.

        Returns
        -------
        ops : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of {element: (descriptor, mask)}.

            * `descriptor`: [4, max_n_terms, n_atoms_plus_virt, nnl_max]
                Represents th

        """
        self._check_keys(features)

        with tf.name_scope(f"Transformer"):
            g2 = self.build_radial_graph(features)
            if self._angular:
                g4 = self.build_angular_graph(features)
            else:
                g4 = None
            return {"radial": g2, "angular": g4}

    def _initialize_placeholders(self):
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
            self._placeholders["row_splits"] = self._create_int_1d(
                name='row_splits', d0=self.n_elements + 1)
            self._placeholders["compositions"] = self._create_float_1d(
                dtype=dtype, name='compositions')
            self._placeholders["g2.ilist"] = self._create_int_1d('g2.ilist')
            self._placeholders["g2.jlist"] = self._create_int_1d('g2.jlist')
            self._placeholders["g2.n1"] = self._create_float_2d(
                dtype=dtype, d0=None, d1=3, name='g2.n1')
            self._placeholders["g2.v2g_map"] = self._create_int_2d(
                d0=None, d1=4, name='g2.v2g_map')

            if self._angular:
                self._placeholders["nnl_a_max"] = self._create_int('nnl_a_max')
                self._placeholders["g4.v2g_map"] = self._create_int_2d(
                    d0=None, d1=4, name='g4.v2g_map')
                self._placeholders["g4.ilist"] = self._create_int_1d('g4.ilist')
                self._placeholders["g4.jlist"] = self._create_int_1d('g4.jlist')
                self._placeholders["g4.klist"] = self._create_int_1d('g4.klist')
                self._placeholders["g4.n1"] = self._create_float_2d(
                    dtype=dtype, d0=None, d1=3, name='g4.n1')
                self._placeholders["g4.n2"] = self._create_float_2d(
                    dtype=dtype, d0=None, d1=3, name='g4.n2')
                self._placeholders["g4.n3"] = self._create_float_2d(
                    dtype=dtype, d0=None, d1=3, name='g4.n3')

        return self._placeholders

    def _get_indexed_slices(self, atoms, vap: VirtualAtomMap):
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
        g2 : G2IndexedSlices
            The radial indexed slices for the target `Atoms` object.
        g4 : G4IndexedSlices
            The angular indexed slices for the target `Atoms` object.

        """

        radial_interactions = {}
        angular_interactions = {}
        for element in self._elements:
            kbody_terms = self._kbody_terms_for_element[element]
            for i, kbody_term in enumerate(kbody_terms):
                if len(get_elements_from_kbody_term(kbody_term)) == 2:
                    radial_interactions[kbody_term] = i
                else:
                    angular_interactions[kbody_term] = i - len(self._elements)

        dtype = get_float_dtype().as_numpy_dtype
        g2 = get_g2_map(atoms=atoms,
                        rc=self._rcut,
                        interactions=radial_interactions,
                        vap=vap,
                        mode=tf_estimator.ModeKeys.PREDICT,
                        nij_max=None,
                        dtype=dtype)
        if self._angular:
            if np.round(self._acut - self._rcut, 2) == 0.0:
                ref_g2 = g2
            else:
                ref_g2 = None
            g4 = get_g4_map(atoms,
                            rc=self._acut,
                            g2=ref_g2,
                            radial_interactions=radial_interactions,
                            angular_interactions=angular_interactions,
                            vap=vap,
                            mode=tf_estimator.ModeKeys.PREDICT,
                            nijk_max=None,
                            dtype=dtype)
        else:
            g4 = None
        return g2, g4

    def _get_np_features(self, atoms: Atoms):
        """
        Return a dict of features (Numpy or Python objects).
        """
        np_dtype = get_float_dtype().as_numpy_dtype

        vap = self.get_vap_transformer(atoms)
        g2, g4 = self._get_indexed_slices(atoms, vap)

        positions = vap.map_positions(atoms.positions)

        # `max_n_atoms` must be used because every element shall have at least
        # one feature row (though it could be all zeros, a dummy or virtual row)
        cell = atoms.get_cell(complete=True)
        volume = atoms.get_volume()
        atom_masks = vap.atom_masks.astype(np_dtype)
        pulay_stress = get_pulay_stress(atoms)
        splits = [1] + [vap.max_occurs[e] for e in self._elements]
        compositions = self._get_compositions(atoms)

        feed_dict = dict()

        feed_dict["positions"] = positions.astype(np_dtype)
        feed_dict["n_atoms_vap"] = np.int32(vap.max_vap_natoms)
        feed_dict["nnl_max"] = np.int32(g2.v2g_map[:, 2].max() + 1)
        feed_dict["atom_masks"] = atom_masks.astype(np_dtype)
        feed_dict["cell"] = cell.array.astype(np_dtype)
        feed_dict["volume"] = np_dtype(volume)
        feed_dict["pulay_stress"] = np_dtype(pulay_stress)
        feed_dict["compositions"] = compositions.astype(np_dtype)
        feed_dict["row_splits"] = np.int32(splits)
        feed_dict.update(g2.as_dict())

        if self._angular:
            feed_dict['nnl_a_max'] = np.int32(g4.v2g_map[:, 2].max() + 1)
            feed_dict.update(g4.as_dict())

        return feed_dict

    def get_feed_dict(self, atoms: Atoms):
        """
        Return the feed dict.
        """
        feed_dict = {}

        if not self._placeholders:
            self._initialize_placeholders()
        placeholders = self._placeholders

        for key, value in self._get_np_features(atoms).items():
            feed_dict[placeholders[key]] = value

        return feed_dict

    def get_constant_features(self, atoms: Atoms):
        """
        Return a dict of constant feature tensors for the given `Atoms`.
        """
        feed_dict = dict()
        with tf.name_scope("Constants"):
            for key, val in self._get_np_features(atoms).items():
                feed_dict[key] = tf.convert_to_tensor(val, name=key)
        return feed_dict
