# coding=utf-8
"""
The implementations of Behler's Symmetry Functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from utils import cutoff, batch_gather_positions
from misc import Defaults, safe_select, AttributeDict
from itertools import chain
from collections import Counter
from sklearn.model_selection import ParameterGrid
from typing import List, Union, Dict
from dataclasses import dataclass

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@dataclass(frozen=True)
class RadialIndexedSlices:
    """
    A `dataclass` contains indexed slices for radial functions.
    
    'v2g_map' : array_like
        A list of (atomi, etai, termi) where atomi is the index of the center
        atom, etai is the index of the `eta` and termi is the index of the
        corresponding 2-body term.
    'ilist' : array_like
        A list of first atom indices.
    'jlist' : array_like
        A list of second atom indices.
    'Slist' : array_like
        A list of (i, j) pairs where i is the index of the center atom and j is
        the index of its neighbor atom.

    """
    v2g_map: Union[np.ndarray, tf.Tensor]
    ilist: Union[np.ndarray, tf.Tensor]
    jlist: Union[np.ndarray, tf.Tensor]
    Slist: Union[np.ndarray, tf.Tensor]

    __slots__ = ["v2g_map", "ilist", "jlist", "Slist"]


@dataclass
class AngularIndexedSlices:
    """
    A `dataclass` contains indexed slices for angular functions.

    'v2g_map' : array_like
        A list of (atomi, termi) where atomi is the index of the center atom and
        termi is the index of the corresponding 3-body term.
    'ij' : array_like
        A list of (i, j) as the indices for r_{i,j}.
    'ik' : array_like
        A list of (i, k) as the indices for r_{i,k}.
    'jk' : array_like
        A list of (j, k) as the indices for r_{j,k}.
    'ijSlist' : array_like
        The cell boundary shift vectors for all r_{i,j}.
    'ikSlist' : array_like
        The cell boundary shift vectors for all r_{i,k}.
    'jkSlist' : array_like
        The cell boundary shift vectors for all r_{j,k}.

    """
    v2g_map: Union[np.ndarray, tf.Tensor]
    ij: Union[np.ndarray, tf.Tensor]
    ik: Union[np.ndarray, tf.Tensor]
    jk: Union[np.ndarray, tf.Tensor]
    ijSlist: Union[np.ndarray, tf.Tensor]
    ikSlist: Union[np.ndarray, tf.Tensor]
    jkSlist: Union[np.ndarray, tf.Tensor]

    __slots__ = ["v2g_map", "ij", "ik", "jk", "ijSlist", "ikSlist", "jkSlist"]


def compute_dimension(kbody_terms: List[str], n_etas, n_betas, n_gammas,
                      n_zetas):
    """
    Compute the total dimension of the feature vector.

    Parameters
    ----------
    kbody_terms : List[str]
        A list of str as all k-body terms.
    n_etas : int
        The number of `eta` for radial functions.
    n_betas : int
        The number of `beta` for angular functions.
    n_gammas : int
        The number of `gamma` for angular functions.
    n_zetas : int
        The number of `zeta` for angular functions.

    Returns
    -------
    total_dim : int
        The total dimension of the feature vector.
    kbody_sizes : List[int]
        The size of each k-body term.

    """
    total_dim = 0
    kbody_sizes = []
    for kbody_term in kbody_terms:
        k = len(get_elements_from_kbody_term(kbody_term))
        if k == 2:
            n = n_etas
        else:
            n = n_gammas * n_betas * n_zetas
        total_dim += n
        kbody_sizes.append(n)
    return total_dim, kbody_sizes


def get_elements_from_kbody_term(kbody_term: str) -> List[str]:
    """
    Return the atoms in the given k-body term.

    Parameters
    ----------
    kbody_term : str
        A str as the k-body term.

    Returns
    -------
    elements : List
        A list of str as the elements of the k-body term.

    """
    sel = [0]
    for i in range(len(kbody_term)):
        if kbody_term[i].isupper():
            sel.append(i + 1)
        else:
            sel[-1] += 1
    atoms = []
    for i in range(len(sel) - 1):
        atoms.append(kbody_term[sel[i]: sel[i + 1]])
    return atoms


def get_kbody_terms(elements: List[str], k_max=3):
    """
    Given a list of unique elements, construct all possible k-body terms and the
    dict mapping terms to each type of element.

    Parameters
    ----------
    elements : list
        A list of unique elements.
    k_max : int
        The maximum k for the many-body expansion.

    Returns
    -------
    terms : List[str]
        A list of str as all k-body terms.
    mapping : dict
        A dict mapping k-body terms to each type of element.
    elements : List[str]
        A list of str as the ordered unique elements.

    """
    elements = sorted(list(set(elements)))
    n = len(elements)
    k_max = max(k_max, 1)
    mapping = {}
    for i in range(n):
        term = "{}{}".format(elements[i], elements[i])
        mapping[elements[i]] = [term]
    if k_max >= 2:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                term = "{}{}".format(elements[i], elements[j])
                mapping[elements[i]].append(term)
    if k_max >= 3:
        for i in range(n):
            center = elements[i]
            for j in range(n):
                for k in range(j, n):
                    suffix = "".join(sorted([elements[j], elements[k]]))
                    term = "{}{}".format(center, suffix)
                    mapping[elements[i]].append(term)
    if k_max >= 4:
        raise ValueError("`k_max>=4` is not supported yet!")
    terms = list(chain(*[mapping[element] for element in elements]))
    return terms, mapping, elements


class IndexTransformer:
    """
    If a dataset has different stoichiometries, a global ordered symbols list
    should be kept. This class is used to transform the local indices of the
    symbols of arbitrary `Atoms` to the global indexing system.
    """
    _ISTART = 1

    def __init__(self, max_occurs: Counter, symbols: List[str]):
        """
        Initialization method.
        """
        self._max_occurs = max_occurs
        self._symbols = symbols
        self._count = sum(max_occurs.values())

        istart = IndexTransformer._ISTART
        elements = sorted(max_occurs.keys())
        offsets = np.cumsum([max_occurs[e] for e in elements])[:-1]
        offsets = np.insert(offsets, 0, 0)
        delta = Counter()
        index_map = {}
        for i, symbol in enumerate(symbols):
            idx_old = i + istart
            idx_new = offsets[elements.index(symbol)] + delta[symbol] + istart
            index_map[idx_old] = idx_new
            delta[symbol] += 1
        reverse_map = {v: k for k, v in index_map.items()}
        index_map[0] = 0
        reverse_map[0] = 0
        self.index_map = index_map
        self.reverse_map = reverse_map

    @property
    def max_occurs(self) -> Counter:
        """
        Return the maximum occurance for each type of element.
        """
        return self._max_occurs

    @property
    def symbols(self) -> List[str]:
        """
        Return a list of str as the ordered chemical symbols of the target
        stoichiometry.
        """
        return self._symbols

    @property
    def reference_symbols(self) -> List[str]:
        """
        Return a list of str as the ordered chemical symbols of the reference
        (global) stoichiometry.
        """
        return sorted(self._max_occurs.elements())

    def map(self, index_or_indices, reverse=False, ignore_extra=False):
        """
        Do the in-place index transformation and return the array.
        """
        if reverse:
            index_map = self.reverse_map
        else:
            index_map = self.index_map
        delta = int(ignore_extra)
        if not hasattr(index_or_indices, "__len__"):
            return index_map[index_or_indices] - delta
        else:
            for i in range(len(index_or_indices)):
                index_or_indices[i] = index_map[index_or_indices[i]] - delta
            return index_or_indices

    def gather(self, params, reverse=False):
        """
        Gather slices from `params` at `axis`.

        `params` should be an array of rank 2 or 3 and `axis = params.ndim - 2`.
        The dimension `params.shape[axis]` must be ether `len(self.symbols)` if
        `reverse == False` or `N = sum(max_occurs.values()) + virtual_atom` if
        `reverse == True`.
        """
        params = np.asarray(params)
        rank = np.ndim(params)
        if rank == 2:
            params = params[np.newaxis, ...]
        if not reverse and params.shape[1] == len(self._symbols):
            params = np.insert(params, 0, 0, axis=1)

        indices = []
        istart = IndexTransformer._ISTART
        if reverse:
            for i in range(istart, istart + len(self._symbols)):
                indices.append(self.index_map[i])
        else:
            for i in range(istart + self._count):
                indices.append(self.reverse_map.get(i, 0))
        params = params[:, indices]
        if rank == 2:
            params = np.squeeze(params, axis=0)
        return params


class SymmetryFunction:
    """
    A tensorflow based implementation of Behler-Parinello's Symmetry Function
    neural network model.
    """

    def __init__(self, rc, max_occurs: Counter, nij_max, nijk_max, eta=None,
                 beta=None, gamma=None, zeta=None, k_max=3):
        """
        Initialization method.
        """
        elements = sorted(max_occurs.keys())
        kbody_terms, mapping, elements = get_kbody_terms(elements, k_max=k_max)
        ndim, kbody_sizes = compute_dimension(
            kbody_terms, Defaults.n_etas, Defaults.n_betas, Defaults.n_gammas,
            Defaults.n_zetas)

        self._rc = rc
        self._mapping = mapping
        self._kbody_terms = kbody_terms
        self._kbody_sizes = kbody_sizes
        self._elements = elements
        self._ndim = ndim
        self._kbody_index = {key: kbody_terms.index(key) for key in kbody_terms}
        self._offsets = np.insert(np.cumsum(kbody_sizes), 0, 0)
        self._eta = safe_select(eta, Defaults.eta)
        self._gamma = safe_select(gamma, Defaults.gamma)
        self._beta = safe_select(beta, Defaults.beta)
        self._zeta = safe_select(zeta, Defaults.zeta)
        self._parameter_grid = ParameterGrid({'beta': self._beta,
                                              'gamma': self._gamma,
                                              'zeta': self._zeta})
        self._n_etas = len(self._eta)
        self._max_occurs = max_occurs
        self._nij_max = nij_max
        self._nijk_max = nijk_max
        self._k_max = k_max
        self._index_transformers = {}

    @property
    def cutoff(self):
        """
        Return the cutoff radius.
        """
        return self._rc

    @property
    def k_max(self):
        """
        The maximum k for the many-body expansion.
        """
        return self._k_max

    @property
    def nij_max(self):
        """
        Return the maximum allowed length of the flatten neighbor list.
        """
        return self._nij_max

    @property
    def nijk_max(self):
        """
        Return the maximum allowed length of the expanded Angle[i,j,k] list.
        """
        return self._nijk_max

    @property
    def ndim(self):
        """
        Return the total dimension of an atom descriptor vector.
        """
        return self._ndim

    @property
    def elements(self) -> List[str]:
        """
        Return a list of str as the sorted unique elements.
        """
        return self._elements

    @property
    def kbody_terms(self) -> List[str]:
        """
        A list of str as the ordered k-body terms.
        """
        return self._kbody_terms

    @property
    def kbody_sizes(self) -> List[int]:
        """
        Return a list of int as the sizes of the k-body terms.
        """
        return self._kbody_sizes

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
        stoichiometry = atoms.get_chemical_formula()
        if stoichiometry not in self._index_transformers:
            self._index_transformers[stoichiometry] = IndexTransformer(
                self._max_occurs, atoms.get_chemical_symbols()
            )
        return self._index_transformers[stoichiometry]

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

    def get_radial_indexed_slices(self, trajectory: List[Atoms]):
        """
        Return the indexed slices for radial functions.
        """
        batch_size = len(trajectory)
        nij_max = self._nij_max
        v2g_map = np.zeros((batch_size, nij_max, 3), dtype=np.int32)
        ilist = np.zeros((batch_size, nij_max), dtype=np.int32)
        jlist = np.zeros((batch_size, nij_max), dtype=np.int32)
        Slist = np.zeros((batch_size, nij_max, 3), dtype=np.int32)
        tlist = np.zeros(nij_max, dtype=np.int32)

        for idx, atoms in enumerate(trajectory):
            symbols = atoms.get_chemical_symbols()
            transformer = self.get_index_transformer(atoms)
            kilist, kjlist, kSlist = neighbor_list('ijS', atoms, self._rc)
            if self._k_max == 1:
                cols = []
                for i in range(len(kilist)):
                    if symbols[kilist[i]] == symbols[kjlist[i]]:
                        cols.append(i)
                kilist = kilist[cols]
                kjlist = kjlist[cols]
                kSlist = kSlist[cols]
            n = len(kilist)
            kilist = self._resize_to_nij_max(kilist, True)
            kjlist = self._resize_to_nij_max(kjlist, True)
            kSlist = self._resize_to_nij_max(kSlist, False)
            ilist[idx] = transformer.map(kilist.copy())
            jlist[idx] = transformer.map(kjlist.copy())
            Slist[idx] = kSlist
            tlist.fill(0)
            for i in range(n):
                symboli = symbols[kilist[i] - 1]
                symbolj = symbols[kjlist[i] - 1]
                tlist[i] = self._kbody_index['{}{}'.format(symboli, symbolj)]
            kilist = transformer.map(kilist)
            v2g_map[idx, :nij_max, 0] = idx
            v2g_map[idx, :nij_max, 1] = kilist
            v2g_map[idx, :nij_max, 2] = self._offsets[tlist]
        return RadialIndexedSlices(v2g_map, ilist, jlist, Slist)

    def get_angular_indexed_slices(self, trajectory: List[Atoms],
                                   rslices: RadialIndexedSlices):
        """
        Return the indexed slices for angular functions.
        """
        if self._k_max < 3:
            return None

        batch_size = len(trajectory)
        v2g_map = np.zeros((batch_size, self._nijk_max, 3), dtype=np.int32)
        ij = np.zeros((batch_size, self._nijk_max, 2), dtype=np.int32)
        ik = np.zeros((batch_size, self._nijk_max, 2), dtype=np.int32)
        jk = np.zeros((batch_size, self._nijk_max, 2), dtype=np.int32)
        ijS = np.zeros((batch_size, self._nijk_max, 3), dtype=np.int32)
        ikS = np.zeros((batch_size, self._nijk_max, 3), dtype=np.int32)
        jkS = np.zeros((batch_size, self._nijk_max, 3), dtype=np.int32)

        for idx, atoms in enumerate(trajectory):
            symbols = atoms.get_chemical_symbols()
            transformer = self.get_index_transformer(atoms)
            indices = {}
            vectors = {}
            for i, atomi in enumerate(rslices.ilist[idx]):
                if atomi == 0:
                    break
                if atomi not in indices:
                    indices[atomi] = []
                    vectors[atomi] = []
                indices[atomi].append(rslices.jlist[idx, i])
                vectors[atomi].append(rslices.Slist[idx, i])
            count = 0
            for atomi, nl in indices.items():
                num = len(nl)
                symboli = symbols[transformer.map(atomi, True, True)]
                prefix = '{}'.format(symboli)
                iSlist = vectors[atomi]
                for j in range(num):
                    atomj = nl[j]
                    symbolj = symbols[transformer.map(atomj, True, True)]
                    for k in range(j + 1, num):
                        atomk = nl[k]
                        symbolk = symbols[transformer.map(atomk, True, True)]
                        suffix = ''.join(sorted([symbolj, symbolk]))
                        term = '{}{}'.format(prefix, suffix)
                        ij[idx, count] = atomi, atomj
                        ik[idx, count] = atomi, atomk
                        jk[idx, count] = atomj, atomk
                        ijS[idx, count] = iSlist[j]
                        ikS[idx, count] = iSlist[k]
                        jkS[idx, count] = iSlist[k] - iSlist[j]
                        index = self._kbody_index[term]
                        v2g_map[idx, count, 0] = idx
                        v2g_map[idx, count, 1] = atomi
                        v2g_map[idx, count, 2] = self._offsets[index]
                        count += 1
        return AngularIndexedSlices(v2g_map, ij=ij, ik=ik, jk=jk, ijSlist=ijS,
                                    ikSlist=ikS, jkSlist=jkS)

    def get_indexed_slices(self, trajectory):
        """
        Return both the radial and angular indexed slices for the trajectory.
        """
        rslices = self.get_radial_indexed_slices(trajectory)
        aslices = self.get_angular_indexed_slices(trajectory, rslices)
        return rslices, aslices

    def get_radial_function_graph(self, R, cells, v2g_map, ilist, jlist, Slist,
                                  batch_size=None):
        """
        The implementation of Behler's radial symmetry function.
        """
        with tf.name_scope("G2"):
            with tf.name_scope("constants"):
                rc2 = tf.constant(self._rc**2, dtype=tf.float64, name='rc2')
                eta = tf.constant(self._eta, dtype=tf.float64, name='eta')
                R = tf.convert_to_tensor(R, dtype=tf.float64, name='R')
                Slist = tf.cast(Slist, dtype=tf.float64, name='Slist')
                cells = tf.convert_to_tensor(
                    cells, dtype=tf.float64, name='cells')
                batch_size = batch_size or R.shape[0]
                max_atoms = R.shape[1]
                v2g_map = tf.convert_to_tensor(
                    v2g_map, dtype=tf.int32, name='v2g_map')
                v2g_map.set_shape([batch_size, self._nij_max, 3])

            with tf.name_scope("rij"):
                Ri = batch_gather_positions(
                    R, ilist, batch_size=batch_size, name='Ri')
                Rj = batch_gather_positions(
                    R, jlist, batch_size=batch_size, name='Rj')
                Dlist = Rj - Ri + tf.einsum('ijk,ikl->ijl', Slist, cells)
                r = tf.norm(Dlist, axis=2, name='r')
                r2 = tf.square(r, name='r2')
                r2c = tf.div(r2, rc2, name='div')

            with tf.name_scope("fc"):
                fc_r = cutoff(r, rc=self._rc, name='fc_r')

            with tf.name_scope("features"):
                shape = tf.constant([batch_size, max_atoms, self._ndim],
                                    dtype=tf.int32, name='shape')
                g = tf.zeros(shape=shape, dtype=tf.float64, name='zeros')
                for i in range(self._n_etas):
                    with tf.name_scope("eta{}".format(i)):
                        vi = tf.exp(-eta[i] * r2c) * fc_r
                        delta = tf.constant(
                            [0, 0, i], dtype=tf.int32, name='delta')
                        v2g_map_i = tf.add(v2g_map, delta, name='v2g_map_i')
                        g += tf.scatter_nd(
                            v2g_map_i, vi, shape, 'g{}'.format(i))
                return g

    def get_angular_function_graph(self, R, cells, v2g_map, ij, ik, jk, ijS,
                                   ikS, jkS, batch_size=None):
        """
        The implementation of Behler's angular symmetry function.
        """

        def _extract(_params):
            """
            A helper function to get `beta`, `gamma` and `zeta`.
            """
            return [tf.constant(_params[key], dtype=tf.float64, name=key)
                    for key in ('beta', 'gamma', 'zeta')]

        with tf.name_scope("G4"):

            with tf.name_scope("constants"):
                one = tf.constant(1.0, dtype=tf.float64)
                two = tf.constant(2.0, dtype=tf.float64)
                rc2 = tf.constant(self._rc**2, dtype=tf.float64, name='rc2')
                cells = tf.convert_to_tensor(
                    cells, dtype=tf.float64, name='clist')
                batch_size = batch_size or R.shape[0]
                max_atoms = R.shape[1]
                v2g_map = tf.convert_to_tensor(
                    v2g_map, dtype=tf.int32, name='v2g_map')
                v2g_map.set_shape([batch_size, self._nijk_max, 3])

            with tf.name_scope("Rij"):
                Ri_ij = batch_gather_positions(R, ij[:, :, 0], batch_size, 'Ri')
                Rj_ij = batch_gather_positions(R, ij[:, :, 1], batch_size, 'Rj')
                ijS = tf.cast(ijS, dtype=tf.float64, name='ijS')
                D_ij = Rj_ij - Ri_ij + tf.einsum('ijk,ikl->ijl', ijS, cells)
                r_ij = tf.norm(D_ij, axis=2)

            with tf.name_scope("Rik"):
                Ri_ik = batch_gather_positions(R, ik[:, :, 0], batch_size, 'Ri')
                Rk_ik = batch_gather_positions(R, ik[:, :, 1], batch_size, 'Rk')
                ikS = tf.cast(ikS, dtype=tf.float64, name='ikS')
                D_ik = Rk_ik - Ri_ik + tf.einsum('ijk,ikl->ijl', ikS, cells)
                r_ik = tf.norm(D_ik, axis=2)

            with tf.name_scope("Rik"):
                Rj_jk = batch_gather_positions(R, jk[:, :, 0], batch_size, 'Rj')
                Rk_jk = batch_gather_positions(R, jk[:, :, 1], batch_size, 'Rk')
                jkS = tf.cast(jkS, dtype=tf.float64, name='jkS')
                D_jk = Rk_jk - Rj_jk + tf.einsum('ijk,ikl->ijl', jkS, cells)
                r_jk = tf.norm(D_jk, axis=2)

            # Compute $\cos{(\theta_{ijk})}$ using the cosine formula
            with tf.name_scope("cosine"):
                upper = tf.square(r_ij) + tf.square(r_ik) - tf.square(r_jk)
                lower = two * tf.multiply(r_ij, r_ik, name='bc')
                theta = tf.div(upper, lower, name='theta')

            # Compute the damping term: f_c(r_{ij}) * f_c(r_{ik}) * f_c(r_{jk})
            with tf.name_scope("fc"):
                fc_r_ij = cutoff(r_ij, self._rc, name='fc_r_ij')
                fc_r_ik = cutoff(r_ik, self._rc, name='fc_r_ik')
                fc_r_jk = cutoff(r_jk, self._rc, name='fc_r_jk')
                fc_r_ijk = fc_r_ij * fc_r_ik * fc_r_jk

            # Compute $R_{ij}^{2} + R_{ik}^{2} + R_{jk}^{2}$
            with tf.name_scope("r2"):
                r2_ij = tf.square(r_ij, name='r2_ij')
                r2_ik = tf.square(r_ik, name='r2_ik')
                r2_jk = tf.square(r_jk, name='r2_jk')
                r2 = r2_ij + r2_ik + r2_jk
                r2c = tf.div(r2, rc2, name='r2_rc2')

            with tf.name_scope("features"):
                shape = tf.constant((batch_size, max_atoms, self._ndim),
                                    dtype=tf.int32, name='shape')
                g = tf.zeros(shape=shape, dtype=tf.float64, name='zeros')
                for i, params in enumerate(self._parameter_grid):
                    with tf.name_scope("p_{}".format(i)):
                        beta, gamma, zeta = _extract(params)
                        c = (one + gamma * theta) ** zeta * two ** (1.0 - zeta)
                        vi = c * tf.exp(-beta * r2c) * fc_r_ijk
                        delta = tf.constant(
                            [0, 0, i], dtype=tf.int32, name='delta')
                        v2g_map_i = tf.add(v2g_map, delta, name='v2g_map_i')
                        g += tf.scatter_nd(
                            v2g_map_i, vi, shape, 'g{}'.format(i))
                return g

    def get_split_sizes(self) -> (List[int], Dict):
        """
        Return the row-wise and column-wise split sizes.
        """
        row_splits = [1, ]
        column_splits = {}
        for i, element in enumerate(self._elements):
            row_splits.append(self._max_occurs[element])
            column_splits[element] = [len(self._elements), i]
        return row_splits, column_splits

    def get_descriptors_graph(self, examples: AttributeDict,
                              batch_size=None):
        """
        Build the tensorflow graph for computing symmetry function descriptors
        from an input batch.

        Parameters
        ----------
        examples : AttributeDict
            A dict returned by `Dataset.next_batch` as the inputs to the graph.
        batch_size : int
            The size of one batch.

        Returns
        -------
        splits : Dict[str, tf.Tensor]
            A list of tensors. `splits[i]` represents the descriptors of element
            type `self.elements[i]`.

        """
        with tf.name_scope("Descriptors"):
            g = self.get_radial_function_graph(
                examples.positions, examples.cell, examples.rv2g,
                examples.ilist, examples.jlist, examples.Slist, batch_size)

            if self._k_max == 3:
                g += self.get_angular_function_graph(
                    examples.positions, examples.cell, examples.av2g,
                    examples.ij, examples.ik, examples.jk,
                    examples.ijS, examples.ikS, examples.jkS, batch_size)

        with tf.name_scope("Split"):
            # Atom 0 is a virtual atom.
            row_splits, column_splits = self.get_split_sizes()
            splits = tf.split(g, row_splits, axis=1, name='row_splits')[1:]
            blocks = []
            # Further split the element arrays to remove redundant zeros
            for i in range(len(splits)):
                element = self._elements[i]
                size_splits, idx = column_splits[element]
                block = tf.split(splits[i], size_splits, axis=2,
                                 name='{}_block'.format(element))[idx]
                blocks.append(block)
            return dict(zip(self._elements, blocks))
