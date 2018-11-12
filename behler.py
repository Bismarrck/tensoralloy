# coding=utf-8
"""
The implementations of Behler's Symmetry Functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from itertools import chain
from collections import Counter
from sklearn.model_selection import ParameterGrid
from typing import List, Union, Dict
from dataclasses import dataclass

from descriptor import AtomicDescriptor
from descriptor import DescriptorTransformer, BatchDescriptorTransformer
from utils import cutoff, batch_gather_positions
from misc import Defaults, AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@dataclass(frozen=True)
class G2IndexedSlices:
    """
    A `dataclass` contains indexed slices for the symmetry function G2.
    
    'v2g_map' : array_like
        A list of (atomi, etai, termi) where atomi is the index of the center
        atom, etai is the index of the `eta` and termi is the index of the
        corresponding 2-body term.
    'ilist' : array_like
        A list of first atom indices.
    'jlist' : array_like
        A list of second atom indices.
    'shift' : array_like
        The cell boundary shift vectors, `shift[k] = Slist[k] @ cell`.

    """
    v2g_map: Union[np.ndarray, tf.Tensor]
    ilist: Union[np.ndarray, tf.Tensor]
    jlist: Union[np.ndarray, tf.Tensor]
    shift: Union[np.ndarray, tf.Tensor]

    __slots__ = ["v2g_map", "ilist", "jlist", "shift"]


@dataclass
class G4IndexedSlices:
    """
    A `dataclass` contains indexed slices for the symmetry function G4.

    'v2g_map' : array_like
        A list of (atomi, termi) where atomi is the index of the center atom and
        termi is the index of the corresponding 3-body term.
    'ij' : array_like
        A list of (i, j) as the indices for r_{i,j}.
    'ik' : array_like
        A list of (i, k) as the indices for r_{i,k}.
    'jk' : array_like
        A list of (j, k) as the indices for r_{j,k}.
    'ij_shift' : array_like
        The cell boundary shift vectors for all r_{i,j}.
    'ik_shift' : array_like
        The cell boundary shift vectors for all r_{i,k}.
    'jk_shift' : array_like
        The cell boundary shift vectors for all r_{j,k}.

    """
    v2g_map: Union[np.ndarray, tf.Tensor]
    ij: Union[np.ndarray, tf.Tensor]
    ik: Union[np.ndarray, tf.Tensor]
    jk: Union[np.ndarray, tf.Tensor]
    ij_shift: Union[np.ndarray, tf.Tensor]
    ik_shift: Union[np.ndarray, tf.Tensor]
    jk_shift: Union[np.ndarray, tf.Tensor]

    __slots__ = ["v2g_map", "ij", "ik", "jk",
                 "ij_shift", "ik_shift", "jk_shift"]


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
        self._n_atoms = sum(max_occurs.values()) + IndexTransformer._ISTART

        istart = IndexTransformer._ISTART
        elements = sorted(max_occurs.keys())
        offsets = np.cumsum([max_occurs[e] for e in elements])[:-1]
        offsets = np.insert(offsets, 0, 0)
        delta = Counter()
        index_map = {}
        mask = np.zeros(self._n_atoms, dtype=bool)
        for i, symbol in enumerate(symbols):
            idx_old = i + istart
            idx_new = offsets[elements.index(symbol)] + delta[symbol] + istart
            index_map[idx_old] = idx_new
            delta[symbol] += 1
            mask[idx_new] = True
        reverse_map = {v: k for k, v in index_map.items()}
        index_map[0] = 0
        reverse_map[0] = 0
        self._mask = mask
        self._index_map = index_map
        self._reverse_map = reverse_map

    @property
    def n_atoms(self):
        """
        Return the number of atoms (including the virtual atom at index 0).
        """
        return self._n_atoms

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

    @property
    def mask(self) -> np.ndarray:
        """
        Return a `bool` array.
        """
        return self._mask

    def map(self, index_or_indices, reverse=False, ignore_extra=False):
        """
        Do the in-place index transformation and return the array.
        """
        if reverse:
            index_map = self._reverse_map
        else:
            index_map = self._index_map
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
                indices.append(self._index_map[i])
        else:
            for i in range(self._n_atoms):
                indices.append(self._reverse_map.get(i, 0))
        params = params[:, indices]
        if rank == 2:
            params = np.squeeze(params, axis=0)
        return params


class SymmetryFunction(AtomicDescriptor):
    """
    A tensorflow based implementation of Behler-Parinello's SymmetryFunction
    descriptor.
    """

    def __init__(self, rc, elements, eta=Defaults.eta, beta=Defaults.beta,
                 gamma=Defaults.gamma, zeta=Defaults.zeta, k_max=3,
                 periodic=True):
        """
        Initialization method.

        Parameters
        ----------
        rc : float
            The cutoff radius.
        elements : List[str]
            A list of str as the ordered elements.
        eta : array_like
            The `eta` for radial functions.
        beta : array_like
            The `beta` for angular functions.
        gamma : array_like
            The `beta` for angular functions.
        zeta : array_like
            The `beta` for angular functions.
        k_max : int
            The maximum k for the many-body expansion.
        periodic : bool
            If False, some Ops of the computation graph will be ignored and this
            can only proceed non-periodic molecules.

        """
        kbody_terms, mapping, elements = get_kbody_terms(elements, k_max=k_max)
        ndim, kbody_sizes = compute_dimension(kbody_terms, len(eta), len(beta),
                                              len(gamma), len(zeta))

        self._rc = rc
        self._k_max = k_max
        self._elements = elements
        self._n_elements = len(elements)
        self._periodic = periodic
        self._mapping = mapping
        self._kbody_terms = kbody_terms
        self._kbody_sizes = kbody_sizes
        self._ndim = ndim
        self._kbody_index = {key: kbody_terms.index(key) for key in kbody_terms}
        self._offsets = np.insert(np.cumsum(kbody_sizes), 0, 0)
        self._eta = np.asarray(eta)
        self._gamma = np.asarray(gamma)
        self._beta = np.asarray(beta)
        self._zeta = np.asarray(zeta)
        self._parameter_grid = ParameterGrid({'beta': self._beta,
                                              'gamma': self._gamma,
                                              'zeta': self._zeta})

    @property
    def cutoff(self):
        """
        Return the cutoff radius.
        """
        return self._rc

    @property
    def k_max(self):
        """
        Return the maximum k for the many-body expansion scheme.
        """
        return self._k_max

    @property
    def elements(self):
        """
        Return a list of str as the sorted unique elements.
        """
        return self._elements

    @property
    def periodic(self):
        """
        Return True if this can be applied to periodic structures.
        For non-periodic molecules some Ops can be ignored.
        """
        return self._periodic

    @property
    def ndim(self):
        """
        Return the total dimension of an atom descriptor vector.
        """
        return self._ndim

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

    def _gather(self, R, ilist, name):
        """
        A wrapper of the gather function.
        """
        return tf.gather(R, ilist, name=name)

    def _get_rij(self, R, ilist, jlist, shift, name):
        """
        Return the subgraph to compute `rij`.
        """
        with tf.name_scope(name):
            Ri = self._gather(R, ilist, 'Ri')
            Rj = self._gather(R, jlist, 'Rj')
            Dij = tf.subtract(Rj, Ri, name='Dij')
            if self._periodic:
                Dij = tf.add(Dij, shift, name='pbc')
            return tf.norm(Dij, axis=-1, name='r')

    @staticmethod
    def _get_v2g_map_delta(index):
        """
        Return the step delta for `v2g_map`.
        """
        return tf.constant([0, index], dtype=tf.int32, name='delta')

    def _get_g2_graph_for_eta(self, index, shape, r2c, fc_r, v2g_map):
        """
        Return the subgraph to compute G2 with the given `eta`.

        Parameters
        ----------
        index : int
            The index of the `eta` to use.
        shape : Sized
            The shape of the descriptor.
        r2c : tf.Tensor
            The `float64` tensor of `r**2 / rc**2`.
        fc_r : tf.Tensor
            The `float64` tensor of `cutoff(r)`.
        v2g_map : tf.Tensor
            The `int32` tensor as the mapping from 'v' to 'g'.

        Returns
        -------
        g : tf.Tensor
            A `float64` tensor with the input `shape` as the fingerprints
            contributed by the radial function G2 with the given `eta`.

        """
        with tf.name_scope(f"eta{index}"):
            eta = tf.constant(self._eta[index], dtype=tf.float64, name='eta')
            delta = self._get_v2g_map_delta(index)
            v_index = tf.exp(-tf.multiply(eta, r2c, 'eta_r2c')) * fc_r
            v2g_map_index = tf.add(v2g_map, delta, f'v2g_map_{index}')
            return tf.scatter_nd(v2g_map_index, v_index, shape, f'g{index}')

    def _get_g_shape(self, placeholders):
        """
        Return the shape of the descriptor matrix.
        """
        return [placeholders.n_atoms, self._ndim]

    def _get_v2g_map(self, placeholders, fn_name: str):
        assert fn_name in ('g2', 'g4')
        return tf.identity(placeholders[fn_name].v2g_map, name='v2g_map')

    def _get_g2_graph(self, placeholders: AttributeDict):
        """
        The implementation of Behler's G2 symmetry function.
        """
        with tf.name_scope("G2"):

            r = self._get_rij(placeholders.positions,
                              placeholders.g2.ilist,
                              placeholders.g2.jlist,
                              placeholders.g2.shift,
                              name='rij')
            r2 = tf.square(r, name='r2')
            rc2 = tf.constant(self._rc**2, dtype=tf.float64, name='rc2')
            r2c = tf.div(r2, rc2, name='div')
            fc_r = cutoff(r, rc=self._rc, name='fc_r')

            with tf.name_scope("v2g_map"):
                v2g_map = self._get_v2g_map(placeholders, 'g2')

            with tf.name_scope("features"):
                shape = self._get_g_shape(placeholders)
                # TODO: maybe `tf.while` can be used here
                blocks = []
                for index in range(len(self._eta)):
                    blocks.append(self._get_g2_graph_for_eta(
                        index, shape, r2c, fc_r, v2g_map)
                    )
                return tf.add_n(blocks, name='g')

    @staticmethod
    def _extract(_params):
        """
        A helper function to get `beta`, `gamma` and `zeta`.
        """
        return [tf.constant(_params[key], dtype=tf.float64, name=key)
                for key in ('beta', 'gamma', 'zeta')]

    def _get_g4_graph_for_params(self, index, shape, theta, r2c, fc_r, v2g_map):
        """
        Return the subgraph to compute angular descriptors with the given
        parameters set.

        Parameters
        ----------
        index : int
            The index of the `eta` to use.
        shape : Sized
            The shape of the descriptor.
        theta : tf.Tensor
            The `float64` tensor of `cos(theta)`.
        r2c : tf.Tensor
            The `float64` tensor of `r**2 / rc**2`.
        fc_r : tf.Tensor
            The `float64` tensor of `cutoff(r)`.
        v2g_map : tf.Tensor
            The `int32` tensor as the mapping from 'v' to 'g'.

        Returns
        -------
        g : tf.Tensor
            A `float64` tensor with the input `shape` as the fingerprints
            contributed by the angular function G4 with the given parameters.

        """
        with tf.name_scope(f"grid{index}"):
            beta, gamma, zeta = self._extract(self._parameter_grid[index])
            delta = self._get_v2g_map_delta(index)
            c = (1.0 + gamma * theta) ** zeta * 2.0 ** (1.0 - zeta)
            v_index = tf.multiply(c * tf.exp(-beta * r2c), fc_r, f'v_{index}')
            v2g_map_index = tf.add(v2g_map, delta, name=f'v2g_map_{index}')
            return tf.scatter_nd(v2g_map_index, v_index, shape, f'g{index}')

    def _get_g4_graph(self, placeholders):
        """
        The implementation of Behler's angular symmetry function.
        """
        with tf.name_scope("G4"):

            rij = self._get_rij(placeholders.positions,
                                placeholders.g4.ij.ilist,
                                placeholders.g4.ij.jlist,
                                placeholders.g4.shift.ij,
                                name='rij')
            rik = self._get_rij(placeholders.positions,
                                placeholders.g4.ik.ilist,
                                placeholders.g4.ik.klist,
                                placeholders.g4.shift.ik,
                                name='rik')
            rjk = self._get_rij(placeholders.positions,
                                placeholders.g4.jk.jlist,
                                placeholders.g4.jk.klist,
                                placeholders.g4.shift.jk,
                                name='rjk')

            rij2 = tf.square(rij, name='rij2')
            rik2 = tf.square(rik, name='rik2')
            rjk2 = tf.square(rjk, name='rjk2')
            rc2 = tf.constant(self._rc ** 2, dtype=tf.float64, name='rc2')
            r2 = tf.add_n([rij2, rik2, rjk2], name='r2')
            r2c = tf.div(r2, rc2, name='r2_rc2')

            with tf.name_scope("cosine"):
                upper = tf.subtract(rij2 + rik2, rjk2, name='upper')
                lower = tf.multiply(2 * rij, rik, name='lower')
                theta = tf.div(upper, lower, name='theta')

            with tf.name_scope("fc"):
                fc_rij = cutoff(rij, self._rc, name='fc_rij')
                fc_rik = cutoff(rik, self._rc, name='fc_rik')
                fc_rjk = cutoff(rjk, self._rc, name='fc_rjk')
                fc_r = tf.multiply(fc_rij, fc_rik * fc_rjk, 'fc_r')

            with tf.name_scope("v2g_map"):
                v2g_map = self._get_v2g_map(placeholders, 'g4')

            with tf.name_scope("features"):
                shape = self._get_g_shape(placeholders)
                blocks = []
                for index in range(len(self._parameter_grid)):
                    blocks.append(self._get_g4_graph_for_params(
                        index, shape, theta, r2c, fc_r, v2g_map))
                return tf.add_n(blocks, name='g')

    def _get_row_split_sizes(self, placeholders):
        """
        Return the sizes of the rowwise splitted subsets of `g`.
        """
        return placeholders.row_splits

    @staticmethod
    def _get_row_split_axis():
        """
        Return the axis to rowwise split `g`.
        """
        return 0

    def _get_column_split_sizes(self):
        """
        Return the sizes of the column-wise splitted subsets of `g`.
        """
        column_splits = {}
        for i, element in enumerate(self._elements):
            column_splits[element] = [len(self._elements), i]
        return column_splits

    @staticmethod
    def _get_column_split_axis():
        """
        Return the axis to column-wise split `g`.
        """
        return 1

    def _split_descriptors(self, g, placeholders) -> Dict[str, tf.Tensor]:
        """
        Split the descriptors into `N_element` subsets.
        """
        with tf.name_scope("Split"):
            row_split_sizes = self._get_row_split_sizes(placeholders)
            row_split_axis = self._get_row_split_axis()
            column_split_sizes = self._get_column_split_sizes()
            column_split_axis = self._get_column_split_axis()
            splits = tf.split(
                g, row_split_sizes, axis=row_split_axis, name='rows')[1:]
            if len(self._elements) > 1:
                # Further split the element arrays to remove redundant zeros
                blocks = []
                for i in range(len(splits)):
                    element = self._elements[i]
                    size_splits, idx = column_split_sizes[element]
                    block = tf.split(splits[i],
                                     size_splits,
                                     axis=column_split_axis,
                                     name='{}_block'.format(element))[idx]
                    blocks.append(block)
            else:
                blocks = splits
            return dict(zip(self._elements, blocks))

    def build_graph(self, placeholders: AttributeDict):
        """
        Get the tensorflow based computation graph of the Symmetry Function.
        """
        with tf.name_scope("Behler"):
            g = self._get_g2_graph(placeholders)
            if self._k_max == 3:
                g += self._get_g4_graph(placeholders)
        return self._split_descriptors(g, placeholders)


class BatchSymmetryFunction(SymmetryFunction):
    """
    A special implementation of Behler-Parinello's Symmetry Function for batch
    training and evaluations.
    """

    def __init__(self, rc, max_occurs: Counter, elements: List[str],
                 nij_max: int, nijk_max: int, batch_size: int, eta=Defaults.eta,
                 beta=Defaults.beta, gamma=Defaults.gamma, zeta=Defaults.zeta,
                 k_max=3, periodic=True):
        """
        Initialization method.
        """
        super(BatchSymmetryFunction, self).__init__(
            rc=rc, elements=elements, eta=eta, beta=beta, gamma=gamma,
            zeta=zeta, k_max=k_max, periodic=periodic)

        self._max_occurs = max_occurs
        self._max_n_atoms = sum(max_occurs.values()) + 1
        self._nij_max = nij_max
        self._nijk_max = nijk_max
        self._batch_size = batch_size

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
    def batch_size(self):
        """
        Return the batch size.
        """
        return self._batch_size

    def _gather(self, R, ilist, name):
        """
        A wrapper of `batch_gather_positions`.
        """
        return batch_gather_positions(R, ilist, self._batch_size, name)

    def _get_g_shape(self, _):
        """
        Return the shape of the descriptor matrix.
        """
        return [self._batch_size, self._max_n_atoms, self._ndim]

    def _get_v2g_map_batch_indexing_matrix(self, fn_name='g2'):
        """
        Return an `int32` matrix of shape `[batch_size, ndim, 3]` to rebuild the
        batch indexing of a `v2g_map`.
        """
        if fn_name == 'g2':
            ndim = self._nij_max
        else:
            ndim = self._nijk_max
        indexing_matrix = np.zeros((self._batch_size, ndim, 3), dtype=np.int32)
        for i in range(self._batch_size):
            indexing_matrix[i] += [i, 0, 0]
        return indexing_matrix

    @staticmethod
    def _get_v2g_map_delta(index):
        return tf.constant([0, 0, index], tf.int32, name='delta')

    def _get_v2g_map(self, placeholders, fn_name: str):
        """

        """
        indexing = self._get_v2g_map_batch_indexing_matrix(fn_name=fn_name)
        return tf.add(placeholders[fn_name].v2g_map, indexing, name='v2g_map')

    def _get_row_split_sizes(self, _):
        row_splits = [1, ]
        for i, element in enumerate(self._elements):
            row_splits.append(self._max_occurs[element])
        return row_splits

    @staticmethod
    def _get_row_split_axis():
        return 1

    @staticmethod
    def _get_column_split_axis():
        return 2


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

    def get_graph(self):
        """
        Return the graph to compute symmetry function descriptors.
        """
        if not self._placeholders:
            self._initialize_placeholders()
        return self.build_graph(self.placeholders)

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
                 nij_max: int, nijk_max: int, batch_size: int, eta=Defaults.eta,
                 beta=Defaults.beta, gamma=Defaults.gamma, zeta=Defaults.zeta,
                 k_max=3, periodic=True):
        """
        Initialization method.
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
