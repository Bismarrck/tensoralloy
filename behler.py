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
from itertools import chain
from collections import namedtuple, Counter
from sklearn.model_selection import ParameterGrid
from typing import List

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


RadialMap = namedtuple(
    'RadialMap',
    ('v2g_map', 'ilist', 'jlist', 'Slist')
)

AngularMap = namedtuple(
    'AngularMap',
    ('v2g_map', 'ij', 'ik', 'jk', 'ijSlist', 'ikSlist', 'jkSlist')
)


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

    def __init__(self, elements_and_counts: Counter, atoms: Atoms,
                 virtual_atom=True):
        """
        Initialization method.
        """
        self.elements_and_counts = elements_and_counts
        self.atoms = atoms
        elements = sorted(elements_and_counts.keys())
        offsets = np.cumsum([elements_and_counts[e] for e in elements])[:-1]
        offsets = np.insert(offsets, 0, 0)
        delta = Counter()
        index_map = {}
        for i, atom in enumerate(atoms):
            symbol = atom.symbol
            extra = int(virtual_atom)
            idx_old = i + extra
            idx_new = offsets[elements.index(symbol)] + delta[symbol] + extra
            index_map[idx_old] = idx_new
            delta[atom.symbol] += 1
        reverse_map = {v: k for k, v in index_map.items()}
        if virtual_atom:
            index_map[0] = 0
            reverse_map[0] = 0
        self.index_map = index_map
        self.reverse_map = reverse_map

    def __call__(self, index_or_indices, reverse=False):
        """
        Do the in-place index transformation and return the array.
        """
        if reverse:
            index_map = self.reverse_map
        else:
            index_map = self.index_map
        if not hasattr(index_or_indices, "__len__"):
            return index_map[index_or_indices]
        else:
            for i in range(len(index_or_indices)):
                index_or_indices[i] = index_map[index_or_indices[i]]
            return index_or_indices


def build_radial_v2g_map(trajectory: List[Atoms], rc, n_etas, nij_max,
                         kbody_terms: List[str], kbody_sizes: List[int],
                         elements_and_counts: Counter):
    """
    Build the values-to-features mapping for radial symmetry functions.

    Parameters
    ----------
    trajectory : List[Atoms]
        A list of `ase.Atoms` objects.
    rc : float
        The cutoff radius.
    n_etas : int
        The number of `eta` for radial symmetry functions.
    nij_max : int
        The maximum size of `ilist`.
    kbody_terms : List[str]
        A list of str as all k-body terms.
    kbody_sizes : List[int]
        A list of int as the sizes of the k-body terms.
    elements_and_counts : Counter
        The ordered unique elements and their maximum occurances.

    Returns
    -------
    rmap : RadialMap
        A namedtuple with these properties:

        'v2g_map' : array_like
            A list of (atomi, etai, termi) where atomi is the index of the
            center atom, etai is the index of the `eta` and termi is the index
            of the corresponding 2-body term.
        'ilist' : array_like
            A list of first atom indices.
        'jlist' : array_like
            A list of second atom indices.
        'Slist' : array_like
            A list of (i, j) pairs where i is the index of the center atom and j
            is the index of its neighbor atom.
        'cell': array_like
            The boundary cell of the structure.

    """

    def _align(alist, is_indices=True):
        if np.ndim(alist) == 1:
            nlist = np.zeros(nij_max, dtype=np.int32)
        else:
            nlist = np.zeros([nij_max] + list(alist.shape[1:]), dtype=np.int32)
        length = len(alist)
        nlist[:length] = alist
        if is_indices:
            nlist[:length] += 1
        return nlist

    batch_size = len(trajectory)
    v2g_map = np.zeros((batch_size, nij_max * n_etas, 3), dtype=np.int32)
    offsets = np.insert(np.cumsum(kbody_sizes)[:-1], 0, 0)
    ilist = np.zeros((batch_size, nij_max), dtype=np.int32)
    jlist = np.zeros((batch_size, nij_max), dtype=np.int32)
    Slist = np.zeros((batch_size, nij_max, 3), dtype=np.int32)
    tlist = np.zeros(nij_max, dtype=np.int32)

    for idx, atoms in enumerate(trajectory):
        transformer = IndexTransformer(elements_and_counts, atoms)
        symbols = atoms.get_chemical_symbols()
        kilist, kjlist, kSlist = neighbor_list('ijS', atoms, rc)
        n = len(kilist)
        kilist = _align(kilist, True)
        kjlist = _align(kjlist, True)
        kSlist = _align(kSlist, False)
        ilist[idx] = transformer(kilist.copy())
        jlist[idx] = transformer(kjlist.copy())
        Slist[idx] = kSlist
        tlist.fill(0)
        for i in range(n):
            symboli = symbols[kilist[i] - 1]
            symbolj = symbols[kjlist[i] - 1]
            tlist[i] = kbody_terms.index('{}{}'.format(symboli, symbolj))
        kilist = transformer(kilist)
        for etai in range(n_etas):
            istart = etai * nij_max
            istop = istart + nij_max
            v2g_map[idx, istart: istop, 0] = idx
            v2g_map[idx, istart: istop, 1] = kilist
            v2g_map[idx, istart: istop, 2] = offsets[tlist] + etai
    return RadialMap(v2g_map, ilist=ilist, jlist=jlist, Slist=Slist)


def build_angular_v2g_map(trajectory: List[Atoms], rmap: RadialMap, nijk_max,
                          kbody_terms: List[str], kbody_sizes: List[int],
                          elements_and_counts: Counter):
    """
    Build the values-to-features mapping for angular symmetry functions.

    Parameters
    ----------
    trajectory : List[Atoms]
        A list of `ase.Atoms` objects.
    rmap : RadialMap
        The mapping for radial symmetry functions.
    nijk_max : int
        The maximum number of `Angle[i,j,k]` that one `Atoms` object has.
    kbody_terms : List[str]
        A list of str as all k-body terms.
    kbody_sizes : List[int]
        A list of int as the sizes of the k-body terms.
    elements_and_counts : Counter
        The ordered unique elements and their maximum occurances.

    Returns
    -------
    amap : AngularMap
        A namedtuple with these properties:

        'v2g_map' : array_like
            A list of (atomi, termi) where atomi is the index of the center atom
            and termi is the index of the corresponding 3-body term.
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
    batch_size = len(trajectory)
    offsets = np.insert(np.cumsum(kbody_sizes)[:-1], 0, 0)
    v2g_map = np.zeros((batch_size, nijk_max, 3), dtype=np.int32)
    ij = np.zeros((2, batch_size, nijk_max), dtype=np.int32)
    ik = np.zeros((2, batch_size, nijk_max), dtype=np.int32)
    jk = np.zeros((2, batch_size, nijk_max), dtype=np.int32)
    ijS = np.zeros((batch_size, nijk_max, 3), dtype=np.int32)
    ikS = np.zeros((batch_size, nijk_max, 3), dtype=np.int32)
    jkS = np.zeros((batch_size, nijk_max, 3), dtype=np.int32)

    for idx, atoms in enumerate(trajectory):
        transformer = IndexTransformer(elements_and_counts, atoms)
        symbols = ['X'] + atoms.get_chemical_symbols()
        nl_indices = {}
        nl_vectors = {}
        for i, atomi in enumerate(rmap.ilist[idx]):
            if atomi == 0:
                break
            nl_indices[atomi] = nl_indices.get(atomi, []) + [rmap.jlist[idx, i]]
            nl_vectors[atomi] = nl_vectors.get(atomi, []) + [rmap.Slist[idx, i]]
        count = 0
        for atomi, nl in nl_indices.items():
            num = len(nl)
            prefix = '{}'.format(symbols[transformer(atomi, True)])
            iSlist = nl_vectors[transformer(atomi, True)]
            for j in range(num):
                atomj = nl[j]
                for k in range(j + 1, num):
                    atomk = nl[k]
                    suffix = ''.join(sorted([
                        symbols[transformer(atomj, True)],
                        symbols[transformer(atomk, True)]])
                    )
                    term = '{}{}'.format(prefix, suffix)
                    if 'X' in term:
                        print('warning')
                    ij[:, idx, count] = atomi, atomj
                    ik[:, idx, count] = atomi, atomk
                    jk[:, idx, count] = atomj, atomk
                    ijS[idx, count] = iSlist[j]
                    ikS[idx, count] = iSlist[k]
                    jkS[idx, count] = iSlist[k] - iSlist[j]
                    v2g_map[idx, count, 0] = idx
                    v2g_map[idx, count, 1] = atomi
                    v2g_map[idx, count, 2] = offsets[kbody_terms.index(term)]
                    count += 1
    return AngularMap(v2g_map, ij=ij, ik=ik, jk=jk, ijSlist=ijS, ikSlist=ikS,
                      jkSlist=jkS)


def radial_function(R, rc, v2g_map, clist, etas, ilist, jlist, Slist,
                    total_dim):
    """
    The implementation of Behler's radial symmetry function for a single
    structure.
    """
    with tf.name_scope("G2"):

        with tf.name_scope("constants"):
            rc2 = tf.constant(rc ** 2, dtype=tf.float64, name='rc2')
            etas = tf.constant(etas, dtype=tf.float64, name='etas')
            R = tf.convert_to_tensor(R, dtype=tf.float64, name='R')
            Slist = tf.convert_to_tensor(Slist, dtype=tf.float64, name='Slist')
            clist = tf.convert_to_tensor(clist, dtype=tf.float64, name='clist')
            batch_size = R.shape[0]
            n_atoms = R.shape[1]

        with tf.name_scope("rij"):
            Ri = batch_gather_positions(R, ilist, name='Ri')
            Rj = batch_gather_positions(R, jlist, name='Rj')
            Dlist = Rj - Ri + tf.einsum('ijk,ikl->ijl', Slist, clist)
            r = tf.norm(Dlist, axis=2, name='r')
            r2 = tf.square(r, name='r2')
            r2c = tf.div(r2, rc2, name='div')

        with tf.name_scope("fc"):
            fc_r = cutoff(r, rc=rc, name='fc_r')

        with tf.name_scope("features"):
            shape = tf.constant([batch_size, n_atoms, total_dim],
                                dtype=tf.int32, name='shape')
            v = tf.exp(-tf.einsum('i,jk->jik', etas, r2c))
            v = tf.einsum('ijk,ik->ijk', v, fc_r)
            v = tf.reshape(v, [batch_size, -1], name='flatten')
            return tf.scatter_nd(v2g_map, v, shape, name='g')


def angular_function(R, rc, v2g_map, clist, grid: ParameterGrid, ij, ik,
                     jk, ijS, ikS, jkS, total_dim):
    """
    The implementation of Behler's angular symmetry function for a single
    structure.
    """
    with tf.name_scope("G4"):

        with tf.name_scope("constants"):
            one = tf.constant(1.0, dtype=tf.float64)
            two = tf.constant(2.0, dtype=tf.float64)
            rc2 = tf.constant(rc**2, dtype=tf.float64, name='rc2')
            clist = tf.convert_to_tensor(clist, dtype=tf.float64, name='clist')
            v2g_base = tf.constant(v2g_map, dtype=tf.int32, name='v2g_base')
            batch_size = R.shape[0]
            n_atoms = R.shape[1]

        with tf.name_scope("Rij"):
            Ri_ij = batch_gather_positions(R, ij[0])
            Rj_ij = batch_gather_positions(R, ij[1])
            ijS = tf.convert_to_tensor(ijS, dtype=tf.float64, name='ijS')
            D_ij = Rj_ij - Ri_ij + tf.einsum('ijk,ikl->ijl', ijS, clist)
            r_ij = tf.norm(D_ij, axis=2)

        with tf.name_scope("Rik"):
            Ri_ik = batch_gather_positions(R, ik[0])
            Rk_ik = batch_gather_positions(R, ik[1])
            ikS = tf.convert_to_tensor(ikS, dtype=tf.float64, name='ijS')
            D_ik = Rk_ik - Ri_ik + tf.einsum('ijk,ikl->ijl', ikS, clist)
            r_ik = tf.norm(D_ik, axis=2)

        with tf.name_scope("Rik"):
            Rj_jk = batch_gather_positions(R, jk[0])
            Rk_jk = batch_gather_positions(R, jk[1])
            jkS = tf.convert_to_tensor(jkS, dtype=tf.float64, name='ijS')
            D_jk = Rk_jk - Rj_jk + tf.einsum('ijk,ikl->ijl', jkS, clist)
            r_jk = tf.norm(D_jk, axis=2)

        # Compute $\cos{(\theta_{ijk})}$ using the cosine formula
        with tf.name_scope("cosine"):
            upper = tf.square(r_ij) + tf.square(r_ik) - tf.square(r_jk)
            lower = two * tf.multiply(r_ij, r_ik, name='bc')
            theta = tf.div(upper, lower, name='theta')

        # Compute the damping term: $f_c(r_{ij}) * f_c(r_{ik}) * f_c(r_{jk})$
        with tf.name_scope("fc"):
            fc_r_ij = cutoff(r_ij, rc, name='fc_r_ij')
            fc_r_ik = cutoff(r_ik, rc, name='fc_r_ik')
            fc_r_jk = cutoff(r_jk, rc, name='fc_r_jk')
            fc_r_ijk = fc_r_ij * fc_r_ik * fc_r_jk

        # Compute $R_{ij}^{2} + R_{ik}^{2} + R_{jk}^{2}$
        with tf.name_scope("r2"):
            r2_ij = tf.square(r_ij, name='r2_ij')
            r2_ik = tf.square(r_ik, name='r2_ik')
            r2_jk = tf.square(r_jk, name='r2_jk')
            r2 = r2_ij + r2_ik + r2_jk
            r2c = tf.div(r2, rc2, name='r2_rc2')

        with tf.name_scope("features"):
            shape = tf.constant((batch_size, n_atoms, total_dim),
                                dtype=tf.int32, name='shape')
            g = tf.zeros(shape=shape, dtype=tf.float64, name='zeros')
            for row, params in enumerate(grid):
                with tf.name_scope("p{}".format(row)):
                    gamma = tf.constant(
                        params['gamma'], dtype=tf.float64, name='gamma')
                    zeta = tf.constant(
                        params['zeta'], dtype=tf.float64, name='zeta')
                    beta = tf.constant(
                        params['beta'], dtype=tf.float64, name='beta')
                    c = (one + gamma * theta)**zeta * two**(1.0 - zeta)
                    v = c * tf.exp(-beta * r2c) * fc_r_ijk
                    step = tf.constant([0, 0, row], dtype=tf.int32, name='step')
                    v2g_row = tf.add(v2g_base, step, name='v2g_row')
                    g = g + tf.scatter_nd(v2g_row, v, shape, 'g{}'.format(row))
            return g
