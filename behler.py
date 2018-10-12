# coding=utf-8
"""
The implementations of Behler's Symmetry Functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from utils import cutoff
from itertools import chain
from collections import namedtuple
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
    return terms, mapping


def build_radial_v2g_map(atoms: Atoms, rc, n_etas, kbody_terms: List[str],
                         kbody_sizes: List[int]):
    """
    Build the values-to-features mapping for radial symmetry functions.

    Parameters
    ----------
    atoms : Atoms
        An `ase.Atoms` object representing a structure.
    rc : float
        The cutoff radius.
    n_etas : int
        The number of `eta` for radial symmetry functions.
    kbody_terms : List[str]
        A list of str as all k-body terms.
    kbody_sizes : List[int]
        A list of int as the sizes of the k-body terms.

    Returns
    -------
    radial : RadialMap
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
    symbols = atoms.get_chemical_symbols()
    ilist, jlist, Slist = neighbor_list('ijS', atoms, rc)
    n = len(ilist)
    tlist = np.zeros_like(ilist)
    for i in range(n):
        tlist[i] = kbody_terms.index(
            '{}{}'.format(symbols[ilist[i]], symbols[jlist[i]]))
    v2g_map = np.zeros((n * n_etas, 2), dtype=np.int32)
    offsets = np.insert(np.cumsum(kbody_sizes)[:-1], 0, 0)
    for etai in range(n_etas):
        istart = etai * n
        istop = istart + n
        v2g_map[istart: istop, 0] = ilist
        v2g_map[istart: istop, 1] = offsets[tlist] + etai
    return RadialMap(v2g_map, ilist=ilist, jlist=jlist, Slist=Slist)


def build_angular_v2g_map(atoms: Atoms, rmap: RadialMap,
                          kbody_terms: List[str], kbody_sizes: List[int]):
    """
    Build the values-to-features mapping for angular symmetry functions.

    Parameters
    ----------
    atoms : Atoms
        An `ase.Atoms` object representing a structure.
    rmap : RadialMap
        The mapping for radial symmetry functions.
    kbody_terms : List[str]
        A list of str as all k-body terms.
    kbody_sizes : List[int]
        A list of int as the sizes of the k-body terms.

    Returns
    -------
    angular : AngularMap
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
    symbols = atoms.get_chemical_symbols()
    offsets = np.insert(np.cumsum(kbody_sizes)[:-1], 0, 0)
    nl_indices = {}
    nl_shifts = {}
    for i, atomi in enumerate(rmap.ilist):
        nl_indices[atomi] = nl_indices.get(atomi, []) + [rmap.jlist[i]]
        nl_shifts[atomi] = nl_shifts.get(atomi, []) + [rmap.Slist[i]]

    total_dim = 0
    for atomi, nl in nl_indices.items():
        n = len(nl)
        total_dim += (n - 1 + 1) * (n - 1) // 2

    v2g_map = np.zeros((total_dim, 2), dtype=np.int32)
    ij = np.zeros_like(v2g_map, dtype=np.int32)
    ik = np.zeros_like(v2g_map, dtype=np.int32)
    jk = np.zeros_like(v2g_map, dtype=np.int32)
    ijS = np.zeros((total_dim, 3), dtype=np.int32)
    ikS = np.zeros((total_dim, 3), dtype=np.int32)
    jkS = np.zeros((total_dim, 3), dtype=np.int32)

    row = 0
    for atomi, nl in nl_indices.items():
        num = len(nl)
        prefix = '{}'.format(symbols[atomi])
        iSlist = nl_shifts[atomi]
        for j in range(num):
            atomj = nl[j]
            for k in range(j + 1, num):
                atomk = nl[k]
                suffix = ''.join(sorted([symbols[atomj], symbols[atomk]]))
                term = '{}{}'.format(prefix, suffix)
                ij[row] = atomi, atomj
                ik[row] = atomi, atomk
                jk[row] = atomj, atomk
                ijS[row] = iSlist[j]
                ikS[row] = iSlist[k]
                jkS[row] = iSlist[k] - iSlist[j]
                v2g_map[row] = atomi, offsets[kbody_terms.index(term)]
                row += 1

    return AngularMap(v2g_map, ij=ij, ik=ik, jk=jk, ijSlist=ijS, ikSlist=ikS,
                      jkSlist=jkS)


def radial_function(R: tf.Tensor, rc, v2g_map, cell, etas, ilist, jlist, Slist,
                    total_dim):
    """
    The implementation of Behler's radial symmetry function for a single
    structure.
    """
    with tf.name_scope("G2"):

        with tf.name_scope("constants"):
            rc2 = tf.constant(rc ** 2, dtype=tf.float64, name='rc2')
            etas = tf.constant(etas, dtype=tf.float64, name='etas')

        with tf.name_scope("rij"):
            Ri = tf.gather(R, ilist, name='Ri')
            Rj = tf.gather(R, jlist, name='Rj')
            Slist = tf.convert_to_tensor(Slist, dtype=tf.float64, name='Slist')
            Dlist = Rj - Ri + tf.matmul(Slist, cell)
            r = tf.norm(Dlist, axis=1, name='r')
            r2 = tf.square(r, name='r2')
            r2c = tf.div(r2, rc2, name='div')

        with tf.name_scope("fc"):
            fc_r = cutoff(r, rc=rc, name='fc_r')

        with tf.name_scope("features"):
            g = tf.Variable(tf.zeros((R.shape[0], total_dim), dtype=tf.float64),
                            name='g', trainable=False)
            v = tf.exp(-tf.tensordot(etas, r2c, axes=0)) * fc_r
            v = tf.reshape(v, [-1], name='flatten')
            g = tf.scatter_nd_add(g, v2g_map, v)
            return tf.convert_to_tensor(g, dtype=tf.float64, name='gr')


def angular_function(R: tf.Tensor, rc, v2g_map, cell, grid: ParameterGrid, ij,
                     ik, jk, ijS, ikS, jkS, total_dim):
    """
    The implementation of Behler's angular symmetry function for a single
    structure.
    """
    with tf.name_scope("G4"):

        with tf.name_scope("constants"):
            one = tf.constant(1.0, dtype=tf.float64)
            two = tf.constant(2.0, dtype=tf.float64)
            rc2 = tf.constant(rc**2, dtype=tf.float64, name='rc2')
            v2g_base = tf.constant(v2g_map, dtype=tf.int32, name='v2g_base')

        with tf.name_scope("Rij"):
            Ri_ij = tf.gather(R, ij[:, 0])
            Rj_ij = tf.gather(R, ij[:, 1])
            ijS = tf.convert_to_tensor(ijS, dtype=tf.float64, name='ijS')
            D_ij = Rj_ij - Ri_ij + tf.matmul(ijS, cell)
            r_ij = tf.norm(D_ij, axis=1)

        with tf.name_scope("Rik"):
            Ri_ik = tf.gather(R, ik[:, 0])
            Rk_ik = tf.gather(R, ik[:, 1])
            ikS = tf.convert_to_tensor(ikS, dtype=tf.float64, name='ijS')
            D_ik = Rk_ik - Ri_ik + tf.matmul(ikS, cell)
            r_ik = tf.norm(D_ik, axis=1)

        with tf.name_scope("Rik"):
            Rj_jk = tf.gather(R, jk[:, 0])
            Rk_jk = tf.gather(R, jk[:, 1])
            jkS = tf.convert_to_tensor(jkS, dtype=tf.float64, name='ijS')
            D_jk = Rk_jk - Rj_jk + tf.matmul(jkS, cell)
            r_jk = tf.norm(D_jk, axis=1)

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
            g = tf.Variable(tf.zeros((R.shape[0], total_dim), dtype=tf.float64),
                            name='g', trainable=False)
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
                    step = tf.constant([0, row], dtype=tf.int32, name='step')
                    v2g_row = tf.add(v2g_base, step, name='v2g_row')
                    g = tf.scatter_nd_add(g, v2g_row, v)
            return tf.convert_to_tensor(g, dtype=tf.float64, name='ga')
