# coding=utf-8
"""
The implementations of Behler's Symmetry Functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from utils import cutoff, pairwise_dist
from itertools import chain
from collections import namedtuple
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


def compute_dimension(terms: List[str], rdim, adim):
    """
    Compute the total dimension of the feature vector.

    Parameters
    ----------
    terms : List[str]
        A list of str as all k-body terms.
    rdim : int
        The total number of radial parameter sets.
    adim : int
        The total number of angular parameter sets.

    Returns
    -------
    ndim : int
        The total dimension of the feature vector.

    """
    ndim = 0
    for term in terms:
        k = len(get_elements_from_kbody_term(term))
        if k == 2:
            ndim += rdim
        else:
            ndim += adim
    return ndim


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


def build_radial_v2g_map(atoms: Atoms, kbody_terms: List[str], rc, n_etas):
    """
    Build the values-to-features mapping for radial symmetry functions.

    Parameters
    ----------
    atoms : Atoms
        An `ase.Atoms` object representing a structure.
    kbody_terms : List[str]
        A list of str as all k-body terms.
    rc : float
        The cutoff radius.
    n_etas : int
        The number of `eta` for radial symmetry functions.

    Returns
    -------
    radial : Pairs
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
    ilist, jlist, Slist = neighbor_list('ijS', atoms, rc)
    n = len(ilist)
    tlist = np.zeros_like(ilist)
    for i in range(n):
        tlist[i] = kbody_terms.index('{}{}'.format(ilist[i], jlist[i]))
    v2g_map = np.zeros_like((n * n_etas, 3), dtype=np.int32)
    for etai in range(n_etas):
        istart = etai * n
        istop = istart + n
        v2g_map[istart: istop, 0] = ilist
        v2g_map[istart: istop, 1] = etai
        v2g_map[istart: istop, 2] = tlist
    return RadialMap(v2g_map, ilist=ilist, jlist=jlist, Slist=Slist)


def build_angular_v2g_map(atoms: Atoms, rmap: RadialMap, kbody_terms):
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
    ij = np.zeros_like(v2g_map)
    ik = np.zeros_like(v2g_map)
    jk = np.zeros_like(v2g_map)
    ijS = np.zeros((total_dim, 3))
    ikS = np.zeros((total_dim, 3))
    jkS = np.zeros((total_dim, 3))

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
                v2g_map[row] = atomi, kbody_terms.index(term)

    return AngularMap(v2g_map, ij=ij, ik=ik, jk=jk, ijSlist=ijS, ikSlist=ikS,
                      jkSlist=jkS)


def radial_function(R: tf.Tensor, rc: float, g: tf.Variable, etas,
                    ilist: tf.Tensor, jlist: tf.Tensor, Slist: tf.Tensor,
                    v2g_map: tf.Tensor, cell: tf.Tensor):
    """
    The implementation of Behler's radial symmetry function for a single
    structure.
    """
    with tf.name_scope("G2"):
        with tf.name_scope("mic"):
            Ri = tf.gather(R, ilist, name='Ri')
            Rj = tf.gather(R, jlist, name='Rj')
            Dlist = Rj - Ri + tf.matmul(Slist, cell)
            dlist = tf.norm(Dlist, axis=1, name='dlist')
        rc2 = tf.constant(rc**2, dtype=tf.float64, name='rc2')
        r = tf.identity(dlist, name='r')
        r2 = tf.square(r, name='r2')
        fc_r = cutoff(r, rc=rc, name='fc_r')
        etas = tf.constant(etas, dtype=tf.float64, name='etas')
        v = tf.div(r2, rc2, name='div')
        v = tf.tensordot(etas, v, 0, name='tensordot')
        v = tf.exp(tf.negative(v, name='neg'), name='exp')
        v = tf.multiply(v, fc_r, name='vfcr')
        v = tf.reshape(v, [-1], name='flatten')
        return tf.scatter_nd_add(g, v2g_map, v)


def angular_function(R: tf.Tensor, rc: float, indices: dict, eta_list: list,
                     gamma_list: list, zeta_list: list, v_to_g_map):
    """
    The implementation of Behler's angular symmetry function for a single
    structure.
    """
    one = tf.constant(1.0, dtype=tf.float64)
    half = tf.constant(0.5, dtype=tf.float64)
    two = tf.constant(2.0, dtype=tf.float64)
    ndim = len(eta_list) * len(gamma_list) * len(zeta_list)
    N = R.shape[0]

    r = pairwise_dist(R, R, name='pdist')
    r2 = tf.square(r)
    rc2 = tf.constant(rc ** 2, dtype=tf.float64, name='rc2')

    r_ij = tf.gather_nd(r, indices['ij'], name='ij3')
    r_ik = tf.gather_nd(r, indices['ik'], name='ik3')
    r_jk = tf.gather_nd(r, indices['jk'], name='jk3')

    # Compute $\cos{(\theta_{ijk})}$ using the cosine formula
    b2c2a2 = tf.square(r_ij) + tf.square(r_ik) - tf.square(r_jk)
    bc = tf.multiply(r_ij, r_ik, name='bc')
    bc = tf.multiply(two, bc, name='2bc')
    cos_theta = tf.div(b2c2a2, bc, name='cos_theta')

    # Compute the damping term: $f_c(r_{ij}) * f_c(r_{ik}) * f_c(r_{jk})$
    fc_r_ij = cutoff(r_ij, rc, name='fc_r_ij')
    fc_r_ik = cutoff(r_ik, rc, name='fc_r_ik')
    fc_r_jk = cutoff(r_jk, rc, name='fc_r_jk')
    fc_r_ijk = fc_r_ij * fc_r_ik * fc_r_jk

    # Compute $R_{ij}^{2} + R_{ik}^{2} + R_{jk}^{2}$
    r2_ij = tf.gather_nd(r2, indices['ij'], name='r2_ij')
    r2_ik = tf.gather_nd(r2, indices['ik'], name='r2_ik')
    r2_jk = tf.gather_nd(r2, indices['jk'], name='r2_jk')
    r2_sum = r2_ij + r2_ik + r2_jk

    g = tf.Variable(tf.zeros((N, ndim), dtype=tf.float64), trainable=False)
    group = []
    row = 0
    for eta in eta_list:
        for gamma in gamma_list:
            for zeta in zeta_list:
                gamma_ = tf.constant(gamma, dtype=tf.float64)
                eta_ = tf.constant(eta, dtype=tf.float64)
                zeta_ = tf.constant(zeta, dtype=tf.float64)
                v = one + tf.multiply(gamma_, cos_theta)
                v = tf.pow(v, zeta_, name='pow')
                v = tf.pow(two, one - zeta_) * v
                v = v * tf.exp(-eta_ * r2_sum / rc2) * fc_r_ijk
                group.append(tf.scatter_nd_add(g, v_to_g_map[row], v))
                row += 1
    with tf.control_dependencies(group):
        return tf.multiply(half, g)
