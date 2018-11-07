# coding=utf-8
"""
This module defines the unit tests for implementations of Behler's symmetry
functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import nose
from unittest import TestCase
from utils import cutoff
from behler import get_kbody_terms, compute_dimension
from behler import SymmetryFunction, IndexTransformer
from nose.tools import assert_less, assert_equal, assert_list_equal
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid
from itertools import product
from typing import List, Union
from collections import Counter
from misc import Defaults, AttributeDict
from dataclasses import dataclass

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


Pd3O2 = Atoms(symbols='Pd3O2', pbc=np.array([True, True, False], dtype=bool),
              cell=np.array([[7.78, 0., 0.],
                             [0., 5.50129076, 0.],
                             [0., 0., 15.37532269]]),
              positions=np.array([[3.89, 0., 8.37532269],
                                  [0., 2.75064538, 8.37532269],
                                  [3.89, 2.75064538, 8.37532269],
                                  [5.835, 1.37532269, 8.5],
                                  [5.835, 7.12596807, 8.]]))

# A permutation of Pd3O2
Pd2O2Pd = Atoms(symbols='Pd2O2Pd',
                pbc=np.array([True, True, False], dtype=bool),
                cell=np.array([[7.78, 0., 0.],
                               [0., 5.50129076, 0.],
                               [0., 0., 15.37532269]]),
                positions=np.array([[3.89, 0., 8.37532269],
                                    [0., 2.75064538, 8.37532269],
                                    [5.835, 1.37532269, 8.5],
                                    [5.835, 7.12596807, 8.],
                                    [3.89, 2.75064538, 8.37532269]]))

qm7m = AttributeDict(
    max_occurs=Counter({'H': 8, 'C': 5, 'O': 2}),
    nij_max=198,
    nijk_max=1217,
    trajectory=read('test_files/qm7m/qm7m.xyz', index=':', format='xyz'),
)

grid = ParameterGrid({'beta': Defaults.beta, 'gamma': Defaults.gamma,
                      'zeta': Defaults.zeta})


@dataclass(frozen=True)
class LegacyRadialIndexedSlices:
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

    Notes
    -----
    This is the legacy version, for testing only.

    """
    v2g_map: Union[np.ndarray, tf.Tensor]
    ilist: Union[np.ndarray, tf.Tensor]
    jlist: Union[np.ndarray, tf.Tensor]
    Slist: Union[np.ndarray, tf.Tensor]

    __slots__ = ["v2g_map", "ilist", "jlist", "Slist"]


@dataclass
class LegacyAngularIndexedSlices:
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

    Notes
    -----
    This is the legacy version, for testing only.

    """
    v2g_map: Union[np.ndarray, tf.Tensor]
    ij: Union[np.ndarray, tf.Tensor]
    ik: Union[np.ndarray, tf.Tensor]
    jk: Union[np.ndarray, tf.Tensor]
    ijSlist: Union[np.ndarray, tf.Tensor]
    ikSlist: Union[np.ndarray, tf.Tensor]
    jkSlist: Union[np.ndarray, tf.Tensor]

    __slots__ = ["v2g_map", "ij", "ik", "jk", "ijSlist", "ikSlist", "jkSlist"]



class IndexTransformerTest(TestCase):
    """
    A set of unit tests for the class `IndexTransformer`.
    """

    def setUp(self):
        """
        Setup the tests.
        """
        symbols = Pd3O2.get_chemical_symbols()
        self.max_occurs = Counter({'Pd': 4, 'O': 5})
        self.clf = IndexTransformer(self.max_occurs, symbols)

    def test_forward(self):
        assert_equal(len(self.clf.reference_symbols), 9)

        array = np.expand_dims([1, 2, 3, 4, 5], axis=1)
        results = self.clf.gather(array, reverse=False).flatten().tolist()
        assert_list_equal(results, [0, 4, 5, 0, 0, 0, 1, 2, 3, 0])

        array = np.expand_dims([7, 1, 2, 3, 4, 5], axis=1)
        results = self.clf.gather(array, reverse=False).flatten().tolist()
        assert_list_equal(results, [7, 4, 5, 7, 7, 7, 1, 2, 3, 7])

    def test_reverse(self):
        array = np.expand_dims([0, 4, 5, 0, 0, 0, 1, 2, 3, 0], axis=1)
        results = self.clf.gather(array, reverse=True).flatten().tolist()
        assert_list_equal(results, [1, 2, 3, 4, 5])

    def test_call(self):
        assert_equal(self.clf.map(1), 6)
        assert_equal(self.clf.map(1, ignore_extra=True), 5)

    def test_mask(self):
        assert_list_equal(self.clf.mask.tolist(),
                          [0, 1, 1, 0, 0, 0, 1, 1, 1, 0])

    def test_permutation(self):
        symbols = Pd2O2Pd.get_chemical_symbols()
        clf = IndexTransformer(self.max_occurs, symbols)
        assert_list_equal(clf.mask.tolist(),
                          [0, 1, 1, 0, 0, 0, 1, 1, 1, 0])
        assert_less(np.abs(clf.gather(Pd2O2Pd.positions) -
                           self.clf.gather(Pd3O2.positions)).max(), 1e-8)


def cutoff_fxn(r, rc):
    """
    The vectorized cutoff function.

    Args:
      r: a `float` or an array as the interatomic distances.
      rc: a `float` as the cutoff radius.

    Returns:
      fr: the damped `r`.

    """
    return (np.cos(np.minimum(r / rc, 1.0) * np.pi) + 1.0) * 0.5


def get_neighbour_list(r, rc):
    """
    A naive implementation of calculating neighbour list for non-periodic
    molecules.

    Args:
      r: a `[N, N]` array as the pairwise interatomic distances.
      rc: a float as the cutoff radius.

    Returns:
      neighbours: a `[N, N]` boolean array as the neighbour results.

    """
    neighbours = r < rc
    np.fill_diagonal(neighbours, False)
    return neighbours


def get_radial_fingerprints_v1(coords, r, rc, etas):
    """
    Calculate the fingerprints from the radial functions for a mono-atomic
    molecule.

    Args:
      coords: a `[N, 3]` array as the cartesian coordinates.
      r: a `[N, N]` array as the pairwise distance matrix.
      rc: a float as the cutoff radius.
      etas: a `List[float]` as the `eta` in the radial functions.

    Returns:
      x: a `[N, M]` array as the radial fingerprints.

    """

    params = np.array(etas)
    ndim = len(params)
    natoms = len(coords)
    x = np.zeros((natoms, ndim))
    nl = get_neighbour_list(r, rc)
    rc2 = rc ** 2
    fr = cutoff_fxn(r, rc)

    for k, etak in enumerate(etas):
        for i in range(natoms):
            v = 0.0
            ri = coords[i]
            for j in range(natoms):
                if not nl[i, j]:
                    continue
                rs = coords[j]
                ris = np.sum(np.square(ri - rs))
                v += np.exp(-etak * ris / rc2) * fr[i, j]
            x[i, k] = v
    return x


def get_augular_fingerprints_v1(coords, r, rc, etas, gammas, zetas):
    """
    Calculate the fingerprints from the augular functions for a mono-atomic
    molecule.

    Args:
      coords: a `[N, 3]` array as the cartesian coordinates.
      r: a `[N, N]` matrix as the pairwise distances.
      rc: a float as the cutoff radius.
      etas: a `List[float]` as the `eta` in the radial functions.
      gammas: a `List[float]` as the `lambda` in the angular functions.
      zetas: a `List[float]` as the `zeta` in the angular functions.

    Returns:
      x: a `[N, M]` array as the augular fingerprints.

    """
    natoms = len(r)
    params = np.array(list(product(etas, gammas, zetas)))
    ndim = len(params)
    rr = r + np.eye(natoms) * rc
    r2 = rr ** 2
    rc2 = rc ** 2
    fc = cutoff_fxn(rr, rc)
    x = np.zeros((natoms, ndim))
    for row, (etak, gammak, zetak) in enumerate(params):
        for i in range(natoms):
            for j in range(natoms):
                if j == i:
                    continue
                for k in range(natoms):
                    if k == i or k == j:
                        continue
                    rij = coords[j] - coords[i]
                    rik = coords[k] - coords[i]
                    theta = np.dot(rij, rik) / (r[i, j] * r[i, k])
                    v = (1 + gammak * theta)**zetak
                    v *= np.exp(-etak * (r2[i, j] + r2[i, k] + r2[j, k]) / rc2)
                    v *= fc[i, j] * fc[j, k] * fc[i, k]
                    x[i, row] += v
        x[:, row] *= 2.0**(1 - zetak)
    return x / 2.0


def legacy_build_radial_v2g_map(atoms: Atoms, rc, n_etas,
                                kbody_terms: List[str], kbody_sizes: List[int]):
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
    radial : RadialIndexedSlices
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
    ilist = ilist.astype(np.int32)
    jlist = jlist.astype(np.int32)
    Slist = Slist.astype(np.int32)
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
    return LegacyRadialIndexedSlices(v2g_map, ilist=ilist, jlist=jlist,
                                     Slist=Slist)


def legacy_build_angular_v2g_map(atoms: Atoms, rmap: LegacyRadialIndexedSlices,
                                 kbody_terms: List[str],
                                 kbody_sizes: List[int]):
    """
    Build the values-to-features mapping for angular symmetry functions.

    Parameters
    ----------
    atoms : Atoms
        An `ase.Atoms` object representing a structure.
    rmap : RadialIndexedSlices
        The mapping for radial symmetry functions.
    kbody_terms : List[str]
        A list of str as all k-body terms.
    kbody_sizes : List[int]
        A list of int as the sizes of the k-body terms.

    Returns
    -------
    angular : AngularIndexedSlices
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
    indices = {}
    vectors = {}
    for i, atomi in enumerate(rmap.ilist):
        if atomi not in indices:
            indices[atomi] = []
            vectors[atomi] = []
        indices[atomi].append(rmap.jlist[i])
        vectors[atomi].append(rmap.Slist[i])
    total_dim = 0
    for atomi, nl in indices.items():
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
    for atomi, nl in indices.items():
        num = len(nl)
        prefix = '{}'.format(symbols[atomi])
        iSlist = vectors[atomi]
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

    return LegacyAngularIndexedSlices(v2g_map, ij=ij, ik=ik, jk=jk, ijSlist=ijS,
                                      ikSlist=ikS, jkSlist=jkS)


def radial_function(R, rc, v2g_map, cell, etas, ilist, jlist, Slist, total_dim):
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
            shape = tf.constant([R.shape[0], total_dim], tf.int32, name='shape')
            v = tf.exp(-tf.tensordot(etas, r2c, axes=0)) * fc_r
            v = tf.reshape(v, [-1], name='flatten')
            return tf.scatter_nd(v2g_map, v, shape, name='g')


def angular_function(R, rc, v2g_map, cell, params_grid, ij, ik, jk, ijS, ikS,
                     jkS, total_dim):
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
            shape = tf.constant((R.shape[0], total_dim), tf.int32, name='shape')
            g = tf.zeros(shape=shape, dtype=tf.float64, name='zeros')
            for row, params in enumerate(params_grid):
                with tf.name_scope("p{}".format(row)):
                    gamma_ = tf.constant(
                        params['gamma'], dtype=tf.float64, name='gamma')
                    zeta_ = tf.constant(
                        params['zeta'], dtype=tf.float64, name='zeta')
                    beta_ = tf.constant(
                        params['beta'], dtype=tf.float64, name='beta')
                    c = (one + gamma_ * theta)**zeta_ * two**(1.0 - zeta_)
                    v = c * tf.exp(-beta_ * r2c) * fc_r_ijk
                    step = tf.constant([0, row], dtype=tf.int32, name='step')
                    v2g_row = tf.add(v2g_base, step, name='v2g_row')
                    g = g + tf.scatter_nd(v2g_row, v, shape, 'g{}'.format(row))
            return g


def legacy_symmetry_function(atoms: Atoms, rc: float, name_scope: str):
    """
    Compute the symmetry function descriptors for unit tests.
    """
    symbols = atoms.get_chemical_symbols()
    kbody_terms, _, _ = get_kbody_terms(list(set(symbols)), k_max=3)
    total_dim, kbody_sizes = compute_dimension(kbody_terms,
                                               Defaults.n_etas,
                                               Defaults.n_betas,
                                               Defaults.n_gammas,
                                               Defaults.n_zetas)
    tf.reset_default_graph()
    tf.enable_eager_execution()

    with tf.name_scope(name_scope):
        R = tf.constant(atoms.positions, dtype=tf.float64, name='R')
        cell = tf.constant(atoms.cell, tf.float64, name='cell')
        rmap = legacy_build_radial_v2g_map(atoms, rc, Defaults.n_etas,
                                           kbody_terms, kbody_sizes)
        amap = legacy_build_angular_v2g_map(atoms, rmap, kbody_terms,
                                            kbody_sizes)
        gr = radial_function(R, rc, rmap.v2g_map, cell, Defaults.eta,
                             rmap.ilist, rmap.jlist, rmap.Slist, total_dim)
        ga = angular_function(R, rc, amap.v2g_map, cell, grid, amap.ij, amap.ik,
                              amap.jk, amap.ijSlist, amap.ikSlist, amap.jkSlist,
                              total_dim)
        return tf.add(gr, ga, name='g'), kbody_terms, kbody_sizes


def test_monoatomic_molecule():
    """
    Test computing descriptors of a single mono-atomic molecule.
    """
    atoms = read('test_files/B28.xyz', index=0, format='xyz')
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc([False, False, False])
    coords = atoms.get_positions()
    rr = pairwise_distances(coords)
    rc = 6.0
    zr = get_radial_fingerprints_v1(coords, rr, rc, Defaults.eta)
    za = get_augular_fingerprints_v1(coords, rr, rc, Defaults.beta,
                                     Defaults.gamma, Defaults.zeta)
    z = np.hstack((zr, za))
    g, _, _ = legacy_symmetry_function(atoms, rc=rc, name_scope='B28')
    assert_less(np.abs(z - g).max(), 1e-8)


def test_single_structure():
    """
    Test computing descriptors of a single multi-elements periodic structure.
    """
    amp = np.load('test_files/amp_Pd3O2.npz')['g']
    g, _, _ = legacy_symmetry_function(Pd3O2, rc=6.5, name_scope='Pd3O2')
    assert_less(np.abs(amp - g).max(), 1e-8)


def get_ij_ijk_max(trajectory, rc, k_max=3) -> (int, int):
    """
    Return the maximum number of unique `R(ij)` and `Angle(i,j,k)` from one
    `Atoms` object.
    """
    nij_max = 0
    nijk_max = 0
    for atoms in trajectory:
        symbols = atoms.get_chemical_symbols()
        ilist, jlist = neighbor_list('ij', atoms, rc)
        if k_max >= 2:
            nij_max = max(len(ilist), nij_max)
        else:
            nij = 0
            for k in range(len(ilist)):
                if symbols[ilist[k]] == symbols[jlist[k]]:
                   nij += 1
            nij_max = max(nij_max, nij)
        if k_max == 3:
            nl = {}
            for i, atomi in enumerate(ilist):
                if atomi not in nl:
                    nl[atomi] = []
                nl[atomi].append(jlist[i])
            ntotal = 0
            for atomi, nlist in nl.items():
                n = len(nlist)
                ntotal += (n - 1 + 1) * (n - 1) // 2
            nijk_max = max(ntotal, nijk_max)
        else:
            nijk_max = 0
    return nij_max, nijk_max


def test_batch_one_element():
    """
    Test computing descriptors of a batch of mono-atomic molecules.
    """
    trajectory = read('test_files/B28.xyz', index='0:2', format='xyz')
    targets = np.zeros((2, 28, 8), dtype=np.float64)
    positions = np.zeros((2, 29, 3), dtype=np.float64)
    rc = 6.0
    nij_max = 756
    nijk_max = 9828
    if nij_max is None:
        nij_max, nijk_max = get_ij_ijk_max(trajectory, rc)

    for i, atoms in enumerate(trajectory):
        atoms.set_cell([20.0, 20.0, 20.0])
        atoms.set_pbc([False, False, False])
        targets[i] = legacy_symmetry_function(atoms, rc, name_scope='B28')[0]
        positions[i, 1:] = atoms.positions

    max_occurs = Counter({'B': 28})
    sf = SymmetryFunction(rc, max_occurs, nij_max=nij_max, nijk_max=nijk_max,
                          k_max=3)
    rslices, aslices = sf.get_indexed_slices(trajectory)

    tf.reset_default_graph()
    tf.enable_eager_execution()

    gr = sf.get_radial_function_graph(positions, rslices.v2g_map, rslices.ilist,
                                      rslices.jlist, rslices.shift)
    ga = sf.get_angular_function_graph(positions, aslices.v2g_map, aslices.ij,
                                       aslices.ik, aslices.jk, aslices.ij_shift,
                                       aslices.ik_shift, aslices.jk_shift)
    g = gr + ga
    values = g.numpy()[:, 1:, :]
    assert_less(np.abs(values - targets).max(), 1e-8)


def test_manybody_k():
    """
    Test computing descriptors for different `k_max`.
    """
    symbols = Pd3O2.get_chemical_symbols()
    rc = 6.0
    max_occurs = Counter(symbols)
    transformer = IndexTransformer(max_occurs, symbols)
    positions = transformer.gather(Pd3O2.positions)[np.newaxis, ...]
    ref, ref_terms, ref_sizes = legacy_symmetry_function(Pd3O2, rc, 'all')
    ref_offsets = np.insert(np.cumsum(ref_sizes), 0, 0)

    for k_max in (1, 2, 3):

        nij_max, nijk_max = get_ij_ijk_max([Pd3O2], rc, k_max=k_max)
        sf = SymmetryFunction(rc, max_occurs, k_max=k_max, nij_max=nij_max,
                              nijk_max=nijk_max)
        rslices, aslices = sf.get_indexed_slices([Pd3O2])

        g = sf.get_radial_function_graph(positions,
                                         v2g_map=rslices.v2g_map,
                                         ilist=rslices.ilist,
                                         jlist=rslices.jlist,
                                         shift=rslices.shift)
        if k_max == 3:
            g += sf.get_angular_function_graph(positions,
                                               v2g_map=aslices.v2g_map,
                                               ij=aslices.ij,
                                               ik=aslices.ik,
                                               jk=aslices.jk,
                                               ij_shift=aslices.ij_shift,
                                               ik_shift=aslices.ik_shift,
                                               jk_shift=aslices.jk_shift)
        columns = []
        for i, ref_term in enumerate(ref_terms):
            if ref_term in sf.kbody_terms:
                columns.extend(range(ref_offsets[i], ref_offsets[i + 1]))
        g = transformer.gather(g.numpy()[0], reverse=True)
        assert_less(np.abs(ref.numpy()[:, columns] - g).max(), 1e-8,
                    msg="Test case at k_max = {} is failed".format(str(k_max)))


def test_batch_multi_elements():
    """
    Test computing descriptors of a batch of multi-elements molecules.
    """
    rc = 6.0
    batch_size = len(qm7m.trajectory)
    n_atoms = sum(qm7m.max_occurs.values())
    kbody_terms, mapping, elements = get_kbody_terms(
        list(qm7m.max_occurs.keys()), k_max=3
    )
    total_dim, kbody_sizes = compute_dimension(kbody_terms,
                                               Defaults.n_etas,
                                               Defaults.n_betas,
                                               Defaults.n_gammas,
                                               Defaults.n_zetas)
    element_offsets = np.insert(
        np.cumsum([qm7m.max_occurs[e] for e in elements]), 0, 0)

    offsets = np.insert(np.cumsum(kbody_sizes)[:-1], 0, 0)
    targets = np.zeros((batch_size, n_atoms + 1, total_dim))
    for i, atoms in enumerate(qm7m.trajectory):
        g, local_terms, local_sizes = legacy_symmetry_function(
            atoms, rc, atoms.get_chemical_formula())
        local_offsets = np.insert(np.cumsum(local_sizes)[:-1], 0, 0)
        row = Counter()
        for k, atom in enumerate(atoms):
            atom_kbody_terms = mapping[atom.symbol]
            j = row[atom.symbol] + element_offsets[elements.index(atom.symbol)]
            for term in atom_kbody_terms:
                if term not in local_terms:
                    continue
                idx = kbody_terms.index(term)
                istart = offsets[idx]
                istop = istart + kbody_sizes[idx]
                idx = local_terms.index(term)
                lstart = local_offsets[idx]
                lstop = lstart + local_sizes[idx]
                targets[i, j + 1, istart: istop] = g[k, lstart: lstop]
            row[atom.symbol] += 1

    nij_max = 198
    nijk_max = 1217
    if nij_max is None:
        nij_max, nijk_max = get_ij_ijk_max(qm7m.trajectory, rc)

    sf = SymmetryFunction(rc, qm7m.max_occurs, k_max=3, nij_max=nij_max,
                          nijk_max=nijk_max)

    rslices, aslices = sf.get_indexed_slices(qm7m.trajectory)

    positions = np.zeros((batch_size, n_atoms + 1, 3))
    for i, atoms in enumerate(qm7m.trajectory):
        positions[i] = sf.get_index_transformer(atoms).gather(atoms.positions)

    gr = sf.get_radial_function_graph(positions,
                                      v2g_map=rslices.v2g_map,
                                      ilist=rslices.ilist,
                                      jlist=rslices.jlist,
                                      shift=rslices.shift)
    ga = sf.get_angular_function_graph(positions,
                                       v2g_map=aslices.v2g_map,
                                       ij=aslices.ij,
                                       ik=aslices.ik,
                                       jk=aslices.jk,
                                       ij_shift=aslices.ij_shift,
                                       ik_shift=aslices.ik_shift,
                                       jk_shift=aslices.jk_shift)
    g = gr + ga
    values = g.numpy()
    assert_less(np.abs(values[:, 1:] - targets[:, 1:]).max(), 1e-8)


def test_splits():
    """
    Test splitting descriptors into blocks.
    """
    symbols = Pd3O2.get_chemical_symbols()
    rc = 6.0
    k_max = 3
    max_occurs = Counter(symbols)
    transformer = IndexTransformer(max_occurs, symbols)
    positions = transformer.gather(Pd3O2.positions)[np.newaxis, ...]
    ref, _, _ = legacy_symmetry_function(Pd3O2, rc, 'all')

    nij_max, nijk_max = get_ij_ijk_max([Pd3O2], rc, k_max=k_max)
    sf = SymmetryFunction(rc, max_occurs, k_max=k_max, nij_max=nij_max,
                          nijk_max=nijk_max)
    rslices, aslices = sf.get_indexed_slices([Pd3O2])

    g = sf.get_radial_function_graph(positions,
                                     v2g_map=rslices.v2g_map,
                                     ilist=rslices.ilist,
                                     jlist=rslices.jlist,
                                     shift=rslices.shift)
    g += sf.get_angular_function_graph(positions,
                                       v2g_map=aslices.v2g_map,
                                       ij=aslices.ij,
                                       ik=aslices.ik,
                                       jk=aslices.jk,
                                       ij_shift=aslices.ij_shift,
                                       ik_shift=aslices.ik_shift,
                                       jk_shift=aslices.jk_shift)

    inputs = sf.split_descriptors(g)
    assert_less(
        np.abs(inputs['O'].numpy()[0] - ref.numpy()[3:, :20]).max(), 1e-8)
    assert_less(
        np.abs(inputs['Pd'].numpy()[0] - ref.numpy()[:3, 20:]).max(), 1e-8)


if __name__ == "__main__":
    nose.run()
