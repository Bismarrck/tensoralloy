#!coding=utf-8
"""
Unit tests of the symmetry function neural network (SFNN).
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from dataclasses import dataclass
from tensorflow_estimator import estimator as tf_estimator
from ase.io import read
from ase import Atoms
from ase.neighborlist import neighbor_list
from os.path import join
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid
from itertools import product
from typing import Union, List

from tensoralloy.descriptor.cutoff import cosine_cutoff
from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.nn.atomic.sf import SymmetryFunctionNN
from tensoralloy.precision import precision_scope
from tensoralloy.test_utils import test_dir, assert_array_equal, Pd3O2
from tensoralloy.utils import get_elements_from_kbody_term, get_kbody_terms

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'



@dataclass
class Defaults:
    """
    Default parameter choices for generating symmetry function descriptors.
    """
    eta = np.array([0.05, 4.0, 20.0, 80.0])
    omega = np.array([0.0])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])


grid = ParameterGrid({'beta': Defaults.beta, 'gamma': Defaults.gamma,
                      'zeta': Defaults.zeta})


def compute_dimension(all_kbody_terms: List[str], n_etas, n_omegas, n_betas,
                      n_gammas, n_zetas):
    """
    Compute the total dimension of the feature vector.

    Parameters
    ----------
    all_kbody_terms : List[str]
        A list of str as all k-body terms.
    n_etas : int
        The number of `eta` for radial functions.
    n_omegas : int
        The number of `omega` for radial functions.
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
    for kbody_term in all_kbody_terms:
        k = len(get_elements_from_kbody_term(kbody_term))
        if k == 2:
            n = n_etas * n_omegas
        else:
            n = n_gammas * n_betas * n_zetas
        total_dim += n
        kbody_sizes.append(n)
    return total_dim, kbody_sizes


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
    molecule. Only `eta` is used. `omega` (Rs) is fixed to zero.

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


def get_radial_fingerprints_v2(coords, r, rc, etas, omegas):
    """
    Calculate the fingerprints from the radial functions for a mono-atomic
    molecule. Both `eta` and `omega` (Rs) are used.

    Args:
      coords: a `[N, 3]` array as the cartesian coordinates.
      r: a `[N, N]` array as the pairwise distance matrix.
      rc: a float as the cutoff radius.
      etas: a `List[float]` as the `eta` in the radial functions.
      omegas: a `List[float]` as the `omega` in the radial functions.

    Returns:
      x: a `[N, M]` array as the radial fingerprints.

    """

    params = np.array(list(product(etas, omegas)))
    ndim = len(params)
    natoms = len(coords)
    x = np.zeros((natoms, ndim))
    nl = get_neighbour_list(r, rc)
    rc2 = rc ** 2
    fr = cutoff_fxn(r, rc)

    for k, (etak, omegak) in enumerate(params):
        for i in range(natoms):
            v = 0.0
            ri = coords[i]
            for j in range(natoms):
                if not nl[i, j]:
                    continue
                rj = coords[j]
                rij = np.linalg.norm(rj - ri)
                ris = (rij - omegak)**2
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
            r2c = tf.math.truediv(r2, rc2, name='div')

        with tf.name_scope("fc"):
            fc_r = cosine_cutoff(r, rc=rc, name='fc_r')

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
            theta = tf.math.truediv(upper, lower, name='theta')

        # Compute the damping term: $f_c(r_{ij}) * f_c(r_{ik}) * f_c(r_{jk})$
        with tf.name_scope("fc"):
            fc_r_ij = cosine_cutoff(r_ij, rc, name='fc_r_ij')
            fc_r_ik = cosine_cutoff(r_ik, rc, name='fc_r_ik')
            fc_r_jk = cosine_cutoff(r_jk, rc, name='fc_r_jk')
            fc_r_ijk = fc_r_ij * fc_r_ik * fc_r_jk

        # Compute $R_{ij}^{2} + R_{ik}^{2} + R_{jk}^{2}$
        with tf.name_scope("r2"):
            r2_ij = tf.square(r_ij, name='r2_ij')
            r2_ik = tf.square(r_ik, name='r2_ik')
            r2_jk = tf.square(r_jk, name='r2_jk')
            r2 = r2_ij + r2_ik + r2_jk
            r2c = tf.math.truediv(r2, rc2, name='r2_rc2')

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


def legacy_symmetry_function(atoms: Atoms, rc: float):
    """
    Compute the symmetry function descriptors for unit tests.
    """
    symbols = atoms.get_chemical_symbols()
    all_kbody_terms, _, _ = get_kbody_terms(list(set(symbols)), angular=True)
    total_dim, kbody_sizes = compute_dimension(
        all_kbody_terms,
        n_etas=len(Defaults.eta),
        n_omegas=len(Defaults.omega),
        n_betas=len(Defaults.beta),
        n_gammas=len(Defaults.gamma),
        n_zetas=len(Defaults.zeta))

    with tf.Graph().as_default():
        R = tf.constant(atoms.positions, dtype=tf.float64, name='R')
        cell = tf.constant(atoms.cell.array, tf.float64, name='cell')
        rmap = legacy_build_radial_v2g_map(atoms, rc, len(Defaults.eta),
                                           all_kbody_terms, kbody_sizes)
        amap = legacy_build_angular_v2g_map(atoms, rmap, all_kbody_terms,
                                            kbody_sizes)
        gr = radial_function(R, rc, rmap.v2g_map, cell, Defaults.eta,
                             rmap.ilist, rmap.jlist, rmap.Slist, total_dim)
        ga = angular_function(R, rc, amap.v2g_map, cell, grid, amap.ij, amap.ik,
                              amap.jk, amap.ijSlist, amap.ikSlist, amap.jkSlist,
                              total_dim)
        g = tf.add(gr, ga, name='g')

        with tf.Session() as sess:
            results = sess.run(g)

        return results, all_kbody_terms, kbody_sizes


def test_monoatomic_molecule():
    """
    Test computing descriptors of a single mono-atomic molecule.
    """
    atoms = read(join(test_dir(), 'B28.xyz'), index=0, format='xyz')
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc([False, False, False])
    coords = atoms.get_positions()
    rr = pairwise_distances(coords)
    rc = 6.0
    zr = get_radial_fingerprints_v1(coords, rr, rc, Defaults.eta)
    za = get_augular_fingerprints_v1(coords, rr, rc, Defaults.beta,
                                     Defaults.gamma, Defaults.zeta)
    g_init = np.hstack((zr, za))
    g_old, _, _ = legacy_symmetry_function(atoms, rc=rc)
    assert_array_equal(g_init, g_old)

    with tf.Graph().as_default():
        clf = UniversalTransformer(['B'], rcut=rc, acut=rc, angular=True,
                                   periodic=False)
        nn = SymmetryFunctionNN(['B'], eta=Defaults.eta, omega=Defaults.omega,
                                gamma=Defaults.gamma, zeta=Defaults.zeta, beta=Defaults.beta)
        nn.attach_transformer(clf)
        op = nn.get_symmetry_function_descriptors(
            clf.get_descriptors(
                clf.get_constant_features(atoms)),
            mode=tf_estimator.ModeKeys.PREDICT)[0]
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            g_new = sess.run(op)
            assert_array_equal(g_old, g_new['B'])


def test_monoatomic_molecule_with_omega():
    """
    Test computing descriptors (eta and omega) of a single mono-atomic molecule.
    """
    atoms = read(join(test_dir(), 'B28.xyz'), index=0, format='xyz')
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc([False, False, False])

    omega = np.array([0.0, 1.0, 2.0])

    coords = atoms.get_positions()
    rr = pairwise_distances(coords)
    rc = 6.0
    z = get_radial_fingerprints_v2(coords, rr, rc, Defaults.eta, omega)

    with precision_scope("high"):
        with tf.Graph().as_default():
            clf = UniversalTransformer(['B'], rcut=rc, angular=False,
                                       periodic=False)
            nn = SymmetryFunctionNN(['B'], eta=Defaults.eta, omega=omega)
            nn.attach_transformer(clf)
            op = nn.get_symmetry_function_descriptors(
                clf.get_descriptors(
                    clf.get_constant_features(atoms)),
                mode=tf_estimator.ModeKeys.PREDICT)[0]
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                assert_array_equal(sess.run(op)['B'], z)


def test_single_structure():
    """
    Test computing descriptors of a single multi-elements periodic structure.
    """
    amp = np.load(join(test_dir(), 'amp_Pd3O2.npz'))['g']

    with precision_scope("high"):
        with tf.Graph().as_default():
            symbols = Pd3O2.get_chemical_symbols()
            elements = sorted(list(set(symbols)))
            rc = 6.5
            clf = UniversalTransformer(elements, rcut=rc, angular=True,
                                       periodic=True)
            nn = SymmetryFunctionNN(elements)
            nn.attach_transformer(clf)
            op = nn.get_symmetry_function_descriptors(
                clf.get_descriptors(
                    clf.get_constant_features(Pd3O2)),
                mode=tf_estimator.ModeKeys.PREDICT)[0]
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                g_new = sess.run(op)
                assert_array_equal(g_new['O'][0], amp[3: 5, 0: 20])
                assert_array_equal(g_new['Pd'][0], amp[0: 3, 20: 40])


if __name__ == "__main__":
    nose.main()
