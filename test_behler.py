# coding=utf-8
"""
This module defines the unit tests for implementations of Behler's symmetry
functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import nose
from behler import get_kbody_terms, compute_dimension
from behler import radial_function, angular_function
from behler import build_radial_v2g_map, build_angular_v2g_map
from behler import batch_build_radial_v2g_map, batch_radial_function
from behler import batch_build_angular_v2g_map, batch_angular_function
from nose.tools import assert_less
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from itertools import product
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid
from misc import skip

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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

    for l, eta in enumerate(etas):
        for i in range(natoms):
            v = 0.0
            ri = coords[i]
            for j in range(natoms):
                if not nl[i, j]:
                    continue
                rs = coords[j]
                ris = np.sum(np.square(ri - rs))
                v += np.exp(-eta * ris / rc2) * fr[i, j]
            x[i, l] = v
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
    fr = cutoff_fxn(rr, rc)
    x = np.zeros((natoms, ndim))
    for l, (eta, gamma, zeta) in enumerate(params):
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
                    v = (1 + gamma * theta)**zeta
                    v *= np.exp(-eta * (r2[i, j] + r2[i, k] + r2[j, k]) / rc2)
                    v *= fr[i, j] * fr[j, k] * fr[i, k]
                    x[i, l] += v
        x[:, l] *= 2.0**(1 - zeta)
    return x / 2.0


def _symmetry_function(atoms: Atoms, rc: float, name_scope: str):
    """
    Compute the symmetry function descriptors for unit tests.
    """
    eta = np.array([0.05, 4., 20., 80.])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])
    grid = ParameterGrid({'beta': beta, 'gamma': gamma, 'zeta': zeta})
    symbols = atoms.get_chemical_symbols()
    kbody_terms, mapping = get_kbody_terms(list(set(symbols)), k_max=3)
    total_dim, kbody_sizes = compute_dimension(kbody_terms, len(eta), len(beta),
                                               len(gamma), len(zeta))
    tf.reset_default_graph()
    tf.enable_eager_execution()

    with tf.name_scope(name_scope):
        R = tf.constant(atoms.positions, dtype=tf.float64, name='R')
        cell = tf.constant(atoms.cell, tf.float64, name='cell')
        rmap = build_radial_v2g_map(atoms, rc, len(eta), kbody_terms, kbody_sizes)
        amap = build_angular_v2g_map(atoms, rmap, kbody_terms, kbody_sizes)

        gr = radial_function(R, rc, rmap.v2g_map, cell, eta, rmap.ilist, rmap.jlist,
                             rmap.Slist, total_dim)
        ga = angular_function(R, rc, amap.v2g_map, cell, grid, amap.ij, amap.ik,
                              amap.jk, amap.ijSlist, amap.ikSlist, amap.jkSlist,
                              total_dim)
        return tf.add(gr, ga, name='g')


@skip
def test_monoatomic_molecule():
    """
    Test `radial_function` and `angular_function` for a mono-atomic molecule.
    """
    atoms = read('test_files/B28.xyz', index=0, format='xyz')
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc([False, False, False])
    coords = atoms.get_positions()
    rr = pairwise_distances(coords)
    eta = [0.05, 4., 20., 80.]
    beta = [0.005, ]
    gamma = [1.0, -1.0]
    zeta = [1.0, 4.0]
    rc = 6.0
    zr = get_radial_fingerprints_v1(coords, rr, rc, eta)
    za = get_augular_fingerprints_v1(coords, rr, rc, beta, gamma, zeta)
    z = np.hstack((zr, za))
    g = _symmetry_function(atoms, rc=rc, name_scope='B28')
    assert_less(np.abs(z - g).max(), 1e-8)


@skip
def test_single_structure():
    """
    Test `radial_function` and `angular_function` for a periodic structure.
    """
    amp = np.load('test_files/amp_Pd3O2.npz')['g']
    atoms = Atoms(symbols='Pd3O2', pbc=np.array([True, True, False], dtype=bool),
                  cell=np.array([[7.78, 0., 0.],
                                 [0., 5.50129076, 0.],
                                 [0., 0., 15.37532269]]),
                  positions=np.array([[3.89, 0., 8.37532269],
                                      [0., 2.75064538, 8.37532269],
                                      [3.89, 2.75064538, 8.37532269],
                                      [5.835, 1.37532269, 8.5],
                                      [5.835, 7.12596807, 8.]]))
    g = _symmetry_function(atoms, rc=6.5, name_scope='Pd3O2')
    assert_less(np.abs(amp - g).max(), 1e-8)


def get_ij_ijk_max(trajectory, rc) -> (int, int):
    """
    Return the maximum number of unique `R(ij)` and `Angle(i,j,k)` from one
    `Atoms` object.
    """
    nij_max = 0
    nijk_max = 0
    for atoms in trajectory:
        ilist, jlist = neighbor_list('ij', atoms, rc)
        nij_max = max(len(ilist), nij_max)
        nl = {}
        for i, atomi in enumerate(ilist):
            nl[atomi] = nl.get(atomi, []) + [jlist[i]]
        ntotal = 0
        for atomi, nlist in nl.items():
            n = len(nlist)
            ntotal += (n - 1 + 1) * (n - 1) // 2
        nijk_max = max(ntotal, nijk_max)
    return nij_max, nijk_max


def test_batch_monoatomic_molecule():
    """
    Test `radial_function` and `angular_function` for several mono-atomic
    molecules.
    """
    trajectory = read('test_files/B28.xyz', index='0:2', format='xyz')
    targets = np.zeros((2, 28, 8), dtype=np.float64)
    positions = np.zeros((2, 29, 3), dtype=np.float64)
    clist = np.zeros((2, 3, 3), dtype=np.float64)
    rc = 6.0
    nij_max = 756
    nijk_max = 9828
    if nij_max is None:
        nij_max, nijk_max = get_ij_ijk_max(trajectory, rc)

    for i, atoms in enumerate(trajectory):
        atoms.set_cell([20.0, 20.0, 20.0])
        atoms.set_pbc([False, False, False])
        targets[i] = _symmetry_function(atoms, rc, name_scope='B28')
        positions[i, 1:] = atoms.positions
        clist[i] = atoms.cell

    eta = np.array([0.05, 4., 20., 80.])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])
    grid = ParameterGrid({'beta': beta, 'gamma': gamma, 'zeta': zeta})

    elements = ['B']
    kbody_terms, mapping = get_kbody_terms(elements, k_max=3)
    total_dim, kbody_sizes = compute_dimension(kbody_terms, len(eta), len(beta),
                                               len(gamma), len(zeta))

    rmap = batch_build_radial_v2g_map(trajectory, rc, len(eta), nij_max,
                                      kbody_terms, kbody_sizes)

    tf.reset_default_graph()
    tf.enable_eager_execution()

    g = batch_radial_function(positions, rc, rmap.v2g_map, clist, eta,
                              rmap.ilist, rmap.jlist, rmap.Slist, total_dim)
    values = g.numpy()[:, 1:, :]
    assert_less(np.abs(values[:, :, :4] - targets[:, :, :4]).max(), 1e-8)

    amap = batch_build_angular_v2g_map(trajectory, rmap, nijk_max,
                                       kbody_terms, kbody_sizes)
    g = batch_angular_function(positions, rc, amap.v2g_map, clist, grid,
                               amap.ij, amap.ik, amap.jk, amap.ijSlist,
                               amap.ikSlist, amap.jkSlist, total_dim)

    values = g.numpy()[:, 1:, :]
    assert_less(np.abs(values[:, :, 4:] - targets[:, :, 4:]).max(), 1e-8)


if __name__ == "__main__":
    nose.run()
