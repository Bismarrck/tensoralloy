# coding=utf-8
"""
This module defines the unit tests for implementations of Behler's symmetry
functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import nose
from behler import get_elements_from_kbody_term, get_kbody_terms
from behler import compute_dimension
from behler import radial_function, angular_function
from behler import build_radial_v2g_map, build_angular_v2g_map
from nose.plugins.skip import SkipTest
from nose.tools import assert_less
from ase import Atoms
from ase.io import read
from itertools import product
from sklearn.metrics import pairwise_distances

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def skip(func):
    """
    A decorator for skipping tests.
    """
    def _():
        raise SkipTest("Test %s is skipped" % func.__name__)
    _.__name__ = func.__name__
    return _


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


def get_radial_fingerprints(coords, r, rc, etas):
    """
    Return the fingerprints from the radial gaussian functions.

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


def get_augular_fingerprints_naive(coords, r, rc, etas, gammas, zetas):
    """
    Return the fingerprints from the augular functions.

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


@skip
def test_single():
    """
    Test the single version of `radial_function` and `angular_function`.
    """
    atoms = read('test_files/B28.xyz', index=0, format='xyz')
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc([False, False, False])
    coords = atoms.get_positions()
    rr = pairwise_distances(coords)

    eta = [0.05, 4., 20., 80.]
    angular_eta = [0.005, ]
    gamma = [1.0, -1.0]
    zeta = [1.0, 4.0]
    rc = 6.0
    natoms = len(atoms)

    ndim_angular = len(angular_eta) * len(zeta) * len(gamma)
    ndim_radial = len(eta)

    p_v2g_map, pairs = build_radial_v2g_map(atoms, rc, ndim_radial)
    t_v2g_map, triples = build_angular_v2g_map(pairs, ndim_angular)

    with tf.Graph().as_default():
        R = tf.placeholder(tf.float64, shape=(natoms, 3), name='R')
        gr = radial_function(R, rc, eta, pairs['ij'], p_v2g_map)
        ga = angular_function(R, rc, triples, angular_eta, gamma, zeta,
                              t_v2g_map)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            gr_vals, ga_vals = sess.run([gr, ga],
                                        feed_dict={R: atoms.get_positions()})

        zr_vals = get_radial_fingerprints(coords, rr, rc, eta)
        za_vals = get_augular_fingerprints_naive(coords, rr, rc, angular_eta,
                                                 gamma, zeta)

        assert_less(np.linalg.norm(gr_vals - zr_vals), 1e-6)
        assert_less(np.linalg.norm(ga_vals - za_vals), 1e-6)


def test_main():
    amp = np.load('test_files/amp_Pd3O2.npz')['g']

    eta = np.array([0.05, 4., 20., 80.])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])

    rc = 6.5
    atoms = Atoms(symbols='Pd3O2', pbc=np.array([True, True, False], dtype=bool),
                  cell=np.array([[7.78, 0., 0.],
                                 [0., 5.50129076, 0.],
                                 [0., 0., 15.37532269]]),
                  positions=np.array([[3.89, 0., 8.37532269],
                                      [0., 2.75064538, 8.37532269],
                                      [3.89, 2.75064538, 8.37532269],
                                      [5.835, 1.37532269, 8.5],
                                      [5.835, 7.12596807, 8.]]))

    symbols = atoms.get_chemical_symbols()
    kbody_terms, mapping = get_kbody_terms(list(set(symbols)), k_max=3)
    total_dim, kbody_sizes = compute_dimension(kbody_terms, len(eta), len(beta),
                                               len(gamma), len(zeta))

    rmap = build_radial_v2g_map(atoms, rc, len(eta), kbody_terms, kbody_sizes)

    tf.reset_default_graph()
    tf.enable_eager_execution()

    R = tf.constant(atoms.positions, dtype=tf.float64, name='R')
    cell = tf.constant(atoms.cell, tf.float64, name='cell')

    gr = radial_function(R, rc, rmap.v2g_map, cell, eta, rmap.ilist, rmap.jlist,
                         rmap.Slist, total_dim)
    gr = gr.numpy()
    d1 = np.abs(amp[:, 0: 8] - gr[:, :8])
    d2 = np.abs(amp[:, 20: 28] - gr[:, 20: 28])
    assert_less(d1.max(), 1e-8)
    assert_less(d2.max(), 1e-8)


if __name__ == "__main__":
    nose.run()
