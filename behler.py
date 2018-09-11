# coding=utf-8
"""
The implementations of Behler's Symmetry Functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.io import read
from utilities import cutoff, pairwise_dist
from sklearn.metrics import pairwise_distances

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
    from itertools import product

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


def radial_function(R: tf.Tensor, rc: float, eta_list: list, ij: list,
                    v_to_g_map: list):
    """
    The implementation of Behler's radial symmetry function for a single
    structure.
    """
    ndim = len(eta_list)
    N = R.shape[0]
    r = pairwise_dist(R, R, name='pdist')
    rc2 = tf.constant(rc ** 2, dtype=tf.float64, name='rc2')
    eta = tf.constant(eta_list, dtype=tf.float64, name='eta')
    r = tf.gather_nd(r, ij, name='rd')
    rd2 = tf.square(r, name='rd2')
    fc_dd = cutoff(r, rc, name='cutoff')
    g = tf.Variable(tf.zeros((N, ndim), dtype=tf.float64), trainable=False)
    v = tf.negative(tf.tensordot(eta, tf.div(rd2, rc2), axes=0))
    v = tf.exp(v, name='exp')
    v = tf.multiply(v, fc_dd, name='damped')
    ops = []
    for row in range(ndim):
        ops.append(tf.scatter_nd_add(g, v_to_g_map[row], v[row]))
    with tf.control_dependencies(ops):
        return tf.identity(g)


def angular_function(R: tf.Tensor, rc: float, indices: dict, eta_list: list,
                     gamma_list: list, zeta_list: list, v_to_g_map: list):
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
    damp = fc_r_ij * fc_r_ik * fc_r_jk

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
                gamma_ = tf.constant(
                    gamma, dtype=tf.float64, name='g{}3'.format(row))
                eta_ = tf.constant(
                    eta, dtype=tf.float64, name='e{}3'.format(row))
                zeta_ = tf.constant(
                    zeta, dtype=tf.float64, name='z{}3'.format(row))

                v = one + tf.multiply(gamma_, cos_theta)
                v = tf.pow(v, zeta_, name='pow')
                v = tf.pow(two, one - zeta_) * v
                v = v * tf.exp(-eta_ * r2_sum / rc2) * damp
                group.append(tf.scatter_nd_add(g, v_to_g_map[row], v))
                row += 1
    with tf.control_dependencies(group):
        return tf.multiply(half, g)


def test_radial(atoms: Atoms, rc: float, eta_list: list, ij: list, vi: list):
    """
    Test the function: `inference_aa_radial`
    """
    N = len(atoms)
    graph = tf.Graph()

    with graph.as_default():
        R = tf.placeholder(tf.float64, shape=(N, 3), name='R')
        x = radial_function(R, rc, eta_list, ij, vi)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            x_val = sess.run(x, feed_dict={R: atoms.get_positions()})
        z_val = get_radial_fingerprints(
            atoms.get_positions(),
            pairwise_distances(atoms.get_positions()),
            rc,
            eta_list
        )
        print(np.abs(x_val - z_val).max())


def main(aa_radial=False):
    """
    The main function.
    """
    atoms = read('B28.xyz', index=0, format='xyz')
    assert isinstance(atoms, Atoms)

    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc([False, False, False])

    eta_list = [0.05, 4., 20., 80.]
    angular_eta_list = [0.005, ]
    gamma_list = [1.0, -1.0]
    zeta_list = [1.0, 4.0]

    ndim = len(eta_list)
    rc = 6.0
    ii, jj = neighbor_list('ij', atoms, rc)
    ij = list(zip(ii, jj))
    vi = []
    for k in range(ndim):
        vi.append([(ii[idx], k) for idx in range(len(ii))])

    angle_list = {'ij': [], 'ik': [], 'jk': [], 'ii': []}
    nl = {}
    for idx, atomi in enumerate(ii):
        nl[atomi] = nl.get(atomi, []) + [jj[idx]]

    for atomi, jlist in nl.items():
        for atomj in jlist:
            for atomk in jlist:
                if atomj == atomk:
                    continue
                angle_list['ij'].append((atomi, atomj))
                angle_list['ik'].append((atomi, atomk))
                angle_list['jk'].append((atomj, atomk))
                angle_list['ii'].append(atomi)

    vi3 = []
    ndim3 = len(angular_eta_list) * len(gamma_list) * len(zeta_list)
    for k in range(ndim3):
        vi3.append([(angle_list['ii'][idx], k) for idx in range(len(angle_list['ii']))])

    with tf.Graph().as_default():

        R = tf.placeholder(tf.float64, shape=(len(atoms), 3), name='R')
        g = angular_function(R, rc, angle_list, angular_eta_list, gamma_list, zeta_list, vi3)

        with tf.Session() as sess:

            tf.global_variables_initializer().run()
            g_vals = sess.run(g, feed_dict={R: atoms.get_positions()})

        z_vals = get_augular_fingerprints_naive(
            atoms.get_positions(),
            pairwise_distances(atoms.get_positions()),
            rc,
            angular_eta_list,
            gamma_list,
            zeta_list,
        )

        print(g_vals)
        print(z_vals)


    if aa_radial:
        test_radial(atoms, rc, eta_list, ij, vi)


if __name__ == "__main__":
    main()
