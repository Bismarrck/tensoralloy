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


def inference_aa_radial(r: tf.Tensor, rc: float, eta: list, ij, vi):
    """
    The implementation of Behler's radial symmetry function.
    """
    ndim = len(eta)
    N = r.shape[0]
    rd = pairwise_dist(r, r, name='pdist')
    rc2 = tf.constant(rc ** 2, dtype=tf.float64, name='rc2')
    eta = tf.constant(eta, dtype=tf.float64, name='eta')
    rd = tf.gather_nd(rd, ij, name='rd')
    rd2 = tf.square(rd, name='rd2')
    fc_dd = cutoff(rd, rc, name='cutoff')
    g = tf.Variable(tf.zeros((N, ndim), dtype=tf.float64), trainable=False)
    v = tf.negative(tf.tensordot(eta, tf.div(rd2, rc2), axes=0))
    v = tf.exp(v, name='exp')
    v = tf.multiply(v, fc_dd, name='damped')
    ops = []
    for row in range(ndim):
        ops.append(tf.scatter_nd_add(g, vi[row], v[row]))
    with tf.control_dependencies(ops):
        return tf.identity(g)


def test_aa_radial(atoms: Atoms, rc: float, eta_list: list, ij: list, vi: list):
    """
    Test the function: `inference_aa_radial`
    """
    N = len(atoms)
    graph = tf.Graph()

    with graph.as_default():
        R = tf.placeholder(tf.float64, shape=(N, 3), name='R')
        x = inference_aa_radial(R, rc, eta_list, ij, vi)
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

    natoms = len(atoms)
    coords = atoms.get_positions()
    tlist = []
    dist = pairwise_distances(atoms.get_positions())

    for i in range(natoms):
        for j in range(natoms):
            if j == i:
                continue
            if dist[i, j] > rc:
                continue
            for k in range(natoms):
                if k == i or k == j:
                    continue
                if dist[i, k] > rc:
                    continue
                rij = coords[j] - coords[i]
                rik = coords[k] - coords[i]
                tlist.append(np.dot(rij, rik) / (dist[i, j] * dist[i, k]))

    with tf.Graph().as_default():

        one = tf.constant(1.0, dtype=tf.float64)
        half = tf.constant(0.5, dtype=tf.float64)
        two = tf.constant(2.0, dtype=tf.float64)

        R = tf.placeholder(tf.float64, shape=(len(atoms), 3), name='R')
        r = pairwise_dist(R, R, name='pdist')
        r2 = tf.square(r)
        rc2 = tf.constant(rc ** 2, dtype=tf.float64, name='rc2')

        r_ij3 = tf.gather_nd(r, angle_list['ij'], name='ij3')
        r_ik3 = tf.gather_nd(r, angle_list['ik'], name='ik3')
        r_jk3 = tf.gather_nd(r, angle_list['jk'], name='jk3')

        b2c2a2 = tf.square(r_ij3) + tf.square(r_ik3) - tf.square(r_jk3)
        bc = tf.multiply(r_ij3, r_ik3, name='bc')
        bc = tf.multiply(two, bc, name='2bc')
        cos_theta = tf.div(b2c2a2, bc, name='cos_theta')

        fc_r_ij3 = cutoff(r_ij3, rc, name='fc_r_ij3')
        fc_r_ik3 = cutoff(r_ik3, rc, name='fc_r_ik3')
        fc_r_jk3 = cutoff(r_jk3, rc, name='fc_r_jk3')
        damp = fc_r_ij3 * fc_r_ik3 * fc_r_jk3

        r2_ij3 = tf.gather_nd(r2, angle_list['ij'], name='r2_ij3')
        r2_ik3 = tf.gather_nd(r2, angle_list['ik'], name='r2_ik3')
        r2_jk3 = tf.gather_nd(r2, angle_list['jk'], name='r2_jk3')
        r2_sum = r2_ij3 + r2_ik3 + r2_jk3

        x = tf.Variable(tf.zeros((natoms, ndim3), dtype=tf.float64), name='angular', trainable=False)

        ops = []
        row = 0
        for eta in angular_eta_list:
            for gamma in gamma_list:
                for zeta in zeta_list:
                    gamma_ = tf.constant(gamma, dtype=tf.float64, name='g{}3'.format(row))
                    eta_ = tf.constant(eta, dtype=tf.float64, name='e{}3'.format(row))
                    zeta_ = tf.constant(zeta, dtype=tf.float64, name='z{}3'.format(row))
                    v = one + tf.multiply(gamma_, cos_theta)
                    v = tf.pow(v, zeta_, name='pow')
                    v = tf.pow(two, one - zeta_) * v * tf.exp(-eta_ * r2_sum / rc2) * damp
                    ops.append(tf.scatter_nd_add(x, vi3[row], v))
                    row += 1

        with tf.control_dependencies(ops):
            x = tf.multiply(half, x)

        with tf.Session() as sess:

            tf.global_variables_initializer().run()

            x_vals = sess.run(x, feed_dict={R: atoms.get_positions()})

        z_vals = get_augular_fingerprints_naive(
            atoms.get_positions(),
            pairwise_distances(atoms.get_positions()),
            rc,
            angular_eta_list,
            gamma_list,
            zeta_list,
        )

        print(x_vals)
        print(z_vals)


    if aa_radial:
        test_aa_radial(atoms, rc, eta_list, ij, vi)


if __name__ == "__main__":
    main()
