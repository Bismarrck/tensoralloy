# coding=utf-8
"""

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


def get_neighbour_list(r, cutoff):
    """
    A naive implementation of calculating neighbour list for non-periodic
    molecules.

    Args:
      r: a `[N, N]` array as the pairwise interatomic distances.
      cutoff: a float as the cutoff radius.

    Returns:
      neighbours: a `[N, N]` boolean array as the neighbour results.

    """
    neighbours = r < cutoff
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


atoms = read('B28.xyz', index=0, format='xyz')
assert isinstance(atoms, Atoms)

atoms.set_cell([10.0, 10.0, 10.0])
atoms.set_pbc([False, False, False])

eta_list = [0.05, 4., 20., 80.]
N = len(atoms)
ndim = len(eta_list)
rc = 6.0
rs = 0.0

ii, jj = neighbor_list('ij', atoms, rc)
vi = []
for k in range(ndim):
    vi.append([(ii[idx], k) for idx in range(len(ii))])

ij = list(zip(ii, jj))


graph = tf.Graph()

with graph.as_default():

    R = tf.placeholder(tf.float64, shape=(N, 3), name='R')
    rd = pairwise_dist(R, R, name='dist')

    rc2 = tf.constant(rc**2, dtype=tf.float64, name='rc2')
    eta = tf.constant(eta_list, dtype=tf.float64, name='eta')
    dd = tf.gather_nd(rd, ij)
    fc_dd = cutoff(dd, rc, name='cutoff')

    x = tf.Variable(tf.zeros((N, ndim), dtype=tf.float64))
    v = tf.negative(tf.tensordot(eta, tf.div(tf.square(dd), rc2), axes=0))
    v = tf.exp(v, name='exp')
    v = tf.multiply(v, fc_dd, name='damped')

    ops = []
    for row in range(ndim):
        ops.append(tf.scatter_nd_add(x, vi[row], v[row]))
    op = tf.group(ops)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        v_val, x_val, _ = sess.run([v, x, op], feed_dict={R: atoms.get_positions()})

    z_val = get_radial_fingerprints(atoms.get_positions(),
                                    pairwise_distances(atoms.get_positions()),
                                    rc,
                                    eta_list)

    print(np.abs(x_val - z_val).max())
