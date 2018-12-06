# coding=utf-8
"""
This module defines the empirical potential of Al-Fe proposed by Mendelev et al.
at 2011.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.nn.eam.potentials import EamFSPotential

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AlFeMsah11(EamFSPotential):
    """
    The Al-Fe proposed by Mendelev et al. at 2011.

    References
    ----------
    M.I. Mendelev, et al., J. Mater. Res. 20 (2011) 208.

    """
    defaults = {"Al": {}, "Fe": {}}

    def __init__(self):
        super(AlFeMsah11, self).__init__()

    def phi(self, r: tf.Tensor, kbody_term: str):
        """
        The pairwise potential function.

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, 1, max_n_element, nnl, 1]`.
        kbody_term : str
            The corresponding k-body term.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        pass

    def rho(self, r: tf.Tensor, kbody_term: str):
        """
        Return the Op to compute electron density `rho(r)`.

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, 1, max_n_element, nnl, 1]`.
        kbody_term : str
            The corresponding k-body term.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        pass

    def embed(self, rho: tf.Tensor, element: str):
        """
        Return the Op to compute the embedding energy F(rho(r)).

        Parameters
        ----------
        rho : tf.Tensor
            A 3D tensor of shape `[batch_size, max_n_element, 1]` where
            `max_n_element` is the maximum occurace of `element`.
        element : str
            An element symbol.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        with tf.name_scope("Mash11/Embed") as scope:
            with tf.variable_scope(f"{scope}/{element}"):
                if element == 'Al':
                    c1 = tf.constant(
                        0.000093283590195398, dtype=tf.float64, name='c1')
                    c2 = tf.constant(
                        0.0023491751192724, dtype=tf.float64, name='c2')
                    eps = tf.constant(1e-12, dtype=tf.float64, name='eps')
                    idx = tf.where(tf.greater_equal(rho, eps), name='idx')
                    shape = rho.shape
                    x = tf.gather_nd(rho, idx)
                    y = -tf.sqrt(x) + c1 * tf.pow(x, 2) - c2 * x * tf.log(x)
                    return tf.scatter_nd(idx, y, shape, name='embed')
                else:
                    c3 = tf.constant(
                        0.00067314115586063, dtype=tf.float64, name='c3')
                    c4 = tf.constant(
                        0.000000076514905604792, dtype=tf.float64, name='c4')
                    y = tf.add(-tf.sqrt(rho),
                               -c3 * tf.pow(rho, 2) + c4 * tf.pow(rho, 4),
                               name='embed')
                    return y
