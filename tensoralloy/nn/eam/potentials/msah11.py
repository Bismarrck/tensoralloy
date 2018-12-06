# coding=utf-8
"""
This module defines the empirical potential of Al-Fe proposed by Mendelev et al.
at 2011.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

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

        def _density(_factors: np.ndarray, _cutoffs: np.ndarray, _order: int):
            """
            A helper function.
            """
            zero = tf.constant(0.0, dtype=r.dtype, name='zero')
            values = []
            for i in range(len(_cutoffs)):
                cutoff = tf.convert_to_tensor(_cutoffs[i], name=f'rc{i}')
                factor = tf.convert_to_tensor(_factors[i], name=f'c{i}')
                values.append(
                    factor * tf.pow(tf.maximum(cutoff - r, zero), _order))
            return tf.add_n(values, name='density')

        with tf.name_scope("Mash11/Rho") as scope:
            with tf.variable_scope(f"{scope}/{kbody_term}"):
                if kbody_term == 'AlAl':
                    factors = np.asarray([
                        0.00019850823042883, 0.10046665347629,
                        1.0054338881951E-01, 0.099104582963213,
                        0.090086286376778, 0.0073022698419468,
                        0.014583614223199, -0.0010327381407070,
                        0.0073219994475288, 0.0095726042919017])
                    cutoffs = np.asarray([2.5, 2.6, 2.7, 2.8, 3.0,
                                          3.4, 4.2, 4.8, 5.6, 6.5])
                    order = 4
                elif kbody_term == 'FeFe':
                    factors = np.asarray(
                        [11.686859407970, -0.014710740098830, 0.47193527075943])
                    cutoffs = np.asarray([2.4, 3.2, 4.2])
                    order = 3
                else:
                    factors = np.asarray([
                        0.010015421408039, 0.0098878643929526,
                        0.0098070326434207, 0.0084594444746494,
                        0.0038057610928282, -0.0014091094540309,
                        0.0074410802804324])
                    cutoffs = np.asarray([2.4, 2.5, 2.6, 2.8, 3.1,
                                          5.0, 6.2])
                    order = 4
                return _density(factors, cutoffs, order)

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
