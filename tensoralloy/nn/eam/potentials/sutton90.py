# coding=utf-8
"""
This module defines the empirical potential of Ag proposed by Sutton et al. at
1990.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.nn.utils import log_tensor

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AgSutton90(EamAlloyPotential):
    """
    The Ag potential proposed by Sutton et al. at 1990.

    References
    ----------
    A.P. Sutton, and J. Chen, Philos. Mag. Lett. 61 (1990) 139.

    """

    defaults = {'Ag': {'a': 2.928323832},
                'AgAg': {'b': 2.485883762}}

    def __init__(self):
        """
        Initialization method.
        """
        super(AgSutton90, self).__init__()

    def phi(self, r: tf.Tensor, kbody_term: str, variable_scope: str,
            verbose=False):
        """
        The pairwise potential function:

            phi(r) = (b / r)**12

        """
        with tf.variable_scope('Sutton'):
            one = tf.constant(1.0, r.dtype, name='one')
            b = self._get_variable('b', r.dtype, kbody_term, variable_scope)
            with tf.name_scope("ussafe_div"):
                r = tf.div_no_nan(one, r, name='r_inv')
            phi = tf.pow(b * r, 12, name='phi')
            if verbose:
                log_tensor(phi)
            return phi

    def rho(self, r: tf.Tensor, element: str, variable_scope: str,
            verbose=False):
        """
        The electron density function:

            rho(r) = (a / r)**6

        """
        with tf.variable_scope('Sutton'):
            one = tf.constant(1.0, r.dtype, name='one')
            a = self._get_variable('a', r.dtype, element, variable_scope)
            with tf.name_scope("ussafe_div"):
                r = tf.div_no_nan(one, r, name='r_inv')
            rho = tf.pow(a * r, 6, name='rho')
            if verbose:
                log_tensor(rho)
            return rho

    def embed(self, rho: tf.Tensor, element: str, variable_scope: str,
              verbose=False):
        """
        The embedding function:

            F(rho) = -sqrt(rho)

        """
        with tf.variable_scope('Sutton'):
            embed = tf.negative(tf.sqrt(rho), name='embed')
            if verbose:
                log_tensor(embed)
            return embed
