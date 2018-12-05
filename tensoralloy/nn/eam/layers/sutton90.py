# coding=utf-8
"""
This module defines the empirical potential of Ag proposed by Sutton et al. at
1990.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.nn.eam.layers.layers import PotentialFunctionLayer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AgSutton90(PotentialFunctionLayer):
    """
    The Ag potential proposed by Sutton et al. at 1990.

    References
    ----------
    A.P. Sutton, and J. Chen, Philos. Mag. Lett. 61 (1990) 139.

    """

    defaults = {'AgAg': {'a': 2.928323832, 'b': 2.485883762}}

    def __init__(self):
        """
        Initialization method.
        """
        super(AgSutton90, self).__init__(allowed_kbody_terms=['AgAg'])

    def phi(self, r: tf.Tensor, kbody_term: str):
        """
        The pairwise potential function:

            phi(r) = (b / r)**12

        """
        with tf.variable_scope('Sutton'):
            one = tf.constant(1.0, r.dtype, name='one')
            b = self._get_var('b', r.dtype, kbody_term)
            with tf.name_scope("ussafe_div"):
                r = tf.div_no_nan(one, r, name='r_inv')
            return tf.pow(b * r, 12, name='phi')

    def rho(self, r: tf.Tensor, kbody_term: str, **kwargs):
        """
        The electron density function:

            rho(r) = (a / r)**6

        """
        with tf.variable_scope('Sutton'):
            one = tf.constant(1.0, r.dtype, name='one')
            a = self._get_var('a', r.dtype, kbody_term)
            with tf.name_scope("ussafe_div"):
                r = tf.div_no_nan(one, r, name='r_inv')
            return tf.pow(a * r, 6, name='rho')

    def embed(self, rho: tf.Tensor, element: str):
        """
        The embedding function:

            F(rho) = -sqrt(rho)

        """
        with tf.variable_scope('Sutton'):
            return tf.negative(tf.sqrt(rho), name='embed')
