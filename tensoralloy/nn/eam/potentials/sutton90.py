# coding=utf-8
"""
This module defines the empirical potential of Ag proposed by Sutton et al. at
1990.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.nn.utils import log_tensor
from tensoralloy.extension.grad_ops import safe_pow

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AgSutton90(EamAlloyPotential):
    """
    The Ag potential proposed by Sutton et al. at 1990.

    References
    ----------
    A.P. Sutton, and J. Chen, Philos. Mag. Lett. 61 (1990) 139.

    """

    def __init__(self):
        """
        Initialization method.
        """
        super(AgSutton90, self).__init__()

    @property
    def defaults(self):
        """
        The default parameters.
        """
        defaults = {'Ag': {'a': 2.928323832},
                    'AgAg': {'b': 2.485883762}}
        return defaults

    def phi(self, r: tf.Tensor, kbody_term: str, variable_scope: str,
            verbose=False):
        """
        The pairwise potential function:

            phi(r) = (b / r)**12

        """
        with tf.variable_scope('Sutton'):
            assert isinstance(variable_scope, str)
            variable_scope = f"{variable_scope}/{kbody_term}"
            one = tf.constant(1.0, r.dtype, name='one')
            b = self._get_variable('b', r.dtype, kbody_term, variable_scope)
            with tf.name_scope("ussafe_div"):
                r = tf.div_no_nan(one, r, name='r_inv')
            scale = tf.constant(12.0, dtype=r.dtype, name='scale')
            phi = tf.identity(safe_pow(b * r, scale), 'phi')
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
            assert isinstance(variable_scope, str)
            variable_scope = f"{variable_scope}/{element}"
            one = tf.constant(1.0, r.dtype, name='one')
            a = self._get_variable('a', r.dtype, element, variable_scope)
            with tf.name_scope("ussafe_div"):
                r = tf.div_no_nan(one, r, name='r_inv')
            six = tf.constant(6.0, dtype=r.dtype, name='six')
            rho = tf.identity(safe_pow(a * r, six), name='rho')
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

    def dipole(self,
               r: tf.Tensor,
               kbody_term: str,
               variable_scope: str,
               verbose=False):
        """
        Ag/Sutton90 does not support calculating dipole.
        """
        raise Exception(
            f"{self.__class__.__name__} does not have dipole term.")

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   verbose=False):
        """
        Ag/Sutton90 does not support calculating dipole.
        """
        raise Exception(
            f"{self.__class__.__name__} does not have quadrupole term.")
