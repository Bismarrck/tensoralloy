#!coding=utf-8
"""
This module introduces EAM potentials for Pu and other actinides developed by
R.W. Grimes.

The long-range electrostatic term is ignored.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.nn.eam.potentials.generic import morse, buckingham
from tensoralloy.nn.utils import log_tensor

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class RWGrimes(EamAlloyPotential):
    """
    Journal of Nuclear Materials 461 (2015) 206–214
    """

    def __init__(self):
        """
        Initialization method.
        """
        super(RWGrimes, self).__init__()

    @property
    def defaults(self):
        """
        The default parameters.
        """
        defaults = {'PuPu': {'A': 18600.0, 'rho': 0.2637, 'C': 0.0,
                             'D': 0.70185, 'gamma': 1.98008, 'r0': 2.34591},
                    'Pu': {'G': 2.168, 'n': 3980.058}}
        return defaults

    def phi(self, r: tf.Tensor, kbody_term: str, variable_scope: str,
            verbose=False):
        """
        The pairwise potential function.
        """
        with tf.variable_scope('RWGrimes'):
            dtype = r.dtype
            assert isinstance(variable_scope, str)
            variable_scope = f"{variable_scope}/{kbody_term}"
            A = self._get_variable('A', dtype, kbody_term, variable_scope)
            rho = self._get_variable('rho', dtype, kbody_term, variable_scope)
            C = self._get_variable('C', dtype, kbody_term, variable_scope)
            D = self._get_variable('D', dtype, kbody_term, variable_scope)
            gamma = self._get_variable(
                'gamma', dtype, kbody_term, variable_scope)
            r0 = self._get_variable('r0', dtype, kbody_term, variable_scope)

            phi = tf.math.add(morse(r, D, gamma, r0),
                              buckingham(r, A, rho, C),
                              name='phi')
            if verbose:
                log_tensor(phi)
            return phi

    def rho(self, r: tf.Tensor, element: str, variable_scope: str,
            verbose=False):
        """
        The electron density function.
        """
        with tf.variable_scope('RWGrimes'):
            dtype = r.dtype
            assert isinstance(variable_scope, str)
            variable_scope = f"{variable_scope}/{element}"
            n = self._get_variable('n', dtype, element, variable_scope)
            half = tf.constant(0.5, dtype=dtype, name='half')
            one = tf.constant(1.0, dtype=dtype, name='two')
            twenty = tf.constant(20.0, dtype=dtype, name='twenty')
            order = tf.constant(8.0, dtype=tf.int32, name='order')

            rs = tf.math.pow(r, order, name='rs')
            left = tf.div_no_nan(n, rs)
            right = half + half * tf.math.erf(twenty * (r - one - half))

            rho = tf.multiply(left, right, name='rho')
            if verbose:
                log_tensor(rho)
            return rho

    def embed(self, rho: tf.Tensor, element: str, variable_scope: str,
              verbose=False):
        """
        The embedding function.
        """
        with tf.variable_scope('RWGrimes'):
            dtype = rho.dtype
            assert isinstance(variable_scope, str)
            variable_scope = f"{variable_scope}/{element}"
            G = self._get_variable('G', dtype, element, variable_scope)
            embed = tf.multiply(-tf.sqrt(rho), G, name='embed')
            if verbose:
                log_tensor(embed)
            return embed

    def dipole(self,
               r: tf.Tensor,
               kbody_term: str,
               variable_scope: str,
               verbose=False):
        """
        RWGrimes does not support calculating dipole.
        """
        raise Exception(
            f"{self.__class__.__name__} does not have dipole term.")

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   verbose=False):
        """
        RWGrimes does not support calculating dipole.
        """
        raise Exception(
            f"{self.__class__.__name__} does not have quadrupole term.")
