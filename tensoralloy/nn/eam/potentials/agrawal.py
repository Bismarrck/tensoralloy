#!coding=utf-8
"""
An embedded atom method potential of beryllium
Modelling Simul. Mater. Sci. Eng. 21 (2013) 085001
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.extension.grad_ops import safe_pow
from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.nn.eam.potentials.generic import density_exp, morse
from tensoralloy.nn.utils import log_tensor
from tensoralloy.utils import get_elements_from_kbody_term

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def morse_prime(r, d, gamma, r0, name='morse'):
    """
    The derivative of `more` with respect to `r`.
    """
    with tf.name_scope(name, 'MorsePrime'):
        r = tf.convert_to_tensor(r, name='r')
        diff = tf.math.subtract(r, r0, name='diff')
        d = tf.convert_to_tensor(d, name='D')
        gd = tf.multiply(gamma, diff, name='g_diff')
        dtype = r.dtype
        two = tf.constant(2.0, dtype=dtype)
        c = tf.exp(-gd) - tf.exp(-two * gd)
        return tf.multiply(c, d * gamma * two, name=name)


class AgrawalBe(EamAlloyPotential):
    """
    A modified implementation of `Zjw04`.
    """

    def __init__(self):
        """
        Initialization method.
        """
        super(AgrawalBe, self).__init__()

        self._name = 'AgraBe'

    @property
    def defaults(self):
        """
        The default parameters of Zjw04xc.
        """
        return {"Be": {"A": 1.597, "B": 9.49713, "D": 0.41246, "alpha": 0.36324,
                       "re": 2.29, "F0": -2.0393, "F1": 12.6178,
                       "beta": 0.18752, "gamma": -2.28827, "m": 10, "rc": 5.0}}

    def rho(self, r: tf.Tensor, element: str, variable_scope: str,
            fixed=False, verbose=False):
        """
        The electron density function rho(r).
        """
        dtype = r.dtype
        with tf.name_scope(f"{self._name}/Rho/{element}"):
            A = self._get_shared_variable("A", dtype, element, fixed)
            B = self._get_shared_variable("B", dtype, element, fixed)
            re = self._get_shared_variable("re", dtype, element, fixed)
            rc = self._get_shared_variable("rc", dtype, element, fixed)
            m = self._get_shared_variable("m", dtype, element, fixed)
            one = tf.constant(1.0, dtype=dtype, name='one')
            # b = tf.multiply(B, re, "b")
            # rho = density_exp(r, A, b, re, name="rho")
            rho0 = tf.multiply(A, tf.exp(-B * (r - re)), name='rho0')
            rho1 = A * tf.exp(-B * (rc - re))
            drho = -A * B * tf.exp(-B * (rc - re))
            rho2 = rc / m * (one - (r / rc)**m) * drho
            rho = rho0 - rho1 + rho2
            if verbose:
                log_tensor(rho)
            return rho

    def embed(self, rho: tf.Tensor, element: str, variable_scope: str,
              fixed=False, verbose=False):
        """
        The embedding energy function F(rho).

        Parameters
        ----------
        rho : tf.Tensor
            A 3D tensor of shape `[batch_size, max_n_element, 1]` where
            `max_n_element` is the maximum occurs of `element`.
        element : str
            An element symbol.
        variable_scope : str
            The scope for variables of this potential function.
        fixed : bool
            If True, values of variables of this function will be fixed.
        verbose : bool
            A bool. If True, key tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        dtype = rho.dtype

        with tf.name_scope(f"{self._name}/Embed/{element}"):
            one = tf.constant(1.0, dtype=dtype, name='one')
            eps = tf.constant(1e-12, dtype=dtype, name='eps')
            beta = self._get_shared_variable('beta', dtype, element, fixed)
            gamma = self._get_shared_variable('gamma', dtype, element, fixed)
            F0 = self._get_shared_variable('F0', dtype, element, fixed)
            F1 = self._get_shared_variable('F1', dtype, element, fixed)
            x = safe_pow(rho, beta)
            y = safe_pow(rho, gamma)
            logrho = tf.log(tf.maximum(rho, eps), name='logrho')
            embed = tf.add(F0 * (one - beta * logrho) * x,
                           F1 * y, name='embed')
            if verbose:
                log_tensor(embed)
            return embed

    def phi(self,
            r: tf.Tensor,
            kbody_term: str,
            variable_scope: str,
            fixed=False,
            verbose=False):
        """
        The pairwise potential function phi(r).
        """

        dtype = r.dtype
        element = get_elements_from_kbody_term(kbody_term)[0]

        with tf.name_scope(f"{self._name}/Phi/{kbody_term}"):
            one = tf.constant(1.0, dtype=dtype, name='one')
            D = self._get_shared_variable('D', dtype, element, fixed)
            alpha = self._get_shared_variable('alpha', dtype, element, fixed)
            re = self._get_shared_variable('re', dtype, element, fixed)
            rc = self._get_shared_variable('rc', dtype, element, fixed)
            m = self._get_shared_variable('m', dtype, element, fixed)
            phi0 = morse(r, D, alpha, re, name='phi0')
            phi1 = tf.negative(morse(rc, D, alpha, re), name='phi1')
            z = safe_pow(r / rc, m)
            dphi = morse_prime(rc, D, alpha, re, name='dphi')
            phi2 = tf.multiply(rc / m, (one - z) * dphi, name='phi3')
            phi = tf.add(phi0 + phi1, phi2, name='phi')
            if verbose:
                log_tensor(phi)
            return phi

    def dipole(self,
               r: tf.Tensor,
               kbody_term: str,
               variable_scope: str,
               fixed=False,
               verbose=False):
        """ The dipole function. """
        raise NotImplemented()

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   fixed=False,
                   verbose=False):
        """ The quadrupole function. """
        raise NotImplemented()
