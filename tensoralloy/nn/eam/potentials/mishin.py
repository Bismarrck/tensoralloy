#!coding=utf-8
"""
This module defines the empirical functions for ADP developed by Y. Mishin
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.nn.eam.potentials.generic import mishin_cutoff, mishin_polar
from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.nn.utils import log_tensor
from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.precision import get_float_dtype
from tensoralloy.extension.grad_ops import safe_pow

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class MishinH(EamAlloyPotential):
    """
    The ADP potential functions of the Mishin H-style.

    References
    ----------
    F. Apostol and Y. Mishin, PHYSICAL REVIEW B 82, 144115 (2010)

    """

    def __init__(self):
        """
        Initialization method.
        """
        super(MishinH, self).__init__()

        self._name = 'MishinH'
        self._implemented_potentials = ('rho', 'phi', 'embed', 'dipole',
                                        'quadrupole')

    @property
    def defaults(self):
        """
        Return the default parameters.
        """
        params = {
            # Self-fitted parameters
            "Mo": {
                "s1": -2.00695289e-01, "s2": -3.12178751e-04,
                "s3": 7.86343222e-05, "s4": 5.29721645e+00,
                "s5": 3.79481951e-02, "s6": 1.11800974e+02, "s7": 4.05948858e+00
            },
            "Al": {
                "s1": -3.72848864e-01, "s2": 6.52035828e-03,
                "s3": 9.71742655e-05, "s4": 7.64264116e+00,
                "s5": 6.88604789e-02, "s6": 1.55694016e+01, "s7": 5.38646368e+00
            },
            # Parameters fitted by Mishin
            "H": {
                "s1": 8.08612, "s2": 1.46294e-2, "s3": -6.86143e-3,
                "s4": 3.19616, "s5": 1.17247e-1, "s6": 50, "s7": 15e5,
            },
            "NiNi": {
                "d1": 4.4657e-3, "d2": -1.3702e0, "d3": -0.9611e-1,
                "q1": 6.4502e0, "q2": 0.2608e-1, "q3": -6.0208e0,
                "h": 3.323, "rc": 5.168
            },
            "FeFe": {
                "d1": 1.9135e-1, "d2": -1.0796e0, "d3": -0.8928e-1,
                "q1": -5.8954e-2, "q2": -1.3872e0, "q3": 2.4790e0,
                "h": 6.202, "rc": 5.055
            }
        }
        params['MoMo'] = params['NiNi'].copy()
        params['MoNi'] = params['NiNi'].copy()
        params['BeBe'] = params['MoMo'].copy()
        return params

    def phi(self, r: tf.Tensor, kbody_term: str, scope: str,
            fixed=False, verbose=False):
        """
        The pairwise potential function.

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, 1, max_n_element, nnl, 1]`.
        kbody_term : str
            The corresponding k-body term.
        scope : str
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
        el_a, el_b = get_elements_from_kbody_term(kbody_term)

        with tf.name_scope(f"{self._name}/Phi/{kbody_term}"):
            dtype = r.dtype
            variable_scope = f"{scope}/{kbody_term}"
            args = (dtype, kbody_term, variable_scope, fixed)
            V0 = self._get_variable("V0", *args)
            alpha = self._get_variable("alpha", *args)
            beta = self._get_variable("beta", *args)
            gamma = self._get_variable("gamma", *args)
            R0 = self._get_variable("R0", *args)
            R1 = self._get_variable("R1", *args)
            A1 = self._get_variable("A1", *args)
            A2 = self._get_variable("A2", *args)
            A3 = self._get_variable("A3", *args)

            if el_a == el_b:
                rc = self._get_shared_variable('rc', dtype, el_a, fixed)
                h = self._get_shared_variable('h', dtype, el_a, fixed)
            else:
                rc = self._get_variable('rc', *args)
                h = self._get_variable('h', *args)

            drc = tf.subtract(r, rc, name='dr')
            drh = tf.math.truediv(drc, h, name='drh')
            psi = mishin_cutoff(drh)

            dr0 = tf.math.subtract(r, R0, name='dr0')
            dr1 = tf.math.subtract(r, R1, name='dr1')
            bdr0 = tf.math.multiply(beta, dr0, 'bdr0')

            phi1 = V0 * (tf.exp(-alpha * bdr0) - alpha * tf.exp(-bdr0))
            phi3 = A2 * bdr0
            phi4 = A3 * tf.exp(-gamma * tf.square(dr1))

            left = tf.add_n([phi1, A1, phi3, phi4], name='left')
            phi = tf.multiply(left, psi, name='phi')
            if verbose:
                log_tensor(phi)
            return phi

    def rho(self, r: tf.Tensor, element: str, variable_scope: str,
            fixed=False, verbose=False):
        """
        The electron density function rho(r).

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, max_n_terms, 1, nnl, 1]`.
        element : str
            The corresponding element.
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
        with tf.name_scope(f"{self._name}/Rho/{element}"):
            A0 = self._get_shared_variable('A0', r.dtype, element, fixed)
            B0 = self._get_shared_variable('B0', r.dtype, element, fixed)
            C0 = self._get_shared_variable('B0', r.dtype, element, fixed)
            rc = self._get_shared_variable('rc', r.dtype, element, fixed)
            z1 = self._get_shared_variable('z1', r.dtype, element, fixed)
            z2 = self._get_shared_variable('z2', r.dtype, element, fixed)
            a1 = self._get_shared_variable('a1', r.dtype, element, fixed)
            a2 = self._get_shared_variable('a2', r.dtype, element, fixed)
            h = self._get_shared_variable('h', r.dtype, element, fixed)

            dr = tf.subtract(r, rc, name='dr')
            drh = tf.math.truediv(dr, h, name='drh')
            psi = mishin_cutoff(drh)

            rz1 = safe_pow(r, z1)
            e1 = safe_pow(-r * a1)
            rz2 = safe_pow(r, z2)
            e2 = safe_pow(-r * a2)

            c = tf.add_n([tf.math.multiply(A0, rz1 * e1, 'A'),
                          tf.math.multiply(B0, rz2 * e2, 'B'),
                          C0], name='c')
            rho = tf.multiply(c, psi, name='rho')

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

            s1 = self._get_shared_variable('s1', dtype, element, fixed)
            s2 = self._get_shared_variable('s2', dtype, element, fixed)
            s3 = self._get_shared_variable('s3', dtype, element, fixed)
            s4 = self._get_shared_variable('s4', dtype, element, fixed)
            s5 = self._get_shared_variable('s5', dtype, element, fixed)
            s6 = self._get_shared_variable('s6', dtype, element, fixed)
            s7 = self._get_shared_variable('s7', dtype, element, fixed)

            one = tf.constant(1.0, dtype=dtype, name='one')
            rho2 = tf.square(rho, name='rho2')
            rho3 = tf.multiply(rho, rho2, name='rho3')
            rho4 = tf.square(rho2, name='rho4')

            with tf.name_scope("Safe"):
                eps = tf.convert_to_tensor(get_float_dtype().eps, dtype, 'eps')
                rho_eps = tf.add(rho, eps, 'rho')
                rhos5 = safe_pow(rho_eps, s5)

            with tf.name_scope("Omega"):
                """
                The damping function (Eq. 10) Omega(rho):

                    Omega(x) = 1 - (1 - s6 * x**2) / (1 + s7 * x**2)

                """
                a = tf.math.subtract(one, s6 * rho2)
                b = tf.math.add(one, s7 * rho4)
                right = tf.div_no_nan(a, b, name='right')
                omega = tf.subtract(one, right, name='omega')

            values = [
                tf.math.multiply(s1, rho, 's1rho'),
                tf.math.multiply(s2, rho2, 's2rho'),
                tf.math.multiply(s3, rho3, 's3rho'),
                tf.math.multiply(-s4, rhos5, 's4rhos5'),
            ]

            embed = tf.multiply(tf.add_n(values, name='sum'),
                                omega,
                                name='embed')
            if verbose:
                log_tensor(embed)
            return embed

    def dipole(self,
               r: tf.Tensor,
               kbody_term: str,
               variable_scope: str,
               fixed=False,
               verbose=False):
        """
        The dipole function.
        """
        with tf.name_scope(f"{self._name}/Dipole/{kbody_term}"):
            assert isinstance(variable_scope, str)
            variable_scope = f"{variable_scope}/{kbody_term}"
            dtype = r.dtype
            args = (dtype, kbody_term, variable_scope, fixed)
            d1 = self._get_variable("d1", *args)
            d2 = self._get_variable("d2", *args)
            d3 = self._get_variable("d3", *args)
            rc = self._get_shared_variable('rc', dtype, kbody_term, fixed)
            h = self._get_shared_variable('h', dtype, kbody_term, fixed)
            dipole = mishin_polar(r, d1, d2, d3, rc, h, name="U")
            if verbose:
                log_tensor(dipole)
            return dipole

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   fixed=False,
                   verbose=False):
        """
        The quadrupole function.
        """
        with tf.name_scope(f"{self._name}/Quadrupole/{kbody_term}"):
            assert isinstance(variable_scope, str)
            variable_scope = f"{variable_scope}/{kbody_term}"
            dtype = r.dtype
            args = (dtype, kbody_term, variable_scope, fixed)
            q1 = self._get_variable("q1", *args)
            q2 = self._get_variable("q2", *args)
            q3 = self._get_variable("q3", *args)
            rc = self._get_shared_variable('rc', dtype, kbody_term, fixed)
            h = self._get_shared_variable('h', dtype, kbody_term, fixed)
            quadrupole = mishin_polar(r, q1, q2, q3, rc, h, name='W')
            if verbose:
                log_tensor(quadrupole)
            return quadrupole
