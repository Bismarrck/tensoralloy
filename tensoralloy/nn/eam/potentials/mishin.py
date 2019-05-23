#!coding=utf-8
"""
This module defines the empirical functions for ADP developed by Y. Mishin
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from os.path import join

from tensoralloy.nn.eam.potentials.generic import mishin_cutoff, mishin_polar
from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.nn.utils import log_tensor
from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.precision import get_float_dtype
from tensoralloy.extension.grad_ops import safe_pow
from tensoralloy.extension.interp.cubic import CubicInterpolator
from tensoralloy.test_utils import test_dir
from tensoralloy.io.lammps import read_adp_setfl

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class MishinH(EamAlloyPotential):
    """
    The ADP potential functions of the Mishin H-style.

    References
    ----------
    F. Apostol and Y. Mishin, PHYSICAL REVIEW B 82, 144115 2010

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
            }
        }
        return params

    def phi(self, r: tf.Tensor, kbody_term: str, variable_scope: str,
            verbose=False):
        """
        The pairwise potential function.

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, 1, max_n_element, nnl, 1]`.
        kbody_term : str
            The corresponding k-body term.
        variable_scope : str
            The scope for variables of this potential function.
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
            key = kbody_term
            V0 = self._get_variable("V0", dtype, key, variable_scope)
            alpha = self._get_variable("alpha", dtype, key, variable_scope)
            beta = self._get_variable("beta", dtype, key, variable_scope)
            gamma = self._get_variable("gamma", dtype, key, variable_scope)
            R0 = self._get_variable("R0", dtype, key, variable_scope)
            R1 = self._get_variable("R1", dtype, key, variable_scope)
            A1 = self._get_variable("A1", dtype, key, variable_scope)
            A2 = self._get_variable("A2", dtype, key, variable_scope)
            A3 = self._get_variable("A3", dtype, key, variable_scope)

            if el_a == el_b:
                rc = self._get_shared_variable('rc', dtype, el_a)
                h = self._get_shared_variable('h', dtype, el_a)
            else:
                rc = self._get_variable('rc', dtype, kbody_term, variable_scope)
                h = self._get_variable('h', dtype, kbody_term, variable_scope)

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
            verbose=False):
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
        verbose : bool
            A bool. If True, key tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        with tf.name_scope(f"{self._name}/Rho/{element}"):
            A0 = self._get_shared_variable('A0', r.dtype, element)
            B0 = self._get_shared_variable('B0', r.dtype, element)
            C0 = self._get_shared_variable('B0', r.dtype, element)
            rc = self._get_shared_variable('rc', r.dtype, element)
            z1 = self._get_shared_variable('z1', r.dtype, element)
            z2 = self._get_shared_variable('z2', r.dtype, element)
            a1 = self._get_shared_variable('a1', r.dtype, element)
            a2 = self._get_shared_variable('a2', r.dtype, element)
            h = self._get_shared_variable('h', r.dtype, element)

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
              verbose=False):
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
        verbose : bool
            A bool. If True, key tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        dtype = rho.dtype

        with tf.name_scope(f"{self._name}/Embed/{element}"):

            s1 = self._get_shared_variable('s1', dtype, element)
            s2 = self._get_shared_variable('s2', dtype, element)
            s3 = self._get_shared_variable('s3', dtype, element)
            s4 = self._get_shared_variable('s4', dtype, element)
            s5 = self._get_shared_variable('s5', dtype, element)
            s6 = self._get_shared_variable('s6', dtype, element)
            s7 = self._get_shared_variable('s7', dtype, element)

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
               verbose=False):
        """
        The dipole function.
        """
        with tf.name_scope(f"{self._name}/Dipole/{kbody_term}"):
            dtype = r.dtype
            d1 = self._get_variable("d1", dtype, kbody_term, variable_scope)
            d2 = self._get_variable("d2", dtype, kbody_term, variable_scope)
            d3 = self._get_variable("d3", dtype, kbody_term, variable_scope)
            rc = self._get_variable('rc', dtype, kbody_term, variable_scope)
            h = self._get_variable('h', dtype, kbody_term, variable_scope)
            dipole = mishin_polar(r, d1, d2, d3, rc, h, name="U")
            if verbose:
                log_tensor(dipole)
            return dipole

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   verbose=False):
        """
        The quadrupole function.
        """
        with tf.name_scope(f"{self._name}/Quadrupole/{kbody_term}"):
            dtype = r.dtype
            q1 = self._get_variable("q1", dtype, kbody_term, variable_scope)
            q2 = self._get_variable("q2", dtype, kbody_term, variable_scope)
            q3 = self._get_variable("q3", dtype, kbody_term, variable_scope)
            rc = self._get_variable('rc', dtype, kbody_term, variable_scope)
            h = self._get_variable('h', dtype, kbody_term, variable_scope)
            quadrupole = mishin_polar(r, q1, q2, q3, rc, h, name='W')
            if verbose:
                log_tensor(quadrupole)
            return quadrupole


class MishinTa(EamAlloyPotential):
    """
    The ADP potential functions of the Mishin Ta-style.

    References
    ----------
    Y. Mishin and A.Y. Lozovoi, Acta Materialia 54 (2006) 5013–5026

    """

    def __init__(self):
        """
        Initialization method.
        """
        super(MishinTa, self).__init__()

        self._name = 'MishinTa'
        self._implemented_potentials = ('rho', 'phi', 'embed', 'dipole',
                                        'quadrupole')

    @property
    def defaults(self):
        """
        Return the default parameters.
        """
        return {}

    def rho(self, r: tf.Tensor, element: str, variable_scope: str,
            verbose=False):
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
        verbose : bool
            A bool. If True, key tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        with tf.name_scope(f"{self._name}/Rho/{element}"):
            A0 = self._get_shared_variable('A0', r.dtype, element)
            B0 = self._get_shared_variable('B0', r.dtype, element)
            C0 = self._get_shared_variable('B0', r.dtype, element)
            rc = self._get_shared_variable('rc', r.dtype, element)
            r0 = self._get_shared_variable('r0', r.dtype, element)
            y = self._get_shared_variable('y', r.dtype, element)
            gamma = self._get_shared_variable('gamma', r.dtype, element)
            h = self._get_shared_variable('h', r.dtype, element)
            one = tf.constant(1.0, dtype=r.dtype, name='one')

            dr = tf.subtract(r, rc, name='dr')
            drh = tf.math.truediv(dr, h, name='drh')
            psi = mishin_cutoff(drh)

            z = tf.math.subtract(r, r0, name='z')
            egz = tf.exp(-gamma * z)
            c = tf.add((one + B0 * egz) * A0 * egz * safe_pow(z, y), C0, name='c')

            rho = tf.multiply(c, psi, name='rho')
            if verbose:
                log_tensor(rho)
            return rho

    def embed(self, rho: tf.Tensor, element: str, variable_scope: str,
              verbose=False):
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
        verbose : bool
            A bool. If True, key tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        dtype = rho.dtype

        with tf.name_scope(f"{self._name}/Embed/{element}"):

            pass


class AlCuSplineAdp(EamAlloyPotential):
    """
    The Cubic Spline form of the Al-Cu ADP potential.

    References
    ----------
    PHYSICAL REVIEW B 83, 054116 (2011)

    """

    def __init__(self, interval=1):
        """
        Initialization method.
        """
        super(AlCuSplineAdp, self).__init__()

        filename = join(test_dir(), 'lammps', 'AlCu.adp')
        self._adpfl = read_adp_setfl(filename)
        self._name = "AlCuAdp"
        self._interval = interval

    def rho(self, r: tf.Tensor, element: str, variable_scope: str,
            verbose=False):
        """
        The electron density function.
        """
        with tf.name_scope(f"{self._name}/Rho/{element}"):
            nr = self._adpfl.nr
            x = self._adpfl.rho[element][0][0:nr:self._interval]
            y = self._adpfl.rho[element][1][0:nr:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(r, name='shape')
            rho = f.evaluate(tf.reshape(r, (-1,), name='r/flat'))
            rho = tf.reshape(rho, shape, name='rho')
            if verbose:
                log_tensor(rho)
            return rho

    def embed(self,
              rho: tf.Tensor,
              element: str,
              variable_scope: str,
              verbose=False):
        """
        The embedding function.
        """
        with tf.name_scope(f"{self._name}/Embed/{element}"):
            nrho = self._adpfl.nrho
            x = self._adpfl.frho[element][0][0:nrho:self._interval]
            y = self._adpfl.frho[element][1][0:nrho:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(rho, name='shape')
            frho = f.evaluate(tf.reshape(rho, (-1,), name='rho/flat'))
            frho = tf.reshape(frho, shape, name='frho')
            if verbose:
                log_tensor(frho)
            return frho

    def phi(self,
            r: tf.Tensor,
            kbody_term: str,
            variable_scope: str,
            verbose=False):
        """
        The pairwise interaction function.
        """
        with tf.name_scope(f"{self._name}/Phi/{kbody_term}"):
            nr = self._adpfl.nr
            x = self._adpfl.phi[kbody_term][0][0:nr:self._interval]
            y = self._adpfl.phi[kbody_term][1][0:nr:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(r, name='shape')
            phi = f.evaluate(tf.reshape(r, (-1, ), name='r/flat'))
            phi = tf.reshape(phi, shape, name='phi')
            if verbose:
                log_tensor(phi)
            return phi

    def dipole(self,
               r: tf.Tensor,
               kbody_term: str,
               variable_scope: str,
               verbose=False):
        """
        The dipole function.
        """
        with tf.name_scope(f"{self._name}/Dipole/{kbody_term}"):
            nr = self._adpfl.nr
            x = self._adpfl.dipole[kbody_term][0][0:nr:self._interval]
            y = self._adpfl.dipole[kbody_term][1][0:nr:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(r, name='shape')
            dipole = f.evaluate(tf.reshape(r, (-1, ), name='r/flat'))
            dipole = tf.reshape(dipole, shape, name='dipole')
            if verbose:
                log_tensor(dipole)
            return dipole

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   verbose=False):
        """
        The quadrupole function.
        """
        with tf.name_scope(f"{self._name}/Quadrupole/{kbody_term}"):
            nr = self._adpfl.nr
            x = self._adpfl.quadrupole[kbody_term][0][0:nr:self._interval]
            y = self._adpfl.quadrupole[kbody_term][1][0:nr:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(r, name='shape')
            quadrupole = f.evaluate(tf.reshape(r, (-1,), name='r/flat'))
            quadrupole = tf.reshape(quadrupole, shape, name='quadrupole')
            if verbose:
                log_tensor(quadrupole)
            return quadrupole
