# coding=utf-8
"""
This module defines the empirical potentials of Al-Cu proposed by Zhou et al. at
2004.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import scatter_nd_non_aliasing_add

from tensoralloy.nn.layers import PotentialFunctionLayer
from tensoralloy.utils import get_elements_from_kbody_term

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AlCuZJW04(PotentialFunctionLayer):
    """
    The Al-Cu potential proposed by Zhou et al. at 2004.

    References
    ----------
    Zhou, R. Johnson and H. Wadley, Phys. Rev. B. 69 (2004) 144113.

    """

    defaults = {
        'AlAl': {
            're': 2.863924, 'fe': 1.403115, 'A': 0.314873, 'B': 0.365551,
            'kappa': 0.379846, 'lamda': 0.759692, 'gamma': 6.613165,
            'omega': 3.527021, 'F0': -2.83, 'F1': 0.0, 'F2': 0.622245,
            'F3': -2.488244, 'eta': 0.785902, 'Fe': -2.824528, 'Fn0': -2.807602,
            'Fn1': -0.301435, 'Fn2': 1.258562, 'Fn3': -1.247604,
            'rho_e': 20.418205, 'rho_s': 23.195740,
        },
        'CuCu': {
            're': 2.556162, 'fe': 1.554485, 'A': 0.396620, 'B': 0.548085,
            'kappa': 0.308782, 'lamda': 0.756515, 'gamma': 8.127620,
            'omega': 4.334731, 'F0': -2.19, 'F1': 0.0, 'F2': 0.561830,
            'F3': -2.100595, 'eta': 0.310490, 'Fe': -2.186568, 'Fn0': -2.170269,
            'Fn1': -0.263788, 'Fn2': 1.088878, 'Fn3': -0.817603,
            'rho_e': 21.175871, 'rho_s': 21.175395,
        }
    }

    def __init__(self):
        super(AlCuZJW04, self).__init__(
            allowed_kbody_terms=['AlAl', 'AlCu', 'CuCu'])

    @staticmethod
    def _exp_func(re, a, b, c, one, name=None):
        def func(r):
            """
            A helper function to get a function of the form:

                a * exp(-b * (r / re - 1)) / (1 + (r / re - c)**20)

            """
            r_re = tf.div(r, re)
            upper = a * tf.exp(-b * (r_re - one))
            lower = one + tf.pow(r_re - c, 20)
            return tf.div(upper, lower, name=name)
        return func

    def phi(self, r: tf.Tensor, kbody_term: str):
        """
        The pairwise potential function:

            phi(r) = \frac{
                A\exp{[-\gamma(r_{ij}/r_{e} - 1)]}}{
                1 + (r_{ij}/r_{e} - \kappa)^{20}} - \frac{
                B\exp{[-\omega(r_{ij}/r_{e} - 1)]}}{
                1 + (r_{ij}/r_{e} - \lambda)^{20}
            }

        for AlAl, CuCu and:

            phi(r) = \frac{1}{2}\left[
                \frac{\rho_{\beta}(r)}{\rho_{\alpha}(r)}\phi_{\alpha\alpha}(r) +
                \frac{\rho_{\alpha}(r)}{\rho_{\beta}(r)}\phi_{\beta\beta}(r)
            \right]

        for Al-Cu where `\alpha` and `\beta` denotes 'Al' or 'Cu'.

        """
        if kbody_term in ('AlAl', 'CuCu'):
            element = get_elements_from_kbody_term(kbody_term)[0]
            with tf.variable_scope(f"ZJW04/Phi/{element}"):
                re = self._get_var('re', r.dtype, kbody_term, shared=True)
                A = self._get_var('A', r.dtype, kbody_term, shared=True)
                B = self._get_var('B', r.dtype, kbody_term, shared=True)
                gamma = self._get_var('gamma', r.dtype, kbody_term, shared=True)
                omega = self._get_var('omega', r.dtype, kbody_term, shared=True)
                kappa = self._get_var('kappa', r.dtype, kbody_term, shared=True)
                lamda = self._get_var('lamda', r.dtype, kbody_term, shared=True)
                one = tf.constant(1.0, dtype=r.dtype, name='one')
                return tf.subtract(
                    self._exp_func(re, A, gamma, kappa, one, name='A')(r),
                    self._exp_func(re, B, omega, lamda, one, name='B')(r),
                    name='phi')
        else:
            with tf.name_scope('AlAl'):
                phi_al = self.phi(r, 'AlAl')
                rho_al = self.rho(r, 'AlAl')
            with tf.name_scope("CuCu"):
                phi_cu = self.phi(r, 'CuCu')
                rho_cu = self.rho(r, 'CuCu')
            half = tf.constant(0.5, dtype=r.dtype, name='half')
            return tf.multiply(half,
                               tf.add(tf.div(rho_al, rho_cu) * phi_cu,
                                      tf.div(rho_cu, rho_al) * phi_al),
                               name='phi')

    def rho(self, r: tf.Tensor, kbody_term: str, **kwargs):
        """
        The electron density function rho(r).
        """
        if kbody_term == 'AlCu':
            split_sizes = kwargs['split_sizes']
            r_al, r_cu = tf.split(r, num_or_size_splits=split_sizes, axis=2)
            with tf.name_scope("AlAl"):
                rho_al = self.rho(r_al, 'AlAl')
            with tf.name_scope("CuCu"):
                rho_cu = self.rho(r_cu, 'CuCu')
            return tf.concat((rho_al, rho_cu), axis=2, name='rho')
        else:
            element = get_elements_from_kbody_term(kbody_term)[0]
            with tf.variable_scope(f"ZJW04/Rho/{element}"):
                re = self._get_var('re', r.dtype, kbody_term, shared=True)
                fe = self._get_var('fe', r.dtype, kbody_term, shared=True)
                omega = self._get_var('omega', r.dtype, kbody_term, shared=True)
                lamda = self._get_var('lamda', r.dtype, kbody_term, shared=True)
                one = tf.constant(1.0, dtype=r.dtype, name='one')
                return self._exp_func(re, fe, omega, lamda, one, name='rho')(r)

    def embed(self, rho: tf.Tensor, element: str):
        """
        The embedding energy function F(rho).

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
        kbody_term = f"{element}{element}"
        dtype = rho.dtype

        with tf.variable_scope(f"ZJW04/Embed/{element}"):
            Fn0 = self._get_var('Fn0', dtype, kbody_term, shared=True)
            Fn1 = self._get_var('Fn1', dtype, kbody_term, shared=True)
            Fn2 = self._get_var('Fn2', dtype, kbody_term, shared=True)
            Fn3 = self._get_var('Fn3', dtype, kbody_term, shared=True)
            F0 = self._get_var('F0', dtype, kbody_term, shared=True)
            F1 = self._get_var('F1', dtype, kbody_term, shared=True)
            F2 = self._get_var('F2', dtype, kbody_term, shared=True)
            F3 = self._get_var('F3', dtype, kbody_term, shared=True)
            eta = self._get_var('eta', dtype, kbody_term, shared=True)
            rho_e = self._get_var('rho_e', dtype, kbody_term, shared=True)
            rho_s = self._get_var('rho_s', dtype, kbody_term, shared=True)
            Fe = self._get_var('Fe', dtype, kbody_term, shared=True)
            rho_n = tf.convert_to_tensor(
                self._params[kbody_term]['rho_e'] * 0.85, dtype, name='rho_n')
            rho_0 = tf.convert_to_tensor(
                self._params[kbody_term]['rho_e'] * 1.15, dtype, name='rho_0')
            one = tf.constant(1.0, dtype, name='one')

            def embed1(_rho):
                """
                rho < rho_n
                """
                Fn = [Fn0, Fn1, Fn2, Fn3]
                x = _rho / rho_n - one
                return tf.add_n([tf.multiply(Fn[i], tf.pow(x, i))
                                 for i in range(len(Fn))], name='e1')

            def embed2(_rho):
                """
                rho_n <= rho < rho_0
                """
                F = [F0, F1, F2, F3]
                x = _rho / rho_e - one
                return tf.add_n([tf.multiply(F[i], tf.pow(x, i))
                                 for i in range(len(F))], name='e2')

            def embed3(_rho):
                """
                rho_0 <= rho
                """
                x = _rho / rho_s
                lnx = tf.log(x)
                return tf.multiply(Fe * (one - eta * lnx),
                                   tf.pow(x, eta), name='e3')

            shape = rho.shape
            idx1 = tf.where(tf.less(rho, rho_n), name='idx1')
            idx2 = tf.where(tf.logical_and(tf.greater_equal(rho, rho_n),
                                           tf.less(rho, rho_0)), name='idx2')
            idx3 = tf.where(tf.greater_equal(rho, rho_0), name='idx3')

            embed = tf.zeros(shape=shape, dtype=dtype, name='y')
            embed = scatter_nd_non_aliasing_add(
                embed, idx1, embed1(tf.gather_nd(rho, idx1)), name='y1')
            embed = scatter_nd_non_aliasing_add(
                embed, idx2, embed2(tf.gather_nd(rho, idx2)), name='y2')
            embed = scatter_nd_non_aliasing_add(
                embed, idx3, embed3(tf.gather_nd(rho, idx3)), name='y3')
            return embed
