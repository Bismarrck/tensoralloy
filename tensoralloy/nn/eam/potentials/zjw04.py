# coding=utf-8
"""
This module defines the empirical potential of Al-Cu proposed by Zhou et al. at
2004.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.utils import get_elements_from_kbody_term

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class Zjw04(EamAlloyPotential):
    """
    A set of eam/alloy potentials proposed by Zhou et al. at 2004.

    References
    ----------
    Zhou, R. Johnson and H. Wadley, Phys. Rev. B. 69 (2004) 144113.

    """

    defaults = {
        'Al': {
            're': 2.863924, 'fe': 1.403115, 
            'rho_e': 20.418205, 'rho_s': 23.195740, 
            'alpha': 6.613165, 'beta': 3.527021, 
            'A': 0.314873, 'B': 0.365551,
            'kappa': 0.379846, 'lamda': 0.759692,  
            'Fn0': -2.807602, 'Fn1': -0.301435, 
            'Fn2': 1.258562, 'Fn3': -1.247604,
            'F0': -2.83, 'F1': 0.0, 'F2': 0.622245, 'F3': -2.488244, 
            'eta': 0.785902, 'Fe': -2.824528,
        },
        'Cu': {
            're': 2.556162, 'fe': 1.554485, 
            'rho_e': 21.175871, 'rho_s': 21.175395,
            'alpha': 8.127620, 'beta': 4.334731,
            'A': 0.396620, 'B': 0.548085,
            'kappa': 0.308782, 'lamda': 0.756515, 
            'Fn0': -2.170269, 'Fn1': -0.263788, 
            'Fn2': 1.088878,  'Fn3': -0.817603,
            'F0': -2.19, 'F1': 0.0, 'F2': 0.561830, 'F3': -2.100595, 
            'eta': 0.310490, 'Fe': -2.186568,
        },
        'Ni': {
            're': 2.488746, 'fe': 2.007018,
            'rho_e': 27.562015, 'rho_s': 27.930410,
            'alpha': 8.383453, 'beta': 4.471175,
            'A': 0.429046, 'B': 0.633531,
            'kappa': 0.443599, 'lamda': 0.820658,
            'Fn0': -2.693513, 'Fn1': -0.076445,
            'Fn2': 0.241442, 'Fn3': -2.375626,
            'F0': -2.70, 'F1': 0.0, 'F2': 0.265390, 'F3': -0.152856,
            'eta': 0.469000, 'Fe': -2.699486,
        },
        'Ag': {
            're': 2.891814, 'fe': 1.106232,
            'rho_e': 14.604100, 'rho_s': 14.604144,
            'alpha': 9.132010, 'beta': 4.870405,
            'A': 0.277758, 'B': 0.419611,
            'kappa': 0.339710, 'lamda': 0.750758,
            'Fn0': -1.729364, 'Fn1': -0.255882,
            'Fn2': 0.912050, 'Fn3': -0.561432,
            'F0': -1.75, 'F1': 0.0, 'F2': 0.744561, 'F3': -1.150650,
            'eta': 0.783924, 'Fe': -1.748423,
        },
        'Mo': {
            're': 2.728100, 'fe': 2.723710,
            'rho_e': 29.354065, 'rho_s': 29.354065,
            'alpha': 8.393531, 'beta': 4.476550,
            'A': 0.708787, 'B': 1.120373,
            'kappa': 0.137640, 'lamda': 0.275280,
            'Fn0': -3.692913, 'Fn1': -0.178812,
            'Fn2': 0.380450, 'Fn3': -3.133650,
            'F0': -3.71, 'F1': 0.0, 'F2': 0.875874, 'F3': 0.776222,
            'eta': 0.790879, 'Fe': -3.712093,
        },
        'Co': {
            're': 2.505979, 'fe': 1.975299,
            'rho_e': 27.206789, 'rho_s': 27.206789,
            'alpha': 8.679625, 'beta': 4.629134,
            'A': 0.421378, 'B': 0.640107,
            'kappa': 0.5, 'lamda': 1.0,
            'Fn0': -2.541799, 'Fn1': -0.219415,
            'Fn2': 0.733381, 'Fn3': -1.589003,
            'F0': -2.56, 'F1': 0.0, 'F2': 0.705845, 'F3': -0.687140,
            'eta': 0.694608, 'Fe': -2.559307,
        },
        'Mg': {
            're': 3.196291, 'fe': 0.544323,
            'rho_e': 7.132600, 'rho_s': 7.132600,
            'alpha': 10.228708, 'beta': 5.455311,
            'A': 0.137518, 'B': 0.225930,
            'kappa': 0.5, 'lamda': 1.0,
            'Fn0': -0.896473, 'Fn1': -0.044291,
            'Fn2': 0.162232, 'Fn3': -0.689950,
            'F0': -0.90, 'F1': 0.0, 'F2': 0.122838, 'F3': -0.226010,
            'eta': 0.431425, 'Fe': -0.899702,
        },
        'Fe': {
            're': 2.481987, 'fe': 1.885957,
            'rho_e': 20.041463, 'rho_s': 20.041463,
            'alpha': 9.818270, 'beta': 5.236411,
            'A': 0.392811, 'B': 0.646243,
            'kappa': 0.170306, 'lamda': 0.340613,
            'Fn0': -2.534992, 'Fn1': -0.059605,
            'Fn2': 0.193065, 'Fn3': -2.282322,
            'F0': -2.54, 'F1': 0.0, 'F2': 0.200269, 'F3': -0.148770,
            'eta': 0.391750, 'Fe': -2.539945,
        }
    }

    def __init__(self):
        super(Zjw04, self).__init__()

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
        if kbody_term in ('AlAl', 'CuCu'):
            element = get_elements_from_kbody_term(kbody_term)[0]
            with tf.variable_scope(f"ZJW04/Phi/{element}"):
                re = self._get_var('re', r.dtype, element, shared=True)
                A = self._get_var('A', r.dtype, element, shared=True)
                B = self._get_var('B', r.dtype, element, shared=True)
                alpha = self._get_var('alpha', r.dtype, element, shared=True)
                beta = self._get_var('beta', r.dtype, element, shared=True)
                kappa = self._get_var('kappa', r.dtype, element, shared=True)
                lamda = self._get_var('lamda', r.dtype, element, shared=True)
                one = tf.constant(1.0, dtype=r.dtype, name='one')
                return tf.subtract(
                    self._exp_func(re, A, alpha, kappa, one, name='A')(r),
                    self._exp_func(re, B, beta, lamda, one, name='B')(r),
                    name='phi')
        else:
            with tf.name_scope('AlAl'):
                phi_al = self.phi(r, 'AlAl')
                rho_al = self.rho(r, 'Al')
            with tf.name_scope("CuCu"):
                phi_cu = self.phi(r, 'CuCu')
                rho_cu = self.rho(r, 'Cu')
            half = tf.constant(0.5, dtype=r.dtype, name='half')
            return tf.multiply(half,
                               tf.add(tf.div(rho_al, rho_cu) * phi_cu,
                                      tf.div(rho_cu, rho_al) * phi_al),
                               name='phi')

    def rho(self, r: tf.Tensor, element: str):
        """
        The electron density function rho(r).

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, max_n_terms, 1, nnl, 1]`.
        element : str
            The corresponding element.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        with tf.variable_scope(f"ZJW04/Rho/{element}"):
            re = self._get_var('re', r.dtype, element, shared=True)
            fe = self._get_var('fe', r.dtype, element, shared=True)
            beta = self._get_var('beta', r.dtype, element, shared=True)
            lamda = self._get_var('lamda', r.dtype, element, shared=True)
            one = tf.constant(1.0, dtype=r.dtype, name='one')
            return self._exp_func(re, fe, beta, lamda, one, name='rho')(r)

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
        dtype = rho.dtype

        with tf.variable_scope(f"ZJW04/Embed/{element}"):
            Fn0 = self._get_var('Fn0', dtype, element, shared=True)
            Fn1 = self._get_var('Fn1', dtype, element, shared=True)
            Fn2 = self._get_var('Fn2', dtype, element, shared=True)
            Fn3 = self._get_var('Fn3', dtype, element, shared=True)
            F0 = self._get_var('F0', dtype, element, shared=True)
            F1 = self._get_var('F1', dtype, element, shared=True)
            F2 = self._get_var('F2', dtype, element, shared=True)
            F3 = self._get_var('F3', dtype, element, shared=True)
            eta = self._get_var('eta', dtype, element, shared=True)
            rho_e = self._get_var('rho_e', dtype, element, shared=True)
            rho_s = self._get_var('rho_s', dtype, element, shared=True)
            Fe = self._get_var('Fe', dtype, element, shared=True)
            rho_n = tf.convert_to_tensor(
                self._params[element]['rho_e'] * 0.85, dtype, name='rho_n')
            rho_0 = tf.convert_to_tensor(
                self._params[element]['rho_e'] * 1.15, dtype, name='rho_0')
            one = tf.constant(1.0, dtype, name='one')

            def embed1(_rho):
                """
                rho < rho_n
                """
                with tf.name_scope("e1"):
                    Fn = [Fn0, Fn1, Fn2, Fn3]
                    x = _rho / rho_n - one
                    return tf.add_n([tf.multiply(Fn[i], tf.pow(x, i))
                                     for i in range(len(Fn))], name='e1')

            def embed2(_rho):
                """
                rho_n <= rho < rho_0
                """
                with tf.name_scope("e2"):
                    F = [F0, F1, F2, F3]
                    x = _rho / rho_e - one
                    return tf.add_n([tf.multiply(F[i], tf.pow(x, i))
                                     for i in range(len(F))], name='e2')

            def embed3(_rho):
                """
                rho_0 <= rho
                """
                with tf.name_scope("e3"):
                    x = _rho / rho_s
                    lnx = tf.log(x)
                    return tf.multiply(Fe * (one - eta * lnx),
                                       tf.pow(x, eta), name='e3')

            shape = rho.shape
            idx1 = tf.where(tf.less(rho, rho_n), name='idx1')
            idx2 = tf.where(tf.logical_and(tf.greater_equal(rho, rho_n),
                                           tf.less(rho, rho_0)), name='idx2')
            idx3 = tf.where(tf.greater_equal(rho, rho_0), name='idx3')

            values = [
                tf.scatter_nd(idx1, embed1(tf.gather_nd(rho, idx1)), shape),
                tf.scatter_nd(idx2, embed2(tf.gather_nd(rho, idx2)), shape),
                tf.scatter_nd(idx3, embed3(tf.gather_nd(rho, idx3)), shape),
            ]
            return tf.add_n(values, name='embed')
