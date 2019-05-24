# coding=utf-8
"""
This module defines the empirical potential of Al-Cu proposed by Zhou et al. at
2004.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.nn.utils import log_tensor
from tensoralloy.nn.eam.potentials.generic import zhou_exp
from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.extension.grad_ops import safe_pow

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


zjw04_defaults = {
    'Al': {
        'r_eq': 2.863924, 'f_eq': 1.403115,
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
        'r_eq': 2.556162, 'f_eq': 1.554485,
        'rho_e': 21.175871, 'rho_s': 21.175395,
        'alpha': 8.127620, 'beta': 4.334731,
        'A': 0.396620, 'B': 0.548085,
        'kappa': 0.308782, 'lamda': 0.756515,
        'Fn0': -2.170269, 'Fn1': -0.263788,
        'Fn2': 1.088878, 'Fn3': -0.817603,
        'F0': -2.19, 'F1': 0.0, 'F2': 0.561830, 'F3': -2.100595,
        'eta': 0.310490, 'Fe': -2.186568,
    },
    'Ni': {
        'r_eq': 2.488746, 'f_eq': 2.007018,
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
        'r_eq': 2.891814, 'f_eq': 1.106232,
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
        'r_eq': 2.728100, 'f_eq': 2.723710,
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
        'r_eq': 2.505979, 'f_eq': 1.975299,
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
        'r_eq': 3.196291, 'f_eq': 0.544323,
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
        'r_eq': 2.481987, 'f_eq': 1.885957,
        'rho_e': 20.041463, 'rho_s': 20.041463,
        'alpha': 9.818270, 'beta': 5.236411,
        'A': 0.392811, 'B': 0.646243,
        'kappa': 0.170306, 'lamda': 0.340613,
        'Fn0': -2.534992, 'Fn1': -0.059605,
        'Fn2': 0.193065, 'Fn3': -2.282322,
        'F0': -2.54, 'F1': 0.0, 'F2': 0.200269, 'F3': -0.148770,
        'eta': 0.391750, 'Fe': -2.539945,
    },
    'Pd': {
        'r_eq': 2.750897, 'f_eq': 1.595417,
        'rho_e': 21.335246, 'rho_s': 21.940073,
        'alpha': 8.697397, 'beta': 4.638612,
        'A': 0.406763, 'B': 0.598880,
        'kappa': 0.397263, 'lamda': 0.754799,
        'Fn0': -2.321006, 'Fn1': -0.473983,
        'Fn2': 1.615343, 'Fn3': -0.231681,
        'F0': -2.36, 'F1': 0.0, 'F2': 1.481742, 'F3': -1.675615,
        'eta': 1.13, 'Fe': -2.352753,
    },
    'W': {
        'r_eq': 2.740840, 'f_eq': 3.487340,
        'rho_e': 37.234847, 'rho_s': 37.234847,
        'alpha': 8.900114, 'beta': 4.746728,
        'A': 0.882435, 'B': 1.394592,
        'kappa': 0.139209, 'lamda': 0.278417,
        'Fn0': -4.946281, 'Fn1': -0.148818,
        'Fn2': 0.365057, 'Fn3': -4.432406,
        'F0': -4.96, 'F1': 0.0, 'F2': 0.661935, 'F3': 0.348147,
        'eta': -0.582714, 'Fe': -4.961306,
    },
    'Ta': {
        'r_eq': 2.860082, 'f_eq': 3.086341,
        'rho_e': 33.787168, 'rho_s': 33.787168,
        'alpha': 8.489528, 'beta': 4.527748,
        'A': 0.611679, 'B': 1.032101,
        'kappa': 0.176977, 'lamda': 0.353954,
        'Fn0': -5.103845, 'Fn1': -0.405524,
        'Fn2': 1.112997, 'Fn3': -3.585325,
        'F0': -5.14, 'F1': 0.0, 'F2': 1.640098, 'F3': 0.221375,
        'eta': 0.848843, 'Fe': -5.141526,
    },
    'Zr': {
        'r_eq': 3.199978, 'f_eq': 2.230909,
        'rho_e': 30.879991, 'rho_s': 30.879991,
        'alpha': 8.559190, 'beta': 4.564902,
        'A': 0.424667, 'B': 0.640054,
        'kappa': 0.5, 'lamda': 1.0,
        'Fn0': -4.485793, 'Fn1': -0.293129,
        'Fn2': 0.990148, 'Fn3': -3.202516,
        'F0': -4.51, 'F1': 0.0, 'F2': 0.928602, 'F3': -0.981870,
        'eta': 0.597133, 'Fe': -4.509025,
    },
}


class Zjw04(EamAlloyPotential):
    """
    A set of eam/alloy potentials proposed by Zhou et al. at 2004.

    References
    ----------
    Zhou, R. Johnson and H. Wadley, Phys. Rev. B. 69 (2004) 144113.

    """

    def __init__(self):
        """
        Initialization method.

        All embed function related parameters are fixed. The original embed
        function is a piecewise function so direct optimizaition will break the
        continuity at endpoints of segments.

        """
        fixed = {element: ['F0', 'F1', 'F2', 'F3', 'Fn0', 'Fn1', 'Fn2', 'Fn3',
                           'Fe', 'eta', 'rho_e', 'rho_s', 'r_eq']
                 for element in zjw04_defaults}
        super(Zjw04, self).__init__(fixed=fixed)
        self._name = 'Zjw04'

    @property
    def defaults(self):
        """
        Return the default parameters.
        """
        return zjw04_defaults

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
        if el_a == el_b:
            with tf.name_scope(f"{self._name}/Phi/{el_a}"):
                r_eq = self._get_shared_variable('r_eq', r.dtype, el_a)
                A = self._get_shared_variable('A', r.dtype, el_a)
                B = self._get_shared_variable('B', r.dtype, el_a)
                alpha = self._get_shared_variable('alpha', r.dtype, el_a)
                beta = self._get_shared_variable('beta', r.dtype, el_a, )
                kappa = self._get_shared_variable('kappa', r.dtype, el_a)
                lamda = self._get_shared_variable('lamda', r.dtype, el_a)
                phi = tf.subtract(
                    zhou_exp(r, A=A, B=alpha, C=kappa, r_eq=r_eq, name='A'),
                    zhou_exp(r, A=B, B=beta, C=lamda, r_eq=r_eq, name='B'),
                    name='phi')
                if verbose:
                    log_tensor(phi)
                return phi
        else:
            with tf.name_scope(f'{el_a}{el_a}'):
                phi_a = self.phi(r, f'{el_a}{el_a}', variable_scope, False)
                rho_a = self.rho(r, f'{el_a}', variable_scope, False)
            with tf.name_scope(f"{el_b}{el_b}"):
                phi_b = self.phi(r, f'{el_b}{el_b}', variable_scope, False)
                rho_b = self.rho(r, f'{el_b}', variable_scope, False)
            half = tf.constant(0.5, dtype=r.dtype, name='half')
            phi = tf.multiply(half,
                              tf.add(tf.math.divide(rho_a, rho_b) * phi_b,
                                     tf.math.divide(rho_b, rho_a) * phi_a),
                              name='phi')
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
            r_eq = self._get_shared_variable('r_eq', r.dtype, element)
            f_eq = self._get_shared_variable('f_eq', r.dtype, element)
            beta = self._get_shared_variable('beta', r.dtype, element)
            lamda = self._get_shared_variable('lamda', r.dtype, element)
            rho = zhou_exp(r, A=f_eq, B=beta, C=lamda, r_eq=r_eq)
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
            Fn0 = self._get_shared_variable('Fn0', dtype, element)
            Fn1 = self._get_shared_variable('Fn1', dtype, element)
            Fn2 = self._get_shared_variable('Fn2', dtype, element)
            Fn3 = self._get_shared_variable('Fn3', dtype, element)
            F0 = self._get_shared_variable('F0', dtype, element)
            F1 = self._get_shared_variable('F1', dtype, element)
            F2 = self._get_shared_variable('F2', dtype, element)
            F3 = self._get_shared_variable('F3', dtype, element)
            eta = self._get_shared_variable('eta', dtype, element)
            rho_e = self._get_shared_variable('rho_e', dtype, element)
            rho_s = self._get_shared_variable('rho_s', dtype, element)
            Fe = self._get_shared_variable('Fe', dtype, element)
            rho_n = tf.multiply(tf.constant(0.85, dtype=dtype, name='lb'),
                                rho_e, name='rho_n')
            rho_0 = tf.multiply(tf.constant(1.15, dtype=dtype, name='ub'),
                                rho_e, name='rho_0')
            one = tf.constant(1.0, dtype, name='one')
            two = tf.constant(2.0, dtype, name='two')
            three = tf.constant(3.0, dtype, name='three')

            def embed1(_rho):
                """
                rho < rho_n
                """
                x1 = tf.subtract(tf.math.divide(_rho, rho_n), one, name='x')
                x2 = safe_pow(x1, two)
                x3 = safe_pow(x1, three)
                e11 = tf.multiply(Fn1, x1, 'Fn1e1')
                e12 = tf.multiply(Fn2, x2, 'Fn2e2')
                e13 = tf.multiply(Fn3, x3, 'Fn3e3')
                return tf.add(Fn0, e11 + e12 + e13, name='e1')

            def embed2(_rho):
                """
                rho_n <= rho < rho_0

                Notes
                -----
                1. `x` may be zero because 0.85 * rho_e <= rho < 1.15 * rho_e
                   and `tf.pow(0, 0)` is `nan` which leads to inf loss.
                2. `x` will be differentiated twice when computing gradients of
                   force loss w.r.t. Zjw04 parameters. However the current
                   implementation of `tf.pow(x, y)` will return NaN but not zero
                   when computing
                       `tf.gradients(tf.gradientx(tf.pow(x, 1), x), x)`
                   if `x` is zero.

                TODO: fix `tf.pow` and create a pull request.

                """
                with tf.name_scope("e2"):
                    x1 = _rho / rho_e - one
                    x2 = safe_pow(x1, two)
                    x3 = safe_pow(x1, three)
                    e11 = tf.multiply(F1, x1, 'Fn1e1')
                    e12 = tf.multiply(F2, x2, 'Fn2e2')
                    e13 = tf.multiply(F3, x3, 'Fn3e3')
                    return tf.add(F0, tf.add_n([e11, e12, e13], name='e2_123'),
                                  name='e2')

            def embed3(_rho):
                """
                rho_0 <= rho
                """
                with tf.name_scope("e3"):
                    x_e3 = _rho / rho_s
                    lnx = tf.log(x_e3)
                    return tf.multiply(Fe * (one - eta * lnx),
                                       safe_pow(x_e3, eta), name='e3')

            idx1 = tf.where(tf.less(rho, rho_n), name='idx1')
            idx2 = tf.where(tf.logical_and(tf.greater_equal(rho, rho_n),
                                           tf.less(rho, rho_0)), name='idx2')
            idx3 = tf.where(tf.greater_equal(rho, rho_0), name='idx3')
            shape = tf.shape(rho, name='shape', out_type=idx1.dtype)

            values = [
                tf.scatter_nd(idx1, embed1(tf.gather_nd(rho, idx1)), shape),
                tf.scatter_nd(idx2, embed2(tf.gather_nd(rho, idx2)), shape),
                tf.scatter_nd(idx3, embed3(tf.gather_nd(rho, idx3)), shape),
            ]
            embed = tf.add_n(values, name='embed')
            if verbose:
                log_tensor(embed)
            return embed

    def dipole(self,
               r: tf.Tensor,
               kbody_term: str,
               variable_scope: str,
               verbose=False):
        """
        Zjw04 does not support calculating dipole.
        """
        raise Exception(
            f"{self.__class__.__name__} does not have dipole term.")

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   verbose=False):
        """
        Zjw04 does not support calculating dipole.
        """
        raise Exception(
            f"{self.__class__.__name__} does not have quadrupole term.")


class Zjw04xc(Zjw04):
    """
    A modified implementation of `Zjw04`.
    """

    def __init__(self):
        """
        Initialization method.
        """
        super(Zjw04xc, self).__init__()

        self._fixed = {element: ['r_eq'] for element in self.defaults.keys()}
        self._name = 'Zjw04xc'

    @property
    def defaults(self):
        """
        The default parameters of Zjw04xc.
        """
        params = zjw04_defaults.copy()
        params['Pu'] = zjw04_defaults['Zr'].copy()
        return params

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
            one = tf.constant(1.0, dtype=dtype, name='one')
            two = tf.constant(2.0, dtype=dtype, name='two')
            three = tf.constant(3.0, dtype=dtype, name='three')
            Fn0 = self._get_shared_variable('Fn0', dtype, element)
            Fn1 = self._get_shared_variable('Fn1', dtype, element)
            Fn2 = self._get_shared_variable('Fn2', dtype, element)
            Fn3 = self._get_shared_variable('Fn3', dtype, element)
            F0 = self._get_shared_variable('F0', dtype, element)
            F1 = self._get_shared_variable('F1', dtype, element)
            F2 = self._get_shared_variable('F2', dtype, element)
            F3 = self._get_shared_variable('F3', dtype, element)
            eta = self._get_shared_variable('eta', dtype, element)
            rho_e = self._get_shared_variable('rho_e', dtype, element)
            rho_s = self._get_shared_variable('rho_s', dtype, element)
            Fe = self._get_shared_variable('Fe', dtype, element)
            rho_n = tf.multiply(tf.constant(0.85, dtype=dtype, name='lb'),
                                rho_e, name='rho_n')
            rho_0 = tf.multiply(tf.constant(1.15, dtype=dtype, name='ub'),
                                rho_e, name='rho_0')

            def embed1(_rho):
                """
                rho < rho_n
                """
                x1 = tf.subtract(tf.math.truediv(_rho, rho_n), one, name='x')
                x2 = safe_pow(x1, two)
                x3 = safe_pow(x1, three)
                e11 = tf.multiply(Fn1, x1, 'Fn1e1')
                e12 = tf.multiply(Fn2, x2, 'Fn2e2')
                e13 = tf.multiply(Fn3, x3, 'Fn3e3')
                return tf.add(Fn0, e11 + e12 + e13, name='e1')

            def embed2(_rho):
                """
                rho_n <= rho < rho_0

                Notes
                -----
                1. `x` may be zero because 0.85 * rho_e <= rho < 1.15 * rho_e
                   and `tf.pow(0, 0)` is `nan` which leads to inf loss.
                2. `x` will be differentiated twice when computing gradients of
                   force loss w.r.t. Zjw04 parameters. However the current
                   implementation of `tf.pow(x, y)` will return NaN but not zero
                   when computing
                       `tf.gradients(tf.gradientx(tf.pow(x, 1), x), x)`
                   if `x` is zero.

                """
                with tf.name_scope("e2"):
                    x = tf.subtract(tf.math.truediv(_rho, rho_e), one, name='x')
                    e2 = [tf.multiply(F1, x, 'F1e1'),
                          tf.multiply(F2, safe_pow(x, two), 'F2e2'),
                          tf.multiply(F3, safe_pow(x, three), 'F3e3')]
                    return tf.add(F0, tf.add_n(e2, name='e2_123'),
                                  name='e2')

            def embed3(_rho):
                """
                rho_0 <= rho
                """
                with tf.name_scope("e3"):
                    eps = tf.constant(1e-8, dtype, name='eps')
                    x = tf.add(tf.math.divide(_rho, rho_s), eps, name='x')
                    lnx = tf.log(x)
                    return tf.multiply(Fe * (one - eta * lnx),
                                       safe_pow(x, eta), name='e3')

            y1 = embed1(rho)
            y2 = embed2(rho)
            y3 = embed3(rho)

            c1 = tf.sigmoid(tf.multiply(two, rho_n - rho))
            c3 = tf.sigmoid(tf.multiply(two, rho - rho_0))
            c2 = tf.subtract(one, tf.add(c1, c3))

            embed = tf.add_n([tf.multiply(c1, y1, name='c1e1'),
                              tf.multiply(c2, y2, name='c2e2'),
                              tf.multiply(c3, y3, name='c3e3')], name='embed')

            if verbose:
                log_tensor(embed)

            return embed


class Zjw04uxc(Zjw04xc):
    """
    An unrestricted implementation of `Zjw04xc`. `r_eq` is treated as a plain
    variable, but not the equilibrium spacing between neighbors, in this
    potential.
    """

    def __init__(self):
        """
        Initialization method.
        """
        super(Zjw04uxc, self).__init__()

        self._fixed = {}
        self._name = 'Zjw04uxc'


class Zjw04xcp(Zjw04xc):
    """
    A modified version of `Zjw04xc`. The pairwise interaction of AB is also
    described by the .
    """

    def __init__(self):
        super(Zjw04xcp, self).__init__()

        self._name = "Zjw04xcp"
        self._fixed = {
            element: ['F0', 'F1', 'F2', 'F3', 'Fn0', 'Fn1', 'Fn2', 'Fn3',
                      'Fe', 'eta', 'rho_e', 'rho_s', 'r_eq']
            for element in zjw04_defaults.keys()}
