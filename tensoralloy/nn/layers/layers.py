# coding=utf-8
"""
This module defines general classical potential function layers for NN-EAM.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import List, Dict, Union

from tensoralloy.misc import safe_select
from tensoralloy.utils import get_elements_from_kbody_term

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


any_kbody_term = 'ANY'

# The collection of shared variables
SHARED_VARS = 'shared_variables'


def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 collections=None,
                 validate_shape=True,
                 shared=False):
    """
    A wrapper of `tf.get_variable`.
    """
    _collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if trainable:
        _collections.append(tf.GraphKeys.TRAINABLE_VARIABLES)
    if shared:
        _collections.append(SHARED_VARS)
    if collections is not None:
        _collections += list(collections)
    return tf.get_variable(name, shape=shape, dtype=dtype,
                           initializer=initializer, regularizer=regularizer,
                           trainable=trainable, collections=collections,
                           reuse=tf.AUTO_REUSE,
                           validate_shape=validate_shape)


class PotentialFunctionLayer:
    """
    A `PotentialFunctionLayer` represents a tensorflow based implemetation of a
    classical potential function.
    """

    # The default parameters and (initial) values. Only scalar values are
    # supported.
    defaults = {any_kbody_term: {}}

    def __init__(self, allowed_kbody_terms: Union[List[str], None]=None,
                 params=None, fixed=None):
        """
        Initialization method.

        Parameters
        ----------
        allowed_kbody_terms : List[str] or None
            A list of str as the allowed k-body terms or None so that there will
            be no restriction for this function.
        params : Dict[str, Dict[str, Union[float, int]]] or None
            A nested dict.
        fixed : Dict[str, List[str]] or None
            A dict of (kbody_term, [var1, ...]) where `[var1, ...]` is a list of
            str as the fixed (non-trainable) parameters for `kbody_term`.

        """
        self._allowed_kbody_terms = safe_select(allowed_kbody_terms,
                                                [any_kbody_term])

        fixed = safe_select(fixed, {})
        params = safe_select(params, {})

        self._params = self.defaults.copy()
        for kbody_term, values in params.items():
            if kbody_term in self._params:
                self._params[kbody_term].update(values)
            elif any_kbody_term in self._params:
                d = self._params[any_kbody_term].copy()
                d.update(values)
                self._params[kbody_term] = d
            else:
                raise ValueError(
                    f"{self.__class__.__name__} does not support {kbody_term}")

        self._fixed = {}
        for kbody_term in fixed:
            if self.__contains__(kbody_term):
                self._fixed[kbody_term] = list(fixed[kbody_term])

        self._shared_vars = {}

    @property
    def allowed_kbody_terms(self) -> List[str]:
        """
        Return a list of str as the allowed kbody-terms for this potential.
        """
        return self._allowed_kbody_terms

    @property
    def params(self):
        """
        Return the parameters.
        """
        return self._params

    @property
    def fixed(self):
        """
        Return whether a parameter is trainable or not.
        """
        return self._fixed

    def __contains__(self, kbody_term):
        """
        Return True if `kbody_term` is allowed.
        """
        _in = kbody_term in self._allowed_kbody_terms
        _any = any_kbody_term in self._allowed_kbody_terms
        return _in or _any

    def _get_var(self, parameter, dtype, kbody_term, shared=False):
        """
        A helper function to initialize a new `tf.Variable`.
        """
        if kbody_term not in self._params:
            if any_kbody_term in self._params:
                kbody_term = any_kbody_term
        shared_key = f"{kbody_term}.{parameter}"
        if shared and shared in self._shared_vars:
            return self._shared_vars[shared_key]
        trainable = parameter in self._fixed[kbody_term]
        var = get_variable(name=parameter, dtype=dtype,
                           initializer=tf.constant_initializer(
                               value=self._params[kbody_term][parameter],
                               dtype=dtype),
                           shared=shared,
                           trainable=trainable)
        if shared:
            self._shared_vars[shared_key] = var
        return var

    def rho(self, r: tf.Tensor, kbody_term: str):
        """
        Return the Op to compute electron density `rho(r)`.
        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")

    def phi(self, r: tf.Tensor, kbody_term: str):
        """
        Return the Op to compute pairwise potential `phi(r)`.
        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")

    def embed(self, rho: tf.Tensor, kbody_term: str):
        """
        Return the Op to compute the embedding energy F(rho(r)).
        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")


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

    def rho(self, r: tf.Tensor, kbody_term: str):
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

    def embed(self, rho: tf.Tensor, kbody_term: str):
        """
        The embedding function:

            F(rho) = -sqrt(rho)

        """
        with tf.variable_scope('Sutton'):
            return tf.negative(tf.sqrt(rho), name='embed')


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
            'omega': 3.527021,
        },
        'CuCu': {
            're': 2.556162, 'fe': 1.554485, 'A': 0.396620, 'B': 0.548085,
            'kappa': 0.308782, 'lamda': 0.756515, 'gamma': 8.127620,
            'omega': 4.334731,
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
            r_re_1 = r_re - one
            upper = a * tf.exp(-b * r_re_1)
            lower = one + tf.pow(re - c, 20)
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
        if kbody_term in ['AlAl', 'CuCu']:
            with tf.variable_scope("ZJW04"):
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
            phi_al = self.phi(r, 'AlAl')
            phi_cu = self.phi(r, 'CuCu')
            rho_al = self.rho(r, 'AlAl')
            rho_cu = self.rho(r, 'CuCu')
            half = tf.constant(0.5, dtype=r.dtype, name='half')
            return tf.multiply(half,
                               tf.add(tf.div_no_nan(rho_al, rho_cu) * phi_cu,
                                      tf.div_no_nan(rho_cu, rho_al) * phi_al),
                               name='phi')

    def rho(self, r: tf.Tensor, kbody_term: str, **kwargs):
        """
        The electron density function rho(r).
        """
        if kbody_term == 'AlCu':
            split_sizes = kwargs['split_sizes']
            r_al, r_cu = tf.split(r, num_or_size_splits=split_sizes, axis=2)
            rho_al = self.rho(r_al, 'AlAl')
            rho_cu = self.rho(r_cu, 'CuCu')
            return tf.concat((rho_al, rho_cu), axis=2, name='rho')
        else:
            element = get_elements_from_kbody_term(kbody_term)[0]
            with tf.variable_scope(f"ZJW04_{element}"):
                re = self._get_var('re', r.dtype, kbody_term, shared=True)
                fe = self._get_var('fe', r.dtype, kbody_term, shared=True)
                omega = self._get_var('omega', r.dtype, kbody_term, shared=True)
                lamda = self._get_var('lamda', r.dtype, kbody_term, shared=True)
                one = tf.constant(1.0, dtype=r.dtype, name='one')
                return self._exp_func(re, fe, omega, lamda, one, name='rho')(r)

    def embed(self, rho: tf.Tensor, kbody_term: str):
        pass
