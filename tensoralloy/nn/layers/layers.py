# coding=utf-8
"""
This module defines general classical potential function layers for NN-EAM.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import List, Dict, Union

from tensoralloy.misc import safe_select

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


any_kbody_term = 'ANY'

# The collection of shared variables
SHARED_VARS = 'shared_variables'


def get_variable(name,
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
    return tf.get_variable(name, shape=(), dtype=dtype,
                           initializer=initializer, regularizer=regularizer,
                           trainable=trainable, collections=collections,
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

    def _check_kbody_term(self, kbody_term):
        """
        Convert `kbody_term` to `any_kbody_term` if needed.
        """
        if kbody_term not in self._params:
            if any_kbody_term in self._params:
                kbody_term = any_kbody_term
        return kbody_term

    def _is_trainable(self, parameter, kbody_term):
        """
        Return True if the parameter is trainable.
        """
        kbody_term = self._check_kbody_term(kbody_term)
        if kbody_term not in self._fixed:
            return True
        elif parameter not in self._fixed[kbody_term]:
            return True
        return False

    def _get_var(self, parameter, dtype, kbody_term, shared=False):
        """
        A helper function to initialize a new `tf.Variable`.
        """
        kbody_term = self._check_kbody_term(kbody_term)

        shared_key = f"{kbody_term}.{parameter}"
        if shared:
            if shared_key in self._shared_vars:
                return self._shared_vars[shared_key]

        trainable = self._is_trainable(parameter, kbody_term)
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

    def embed(self, rho: tf.Tensor, element: str):
        """
        Return the Op to compute the embedding energy F(rho(r)).
        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")
