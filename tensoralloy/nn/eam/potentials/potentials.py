# coding=utf-8
"""
This module defines general classical potential function layers for NN-EAM.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List, Dict
from abc import ABC

from tensoralloy.utils import GraphKeys, safe_select

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_variable(name,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 collections=None,
                 validate_shape=True):
    """
    A wrapper of `tf.get_variable`.
    """
    _collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if trainable:
        _collections.append(tf.GraphKeys.TRAINABLE_VARIABLES)
    if collections is not None:
        assert isinstance(collections, (tuple, list, set))
        for collection in collections:
            if collection not in _collections:
                _collections.append(collection)
    return tf.get_variable(name, shape=(), dtype=dtype,
                           initializer=initializer, regularizer=regularizer,
                           trainable=trainable, collections=_collections,
                           validate_shape=validate_shape)


class EmpiricalPotential:
    """
    An `EmpiricalPotential` represents a tensorflow based implemetation of an
    empirical EAM potential.
    """

    # The default parameters and (initial) values. Only scalar values are
    # supported.
    defaults = {}

    def __init__(self, params=None, fixed=None):
        """
        Initialization method.

        Parameters
        ----------
        params : Dict[str, Dict[str, Union[float, int]]] or None
            A nested dict to update the default values.
        fixed : Dict[str, List[str]] or None
            A dict of (kbody_term, [var1, ...]) where `[var1, ...]` is a list of
            str as the fixed (non-trainable) parameters for `kbody_term`.

        """
        fixed = safe_select(fixed, {})
        params = safe_select(params, {})

        self._params = self.defaults.copy()
        for section, values in params.items():
            if section in self._params:
                self._params[section].update(values)
            else:
                raise KeyError(
                    f"{self.__class__.__name__} does not support `{section}`")

        self._fixed = {}
        for section in fixed:
            if self.__contains__(section):
                self._fixed[section] = list(fixed[section])

        self._shared_variables = {}

    @property
    def params(self):
        """
        Return the parameters.
        """
        return self._params

    @property
    def fixed(self):
        """
        Return a dict of fixed parameters and their sections.
        """
        return self._fixed

    def __contains__(self, section):
        """
        Return True if `kbody_term` is allowed.
        """
        return section in self._params

    def _is_trainable(self, parameter, section):
        """
        Return True if the parameter is trainable.
        """
        if section not in self._fixed:
            return True
        elif parameter not in self._fixed[section]:
            return True
        return False

    def _get_variable(self, parameter, dtype, section, variable_scope: str):
        """
        A function to initialize a new local `tf.Variable`.
        Local variables cannot be used by potential functions outside `section`.
        Local variables will be placed under the given variable scope.

        Parameters
        ----------
        parameter : str
            The name of the variable.
        dtype : DType
            The data type of the variable.
        section : str
            The section to which the parameter belongs.
        variable_scope : str
            The created variable will be placed under this scope.

        Returns
        -------
        var : tf.Variable
            The corresponding variable.

        """
        trainable = self._is_trainable(parameter, section)
        with tf.variable_scope(variable_scope, reuse=False):
            var = get_variable(name=parameter, dtype=dtype,
                               initializer=tf.constant_initializer(
                                   value=self._params[section][parameter],
                                   dtype=dtype),
                               collections=[
                                   GraphKeys.EAM_POTENTIAL_VARIABLES,
                                   tf.GraphKeys.MODEL_VARIABLES,
                               ],
                               trainable=trainable)
            return var

    def _get_shared_variable(self, parameter: str, dtype, section: str):
        """
        Return a shared variable. A shared variable may be reused by potential
        functions in different sections.

        As an example, all variables in `zjw04` are shared. The param `Al.re` is
        used by `Al.rho()` and `Al.embed()`.

        Parameters
        ----------
        parameter : str
            The name of the variable.
        dtype : DType
            The data type of the variable.
        section : str
            The section to which the parameter belongs.

        Returns
        -------
        var : tf.Variable
            The corresponding variable.

        """
        tag = f"{section}.{parameter}"
        if tag in self._shared_variables:
            return self._shared_variables[tag]

        with tf.variable_scope(f"Shared/{section}", reuse=tf.AUTO_REUSE):
            trainable = self._is_trainable(parameter, section)
            var = get_variable(name=parameter, dtype=dtype,
                               initializer=tf.constant_initializer(
                                   value=self._params[section][parameter],
                                   dtype=dtype),
                               collections=[
                                   GraphKeys.EAM_POTENTIAL_VARIABLES,
                                   tf.GraphKeys.MODEL_VARIABLES],
                               trainable=trainable)
            self._shared_variables[tag] = var
            return var

    def rho(self,
            r: tf.Tensor,
            element_or_kbody_term: str,
            variable_scope: str):
        """
        Return the Op to compute electron density `rho(r)`.
        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")

    def phi(self,
            r: tf.Tensor,
            kbody_term: str,
            variable_scope: str):
        """
        Return the Op to compute pairwise potential `phi(r)`.

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, 1, max_n_element, nnl, 1]`.
        kbody_term : str
            The corresponding k-body term.
        variable_scope : str
            The scope for variables of this potential function.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")

    def embed(self,
              rho: tf.Tensor,
              element: str,
              variable_scope: str):
        """
        Return the Op to compute the embedding energy F(rho(r)).

        Parameters
        ----------
        rho : tf.Tensor
            A 3D tensor of shape `[batch_size, max_n_element, 1]` where
            `max_n_element` is the maximum occurace of `element`.
        element : str
            An element symbol.
        variable_scope : str
            The scope for variables of this potential function.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")


class EamAlloyPotential(EmpiricalPotential, ABC):
    """
    This class represents an `EAM/Alloy` style empirical potential.
    """

    def rho(self, r: tf.Tensor, element: str, variable_scope: str):
        """
        Return the Op to compute electron density `rho(r)`.

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, max_n_terms, 1, nnl, 1]`.
        element : str
            The corresponding element.
        variable_scope : str
            The scope for variables of this potential function.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")


class EamFSPotential(EmpiricalPotential, ABC):
    """
    This class represents an `EAM/Finnis-Sinclair` style empirical potential.
    """

    def rho(self, r: tf.Tensor, kbody_term: str, variable_scope: str):
        """
        Return the Op to compute electron density `rho(r)`.

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, 1, max_n_element, nnl, 1]`.
        kbody_term : str
            The corresponding k-body term.
        variable_scope : str
            The scope for variables of this potential function.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")
