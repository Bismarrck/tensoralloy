# coding=utf-8
"""
This module defines general classical potential function layers for NN-EAM.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import List, Dict
from abc import ABC

from tensoralloy.misc import safe_select
from tensoralloy.nn.utils import GraphKeys

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

        self._shared_vars = {}

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

    def _get_var(self, parameter, dtype, section, shared=False):
        """
        A helper function to initialize a new `tf.Variable`.
        """
        tag = f"{section}.{parameter}"
        if shared:
            if tag in self._shared_vars:
                return self._shared_vars[tag]

        trainable = self._is_trainable(parameter, section)
        var = get_variable(name=parameter, dtype=dtype,
                           initializer=tf.constant_initializer(
                               value=self._params[section][parameter],
                               dtype=dtype),
                           collections=[
                               GraphKeys.EAM_POTENTIAL_VARIABLES,
                               tf.GraphKeys.MODEL_VARIABLES,
                           ],
                           trainable=trainable)
        if shared:
            self._shared_vars[tag] = var
        return var

    def rho(self, *args):
        """
        Return the Op to compute electron density `rho(r)`.
        """
        raise NotImplementedError(
            "This method must be overridden by its subclass!")

    def phi(self, r: tf.Tensor, kbody_term: str):
        """
        Return the Op to compute pairwise potential `phi(r)`.

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
        raise NotImplementedError(
            "This method must be overridden by its subclass!")

    def embed(self, rho: tf.Tensor, element: str):
        """
        Return the Op to compute the embedding energy F(rho(r)).

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
        raise NotImplementedError(
            "This method must be overridden by its subclass!")


class EamAlloyPotential(EmpiricalPotential, ABC):
    """
    This class represents an `EAM/Alloy` style empirical potential.
    """

    def rho(self, r: tf.Tensor, element: str):
        """
        Return the Op to compute electron density `rho(r)`.

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
        raise NotImplementedError(
            "This method must be overridden by its subclass!")


class EamFSPotential(EmpiricalPotential, ABC):
    """
    This class represents an `EAM/Finnis-Sinclair` style empirical potential.
    """

    def rho(self, r: tf.Tensor, kbody_term: str):
        """
        Return the Op to compute electron density `rho(r)`.

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
        raise NotImplementedError(
            "This method must be overridden by its subclass!")
