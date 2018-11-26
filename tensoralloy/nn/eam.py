# coding=utf-8
"""
This module defines the neural network based implementation of the EAM model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import List

from tensoralloy.misc import AttributeDict
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.basic import BasicNN

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class EamNN(BasicNN):
    """
    The implementation of nn-EAM.
    """

    def _get_energy(self, outputs: (List[tf.Tensor], List[tf.Tensor]),
                    features: AttributeDict, verbose=True):
        """
        Return the Op to compute total energy of nn-EAM.

        Parameters
        ----------
        outputs : List[List[tf.Tensor], List[tf.Tensor]]
            A tuple of `List[tf.Tensor]`.
        features : AttributeDict
            A dict of input features.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        energy : tf.Tensor
            The total energy tensor.

        """
        with tf.name_scope("Energy"):
            with tf.name_scope("Atomic"):
                values = []
                for i, (phi, embed) in enumerate(zip(*outputs)):
                    values.append(tf.add(phi, embed, name=self._elements[i]))
                y_atomic = tf.concat(values, axis=1, name='atomic')
            shape = y_atomic.shape
            with tf.name_scope("mask"):
                mask = tf.split(
                    features.mask, [1, -1], axis=1, name='split')[1]
                y_mask = tf.multiply(y_atomic, mask, name='mask')
                y_mask.set_shape(shape)
            energy = tf.reduce_sum(
                y_mask, axis=1, keepdims=False, name='energy')
            if verbose:
                log_tensor(energy)
            return energy

    def _build_phi_nn(self, features: AttributeDict, verbose=False):
        """
        Return the outputs of the pairwise interactions, `Phi(r)`.
        """
        activation_fn = get_activation_fn(self._activation)
        outputs = []
        with tf.name_scope("Phi"):
            for i, element in enumerate(self._elements):
                with tf.variable_scope(element):
                    # Convert `x` to a 5D tensor.
                    x = tf.expand_dims(features.descriptors[element],
                                       axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    hidden_sizes = self._hidden_sizes[element]
                    y = self._get_1x1conv_nn(x, activation_fn, hidden_sizes,
                                             verbose=verbose)
                    # `y` here will be reduced to a 2D tensor of shape
                    # `[batch_size, max_n_atoms]`
                    y = tf.reduce_sum(y, axis=(2, 3, 4), keepdims=False,
                                      name='atomic')
                    if verbose:
                        log_tensor(y)
                    outputs.append(y)
            return outputs

    def _build_embed_nn(self, features: AttributeDict, verbose=False):
        """
        Return the outputs of the embedding energy, `F(rho(r))`.
        """
        activation_fn = get_activation_fn(self._activation)
        outputs = []
        with tf.name_scope("Embed"):
            for i, element in enumerate(self._elements):
                hidden_sizes = self._hidden_sizes[element]
                with tf.variable_scope(element):
                    with tf.variable_scope("Rho"):
                        x = tf.expand_dims(features.descriptors[element],
                                           axis=-1, name='input')
                        if verbose:
                            log_tensor(x)
                        y = self._get_1x1conv_nn(x, activation_fn, hidden_sizes,
                                                 verbose=verbose)
                        rho = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                        if verbose:
                            log_tensor(rho)
                    with tf.variable_scope("F"):
                        rho = tf.expand_dims(rho, axis=-1, name='rho')
                        y = self._get_1x1conv_nn(
                            rho, activation_fn, hidden_sizes, verbose=verbose)
                        embed = tf.reduce_sum(y, axis=(2, 3), name='embed')
                        if verbose:
                            log_tensor(embed)
                        outputs.append(embed)
            return outputs

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Return the nn-EAM model.
        """
        with tf.name_scope("nnEAM"):
            outputs = (
                self._build_phi_nn(features, verbose=verbose),
                self._build_embed_nn(features, verbose=verbose)
            )
            return outputs
