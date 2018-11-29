# coding=utf-8
"""
This module defines the neural network based implementation of the EAM model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from typing import List

from tensoralloy.misc import AttributeDict, Defaults
from tensoralloy.utils import get_kbody_terms, get_elements_from_kbody_term
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.basic import BasicNN

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class EamNN(BasicNN):
    """
    The tensorflow/CNN based implementation of the Embedded-Atom Method.
    """

    def __init__(self, symmetric=False, *args, **kwargs):
        """
        Initialization method.
        """
        super(EamNN, self).__init__(*args, **kwargs)

        self._symmetric = symmetric and len(self._elements) > 1

    @property
    def symmetric(self):
        """
        If True, AB and BA will be considered to be equal.
        """
        return self._symmetric

    @property
    def all_kbody_terms(self):
        """
        Return a list of str as all the k-body terms for this model.
        """
        return self._all_kbody_terms

    def _convert_to_dict(self, hidden_sizes):
        all_kbody_terms, kbody_terms, _ = get_kbody_terms(
            self._elements, k_max=2)

        self._kbody_terms = kbody_terms
        self._all_kbody_terms = all_kbody_terms

        results = {}
        for element in self._all_kbody_terms:
            if isinstance(hidden_sizes, dict):
                sizes = np.asarray(
                    hidden_sizes.get(element, Defaults.hidden_sizes),
                    dtype=np.int)
            else:
                sizes = np.atleast_1d(hidden_sizes).astype(np.int)
            assert (sizes > 0).all()
            results[element] = sizes.tolist()
        return results

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

    def _build_phi_nn(self, partitions: AttributeDict, verbose=False):
        """
        Return the outputs of the pairwise interactions, `Phi(r)`.
        """
        activation_fn = get_activation_fn(self._activation)
        outputs = []
        with tf.name_scope("Phi"):
            half = tf.constant(0.5, dtype=tf.float64, name='half')
            for kbody_term, (xi, mi) in partitions.items():
                with tf.variable_scope(kbody_term):
                    # Convert `x` to a 5D tensor.
                    x = tf.expand_dims(xi, axis=-1, name='input')
                    if verbose:
                        log_tensor(x)

                    hidden_sizes = self._hidden_sizes[kbody_term]
                    y = self._get_1x1conv_nn(x, activation_fn, hidden_sizes,
                                             verbose=verbose)

                    # Apply the value mask
                    y = tf.multiply(y, mi, name='masked')

                    # `y` here will be reduced to a 2D tensor of shape
                    # `[batch_size, max_n_atoms]`
                    y = tf.reduce_sum(y, axis=(2, 3, 4), keepdims=False)
                    y = tf.multiply(y, half, name='atomic')

                    if verbose:
                        log_tensor(y)
                    outputs.append(y)
            return outputs

    def _build_embed_nn(self, partitions: AttributeDict, verbose=False):
        """
        Return the outputs of the embedding energy, `F(rho(r))`.
        """
        activation_fn = get_activation_fn(self._activation)
        outputs = []
        with tf.name_scope("Embed"):
            for kbody_term, (xi, mi) in partitions.items():
                hidden_sizes = self._hidden_sizes[kbody_term]
                with tf.variable_scope(kbody_term):
                    with tf.variable_scope("Rho"):
                        x = tf.expand_dims(xi, axis=-1, name='input')
                        if verbose:
                            log_tensor(x)
                        y = self._get_1x1conv_nn(x, activation_fn, hidden_sizes,
                                                 verbose=verbose)
                        # Apply the mask to rho.
                        y = tf.multiply(y, mi, name='masked')

                        rho = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                        if verbose:
                            log_tensor(rho)
                    with tf.variable_scope("F"):
                        rho = tf.expand_dims(rho, axis=-1, name='rho')
                        if verbose:
                            log_tensor(rho)
                        y = self._get_1x1conv_nn(
                            rho, activation_fn, hidden_sizes, verbose=verbose)
                        embed = tf.reduce_sum(y, axis=(2, 3), name='embed')
                        if verbose:
                            log_tensor(embed)
                        outputs.append(embed)
            return outputs

    def _dynamic_partition(self, features: AttributeDict):
        """
        Split the descriptors to `Np` partitions where `Np` is the total number
        of unique k-body terms. If `self.symmetric` is False, `Np` is equal to
        `N**2` where N is the number of elements. If `self.symmetric` is True,
        `Np` will be `N * (N + 1) / 2`.

        Returns
        -------
        partitions : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are unique kbody terms and values are tuples of
            (gi, mi) where `gi` represents the descriptors and `mi` is the value
            mask.

        """
        partitions = AttributeDict()

        with tf.name_scope("Partition"):
            for element in self._elements:
                kbody_terms = self._kbody_terms[element]
                g, mask = features.descriptors[element]
                num = len(kbody_terms)
                glists = tf.split(g, num_or_size_splits=num, axis=1)
                mlists = tf.split(mask, num_or_size_splits=num, axis=1)
                for i, (gi, mi) in enumerate(zip(glists, mlists)):
                    kbody_term = kbody_terms[i]
                    if self._symmetric:
                        kbody_term = ''.join(
                            sorted(get_elements_from_kbody_term(kbody_term)))
                        if kbody_term in partitions:
                            _gi, _mi = partitions[kbody_term]
                            gi = tf.concat((_gi, gi), axis=2)
                            mi = tf.concat((_mi, mi), axis=2)
                    partitions[kbody_term] = (gi, mi)
            return partitions

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Return the nn-EAM model.
        """
        with tf.name_scope("nnEAM"):
            partitions = self._dynamic_partition(features)
            outputs = (
                self._build_phi_nn(partitions, verbose=verbose),
                self._build_embed_nn(partitions, verbose=verbose)
            )
            return outputs
