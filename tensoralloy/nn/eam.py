# coding=utf-8
"""
This module defines the neural network based implementation of the EAM model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from typing import List, Dict
from collections import Counter

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

    def _get_energy(self, outputs: List[tf.Tensor], features: AttributeDict,
                    verbose=True):
        """
        Return the Op to compute total energy of nn-EAM.

        Parameters
        ----------
        outputs : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the unmasked
            atomic energies.
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
            y_atomic = tf.identity(outputs, name='atomic')
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
        outputs = {}
        with tf.name_scope("Phi"):
            half = tf.constant(0.5, dtype=tf.float64, name='half')
            for kbody_term, (value, mask) in partitions.items():
                with tf.variable_scope(kbody_term):
                    # Convert `x` to a 5D tensor.
                    x = tf.expand_dims(value, axis=-1, name='input')
                    if verbose:
                        log_tensor(x)

                    hidden_sizes = self._hidden_sizes[kbody_term]
                    y = self._get_1x1conv_nn(x, activation_fn, hidden_sizes,
                                             verbose=verbose)

                    # Apply the value mask
                    y = tf.multiply(y, tf.expand_dims(mask, axis=-1),
                                    name='masked')

                    # `y` here will be reduced to a 2D tensor of shape
                    # `[batch_size, max_n_atoms]`
                    y = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                    y = tf.squeeze(y, axis=1)
                    y = tf.multiply(y, half, name='atomic')
                    if verbose:
                        log_tensor(y)
                    outputs[kbody_term] = y
            return self._dynamic_stitch(outputs)

    def _build_rho_nn(self, partitions: AttributeDict, verbose=False):
        """
        Return the outputs of the electron densities, `rho(r)`.
        """
        activation_fn = get_activation_fn(self._activation)
        outputs = {}
        with tf.name_scope("Rho"):
            for kbody_term, (value, mask) in partitions.items():
                hidden_sizes = self._hidden_sizes[kbody_term]
                with tf.variable_scope(kbody_term):
                    x = tf.expand_dims(value, axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    y = self._get_1x1conv_nn(x, activation_fn, hidden_sizes,
                                             verbose=verbose)
                    # Apply the mask to rho.
                    y = tf.multiply(y, tf.expand_dims(mask, axis=-1),
                                    name='masked')

                    y = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                    rho = tf.squeeze(y, axis=1, name='rho')
                    if verbose:
                        log_tensor(rho)
                    outputs[kbody_term] = rho
            return self._dynamic_stitch(outputs)

    def _build_embed_nn(self, rho: tf.Tensor, max_occurs: Counter,
                        verbose=True):
        """
        Return the embedding energy, `F(rho)`.

        Parameters
        ----------
        rho : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the electron
            density of each atom.
        max_occurs : Counter
            The maximum occurance of each type of element.

        Returns
        -------
        embed : tf.Tensor
            The embedding energy. Has the same shape with `rho`.

        """
        activation_fn = get_activation_fn(self._activation)
        split_sizes = [max_occurs[el] for el in self._elements]

        with tf.name_scope("Embed"):
            splits = tf.split(rho, num_or_size_splits=split_sizes, axis=1)
            values = []
            for i, element in enumerate(self._elements):
                hidden_sizes = self._hidden_sizes[f"{element}{element}"]
                with tf.variable_scope(element):
                    x = tf.expand_dims(splits[i], axis=-1, name=element)
                    if verbose:
                        log_tensor(x)
                    y = self._get_1x1conv_nn(x, activation_fn, hidden_sizes,
                                             verbose=verbose)
                    embed = tf.squeeze(y, axis=2, name='atomic')
                    if verbose:
                        log_tensor(embed)
                    values.append(embed)
            return tf.concat(values, axis=1)

    def _dynamic_stitch(self, outputs: Dict[str, tf.Tensor]):
        """
        The reverse of `dynamic_partition`. Interleave the kbody-term centered
        `outputs` of type `Dict[kbody_term, tensor]` to element centered values
        of type `Dict[element, tensor]`.

        Parameters
        ----------
        outputs : Dict[str, tf.Tensor]
            A dict. The keys are unique kbody-terms and values are 2D tensors
            with shape `[batch_size, max_n_elements]` where `max_n_elements`
            denotes the maximum occurances of the center element of the
            correponding kbody-term.

        Returns
        -------
        atomic : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the energies
            of the real atoms.

        """
        with tf.name_scope("Stitch"):
            stacks = {}
            results = []
            for kbody_term, value in outputs.items():
                elements = get_elements_from_kbody_term(kbody_term)
                if self._symmetric and elements[0] != elements[1]:
                    results.append(value)
                else:
                    center = elements[0]
                    if center not in stacks:
                        stacks[center] = [value]
                    else:
                        stacks[center].append(value)
            results.append(
                tf.concat([tf.add_n(stacks[el], name=el)
                           for el in self._elements], axis=1))
            if not self._symmetric:
                return tf.identity(results[0], name='sum')
            else:
                return tf.add_n(results, name='sum')

    def _dynamic_partition(self, features: AttributeDict):
        """
        Split the descriptors of type `Dict[element, (tensor, mask)]` to `Np`
        partitions where `Np` is the total number of unique k-body terms.

        If `self.symmetric` is False, `Np` is equal to `N**2`.
        If `self.symmetric` is True, `Np` will be `N * (N + 1) / 2`.

        Here N denotes the total number of elements.

        Returns
        -------
        partitions : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are unique kbody terms and values are tuples of
            (gi, mi) where `gi` represents the descriptors and `mi` is the value
            mask.
        max_occurs : Counter
            The maximum occurance of each type of element.

        """
        partitions = AttributeDict()
        max_occurs = {}

        with tf.name_scope("Partition"):
            for element in self._elements:
                kbody_terms = self._kbody_terms[element]
                values, masks = features.descriptors[element]
                max_occurs[element] = values.shape[2].value
                num = len(kbody_terms)
                glists = tf.split(values, num_or_size_splits=num, axis=1)
                mlists = tf.split(masks, num_or_size_splits=num, axis=1)
                for i, (value, mask) in enumerate(zip(glists, mlists)):
                    kbody_term = kbody_terms[i]
                    if self._symmetric:
                        kbody_term = ''.join(
                            sorted(get_elements_from_kbody_term(kbody_term)))
                        if kbody_term in partitions:
                            value = tf.concat(
                                (partitions[kbody_term][0], value), axis=2)
                            mask = tf.concat(
                                (partitions[kbody_term][1], mask), axis=2)
                    partitions[kbody_term] = (value, mask)
            return partitions, Counter(max_occurs)

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Return the nn-EAM model.
        """
        with tf.name_scope("nnEAM"):
            partitions, max_occurs = self._dynamic_partition(features)

            rho = self._build_rho_nn(partitions, verbose=verbose)
            embed = self._build_embed_nn(rho, max_occurs, verbose=verbose)
            phi = self._build_phi_nn(partitions, verbose=verbose)

            y = tf.add(phi, embed, name='atomic')
            return y
