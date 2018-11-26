# coding=utf-8
"""
This module defines various atomic neural networks.
"""
from __future__ import print_function, absolute_import

from typing import List, Dict

import numpy as np
import tensorflow as tf

from tensoralloy.nn.normalizer import InputNormalizer
from tensoralloy.nn.utils import GraphKeys, get_activation_fn, log_tensor
from tensoralloy.nn.basic import BasicNN
from tensoralloy.misc import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AtomicNN(BasicNN):
    """
    This class represents a general atomic neural network.
    """

    def __init__(self, elements: List[str], hidden_sizes=None, activation=None,
                 forces=False, stress=False, total_pressure=False, l2_weight=0.,
                 normalizer=None, normalization_weights=None):
        """
        Initialization method.

        normalizer : str
            The normalization method. Defaults to 'linear'. Set this to None to
            disable normalization.
        normalization_weights : Dict[str, array_like]
            The initial weights for column-wise normalizing the input atomic
            descriptors.

        """
        super(AtomicNN, self).__init__(
            elements=elements, hidden_sizes=hidden_sizes, activation=activation,
            forces=forces, stress=stress, total_pressure=total_pressure,
            l2_weight=l2_weight)
        self._initial_normalizer_weights = normalization_weights
        self._normalizer = InputNormalizer(method=normalizer)

    @property
    def hidden_sizes(self) -> Dict[str, List[int]]:
        """
        Return the sizes of hidden layers for each element.
        """
        return self._hidden_sizes

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Build 1x1 Convolution1D based atomic neural networks for all elements.
        """
        with tf.variable_scope("ANN"):
            activation_fn = get_activation_fn(self._activation)
            outputs = []
            for i, element in enumerate(self._elements):
                with tf.variable_scope(element):
                    x = tf.identity(features.descriptors[element], name='input')
                    if self._initial_normalizer_weights is not None:
                        x = self._normalizer(
                            x, self._initial_normalizer_weights[element])
                    hidden_sizes = self._hidden_sizes[element]
                    if verbose:
                        log_tensor(x)
                    yi = self._get_1x1conv_nn(
                        x, activation_fn, hidden_sizes, verbose=verbose)
                    yi = tf.squeeze(yi, axis=2, name='atomic')
                    if verbose:
                        log_tensor(yi)
                    outputs.append(yi)
            return outputs

    def _get_energy(self, outputs, features, verbose=True):
        """
        Return the Op to compute total energy.

        Parameters
        ----------
        outputs : List[tf.Tensor]
            A list of `tf.Tensor` as the outputs of the ANNs.
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
            y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
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


class AtomicResNN(AtomicNN):
    """
    A general atomic residual neural network. The ANNs are used to fit residual
    energies.

    The total energy, `y_total`, is expressed as:

        y_total = y_static + y_res

    where `y_static` is the total atomic static energy and this only depends on
    chemical compositions:

        y_static = sum([atomic_static_energy[e] * count(e) for e in elements])

    where `count(e)` represents the number of element `e`.

    """

    def __init__(self, elements: List[str], hidden_sizes=None, activation=None,
                 l2_weight=0., forces=False, stress=False, total_pressure=False,
                 normalizer='linear', normalization_weights=None,
                 atomic_static_energy=None):
        """
        Initialization method.
        """
        super(AtomicResNN, self).__init__(
            elements=elements, hidden_sizes=hidden_sizes, activation=activation,
            l2_weight=l2_weight, forces=forces, stress=stress,
            total_pressure=total_pressure, normalizer=normalizer,
            normalization_weights=normalization_weights)
        self._atomic_static_energy = atomic_static_energy

    def _check_keys(self, features: AttributeDict, labels: AttributeDict):
        """
        Check the keys of `features` and `labels`.
        """
        super(AtomicResNN, self)._check_keys(features, labels)
        assert 'composition' in features

    def _get_energy(self, outputs, features, verbose=True):
        """
        Return the Op to compute total energy (eV).
        """
        with tf.name_scope("Energy"):

            with tf.variable_scope("Static", reuse=tf.AUTO_REUSE):
                if self._atomic_static_energy is None:
                    values = np.ones(len(self._elements), dtype=np.float64)
                else:
                    values = np.asarray(
                        [self._atomic_static_energy[e] for e in self._elements],
                        dtype=np.float64)
                initializer = tf.constant_initializer(values, dtype=tf.float64)
                x = tf.identity(features.composition, name='input')
                z = tf.get_variable("weights",
                                    shape=(len(self._elements)),
                                    dtype=tf.float64,
                                    trainable=True,
                                    collections=[
                                        GraphKeys.STATIC_ENERGY_VARIABLES,
                                        GraphKeys.TRAIN_METRICS,
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        tf.GraphKeys.GLOBAL_VARIABLES],
                                    initializer=initializer)
                xz = tf.multiply(x, z, name='xz')
                y_static = tf.reduce_sum(xz, axis=1, keepdims=False,
                                         name='static')
                if verbose:
                    log_tensor(y_static)

            with tf.name_scope("Residual"):
                y_atomic = tf.concat(outputs, axis=1, name='atomic')
                with tf.name_scope("mask"):
                    mask = tf.split(
                        features.mask, [1, -1], axis=1, name='split')[1]
                    y_mask = tf.multiply(y_atomic, mask, name='mask')
                y_res = tf.reduce_sum(y_mask, axis=1, keepdims=False,
                                      name='residual')

            energy = tf.add(y_static, y_res, name='energy')

            with tf.name_scope("Ratio"):
                ratio = tf.reduce_mean(tf.div(y_static, energy, name='ratio'),
                                       name='avg')
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, ratio)
                tf.summary.scalar(ratio.op.name + '/summary', ratio,
                                  collections=[GraphKeys.TRAIN_SUMMARY, ])

            return energy