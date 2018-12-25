# coding=utf-8
"""
This module defines a variant of `AtomicNN`: `AtomicResNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from typing import List

from tensoralloy.misc import AttributeDict
from tensoralloy.nn.atomic.atomic import AtomicNN
from tensoralloy.nn.utils import GraphKeys, log_tensor

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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

    default_collection = GraphKeys.ATOMIC_RES_NN_VARIABLES

    def __init__(self, elements: List[str], hidden_sizes=None, activation=None,
                 loss_weights=None, minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces'), normalizer='linear',
                 normalization_weights=None, atomic_static_energy=None):
        """
        Initialization method.
        """
        super(AtomicResNN, self).__init__(
            elements=elements, hidden_sizes=hidden_sizes, activation=activation,
            loss_weights=loss_weights, minimize_properties=minimize_properties,
            export_properties=export_properties, normalizer=normalizer,
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

            ndims = features.composition.shape.ndims
            axis = ndims - 1

            with tf.variable_scope("Static", reuse=tf.AUTO_REUSE):
                if self._atomic_static_energy is None:
                    values = np.ones(len(self._elements), dtype=np.float64)
                else:
                    values = np.asarray(
                        [self._atomic_static_energy[e] for e in self._elements],
                        dtype=np.float64)
                initializer = tf.constant_initializer(values, dtype=tf.float64)
                x = tf.identity(features.composition, name='input')
                if ndims == 1:
                    x = tf.expand_dims(x, axis=0, name='1to2')
                if x.shape[1].value is None:
                    x.set_shape([x.shape[0], len(self._elements)])
                z = tf.get_variable("weights",
                                    shape=len(self._elements),
                                    dtype=tf.float64,
                                    trainable=True,
                                    collections=[
                                        GraphKeys.ATOMIC_RES_NN_VARIABLES,
                                        GraphKeys.TRAIN_METRICS,
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        tf.GraphKeys.GLOBAL_VARIABLES,
                                        tf.GraphKeys.MODEL_VARIABLES],
                                    initializer=initializer)
                xz = tf.multiply(x, z, name='xz')
                if ndims == 1:
                    y_static = tf.reduce_sum(xz, keepdims=False, name='static')
                else:
                    y_static = tf.reduce_sum(xz, axis=1, keepdims=False,
                                             name='static')
                if verbose:
                    log_tensor(y_static)

            with tf.name_scope("Residual"):
                y_atomic = tf.concat(outputs, axis=1, name='atomic')
                with tf.name_scope("mask"):
                    if ndims == 1:
                        y_atomic = tf.squeeze(y_atomic, axis=0)
                    mask = tf.split(
                        features.mask, [1, -1], axis=axis, name='split')[1]
                    y_mask = tf.multiply(y_atomic, mask, name='mask')
                y_res = tf.reduce_sum(y_mask, axis=axis, keepdims=False,
                                      name='residual')
                if verbose:
                    log_tensor(y_res)

            energy = tf.add(y_static, y_res, name='energy')
            if verbose:
                log_tensor(energy)

            with tf.name_scope("Ratio"):
                ratio = tf.reduce_mean(tf.div(y_static, energy, name='ratio'),
                                       name='avg')
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, ratio)
                tf.summary.scalar(ratio.op.name + '/summary', ratio)
            return energy
