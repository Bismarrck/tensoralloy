# coding=utf-8
"""
This module defines a variant of `AtomicNN`: `AtomicResNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from typing import List, Dict

from tensoralloy.nn.atomic.atomic import AtomicNN
from tensoralloy.nn.utils import log_tensor
from tensoralloy.utils import GraphKeys, AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

# TODO: fix the atomic energy


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

    def __init__(self,
                 elements: List[str],
                 hidden_sizes=None,
                 activation=None,
                 kernel_initializer='he_normal',
                 minmax_scale=True,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces', 'hessian'),
                 atomic_static_energy=None,
                 fixed_static_energy=False):
        """
        Initialization method.
        """
        super(AtomicResNN, self).__init__(
            elements=elements,
            hidden_sizes=hidden_sizes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            minmax_scale=minmax_scale,
            minimize_properties=minimize_properties,
            export_properties=export_properties)

        self._atomic_static_energy: Dict[str, float] = atomic_static_energy
        self._fixed_static_energy = fixed_static_energy

    @property
    def fixed_static_energy(self):
        """
        Return True if the static energy parameters are fixed.
        """
        return self._fixed_static_energy

    def _check_keys(self, features: AttributeDict, labels: AttributeDict):
        """
        Check the keys of `features` and `labels`.
        """
        super(AtomicResNN, self)._check_keys(features, labels)
        assert 'composition' in features

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        d = super(AtomicResNN, self).as_dict()
        d['atomic_static_energy'] = self._atomic_static_energy
        d['fixed_static_energy'] = self._fixed_static_energy
        return d

    def _get_internal_energy_op(self, outputs, features, name='energy', verbose=True):
        """
        Return the Op to compute internal energy E.
        """

        ndims = features.composition.shape.ndims
        axis = ndims - 1

        with tf.variable_scope("Static", reuse=tf.AUTO_REUSE):
            x = tf.identity(features.composition, name='input')
            if ndims == 1:
                x = tf.expand_dims(x, axis=0, name='1to2')
            if x.shape[1].value is None:
                x.set_shape([x.shape[0], len(self._elements)])
            if self._atomic_static_energy is None:
                values = np.ones(len(self._elements),
                                 dtype=x.dtype.as_numpy_dtype)
            else:
                values = np.asarray(
                    [self._atomic_static_energy[e] for e in self._elements],
                    dtype=x.dtype.as_numpy_dtype)
            initializer = tf.constant_initializer(values, dtype=x.dtype)
            trainable = not self._fixed_static_energy
            collections = [GraphKeys.ATOMIC_RES_NN_VARIABLES,
                           GraphKeys.TRAIN_METRICS,
                           tf.GraphKeys.GLOBAL_VARIABLES,
                           tf.GraphKeys.MODEL_VARIABLES]
            if trainable:
                collections.append(tf.GraphKeys.TRAINABLE_VARIABLES)
            z = tf.get_variable("weights",
                                shape=len(self._elements),
                                dtype=x.dtype,
                                trainable=trainable,
                                collections=collections,
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
                self._y_atomic_op_name = y_mask.name
            y_res = tf.reduce_sum(y_mask, axis=axis, keepdims=False,
                                  name='residual')
            if verbose:
                log_tensor(y_res)

        energy = tf.add(y_static, y_res, name=name)

        with tf.name_scope("Ratio"):
            ratio = tf.reduce_mean(tf.math.truediv(y_static, energy,
                                                   name='ratio'),
                                   name='avg')
            tf.add_to_collection(GraphKeys.TRAIN_METRICS, ratio)
            tf.summary.scalar(ratio.op.name + '/summary', ratio)

        if verbose:
            log_tensor(energy)

        return energy
