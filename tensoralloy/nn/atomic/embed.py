#coding=utf-8
"""
This module defines a variant of `AtomicNN`: `EmbeddedAtomicNN`.
"""
from __future__ import print_function, absolute_import

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


import tensorflow as tf
import numpy as np

from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.nn.atomic.atomic import AtomicNN
from tensoralloy.nn.utils import log_tensor, get_activation_fn
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.precision import get_float_dtype


class EmbeddedAtomicNN(AtomicNN):
    """
    An variant of `AtomicResNN` whose one-body energy is calculated with the
    EAM approach.
    """

    def _get_model_outputs(self,
                           features: dict,
                           descriptors: dict,
                           mode: tf_estimator.ModeKeys,
                           verbose=False):
        """
        Return the Op to compute internal energy E.
        """
        collections = [self.default_collection]

        with tf.variable_scope("ANN"):
            rho_min = tf.constant(1e-8, dtype=get_float_dtype())
            activation_fn = get_activation_fn(self._activation)
            outputs = []
            for element, (value, atom_mask) in descriptors.items():
                with tf.variable_scope(element, reuse=tf.AUTO_REUSE):
                    x = tf.identity(value, name='input')
                    if mode == tf_estimator.ModeKeys.PREDICT:
                        assert x.shape.ndims == 2
                        x = tf.expand_dims(x, axis=0, name='3d')
                    with tf.name_scope("Embed"):
                        rho = tf.identity(x, name='rho')
                        safe_rho = tf.maximum(rho, rho_min, name='rho/safe')
                        log_rho = tf.math.log(safe_rho, name='rho/log')


                    if self._minmax_scale:
                        with tf.name_scope("MinMax"):
                            _collections = [
                                               tf.GraphKeys.GLOBAL_VARIABLES,
                                               tf.GraphKeys.MODEL_VARIABLES,
                                           ] + collections
                            _shape = [1, 1, x.shape[-1]]
                            _dtype = x.dtype
                            _get_initializer = \
                                lambda val: tf.constant_initializer(val, _dtype)

                            xlo = tf.get_variable(
                                name="xlo", shape=_shape, dtype=_dtype,
                                trainable=False, collections=_collections,
                                initializer=_get_initializer(1000.0),
                                aggregation=tf.VariableAggregation.MEAN)
                            xhi = tf.get_variable(
                                name="xhi", shape=_shape, dtype=_dtype,
                                trainable=False, collections=_collections,
                                initializer=_get_initializer(0.0),
                                aggregation=tf.VariableAggregation.MEAN)

                            if mode == tf_estimator.ModeKeys.TRAIN:
                                xmax = tf.reduce_max(x, [0, 1], True, 'xmax')
                                xmin = tf.reshape(
                                    tf.reduce_min(
                                        tf.boolean_mask(x, atom_mask), axis=0),
                                    xmax.shape, name='xmin')
                                xlo_op = tf.assign(xlo, tf.minimum(xmin, xlo))
                                xhi_op = tf.assign(xhi, tf.maximum(xmax, xhi))
                                update_ops = [xlo_op, xhi_op]
                            else:
                                update_ops = []
                            with tf.control_dependencies(update_ops):
                                x = tf.div_no_nan(xhi - x, xhi - xlo, name='x')

                    hidden_sizes = self._hidden_sizes[element]
                    if verbose:
                        log_tensor(x)
                    yi = convolution1x1(
                        x,
                        activation_fn=activation_fn,
                        hidden_sizes=hidden_sizes,
                        l2_weight=1.0,
                        collections=collections,
                        kernel_initializer=self._kernel_initializer,
                        variable_scope=None,
                        verbose=verbose)
                    yi = tf.squeeze(yi, axis=2, name='atomic')
                    if verbose:
                        log_tensor(yi)
                    outputs.append(yi)
            return outputs
