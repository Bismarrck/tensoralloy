# coding=utf-8
"""
This module defines various atomic neural networks.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List, Dict
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.utils import GraphKeys
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.convolutional import convolution1x1

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AtomicNN(BasicNN):
    """
    This class represents a general atomic neural network.
    """

    default_collection = GraphKeys.ATOMIC_NN_VARIABLES

    def __init__(self,
                 elements: List[str],
                 hidden_sizes=None,
                 activation=None,
                 kernel_initializer='he_normal',
                 minmax_scale=True,
                 use_resnet_dt=False,
                 atomic_static_energy=None,
                 use_atomic_static_energy=True,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces', 'hessian')):
        """
        Initialization method.
        """
        super(AtomicNN, self).__init__(
            elements=elements,
            hidden_sizes=hidden_sizes,
            activation=activation,
            minimize_properties=minimize_properties,
            export_properties=export_properties)

        self._kernel_initializer = kernel_initializer
        self._minmax_scale = minmax_scale
        self._use_resnet_dt=use_resnet_dt
        self._atomic_static_energy = atomic_static_energy or {}
        self._use_atomic_static_energy = use_atomic_static_energy

    @property
    def hidden_sizes(self) -> Dict[str, List[int]]:
        """
        Return the sizes of hidden layers for each element.
        """
        return self._hidden_sizes

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        return {"class": self.__class__.__name__,
                "elements": self._elements,
                "hidden_sizes": self._hidden_sizes,
                "activation": self._activation,
                'kernel_initializer': self._kernel_initializer,
                'minmax_scale': self._minmax_scale,
                'use_resnet_dt': self._use_resnet_dt,
                'use_atomic_static_energy': self._use_atomic_static_energy,
                'atomic_static_energy': self._atomic_static_energy,
                "minimize_properties": self._minimize_properties,
                "export_properties": self._export_properties}

    def _get_model_outputs(self,
                           features: dict,
                           descriptors: dict,
                           mode: tf_estimator.ModeKeys,
                           verbose=False):
        """
        Build 1x1 Convolution1D based atomic neural networks for all elements.

        Parameters
        ----------
        features : dict
            A dict of input tensors and the descriptors:
                * 'descriptors'
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'atom_masks' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        descriptors : dict
            A dict of (element, (value, mask)) where `element` represents the
            symbol of an element, `value` is the descriptors of `element` and
            `mask` is None.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        collections = [self.default_collection]

        with tf.variable_scope("ANN"):
            activation_fn = get_activation_fn(self._activation)
            outputs = []
            for element, (value, atom_mask) in descriptors.items():
                with tf.variable_scope(element, reuse=tf.AUTO_REUSE):
                    x = tf.identity(value, name='input')
                    if mode == tf_estimator.ModeKeys.PREDICT:
                        assert x.shape.ndims == 2
                        x = tf.expand_dims(x, axis=0, name='3d')

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

                    if self._use_atomic_static_energy:
                        bias_mean = self._atomic_static_energy.get(element, 0.0)
                    else:
                        bias_mean = 0.0

                    yi = convolution1x1(
                        x,
                        activation_fn=activation_fn,
                        hidden_sizes=hidden_sizes,
                        l2_weight=1.0,
                        collections=collections,
                        output_bias=self._use_atomic_static_energy,
                        output_bias_mean=bias_mean,
                        use_resnet_dt=self._use_resnet_dt,
                        kernel_initializer=self._kernel_initializer,
                        variable_scope=None,
                        verbose=verbose)
                    yi = tf.squeeze(yi, axis=2, name='atomic')
                    if verbose:
                        log_tensor(yi)
                    outputs.append(yi)
            return outputs

    def _get_internal_energy_op(self, outputs, features, name='energy',
                                verbose=True):
        """
        Return the Op to compute internal energy E.

        Parameters
        ----------
        outputs : List[tf.Tensor]
            A list of `tf.Tensor` as the outputs of the ANNs.
        features : dict
            A dict of input tensors.
        name : str
            The name of the output tensor.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        energy : tf.Tensor
            The total energy tensor.

        """
        y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
        ndims = features["atom_masks"].shape.ndims
        axis = ndims - 1
        with tf.name_scope("mask"):
            if ndims == 1:
                y_atomic = tf.squeeze(y_atomic, axis=0)
            mask = tf.split(
                features["atom_masks"], [1, -1], axis=axis, name='split')[1]
            y_mask = tf.multiply(y_atomic, mask, name='mask')
            self._y_atomic_op_name = y_mask.name
        energy = tf.reduce_sum(
            y_mask, axis=axis, keepdims=False, name=name)
        if verbose:
            log_tensor(energy)
        return energy
