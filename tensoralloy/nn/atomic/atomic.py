# coding=utf-8
"""
This module defines various atomic neural networks.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List, Dict
from collections import Counter
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.utils import GraphKeys
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.dataclasses import EnergyOps, LossParameters
from tensoralloy.nn.losses import LossMethod
from tensoralloy.nn import losses as loss_ops
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.atomic.dataclasses import AtomicDescriptors

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
                 fixed_atomic_static_energy=False,
                 temperature_dependent=False,
                 temperature_layers=(128, 128),
                 temperature_activation='softplus',
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
        self._use_resnet_dt = use_resnet_dt
        self._atomic_static_energy = atomic_static_energy or {}
        self._use_atomic_static_energy = use_atomic_static_energy
        self._fixed_atomci_static_energy = fixed_atomic_static_energy
        self._temperature_dependent = temperature_dependent
        self._temperature_layers = temperature_layers
        self._temperature_activation = temperature_activation

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
                'fixed_atomic_static_energy': self._fixed_atomci_static_energy,
                'atomic_static_energy': self._atomic_static_energy,
                'temperature_dependent': self._temperature_dependent,
                'temperature_layers': self._temperature_layers,
                'temperature_activation': self._temperature_activation,
                "minimize_properties": self._minimize_properties,
                "export_properties": self._export_properties}

    def _get_atomic_energy_op_name(self):
        """
        The Op for calculating atomic energies.
        """
        if self._temperature_dependent:
            return "Output/Energy/E/atomic"
        else:
            return "Output/Energy/U/atomic"

    def _get_atomic_descriptors(self,
                                universal_descriptors,
                                mode: tf_estimator.ModeKeys,
                                verbose=True) -> AtomicDescriptors:
        """
        Return the atomic descriptors calaculated based on the universal
        descriptors.
        """
        raise NotImplementedError("")

    @staticmethod
    def _apply_minmax_normalization(x: tf.Tensor,
                                    mask: tf.Tensor,
                                    mode: tf_estimator.ModeKeys,
                                    collections=None):
        """
        Apply the min-max normalization to raw symmetry function descriptors.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        mask : tf.Tensor
            The atom mask.
        mode : tf_estimator.ModeKeys

        collections : List[str]
            Additional collections to place the variables.

        Returns
        -------
        x : tf.Tensor
            Dynamically normalized input tensor.

        """
        with tf.name_scope("MinMax"):
            _collections = [
                tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES
            ]
            if collections is not None:
                _collections += collections
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
                        tf.boolean_mask(x, mask), axis=0),
                    xmax.shape, name='xmin')
                xlo_op = tf.assign(xlo, tf.minimum(xmin, xlo))
                xhi_op = tf.assign(xhi, tf.maximum(xmax, xhi))
                update_ops = [xlo_op, xhi_op]
            else:
                update_ops = []
            with tf.control_dependencies(update_ops):
                return tf.div_no_nan(xhi - x, xhi - xlo, name='x')

    @staticmethod
    def _add_electron_temperature(x: tf.Tensor,
                                  etemperature: tf.Tensor,
                                  element: str,
                                  mode: tf_estimator.ModeKeys,
                                  max_occurs: Counter):
        """
        Add electron temperature to the atomic descriptor tensor for element.
        """
        with tf.name_scope("Temperature"):
            if mode == tf_estimator.ModeKeys.PREDICT:
                d0 = 1
                d1 = max_occurs[element]
            else:
                d0, d1 = x.shape.as_list()[0: 2]
            etemp = tf.reshape(
                etemperature, [1, d0], name='etemp/0')
            etemp = tf.transpose(etemp, name='T')
            etemp = tf.tile(etemp, [d1, 1], name='tiled')
            etemp = tf.reshape(
                etemp, [d0, d1, 1], name='etemp')
            x = tf.concat((x, etemp), axis=2, name='x')
        return x

    def _get_eentropy_outputs(self,
                              h: tf.Tensor,
                              element: str,
                              collections: List[str],
                              verbose=True):
        """
        Model electron entropy S using the given features 'h'.

        Parameters
        ----------
        h : tf.Tensor
            Input features.
        element : str
            The target element.
        collections : List[str]
            A list of str as the collections where the variables should be
            added.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        with tf.variable_scope("S"):
            eentropy = convolution1x1(
                h,
                activation_fn=get_activation_fn(self._activation),
                hidden_sizes=self._hidden_sizes[element],
                num_out=1,
                l2_weight=1.0,
                collections=collections,
                output_bias=False,
                output_bias_mean=0.0,
                use_resnet_dt=self._use_resnet_dt,
                kernel_initializer=self._kernel_initializer,
                variable_scope=None,
                verbose=verbose)
            eentropy = tf.squeeze(eentropy, axis=2, name="atomic")
            return eentropy

    def _get_internal_energy_outputs(self,
                                     h: tf.Tensor,
                                     element: str,
                                     atomic_static_energy: float,
                                     collections: List[str],
                                     verbose=True):
        """
        Model internal energy U using the given features 'h'.

        Parameters
        ----------
        h : tf.Tensor
            Input features.
        element : str
            The target element.
        atomic_static_energy : float
            Atomic static energy, used as the bias unit of the output layer.
        collections : List[str]
            A list of str as the collections where the variables should be
            added.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        with tf.variable_scope("U"):
            energy = convolution1x1(
                h,
                activation_fn=get_activation_fn(self._activation),
                hidden_sizes=self._hidden_sizes[element],
                num_out=1,
                l2_weight=1.0,
                collections=collections,
                output_bias=True,
                output_bias_mean=atomic_static_energy,
                use_resnet_dt=self._use_resnet_dt,
                kernel_initializer=self._kernel_initializer,
                variable_scope=None,
                verbose=verbose)
            energy = tf.squeeze(energy, axis=2, name="atomic")
            return energy

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
        activation_fn = get_activation_fn(self._activation)

        with tf.variable_scope("ANN"):

            outputs = {'energy': [], 'eentropy': [], 'free_energy': []}
            atomic_descriptors = self._get_atomic_descriptors(
                universal_descriptors=descriptors,
                mode=mode)
            for element, x in atomic_descriptors.descriptors.items():
                with tf.variable_scope(element, reuse=tf.AUTO_REUSE):
                    if self._use_atomic_static_energy:
                        bias_mean = self._atomic_static_energy.get(element, 0.0)
                    else:
                        bias_mean = 0.0

                    with tf.variable_scope(element, reuse=tf.AUTO_REUSE):
                        if verbose:
                            log_tensor(x)
                        if self._minmax_scale:
                            x = self._apply_minmax_normalization(
                                x=x,
                                mask=descriptors['atom_masks'][element],
                                mode=mode,
                                collections=collections)
                            if verbose:
                                log_tensor(x)
                        if self._temperature_dependent:
                            x = self._add_electron_temperature(
                                x=x,
                                etemperature=features["etemperature"],
                                element=element,
                                mode=mode,
                                max_occurs=atomic_descriptors.max_occurs)
                            if verbose:
                                log_tensor(x)
                            t = tf.reshape(
                                features["etemperature"], (-1, 1), name='T')
                            etemp_fn = get_activation_fn(
                                self._temperature_activation)
                            h = convolution1x1(
                                x,
                                activation_fn=etemp_fn,
                                hidden_sizes=self._temperature_layers[:-1],
                                num_out=self._temperature_layers[-1],
                                l2_weight=1.0,
                                collections=collections,
                                kernel_initializer=self._kernel_initializer,
                                output_bias=True,
                                output_bias_mean=0.0,
                                use_resnet_dt=self._use_resnet_dt,
                                variable_scope="T",
                                verbose=verbose)
                            s = self._get_eentropy_outputs(
                                h,
                                element=element,
                                collections=collections,
                                verbose=verbose)
                            u = self._get_internal_energy_outputs(
                                h,
                                element=element,
                                atomic_static_energy=bias_mean,
                                collections=collections,
                                verbose=verbose)
                            ts = tf.multiply(t, s, name='TS')
                            y = tf.subtract(u, ts, name="F")
                            outputs['energy'].append(u)
                            outputs['eentropy'].append(s)
                            outputs['free_energy'].append(y)
                        else:
                            y = convolution1x1(
                                x,
                                activation_fn=activation_fn,
                                hidden_sizes=self._hidden_sizes[element],
                                num_out=1,
                                l2_weight=1.0,
                                collections=collections,
                                output_bias=self._use_atomic_static_energy,
                                output_bias_mean=bias_mean,
                                fixed_output_bias=self._fixed_atomci_static_energy,
                                use_resnet_dt=self._use_resnet_dt,
                                kernel_initializer=self._kernel_initializer,
                                variable_scope=None,
                                verbose=verbose)
                            y = tf.squeeze(y, axis=2, name="atomic")
                            if verbose:
                                log_tensor(y)
                            outputs['free_energy'].append(y)
            return outputs

    def _get_energy_ops(self, outputs, features, verbose=True) -> EnergyOps:
        """
        Return the Op to compute internal energy E.

        Parameters
        ----------
        outputs : Dict[str, [tf.Tensor]]
            A list of `tf.Tensor` as the outputs of the ANNs.
        features : dict
            A dict of input tensors.
        name : str
            The name of the output tensor.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        ops : EnergyOps
            The energy tensors.

        """
        ndims = features["atom_masks"].shape.ndims
        axis = ndims - 1
        with tf.name_scope("Mask"):
            mask = tf.split(
                features["atom_masks"], [1, -1], axis=axis, name='split')[1]

        def _apply_mask(name: str):
            name_map = {
                'energy': 'U',
                'eentropy': 'S',
                'free_energy': 'E'
            }
            with tf.name_scope(name_map[name]):
                y_atomic = tf.concat(outputs[name], axis=1, name='atomic/raw')
                if ndims == 1:
                    y_atomic = tf.squeeze(y_atomic, axis=0)
                y_atomic = tf.multiply(y_atomic, mask, name='atomic')
            y_sum = tf.reduce_sum(
                y_atomic, axis=axis, keepdims=False, name=name)
            return y_sum, y_atomic

        if self._temperature_dependent:
            free_energy, atomic_energy = _apply_mask('free_energy')
            eentropy = _apply_mask('eentropy')[0]
            energy = _apply_mask('energy')[0]
        else:
            eentropy = tf.no_op("eentropy")
            energy, atomic_energy = _apply_mask('energy')
            free_energy = energy

        enthalpy = self._get_enthalpy_op(features, free_energy, verbose=verbose)
        if verbose:
            if self._temperature_dependent:
                log_tensor(free_energy)
                log_tensor(eentropy)
            log_tensor(energy)
        return EnergyOps(energy, eentropy, enthalpy, free_energy, atomic_energy)

    def _get_energy_loss(self,
                         predictions,
                         labels,
                         n_atoms,
                         loss_parameters: LossParameters,
                         collections) -> Dict[str, tf.Tensor]:
        """
        The energy loss or energy losses if temperature effect is considered.
        """
        if not self._temperature_dependent:
            return super(AtomicNN, self)._get_energy_loss(
                predictions, labels, n_atoms, loss_parameters, collections)

        losses = {}

        with tf.name_scope("Energy"):
            for scope_name, prop in {"U": "energy",
                                     "F": "free_energy",
                                     "S": "eentropy"}.items():
                if prop in self._minimize_properties:
                    loss = loss_ops.get_energy_loss(
                        labels=labels[prop],
                        predictions=predictions[prop],
                        n_atoms=n_atoms,
                        loss_weight=loss_parameters.energy.weight,
                        per_atom_loss=loss_parameters.energy.per_atom_loss,
                        method=LossMethod[loss_parameters.energy.method],
                        collections=collections,
                        name_scope=scope_name)
                    losses[prop] = loss
        return losses
