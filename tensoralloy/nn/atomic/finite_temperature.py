#!coding=utf-8
"""
A special module for modeling finite-temperature systems.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow_estimator import estimator as tf_estimator
from typing import List, Dict, Union
from collections import Counter

from tensoralloy.utils import GraphKeys
from tensoralloy.nn import losses as loss_ops
from tensoralloy.nn.atomic.atomic import AtomicNN, Descriptor
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.dataclasses import EnergyOps, FiniteTemperatureEnergyOps
from tensoralloy.nn.dataclasses import EnergyOp
from tensoralloy.nn.dataclasses import LossParameters
from tensoralloy.nn.atomic.dataclasses import FiniteTemperatureOptions


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TemperatureDependentAtomicNN(AtomicNN):
    """
    Temperature-dependent atomistic neural network potential.
    """

    default_collection = GraphKeys.ATOMIC_NN_VARIABLES
    scope = "TD"

    def __init__(self, elements: List[str],
                 descriptor: Union[Descriptor, dict],
                 hidden_sizes=None,
                 activation=None,
                 kernel_initializer='he_normal',
                 minmax_scale=True,
                 use_resnet_dt=False,
                 atomic_static_energy=None,
                 use_atomic_static_energy=True,
                 fixed_atomic_static_energy=False,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces'),
                 finite_temperature=FiniteTemperatureOptions()):
        """
        Initialization method.
        """
        super(TemperatureDependentAtomicNN, self).__init__(
            elements=elements, descriptor=descriptor, hidden_sizes=hidden_sizes,
            activation=activation, kernel_initializer=kernel_initializer,
            minmax_scale=minmax_scale, use_resnet_dt=use_resnet_dt,
            atomic_static_energy=atomic_static_energy,
            fixed_atomic_static_energy=fixed_atomic_static_energy,
            use_atomic_static_energy=use_atomic_static_energy,
            minimize_properties=minimize_properties,
            export_properties=export_properties
        )

        if isinstance(finite_temperature, FiniteTemperatureOptions):
            self._finite_temperature = finite_temperature
        else:
            assert isinstance(finite_temperature, dict)
            self._finite_temperature = FiniteTemperatureOptions(
                **finite_temperature)

    @property
    def finite_temperature_options(self):
        """
        Return the options for modeling finite temperature systems.
        """
        return self._finite_temperature

    @property
    def is_finite_temperature(self) -> bool:
        """ Override this method. """
        return True

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        d = super(TemperatureDependentAtomicNN, self).as_dict()
        d['finite_temperature'] = self._finite_temperature.__dict__
        return d

    @staticmethod
    def _add_electron_temperature(x: tf.Tensor,
                                  etemperature: tf.Tensor,
                                  element: str,
                                  mode: tf_estimator.ModeKeys,
                                  max_occurs: Counter):
        """
        Add electron temperature to the atomic descriptor tensor `x`.
        """
        with tf.name_scope("Temperature"):
            if mode == tf_estimator.ModeKeys.PREDICT:
                d0 = 1
                d1 = max_occurs[element]
            else:
                d0, d1 = x.shape.as_list()[0: 2]
            etemp = tf.reshape(
                etemperature, [d0, 1, 1], name='etemp')
            etemp = tf.tile(etemp, [1, d1, 1], name='etemp/tiled')
            x = tf.concat((x, etemp), axis=2, name='x')
        return x, etemp

    def _get_electron_entropy(self,
                              h: tf.Tensor,
                              t: tf.Tensor,
                              element: str,
                              collections: List[str],
                              verbose=True):
        """
        Model electron entropy S with the free electron model.

        Parameters
        ----------
        h : tf.Tensor
            Input features.
        t : tf.Tensor
            The electron temperature tensor.
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
                output_bias=True,
                output_bias_mean=0.0,
                use_resnet_dt=self._use_resnet_dt,
                kernel_initializer=self._kernel_initializer,
                variable_scope=None,
                verbose=verbose)
            eentropy = tf.squeeze(eentropy, axis=2, name="atomic")
            if verbose:
                log_tensor(eentropy)
            return eentropy

    def _get_internal_energy_outputs(self,
                                     h: tf.Tensor,
                                     element: str,
                                     atomic_static_energy: float,
                                     collections: List[str],
                                     verbose=True):
        """
        Model internal energy U using the temperature-dependent atomic
        descriptor 'h'.

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
            if verbose:
                log_tensor(energy)
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

        with tf.variable_scope(self.scope):

            outputs = {'energy': [], 'eentropy': [], 'free_energy': []}
            atomic_descriptors = self._descriptor.calculate(
                transformer=self._transformer,
                universal_descriptors=descriptors,
                mode=mode)
            for element, x in atomic_descriptors.descriptors.items():
                with tf.variable_scope(element, reuse=tf.AUTO_REUSE):
                    if self._use_atomic_static_energy:
                        bias_mean = self._atomic_static_energy.get(element, 0.0)
                    else:
                        bias_mean = 0.0
                    if verbose:
                        log_tensor(x)
                    if self._minmax_scale:
                        x = self._apply_minmax_normalization(
                            x=x,
                            mask=descriptors['atom_masks'][element],
                            mode=mode)
                        if verbose:
                            log_tensor(x)
                    activation_fn = get_activation_fn(
                        self._finite_temperature.activation)
                    layers = self._finite_temperature.layers
                    H = convolution1x1(
                        x,
                        activation_fn=activation_fn,
                        hidden_sizes=layers[:-1],
                        num_out=layers[-1],
                        l2_weight=1.0,
                        collections=collections,
                        kernel_initializer=self._kernel_initializer,
                        output_bias=True,
                        output_bias_mean=0.0,
                        use_resnet_dt=self._use_resnet_dt,
                        variable_scope="H",
                        verbose=verbose)
                    Ht, t = self._add_electron_temperature(
                        x=H,
                        etemperature=features["etemperature"],
                        element=element,
                        mode=mode,
                        max_occurs=atomic_descriptors.max_occurs)
                    T = tf.squeeze(t, axis=2, name='T')
                    S = self._get_electron_entropy(
                        h=Ht,
                        t=T,
                        element=element,
                        collections=collections,
                        verbose=verbose)
                    U = self._get_internal_energy_outputs(
                        h=Ht,
                        element=element,
                        atomic_static_energy=bias_mean,
                        collections=collections,
                        verbose=verbose)
                    U = tf.identity(U, name='U')
                    TS = tf.multiply(T, S, name='TS')
                    E = tf.subtract(U, TS, name='F')
                    outputs['energy'].append(U)
                    outputs['eentropy'].append(S)
                    outputs['free_energy'].append(E)
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

        def _build_energy_op(name: str) -> EnergyOp:
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
            return EnergyOp(total=y_sum, atomic=y_atomic)

        free_energy = _build_energy_op('free_energy')
        eentropy = _build_energy_op('eentropy')
        energy = _build_energy_op('energy')
        enthalpy = self._get_enthalpy_op(features, free_energy.total, verbose)
        if verbose:
            log_tensor(free_energy.total)
            log_tensor(eentropy.total)
            log_tensor(energy.total)
        return FiniteTemperatureEnergyOps(
            energy=energy, enthalpy=enthalpy,
            eentropy=eentropy, free_energy=free_energy)

    def _get_energy_loss(self,
                         predictions,
                         labels,
                         n_atoms,
                         max_train_steps,
                         loss_parameters: LossParameters,
                         collections) -> Dict[str, tf.Tensor]:
        """
        The energy loss or energy losses if temperature effect is considered.
        """

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
                        max_train_steps=max_train_steps,
                        options=loss_parameters[prop],
                        collections=collections,
                        name_scope=scope_name)
                    losses[prop] = loss
        return losses
