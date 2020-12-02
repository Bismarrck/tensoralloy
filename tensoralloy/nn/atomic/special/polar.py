#!coding=utf-8
"""
The special potential for predicting polarizability.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from dataclasses import dataclass
from tensorflow_estimator import estimator as tf_estimator
from typing import Union, List, Dict

from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.atomic import AtomicNN
from tensoralloy.nn.atomic.atomic import Descriptor
from tensoralloy.nn.dataclasses import EnergyOps, EnergyOp, LossParameters
from tensoralloy.nn.dataclasses import EnergyLossOptions
from tensoralloy.nn.utils import log_tensor, get_activation_fn
from tensoralloy.nn import losses as loss_ops
from tensoralloy.utils import GraphKeys

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@dataclass
class PolarOps(EnergyOps):
    """
    The special
    """
    polar: EnergyOp

    def as_dict(self):
        adict = super(PolarOps, self).as_dict()
        adict['polar'] = self.polar.total
        adict['polar/atom'] = self.polar.atomic
        return adict


class PolarNN(AtomicNN):
    """
    A special model for simultaneously predicting energy, forces and polar
    tensor.
    """

    default_collection = GraphKeys.ATOMIC_NN_VARIABLES
    scope = "Polar"

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
                 inner_layers=(128, 128),
                 polar_loss_weight=1.0):
        """
        Initialization method.
        """
        super(PolarNN, self).__init__(
            elements=elements, descriptor=descriptor, hidden_sizes=hidden_sizes,
            activation=activation, kernel_initializer=kernel_initializer,
            minmax_scale=minmax_scale, use_resnet_dt=use_resnet_dt,
            atomic_static_energy=atomic_static_energy,
            fixed_atomic_static_energy=fixed_atomic_static_energy,
            use_atomic_static_energy=use_atomic_static_energy,
            minimize_properties=minimize_properties,
            export_properties=export_properties
        )
        self._inner_layers = inner_layers
        self._polar_loss_weight = polar_loss_weight

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        d = super(PolarNN, self).as_dict()
        d['inner_layers'] = self._inner_layers
        d['polar_loss_weight'] = self._polar_loss_weight
        return d

    def get_polar_outputs(self, features, element, collections, verbose=False):
        with tf.variable_scope("Polar", reuse=tf.AUTO_REUSE):
            polar = convolution1x1(
                features,
                activation_fn=get_activation_fn(self._activation),
                hidden_sizes=self._hidden_sizes[element],
                num_out=6,
                l2_weight=1.0,
                collections=collections,
                output_bias=False,
                use_resnet_dt=self._use_resnet_dt,
                kernel_initializer=self._kernel_initializer,
                variable_scope=None,
                verbose=verbose)
            mu = self._create_variable(
                name="mu", shape=(6, ),
                init_val=(305.823, 306.062, 306.123, -4.652, -4.839, -4.691),
                monitoring=0, trainable=False)
            sigma = self._create_variable(
                name="sigma", shape=(6, ),
                init_val=(6.330, 6.488, 6.583, 16.371, 16.606, 16.727),
                monitoring=0, trainable=False)
            return tf.add(polar * sigma, mu, name="polar/ab")

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

        with tf.variable_scope(self.scope):
            outputs = {'energy': [], 'polar': []}
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
                    output_bias = self._fixed_atomic_static_energy

                    H = convolution1x1(
                        x,
                        activation_fn=activation_fn,
                        hidden_sizes=self._inner_layers[:-1],
                        num_out=self._inner_layers[-1],
                        l2_weight=1.0,
                        collections=collections,
                        kernel_initializer=self._kernel_initializer,
                        output_bias=True,
                        output_bias_mean=0.0,
                        use_resnet_dt=self._use_resnet_dt,
                        variable_scope="H",
                        verbose=verbose)
                    polar = self.get_polar_outputs(
                        H, element, collections, verbose=verbose)
                    y = convolution1x1(
                        H,
                        activation_fn=activation_fn,
                        hidden_sizes=self._hidden_sizes[element],
                        num_out=1,
                        l2_weight=1.0,
                        collections=collections,
                        output_bias=self._use_atomic_static_energy,
                        output_bias_mean=bias_mean,
                        fixed_output_bias=output_bias,
                        use_resnet_dt=self._use_resnet_dt,
                        kernel_initializer=self._kernel_initializer,
                        variable_scope=None,
                        verbose=verbose)
                    y = tf.squeeze(y, axis=2, name="atomic")
                    outputs['energy'].append(y)
                    outputs['polar'].append(polar)
            return outputs

    def _get_energy_ops(self, outputs, features, verbose=True) -> PolarOps:
        """
        Return the Op to compute internal energy E.

        Parameters
        ----------
        outputs : Dict[str, [tf.Tensor]]
            A list of `tf.Tensor` as the outputs of the ANNs.
        features : dict
            A dict of input tensors.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        ops : PolarOps
            The energy tensors.

        """
        ndims = features["atom_masks"].shape.ndims
        axis = ndims - 1
        with tf.name_scope("Mask"):
            mask = tf.split(
                features["atom_masks"], [1, -1], axis=axis, name='split')[1]
            energies = tf.concat(outputs["energy"], axis=1, name='atomic/raw')
            if ndims == 1:
                energies = tf.squeeze(energies, axis=0)
            energies = tf.multiply(energies, mask, name='atomic')

            polars = tf.concat(outputs["polar"], axis=1, name='polar/raw')
            if ndims == 1:
                polars = tf.squeeze(polars, axis=0)
            polars = tf.multiply(polars, mask[..., tf.newaxis],
                                 name='atomic/polar')

        energy = tf.reduce_sum(
            energies, axis=axis, keepdims=False, name='energy')
        polar = tf.reduce_sum(polars, axis=axis, keepdims=False, name='polar')
        if verbose:
            log_tensor(energy)
            log_tensor(polar)
        return PolarOps(energy=EnergyOp(energy, energies),
                        polar=EnergyOp(polar, polars))

    def _get_energy_loss(self,
                         predictions,
                         labels,
                         n_atoms,
                         max_train_steps,
                         loss_parameters: LossParameters,
                         collections) -> Dict[str, tf.Tensor]:
        """
        Return the energy and polar loss Ops.
        """
        losses = {}
        with tf.name_scope("Energy"):
            for scope_name, prop in {"U": "energy",
                                     "P": "polar"}.items():
                if prop in self._minimize_properties:
                    if prop == 'energy':
                        options = loss_parameters.energy
                    else:
                        options = EnergyLossOptions(
                            weight=self._polar_loss_weight)
                    loss = loss_ops.get_energy_loss(
                        labels=labels[prop],
                        predictions=predictions[prop],
                        n_atoms=n_atoms,
                        max_train_steps=max_train_steps,
                        options=options,
                        collections=collections,
                        name_scope=scope_name)
                    losses[prop] = loss
        return losses

    # def _get_eval_energy_metrics(self, labels, predictions, n_atoms):
    #     name_map = {
    #         'energy': 'U',
    #         'polar': 'P'
    #     }
    #     metrics = {}
    #     for prop, desc in name_map.items():
    #         with tf.name_scope(prop.capitalize()):
    #             if prop in self._minimize_properties:
    #                 x = labels[prop]
    #                 y = predictions[prop]
    #                 if prop == 'polar':
    #                     xn = x / n_atoms[..., tf.newaxis]
    #                     yn = y / n_atoms[..., tf.newaxis]
    #                 else:
    #                     xn = x / n_atoms
    #                     yn = y / n_atoms
    #                 ops_dict = {
    #                     f'{desc}/mae': tf.metrics.mean_absolute_error(x, y),
    #                     f'{desc}/mse': tf.metrics.mean_squared_error(x, y),
    #                     f'{desc}/mae/atom': tf.metrics.mean_absolute_error(
    #                         xn, yn)}
    #                 metrics.update(ops_dict)
    #     return metrics

    def _get_eval_energy_metrics(self, labels, predictions, n_atoms):
        metrics = super(PolarNN, self)._get_eval_energy_metrics(
            labels, predictions, n_atoms)
        if 'polar' in self._minimize_properties:
            voigt_notations = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
            for axis, voigt_notation in enumerate(voigt_notations):
                with tf.name_scope(f"P{voigt_notation}"):
                    x = labels['polar'][:, axis] / n_atoms
                    y = predictions['polar'][:, axis] / n_atoms
                    metrics[f'P{voigt_notation}/mae'] = \
                        tf.metrics.mean_absolute_error(x, y)
        return metrics
