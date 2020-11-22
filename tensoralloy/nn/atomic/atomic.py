#!coding=utf-8
"""
This module defines various atomic neural networks.
"""
from __future__ import print_function, absolute_import
from tensoralloy.precision import get_float_dtype

import tensorflow as tf
import json

from monty.json import MSONable, MontyDecoder
from typing import List, Dict, Union
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.utils import GraphKeys
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.dataclasses import EnergyOps, EnergyOp
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.atomic.dataclasses import AtomicDescriptors

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class Descriptor(MSONable):
    """
    The base class for all atomistic descriptors.
    """

    def __init__(self, elements: List[str]):
        """
        Initialization method.

        Parameters
        ----------
        elements : List[str]
            A list of str as the ordered elements.
        """
        self._elements = elements

    def calculate(self,
                  transformer: UniversalTransformer,
                  universal_descriptors,
                  mode: tf_estimator.ModeKeys,
                  verbose=False) -> AtomicDescriptors:
        """
        Calculate atomic descriptors with the unversal descriptors.
        """
        pass

    @property
    def name(self):
        """ Return the name of this descriptor. """
        raise NotImplementedError


class AtomicNN(BasicNN):
    """
    This class represents a general atomic neural network.
    """

    default_collection = GraphKeys.ATOMIC_NN_VARIABLES
    scope = "Atomic"

    def __init__(self,
                 elements: List[str],
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
                 export_properties=('energy', 'forces')):
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
        self._fixed_atomic_static_energy = fixed_atomic_static_energy

        if isinstance(descriptor, dict):
            descriptor = json.loads(json.dumps(descriptor), cls=MontyDecoder)
        self._descriptor = descriptor

    @property
    def hidden_sizes(self) -> Dict[str, List[int]]:
        """
        Return the sizes of hidden layers for each element.
        """
        return self._hidden_sizes

    @property
    def descriptor(self) -> Descriptor:
        """
        Return the attached atomistic descriptor for this potential.
        """
        return self._descriptor

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
                'fixed_atomic_static_energy': self._fixed_atomic_static_energy,
                'atomic_static_energy': self._atomic_static_energy,
                "minimize_properties": self._minimize_properties,
                "export_properties": self._export_properties,
                "descriptor": self._descriptor.as_dict()}
    
    def _create_variable(self, name, shape, trainable=True, init_val=0.0,
                         monitoring=0):
        """
        Create a variable.
        """
        collections = [
            tf.GraphKeys.GLOBAL_VARIABLES, 
            tf.GraphKeys.MODEL_VARIABLES,
            self.default_collection,
        ]
        if trainable:
            collections.append(tf.GraphKeys.TRAINABLE_VARIABLES)
        if monitoring >= 1:
            collections.append(GraphKeys.EVAL_METRICS)
        if monitoring >= 2:
            collections.append(GraphKeys.TRAIN_METRICS)
        dtype = get_float_dtype()
        init = tf.constant_initializer(init_val, dtype)
        return tf.get_variable(name=name, shape=shape, dtype=dtype, 
                               trainable=trainable, collections=collections,
                               initializer=init, 
                               aggregation=tf.VariableAggregation.MEAN)

    def _apply_minmax_normalization(self,
                                    x: tf.Tensor,
                                    mask: tf.Tensor,
                                    mode: tf_estimator.ModeKeys):
        """
        Apply the min-max normalization to raw symmetry function descriptors.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        mask : tf.Tensor
            The atom mask.
        mode : tf_estimator.ModeKeys


        Returns
        -------
        x : tf.Tensor
            Dynamically normalized input tensor.

        """
        with tf.name_scope("MinMax"):
            shape = [1, 1, x.shape[-1]]
            xlo = self._create_variable("xlo", shape, False, init_val=1000.)
            xhi = self._create_variable("xhi", shape, False, init_val=0.)
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

            outputs = {'energy': []}
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
                    y = convolution1x1(
                        x,
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
                    if verbose:
                        log_tensor(y)
                    outputs['energy'].append(y)
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
            y_atomic = tf.concat(outputs["energy"], axis=1, name='atomic/raw')
            if ndims == 1:
                y_atomic = tf.squeeze(y_atomic, axis=0)
            y_atomic = tf.multiply(y_atomic, mask, name='atomic')
        y_sum = tf.reduce_sum(
            y_atomic, axis=axis, keepdims=False, name='energy')
        enthalpy = self._get_enthalpy_op(features, y_sum, verbose=verbose)
        if verbose:
            log_tensor(y_sum)
        return EnergyOps(energy=EnergyOp(y_sum, y_atomic), enthalpy=enthalpy)
