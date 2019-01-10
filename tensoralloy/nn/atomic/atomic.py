# coding=utf-8
"""
This module defines various atomic neural networks.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List, Dict

from tensoralloy.nn.atomic.normalizer import InputNormalizer
from tensoralloy.nn.utils import get_activation_fn, log_tensor, GraphKeys
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.misc import AttributeDict, safe_select

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
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces'),
                 positive_energy_mode=False,
                 normalizer=None,
                 normalization_weights=None):
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
            elements=elements,
            hidden_sizes=hidden_sizes,
            activation=activation,
            minimize_properties=minimize_properties,
            export_properties=export_properties,
            positive_energy_mode=positive_energy_mode)

        self._initial_normalizer_weights = \
            safe_select(normalization_weights, {})
        self._normalizer = InputNormalizer(method=normalizer)

    @property
    def hidden_sizes(self) -> Dict[str, List[int]]:
        """
        Return the sizes of hidden layers for each element.
        """
        return self._hidden_sizes

    def _build_nn(self, features: AttributeDict, mode: tf.estimator.ModeKeys,
                  verbose=False):
        """
        Build 1x1 Convolution1D based atomic neural networks for all elements.

        Parameters
        ----------
        features : AttributeDict
            A dict of input tensors:
                * 'descriptors', a dict of (element, (value, mask)) where
                  `element` represents the symbol of an element, `value` is the
                  descriptors of `element` and `mask` is None.
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cells' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        mode : tf.estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        collections = [self.default_collection]

        with tf.variable_scope("ANN"):
            activation_fn = get_activation_fn(self._activation)
            outputs = []
            for element, (value, _) in features.descriptors.items():
                with tf.variable_scope(element):
                    x = tf.identity(value, name='input')
                    if mode == tf.estimator.ModeKeys.PREDICT:
                        assert x.shape.ndims == 2
                        x = tf.expand_dims(x, axis=0, name='2to3')
                    if self._normalizer.enabled:
                        x = self._normalizer(
                            x, self._initial_normalizer_weights.get(element),
                            collections=collections)
                    hidden_sizes = self._hidden_sizes[element]
                    if verbose:
                        log_tensor(x)
                    yi = convolution1x1(
                        x,
                        activation_fn=activation_fn,
                        hidden_sizes=hidden_sizes,
                        l2_weight=1.0,
                        collections=collections,
                        variable_scope=None,
                        verbose=verbose)
                    yi = tf.squeeze(yi, axis=2, name='atomic')
                    if verbose:
                        log_tensor(yi)
                    outputs.append(yi)
            return outputs

    def _get_energy_op(self, outputs, features, name='energy', verbose=True):
        """
        Return the Op to compute total energy.

        Parameters
        ----------
        outputs : List[tf.Tensor]
            A list of `tf.Tensor` as the outputs of the ANNs.
        features : AttributeDict
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
        ndims = features.mask.shape.ndims
        axis = ndims - 1
        with tf.name_scope("mask"):
            if ndims == 1:
                y_atomic = tf.squeeze(y_atomic, axis=0)
            mask = tf.split(
                features.mask, [1, -1], axis=axis, name='split')[1]
            y_mask = tf.multiply(y_atomic, mask, name='mask')
        energy = tf.reduce_sum(
            y_mask, axis=axis, keepdims=False, name=name)
        if verbose:
            log_tensor(energy)
        return energy
