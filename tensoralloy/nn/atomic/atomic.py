# coding=utf-8
"""
This module defines various atomic neural networks.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import List, Dict

from tensoralloy.nn.atomic.normalizer import InputNormalizer
from tensoralloy.nn.utils import get_activation_fn, log_tensor
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
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        with tf.variable_scope("ANN"):
            activation_fn = get_activation_fn(self._activation)
            outputs = []
            for element, (value, _) in features.descriptors.items():
                with tf.variable_scope(element):
                    x = tf.identity(value, name='input')
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
            A dict of input tensors.
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
