# coding=utf-8
"""
This module defines a general atomic neural network framework.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from misc import Defaults, AttributeDict, safe_select
from typing import List, Dict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_activation_fn(fn_name: str):
    """
    Return the corresponding activation function.
    """
    if fn_name.lower() == 'leaky_relu':
        return Defaults.activation_fn
    elif fn_name.lower() == 'relu':
        return tf.nn.relu
    elif fn_name.lower() == 'tanh':
        return tf.nn.tanh
    elif fn_name.lower() == 'sigmoid':
        return tf.nn.sigmoid
    elif fn_name.lower() == 'softplus':
        return tf.nn.softplus
    elif fn_name.lower() == 'softsign':
        return tf.nn.softsign
    elif fn_name.lower() == 'softmax':
        return tf.nn.softmax
    raise ValueError("The function '{}' cannot be recognized!".format(fn_name))


class AtomicNN:
    """
    This class represents a general atomic neural network.
    """

    def __init__(self, elements: List[str], hidden_sizes=None,
                 activation_fn=None, l2_weight=0.0, forces=False):
        """
        Initialization method.

        Parameters
        ----------
        elements : List[str]
            A list of str as the ordered elements.
        hidden_sizes : List[int] or Dict[str, List[int]]
            A list of int or a dict of (str, list of int) as the sizes of the
            hidden layers.
        activation_fn : Callable
            The activation function to use.
        forces : bool
            If True, atomic forces will be derived.
        l2_weight : float
            The weight of the L2 regularization. If zero, L2 will be disabled.

        """
        self._elements = elements
        self._hidden_sizes = self._convert_to_dict(
            safe_select(hidden_sizes, Defaults.hidden_sizes))
        self._activation_fn = safe_select(activation_fn, Defaults.activation_fn)
        self._forces = forces
        self._l2_weight = max(l2_weight, 0.0)

    @property
    def elements(self):
        """
        Return the ordered elements.
        """
        return self._elements

    @property
    def hidden_sizes(self) -> Dict[str, List[int]]:
        """
        Return the sizes of hidden layers for each element.
        """
        return self._hidden_sizes

    @property
    def forces(self):
        """
        Return True if forces are derived.
        """
        return self._forces

    @property
    def l2_weight(self):
        """
        Return the weight of the L2 loss.
        """
        return self._l2_weight

    def _convert_to_dict(self, hidden_sizes):
        """
        Convert `hidden_sizes` to a dict if needed.
        """
        if isinstance(hidden_sizes, dict):
            return hidden_sizes
        return {element: hidden_sizes for element in self._elements}

    def build(self, features: AttributeDict):
        """
        Build the atomic neural network.

        Parameters
        ----------
        features : AttributeDict
            A dict of input tensors. 'descriptors' of shape `[batch_size, N, D]`
            and 'positions' of `[batch_size, N, 3]` are required.

        Returns
        -------
        predictions : AttributeDict
            A dict of output tensors.

        """
        kernel_initializer = xavier_initializer(
            seed=Defaults.seed, dtype=tf.float64)
        bias_initializer = tf.zeros_initializer(dtype=tf.float64)

        with tf.variable_scope("ANN"):
            outputs = []
            for i, element in enumerate(self._elements):
                with tf.variable_scope(element):
                    x = tf.identity(features.descriptors[element], name='x')
                    hidden_sizes = self._hidden_sizes[element]
                    for j in range(len(hidden_sizes)):
                        with tf.variable_scope('Hidden{}'.format(j + 1)):
                            x = tf.layers.conv1d(
                                inputs=x, filters=hidden_sizes[j],
                                kernel_size=1, strides=1,
                                activation=self._activation_fn,
                                use_bias=True,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                name='1x1Conv{}'.format(i + 1))
                    yi = tf.layers.conv1d(inputs=x, filters=1, kernel_size=1,
                                          strides=1, use_bias=False,
                                          kernel_initializer=kernel_initializer,
                                          name='Output')
                    yi = tf.squeeze(yi, axis=2, name='ae')
                    outputs.append(yi)

        with tf.name_scope("Output"):
            with tf.name_scope("Energy"):
                y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
                y = tf.reduce_sum(y_atomic, axis=1, keepdims=False, name='y')
            if self._forces:
                with tf.name_scope("Forces"):
                    f = tf.gradients(y, features.positions, name='f')[0]
                return AttributeDict(y=y, f=f)
            else:
                return AttributeDict(y=y)

    def add_l2_penalty(self):
        """
        Build a L2 penalty term.

        Returns
        -------
        l2 : tf.Tensor
            A `float64` tensor as the sum of L2 terms of all trainable kernel
            variables.

        """
        with tf.name_scope("Penalty"):
            for var in tf.trainable_variables():
                if 'bias' in var.op.name:
                    continue
                l2 = tf.nn.l2_loss(var, name=var.op.name + "/l2")
                tf.add_to_collection('l2_losses', l2)
            l2_loss = tf.add_n(tf.get_collection('l2_losses'), name='l2_sum')
            weight = tf.convert_to_tensor(
                self._l2_weight, dtype=tf.float64, name='weight')
            return tf.multiply(l2_loss, weight, name='l2')

    def get_total_loss(self, predictions, labels):
        """
        Get the total loss tensor.

        Parameters
        ----------
        predictions : AttributeDict
            A dict of tensors as the predictions. Valid keys are: 'y' and 'f'.
        labels : AttributeDict
            A dict of label tensors as the desired regression targets.

        Returns
        -------
        loss : tf.Tensor
            A `float64` tensor as the total loss.

        """
        with tf.name_scope("Loss"):
            losses = []

            with tf.name_scope("energy"):
                mse = tf.reduce_mean(
                    tf.squared_difference(labels.y, predictions.y), name='mse')
                y_loss = tf.sqrt(mse, name='y_rmse')
                losses.append(y_loss)

            if self._forces:
                with tf.name_scope("forces"):
                    mse = tf.reduce_mean(
                        tf.squared_difference(labels.f, predictions.f),
                        name='mse')
                    f_loss = tf.sqrt(mse, name='f_rmse')
                    losses.append(f_loss)

            if self._l2_weight > 0.0:
                losses.append(self.add_l2_penalty())

            return tf.add_n(losses, name='loss')

    @staticmethod
    def get_train_op(total_loss):
        """
        Return the Op for a training step.
        """
        with tf.name_scope("Optimization"):
            global_step = tf.train.get_or_create_global_step()
            minimize_op = tf.train.AdamOptimizer(0.001).minimize(
                total_loss, global_step)

        with tf.name_scope("Average"):
            variable_averages = tf.train.ExponentialMovingAverage(
                Defaults.variable_moving_average_decay, global_step)
            variables_averages_op = variable_averages.apply(
                tf.trainable_variables())

        return tf.group(minimize_op, variables_averages_op)

    def get_eval_metrics_ops(self, predictions, labels):
        """
        Return a dict of Ops as the evaluation metrics.
        """
        metrics = {
            'y_rmse': tf.metrics.root_mean_squared_error(
                labels.y, predictions.y, name='y_rmse'),
            'y_mae': tf.metrics.mean_absolute_error(
                labels.y, predictions.y, name='y_mae'),
        }
        if self._forces:
            metrics.update({
                'f_rmse': tf.metrics.root_mean_squared_error(
                    labels.f, predictions.f, name='f_rmse'),
                'f_mae': tf.metrics.mean_absolute_error(
                    labels.f, predictions.f, name='f_mae')
            })
        return metrics

    def model_fn(self, features: AttributeDict, labels: AttributeDict,
                 mode: tf.estimator.ModeKeys):
        """
        Initialize a model function for `tf.estimator.Estimator`.

        Parameters
        ----------
        features : AttributeDict
            A dict of input tensors. 'descriptors' of shape `[batch_size, N, D]`
            and 'positions' of `[batch_size, N, 3]` are required.
        labels : AttributeDict
            A dict of reference tensors. 'y' of length `batch_size` is required.
            If `self.forces`, 'f' of shape `[batch_size, N, 3]` are required.
        mode : tf.estimator.ModeKeys
            A `ModeKeys`. Specifies if this is training, evaluation or
            prediction.

        Returns
        -------
        spec : tf.estimator.EstimatorSpec
            Ops and objects returned from a `model_fn` and passed to an
            `Estimator`. `EstimatorSpec` fully defines the model to be run by an
            `Estimator`.

        """
        predictions = self.build(features)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        total_loss = self.get_total_loss(predictions, labels)
        train_op = self.get_train_op(total_loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                              train_op=train_op)

        eval_metrics_ops = self.get_eval_metrics_ops(predictions, labels)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          eval_metric_ops=eval_metrics_ops)
