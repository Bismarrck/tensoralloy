# coding=utf-8
"""
This module defines a general atomic neural network framework.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.opt import NadamOptimizer
from misc import Defaults, AttributeDict, safe_select
from typing import List, Dict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_activation_fn(fn_name: str):
    """
    Return the corresponding activation function.
    """
    if fn_name.lower() == 'leaky_relu':
        return tf.nn.leaky_relu
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


def get_learning_rate(global_step, learning_rate=0.001, decay_function=None,
                      decay_rate=0.99, decay_steps=1000, staircase=False):
    """
    Return a `float64` tensor as the learning rate.
    """
    with tf.name_scope("learning_rate"):
        if decay_function is None:
            learning_rate = tf.constant(
                learning_rate, dtype=tf.float64, name="learning_rate")
        else:
            if decay_function == 'exponential':
                decay_fn = tf.train.exponential_decay
            elif decay_function == 'inverse_time':
                decay_fn = tf.train.inverse_time_decay
            elif decay_function == 'natural_exp':
                decay_fn = tf.train.natural_exp_decay
            else:
                raise ValueError(
                    "'{}' is not supported!".format(decay_function))
            learning_rate = decay_fn(learning_rate,
                                     global_step=global_step,
                                     decay_rate=decay_rate,
                                     decay_steps=decay_steps,
                                     staircase=staircase,
                                     name="learning_rate")
        tf.summary.scalar('learning_rate_at_step', learning_rate)
        return learning_rate


def get_optimizer(learning_rate, method='adam', **kwargs):
    """
    Return a `tf.train.Optimizer`.

    Parameters
    ----------
    learning_rate : tf.Tensor or float
        A float tensor as the learning rate.
    method : str
        A str as the name of the optimizer. Supported are: 'adam', 'adadelta',
        'rmsprop' and 'nadam'.
    kwargs : dict
        Additional arguments for the optimizer.

    Returns
    -------
    optimizer : tf.train.Optimizer
        An optimizer.

    """
    with tf.name_scope("SGD"):
        if method.lower() == 'adam':
            return tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=kwargs.get('beta1', 0.9))
        elif method.lower() == 'nadam':
            return NadamOptimizer(
                learning_rate=learning_rate, beta1=kwargs.get('beta1', 0.9))
        elif method.lower() == 'adadelta':
            return tf.train.AdadeltaOptimizer(
                learning_rate=learning_rate, rho=kwargs.get('rho', 0.95))
        elif method.lower() == 'rmsprop':
            return tf.train.RMSPropOptimizer(
                learning_rate=learning_rate, decay=kwargs.get('decay', 0.9),
                momentum=kwargs.get('momentum', 0.0))
        else:
            raise ValueError(
                "Supported SGD optimizers: adam, nadam, adadelta, rmsprop.")


def log_tensor(tensor: tf.Tensor):
    """
    Print the name and shape of the input Tensor.
    """
    dimensions = ",".join(["{:6d}".format(dim if dim is not None else -1)
                           for dim in tensor.get_shape().as_list()])
    print("{:<36s} : [{}]".format(tensor.op.name, dimensions))


class AtomicNN:
    """
    This class represents a general atomic neural network.
    """

    def __init__(self, elements: List[str], hidden_sizes=None,
                 activation=None, l2_weight=0.0, forces=False):
        """
        Initialization method.

        Parameters
        ----------
        elements : List[str]
            A list of str as the ordered elements.
        hidden_sizes : List[int] or Dict[str, List[int]]
            A list of int or a dict of (str, list of int) as the sizes of the
            hidden layers.
        activation : str
            The name of the activation function to use.
        forces : bool
            If True, atomic forces will be derived.
        l2_weight : float
            The weight of the L2 regularization. If zero, L2 will be disabled.

        """
        self._elements = elements
        self._hidden_sizes = self._convert_to_dict(
            safe_select(hidden_sizes, Defaults.hidden_sizes))
        self._activation = safe_select(activation, Defaults.activation)
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

    def build(self, features: AttributeDict, verbose=True):
        """
        Build the atomic neural network.

        Parameters
        ----------
        features : AttributeDict
            A dict of input tensors. 'descriptors' of shape `[batch_size, N, D]`
            and 'positions' of `[batch_size, N, 3]` are required.
        verbose : bool
            If True, the

        Returns
        -------
        predictions : AttributeDict
            A dict of output tensors.

        """
        kernel_initializer = xavier_initializer(
            seed=Defaults.seed, dtype=tf.float64)
        bias_initializer = tf.zeros_initializer(dtype=tf.float64)
        activation_fn = get_activation_fn(self._activation)

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
                                activation=activation_fn,
                                use_bias=True,
                                reuse=tf.AUTO_REUSE,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                name='1x1Conv{}'.format(i + 1))
                            if verbose:
                                log_tensor(x)
                    yi = tf.layers.conv1d(inputs=x, filters=1, kernel_size=1,
                                          strides=1, use_bias=False,
                                          kernel_initializer=kernel_initializer,
                                          reuse=tf.AUTO_REUSE,
                                          name='Output')
                    yi = tf.squeeze(yi, axis=2, name='y_atomic')
                    if verbose:
                        log_tensor(yi)
                    outputs.append(yi)

        with tf.name_scope("Output"):
            with tf.name_scope("Energy"):
                y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
                with tf.name_scope("mask"):
                    mask = tf.split(
                        features.mask, [1, -1], axis=1, name='split')[1]
                    y_mask = tf.multiply(y_atomic, mask, name='mask')
                y = tf.reduce_sum(y_mask, axis=1, keepdims=False, name='y')
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
            l2 = tf.multiply(l2_loss, weight, name='l2')
            tf.summary.scalar(l2.op.name + '/summary', l2)
            return l2

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
                tf.summary.scalar(y_loss.op.name + '/summary', y_loss)
                losses.append(y_loss)

            if self._forces:
                with tf.name_scope("forces"):
                    mse = tf.reduce_mean(
                        tf.squared_difference(labels.f, predictions.f),
                        name='mse')
                    f_loss = tf.sqrt(mse, name='f_rmse')
                    tf.summary.scalar(f_loss.op.name + '/summary', f_loss)
                    losses.append(f_loss)

            if self._l2_weight > 0.0:
                losses.append(self.add_l2_penalty())

            return tf.add_n(losses, name='loss')

    @staticmethod
    def add_grads_summary(grads_and_vars, collection: str):
        """
        Add summary of the gradients.
        """
        list_of_ops = []

        for grad, var in grads_and_vars:
            if grad is not None:
                norm = tf.norm(grad, name=var.op.name + "/norm")
                tf.add_to_collection(collection, norm)
                with tf.name_scope("gradients/{}/".format(collection)):
                    list_of_ops.append(
                        tf.summary.histogram(var.op.name + "/hist", grad))
                with tf.name_scope("gradients/{}/".format(collection)):
                    list_of_ops.append(
                        tf.summary.scalar(var.op.name + "/norm", norm))

        with tf.name_scope("total_norm/"):
            total_norm = tf.add_n(tf.get_collection(collection))
            list_of_ops.append(tf.summary.scalar(collection, total_norm))
        return list_of_ops

    def get_train_op(self, total_loss, hparams: AttributeDict):
        """
        Return the Op for a training step.
        """
        with tf.name_scope("Optimize"):
            global_step = tf.train.get_or_create_global_step()
            learning_rate = get_learning_rate(
                global_step,
                learning_rate=hparams.learning_rate,
                decay_function=hparams.decay_function,
                decay_rate=hparams.decay_rate,
                decay_steps=hparams.decay_steps,
                staircase=hparams.staircase
            )
            optimizer = get_optimizer(learning_rate, hparams.method)
            grads_and_vars = optimizer.compute_gradients(total_loss)
            apply_gradients_op = optimizer.apply_gradients(
                grads_and_vars, global_step)

            self.add_grads_summary(grads_and_vars, 'joint')

        with tf.name_scope("Average"):
            variable_averages = tf.train.ExponentialMovingAverage(
                Defaults.variable_moving_average_decay, global_step)
            variables_averages_op = variable_averages.apply(
                tf.trainable_variables())

        return tf.group(apply_gradients_op, variables_averages_op)

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
                 mode: tf.estimator.ModeKeys, params: AttributeDict):
        """
        Initialize a model function for `tf.estimator.Estimator`.

        Parameters
        ----------
        features : AttributeDict
            A dict of input tensors with three keys:
                * 'descriptors' of shape `[batch_size, N, D]`
                * 'positions' of `[batch_size, N, 3]`.
                * 'mask' of shape `[batch_size, N]`.
        labels : AttributeDict
            A dict of reference tensors.
                * 'y' of shape `[batch_size, ]` is required.
                * 'f' of shape `[batch_size, N, 3]` is also required if
                  `self.forces == True`.
        mode : tf.estimator.ModeKeys
            A `ModeKeys`. Specifies if this is training, evaluation or
            prediction.
        params : AttributeDict
            Hyperparameters for building models.

        Returns
        -------
        spec : tf.estimator.EstimatorSpec
            Ops and objects returned from a `model_fn` and passed to an
            `Estimator`. `EstimatorSpec` fully defines the model to be run
            by an `Estimator`.

        """
        predictions = self.build(features)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        total_loss = self.get_total_loss(predictions, labels)
        train_op = self.get_train_op(total_loss, hparams=params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                              train_op=train_op)

        eval_metrics_ops = self.get_eval_metrics_ops(predictions, labels)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          eval_metric_ops=eval_metrics_ops)
