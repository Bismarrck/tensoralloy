# coding=utf-8
"""
This module defines a general atomic neural network framework.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.opt import NadamOptimizer
from misc import Defaults, AttributeDict, safe_select
from hooks import ExamplesPerSecondHook, LoggingTensorHook
from typing import List, Dict
from os.path import join

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class GraphKeys:
    """
    Standard names for variable collections.
    """

    # Summary keys
    TRAIN_SUMMARY = 'train_summary'
    EVAL_SUMMARY = 'eval_summary'

    # Variable keys
    NORMALIZE_VARIABLES = 'normalize_vars'
    Y_STATIC_VARIABLES = 'y_static_vars'

    # Metrics Keys
    TRAIN_METRICS = 'train_metrics'
    EVAL_METRICS = 'eval_metrics'


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
        tf.summary.scalar('learning_rate_at_step', learning_rate,
                          collections=[GraphKeys.TRAIN_SUMMARY, ])
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
    tf.logging.info("{:<36s} : [{}]".format(tensor.op.name, dimensions))


def msra_initializer(dtype=tf.float64, seed=Defaults.seed):
    """
    Return the so-called `MSRA` initializer.

    See Also
    --------
    [Delving Deep into Rectifiers](http://arxiv.org/pdf/1502.01852v1.pdf)

    """
    return variance_scaling_initializer(dtype=dtype, seed=seed)


class _InputNormalizer:
    """
    A collection of funcitons for normalizing input descriptors to [0, 1].
    """

    def __init__(self, method='linear'):
        """
        Initialization method.
        """
        assert method in ('linear', 'arctan', '', None)
        if not method:
            self.method = None
            self.scope = 'NoNorm'
        else:
            self.method = method
            self.scope = '{}Norm'.format(method.capitalize())

    def __call__(self, x: tf.Tensor, values=None):
        """
        Apply the normalization.
        """
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if not self.method:
                return tf.identity(x, name='identity')

            if values is None:
                values = np.ones(x.shape[2], dtype=np.float64)
            alpha = tf.get_variable(
                name='alpha',
                shape=x.shape[2],
                dtype=x.dtype,
                initializer=tf.constant_initializer(values, dtype=x.dtype),
                collections=[tf.GraphKeys.TRAINABLE_VARIABLES,
                             tf.GraphKeys.GLOBAL_VARIABLES,
                             GraphKeys.NORMALIZE_VARIABLES],
                trainable=True)
            tf.summary.histogram(
                name=alpha.op.name + '/summary',
                values=alpha,
                collections=[GraphKeys.TRAIN_SUMMARY])
            x = tf.multiply(x, alpha, name='ax')
            if self.method == 'linear':
                x = tf.identity(x, name='x')
            elif self.method == 'arctan':
                x = tf.atan(x, name='x')
            else:
                raise ValueError(
                    f"Unsupported normalization method: {self.method}")
            tf.summary.histogram(
                name=x.op.name + '/summary',
                values=x,
                collections=[GraphKeys.TRAIN_SUMMARY])
            return x


class AtomicNN:
    """
    This class represents a general atomic neural network.
    """

    def __init__(self, elements: List[str], hidden_sizes=None,
                 activation=None, l2_weight=0.0, forces=False, stress=False,
                 normalizer='linear', normalization_weights=None):
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
            If True, atomic forces will be calculated and trained..
        stress : bool
            If True, the stress tensor will be calculated and trained.
        l2_weight : float
            The weight of the L2 regularization. If zero, L2 will be disabled.
        normalizer : str
            The normalization method. Defaults to 'linear'. Set this to None to
            disable normalization.
        normalization_weights : Dict[str, array_like]
            The initial weights for column-wise normalizing the input atomic
            descriptors.

        """
        self._elements = elements
        self._hidden_sizes = self._convert_to_dict(
            safe_select(hidden_sizes, Defaults.hidden_sizes))
        self._activation = safe_select(activation, Defaults.activation)
        self._forces = forces
        self._stress = stress
        self._l2_weight = max(l2_weight, 0.0)
        self._initial_normalizer_weights = normalization_weights
        self._normalizer = _InputNormalizer(method=normalizer)

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
        Return True if atomic forces will be calculated.
        """
        return self._forces

    @property
    def stress(self):
        """
        Return True if the stress tensor will be calculated.
        """
        return self._stress

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

    def _build_atomic_nn(self, features: AttributeDict, verbose=False):
        """
        Build 1x1 Convolution1D based atomic neural networks for all elements.
        """
        with tf.variable_scope("ANN"):
            kernel_initializer = xavier_initializer(
                seed=Defaults.seed, dtype=tf.float64)
            bias_initializer = tf.zeros_initializer(dtype=tf.float64)
            activation_fn = get_activation_fn(self._activation)
            outputs = []
            for i, element in enumerate(self._elements):
                with tf.variable_scope(element):
                    x = tf.identity(features.descriptors[element], name='input')
                    x = self._normalizer(
                        x, self._initial_normalizer_weights[element])
                    hidden_sizes = self._hidden_sizes[element]
                    if verbose:
                        log_tensor(x)
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
            return outputs

    def _get_energy(self, outputs, features, verbose=True):
        """
        Return the Op to compute total energy.

        Parameters
        ----------
        outputs : List[tf.Tensor]
            A list of `tf.Tensor` as the outputs of the ANNs.
        features : AttributeDict
            A dict of input features.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        energy : tf.Tensor
            The total energy tensor.

        """
        with tf.name_scope("Energy"):
            y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
            with tf.name_scope("mask"):
                mask = tf.split(
                    features.mask, [1, -1], axis=1, name='split')[1]
                y_mask = tf.multiply(y_atomic, mask, name='mask')
            energy = tf.reduce_sum(
                y_mask, axis=1, keepdims=False, name='energy')
            if verbose:
                log_tensor(energy)
            return energy

    @staticmethod
    def _get_forces(energy, positions, verbose=True):
        """
        Return the Op to compute atomic forces (eV / Angstrom).
        """
        with tf.name_scope("Forces"):
            dEdR = tf.gradients(energy, positions, name='dEdR')[0]
            # Please remember: f = -dE/dR
            forces = tf.negative(
                tf.split(dEdR, [1, -1], axis=1, name='split')[1],
                name='forces')
            if verbose:
                log_tensor(forces)
            return forces

    @staticmethod
    def _get_stress(energy, cells, verbose=True):
        """
        Return the Op to compute reduced stress tensor (eV).
        """
        with tf.name_scope("Stress"):
            dEdC = tf.gradients(energy, cells, name='dEdC')[0]
            stress = tf.negative(tf.einsum('ijk,ikl->ijl', dEdC, cells),
                                 name='stress')
            if verbose:
                log_tensor(stress)
            return stress

    def build(self, features: AttributeDict, verbose=True):
        """
        Build the atomic neural network.

        Parameters
        ----------
        features : AttributeDict
            A dict of input tensors. 'descriptors' of shape `[batch_size, N, D]`
            and 'positions' of `[batch_size, N, 3]` are required.
        verbose : bool
            If True, the prediction tensors will be logged.

        Returns
        -------
        predictions : AttributeDict
            A dict of output tensors.

        """
        outputs = self._build_atomic_nn(features, verbose)

        with tf.name_scope("Output"):

            predictions = AttributeDict()

            predictions.energy = self._get_energy(
                outputs, features, verbose=verbose)

            if self._forces:
                predictions.forces = self._get_forces(
                    predictions.energy, features.positions, verbose=verbose)

            if self._stress:
                predictions.stress = self._get_stress(
                    predictions.energy, features.cells, verbose=verbose)

            return predictions

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
            tf.summary.scalar(l2.op.name + '/summary', l2,
                              collections=[GraphKeys.TRAIN_SUMMARY, ])

            tf.add_to_collection(GraphKeys.TRAIN_METRICS, l2)
            return l2

    def get_total_loss(self, predictions, labels):
        """
        Get the total loss tensor.

        Parameters
        ----------
        predictions : AttributeDict
            A dict of tensors as the predictions.
                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, N, 3]` is required if
                  `self.forces == True`.
                * 'stress' of shape `[batch_size, 3, 3]` is required if
                  `self.stress == True`.
        labels : AttributeDict
            A dict of label tensors as the desired regression targets.
                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, N, 3]` is required if
                  `self.forces == True`.
                * 'stress' of shape `[batch_size, 3, 3]` is required if
                  `self.stress == True`.

        Returns
        -------
        loss : tf.Tensor
            A `float64` tensor as the total loss.

        """
        with tf.name_scope("Loss"):
            losses = []

            with tf.name_scope("Energy"):
                mse = tf.reduce_mean(
                    tf.squared_difference(labels.energy, predictions.energy),
                    name='mse')
                y_loss = tf.sqrt(mse, name='y_rmse')
                y_mae = tf.reduce_mean(
                    tf.abs(labels.energy - predictions.energy, name='abs'),
                    name='y_mae')
                losses.append(y_loss)

                tf.summary.scalar(y_loss.op.name + '/summary', y_loss,
                                  collections=[GraphKeys.TRAIN_SUMMARY, ])
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, y_mae)
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, y_loss)

            if self._forces:
                with tf.name_scope("Forces"):
                    mse = tf.reduce_mean(
                        tf.squared_difference(
                            labels.forces, predictions.forces),
                        name='mse')
                    # Add a very small 'eps' to the mean squared error to make
                    # sure `mse` is always greater than zero. Otherwise NaN may
                    # occur at `Sqrt_Grad`.
                    with tf.name_scope("safe_sqrt"):
                        eps = tf.constant(1e-14, dtype=tf.float64, name='eps')
                        mse = tf.add(mse, eps)
                    f_loss = tf.sqrt(mse, name='f_rmse')
                    f_mae = tf.reduce_mean(
                        tf.abs(labels.forces - predictions.forces, name='abs'),
                        name='f_mae')
                    losses.append(f_loss)

                    tf.summary.scalar(f_loss.op.name + '/summary', f_loss,
                                      collections=[GraphKeys.TRAIN_SUMMARY, ])
                    tf.add_to_collection(GraphKeys.TRAIN_METRICS, f_mae)
                    tf.add_to_collection(GraphKeys.TRAIN_METRICS, f_loss)

            if self._stress:
                with tf.name_scope("Stress"):
                    mse = tf.reduce_mean(
                        tf.squared_difference(
                            labels.stress, predictions.stress),
                        name='mse')
                    # Add a very small 'eps' to the mean squared error to make
                    # sure `mse` is always greater than zero. Otherwise NaN may
                    # occur at `Sqrt_Grad`.
                    with tf.name_scope("safe_sqrt"):
                        eps = tf.constant(1e-14, dtype=tf.float64, name='eps')
                        mse = tf.add(mse, eps)
                    t_loss = tf.sqrt(mse, name='t_rmse')
                    t_mae = tf.reduce_mean(
                        tf.abs(labels.stress - predictions.stress, name='abs'),
                        name='t_mae')
                    losses.append(t_loss)

                    tf.summary.scalar(t_loss.op.name + '/summary', t_loss,
                                      collections=[GraphKeys.TRAIN_SUMMARY, ])
                    tf.add_to_collection(GraphKeys.TRAIN_METRICS, t_mae)
                    tf.add_to_collection(GraphKeys.TRAIN_METRICS, t_loss)

            if self._l2_weight > 0.0:
                losses.append(self.add_l2_penalty())

        return tf.add_n(losses, name='loss')

    @staticmethod
    def add_grads_summary(grads_and_vars):
        """
        Add summary of the gradients.
        """
        list_of_ops = []
        for grad, var in grads_and_vars:
            if grad is not None:
                norm = tf.norm(grad, name=var.op.name + "/norm")
                list_of_ops.append(norm)
                with tf.name_scope("gradients/"):
                    tf.summary.scalar(var.op.name + "/norm", norm,
                                      collections=[GraphKeys.TRAIN_SUMMARY, ])
        with tf.name_scope("gradients/"):
            total_norm = tf.add_n(list_of_ops, name='sum')
            tf.summary.scalar('total', total_norm,
                              collections=[GraphKeys.TRAIN_SUMMARY, ])

    def get_train_op(self, total_loss, hparams: AttributeDict):
        """
        Return the Op for a training step.
        """
        with tf.name_scope("Optimize"):
            global_step = tf.train.get_or_create_global_step()
            learning_rate = get_learning_rate(
                global_step,
                learning_rate=hparams.opt.learning_rate,
                decay_function=hparams.opt.decay_function,
                decay_rate=hparams.opt.decay_rate,
                decay_steps=hparams.opt.decay_steps,
                staircase=hparams.opt.staircase
            )
            optimizer = get_optimizer(learning_rate, hparams.opt.method)
            grads_and_vars = optimizer.compute_gradients(total_loss)
            apply_gradients_op = optimizer.apply_gradients(
                grads_and_vars, global_step)

            self.add_grads_summary(grads_and_vars)

        with tf.name_scope("Average"):
            variable_averages = tf.train.ExponentialMovingAverage(
                Defaults.variable_moving_average_decay, global_step)
            variables_averages_op = variable_averages.apply(
                tf.trainable_variables())

        return tf.group(apply_gradients_op, variables_averages_op)

    @staticmethod
    def get_logging_tensors(key) -> Dict[str, tf.Tensor]:
        """
        Return a dict of logging tensors.
        """
        tensors = {}
        for tensor in tf.get_collection(key):
            tensors[tensor.op.name] = tensor
        return tensors

    def get_training_hooks(self, hparams) -> List[tf.train.SessionRunHook]:
        """
        Return a list of `tf.train.SessionRunHook` objects for training.

        Parameters
        ----------
        hparams : AttributeDict
            Hyper parameters for this function.

        """
        with tf.name_scope("Hooks"):

            with tf.name_scope("Summary"):
                summary_saver_hook = tf.train.SummarySaverHook(
                    save_steps=hparams.train.summary_steps,
                    output_dir=hparams.train.model_dir,
                    summary_op=tf.summary.merge_all(key=GraphKeys.TRAIN_SUMMARY,
                                                    name='merge'))

            with tf.name_scope("Speed"):
                examples_per_sec_hook = ExamplesPerSecondHook(
                    batch_size=hparams.train.batch_size,
                    every_n_steps=hparams.train.log_steps)

            hooks = [summary_saver_hook, examples_per_sec_hook]

            if len(tf.get_collection(GraphKeys.TRAIN_METRICS)) > 0:
                logging_tensor_hook = LoggingTensorHook(
                    tensors=self.get_logging_tensors(GraphKeys.TRAIN_METRICS),
                    every_n_iter=hparams.train.log_steps,
                    at_end=True,
                )
                hooks.append(logging_tensor_hook)

            if hparams.train.profile_steps:
                with tf.name_scope("Profile"):
                    profiler_hook = tf.train.ProfilerHook(
                        save_steps=hparams.train.profile_steps,
                        output_dir=join(hparams.train.model_dir, 'profile'),
                        show_memory=True)
                hooks.append(profiler_hook)

        return hooks

    def get_evaluation_hooks(self, hparams):
        """
        Return a list of `tf.train.SessionRunHook` objects for evaluation.
        """
        hooks = []
        if len(tf.get_collection(GraphKeys.EVAL_METRICS)) > 0:
            with tf.name_scope("Hooks"):
                with tf.name_scope("Accuracy"):
                    logging_tensor_hook = LoggingTensorHook(
                        tensors=self.get_logging_tensors(
                            GraphKeys.EVAL_METRICS),
                        every_n_iter=hparams.train.eval_steps,
                        at_end=True)
                hooks.append(logging_tensor_hook)
        return hooks

    def get_eval_metrics_ops(self, predictions, labels):
        """
        Return a dict of Ops as the evaluation metrics.
        """
        with tf.name_scope("Metrics"):

            metrics = {
                'y_rmse': tf.metrics.root_mean_squared_error(
                    labels.energy, predictions.energy, name='y_rmse'),
                'y_mae': tf.metrics.mean_absolute_error(
                    labels.energy, predictions.energy, name='y_mae')}

            if self._forces:
                metrics.update({
                    'f_rmse': tf.metrics.root_mean_squared_error(
                        labels.forces, predictions.forces, name='f_rmse'),
                    'f_mae': tf.metrics.mean_absolute_error(
                        labels.forces, predictions.forces, name='f_mae')})

            if self._stress:
                metrics.update({
                    't_rmse': tf.metrics.root_mean_squared_error(
                        labels.stress, predictions.stress, name='t_rmse'),
                    't_mae': tf.metrics.mean_absolute_error(
                        labels.stress, predictions.stress, name='t_mae')})

            return metrics

    def _check_keys(self, features: AttributeDict, labels: AttributeDict):
        """
        Check the keys of `features` and `labels`.
        """
        assert 'descriptors' in features
        assert 'positions' in features
        assert 'cells' in features
        assert 'mask' in features
        assert 'n_atoms' in features
        assert 'energy' in labels
        if self._forces:
            assert 'forces' in labels
        if self._stress:
            assert 'stress' in labels

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
                * 'cells' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, N]`.
                * 'n_atoms' of dtype `int64`.
        labels : AttributeDict
            A dict of reference tensors.
                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, N, 3]` is required if
                  `self.forces == True`.
                * 'stress' of shape `[batch_size, 3, 3]` is required if
                  `self.stress == True`.
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
        self._check_keys(features, labels)

        predictions = self.build(features,
                                 verbose=(mode == tf.estimator.ModeKeys.TRAIN))

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        total_loss = self.get_total_loss(predictions, labels)
        train_op = self.get_train_op(total_loss, hparams=params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            training_hooks = self.get_training_hooks(hparams=params)
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                              train_op=train_op,
                                              training_hooks=training_hooks)

        eval_metrics_ops = self.get_eval_metrics_ops(predictions, labels)
        evaluation_hooks = self.get_evaluation_hooks(hparams=params)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=total_loss,
                                          eval_metric_ops=eval_metrics_ops,
                                          evaluation_hooks=evaluation_hooks)


class AtomicResNN(AtomicNN):
    """
    A general atomic residual neural network. The ANNs are used to fit residual
    energies.

    The total energy, `y_total`, is expressed as:

        y_total = y_static + y_res

    where `y_static` is the total atomic static energy and this only depends on
    chemical compositions:

        y_static = sum([atomic_static_energy[e] * count(e) for e in elements])

    where `count(e)` represents the number of element `e`.

    """

    def __init__(self, elements: List[str], hidden_sizes=None, activation=None,
                 l2_weight=0.0, forces=False, stress=False, normalizer='linear',
                 normalization_weights=None, atomic_static_energy=None):
        """
        Initialization method.
        """
        super(AtomicResNN, self).__init__(
            elements=elements, hidden_sizes=hidden_sizes, activation=activation,
            l2_weight=l2_weight, forces=forces, stress=stress,
            normalizer=normalizer, normalization_weights=normalization_weights)
        self._atomic_static_energy = atomic_static_energy

    def _check_keys(self, features: AttributeDict, labels: AttributeDict):
        """
        Check the keys of `features` and `labels`.
        """
        super(AtomicResNN, self)._check_keys(features, labels)
        assert 'composition' in features

    def _get_energy(self, outputs, features, verbose=True):
        """
        Return the Op to compute total energy (eV).
        """
        with tf.name_scope("Energy"):

            with tf.variable_scope("Static", reuse=tf.AUTO_REUSE):
                if self._atomic_static_energy is None:
                    values = np.ones(len(self._elements), dtype=np.float64)
                else:
                    values = np.asarray(
                        [self._atomic_static_energy[e] for e in self._elements],
                        dtype=np.float64)
                initializer = tf.constant_initializer(values, dtype=tf.float64)
                x = tf.identity(features.composition, name='input')
                z = tf.get_variable("weights",
                                    shape=(len(self._elements)),
                                    dtype=tf.float64,
                                    trainable=True,
                                    collections=[
                                        GraphKeys.Y_STATIC_VARIABLES,
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        tf.GraphKeys.GLOBAL_VARIABLES,
                                        GraphKeys.TRAIN_METRICS],
                                    initializer=initializer)
                xz = tf.multiply(x, z, name='xz')
                y_static = tf.reduce_sum(xz, axis=1, keepdims=False,
                                         name='y_static')
                if verbose:
                    log_tensor(y_static)

            with tf.name_scope("Residual"):
                y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
                with tf.name_scope("mask"):
                    mask = tf.split(
                        features.mask, [1, -1], axis=1, name='split')[1]
                    y_mask = tf.multiply(y_atomic, mask, name='mask')
                y_res = tf.reduce_sum(y_mask, axis=1, keepdims=False,
                                      name='residual')

            energy = tf.add(y_static, y_res, name='energy')

            with tf.name_scope("Ratio"):
                ratio = tf.reduce_mean(tf.div(y_static, energy, name='ratio'),
                                       name='avg')
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, ratio)
                tf.summary.scalar(ratio.op.name + '/summary', ratio,
                                  collections=[GraphKeys.TRAIN_SUMMARY, ])

            return energy
