# coding=utf-8
"""
This module defines the basic neural network for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from os.path import join
from typing import List, Dict

from tensoralloy.nn.utils import sum_of_grads_and_vars_collections
from tensoralloy.nn.hooks import ExamplesPerSecondHook, LoggingTensorHook
from misc import safe_select, Defaults, AttributeDict
from nn import log_tensor, GraphKeys, get_learning_rate, get_optimizer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class BasicNN:
    """
    The base neural network class.
    """

    def __init__(self, elements: List[str], hidden_sizes=None, activation=None,
                 forces=False, stress=False, total_pressure=False, l2_weight=0):
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
            If True, the reduced stress tensor will be calculated and trained.
        total_pressure : bool
            If True, the reduced total pressure will be calculated and trained.
            This option will suppress `stress`.
        l2_weight : float
            The weight of the L2 regularization. If zero, L2 will be disabled.

        """
        self._elements = elements
        self._hidden_sizes = self._convert_to_dict(
            safe_select(hidden_sizes, Defaults.hidden_sizes))
        self._activation = safe_select(activation, Defaults.activation)
        self._forces = forces
        self._stress = stress
        self._total_pressure = total_pressure
        self._l2_weight = max(l2_weight, 0.0)

    @property
    def elements(self):
        """
        Return the ordered elements.
        """
        return self._elements

    @property
    def hidden_sizes(self):
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
    def reduced_stress(self):
        """
        Return True if the reduced stress tensor, dE/dC, will be calculated.
        """
        return self._stress

    @property
    def reduced_total_pressure(self):
        """
        Return True if the reduced total pressure (eV), Trace(virial)/3.0, will
        be calculated.
        """
        return self._total_pressure

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

    def _get_energy(self, outputs, features, verbose=True):
        """
        Return the Op to compute total energy.
        """
        raise NotImplementedError("This method must be overridden!")

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
    def _get_reduced_full_stress_tensor(energy, cells):
        """
        Return the Op to compute the reduced stress tensor `-0.5 * dE/dh @ h`
        where `h` is a column-major cell tensor.
        """
        with tf.name_scope("Stress"):
            factor = tf.constant(-0.5, dtype=tf.float64, name='factor')
            dEdhT = tf.gradients(energy, cells)[0]
            # The cell tensor `h` in text books is column-major while in ASE
            # is row-major. So the Voigt indices and the matrix multiplication
            # below are transposed.
            stress = tf.einsum('ijk,ikl->ijl', cells, dEdhT)
            stress = tf.multiply(factor, stress)
            return stress

    def _get_reduced_stress(self, energy, cells, verbose=True):
        """
        Return the Op to compute the reduced stress (eV) in Voigt format.
        """
        with tf.name_scope("VoigtStress"):
            stress = self._get_reduced_full_stress_tensor(energy, cells)
            with tf.name_scope("Voigt"):
                voigt = tf.convert_to_tensor(
                    [[0, 0], [1, 1], [2, 2], [1, 2], [2, 0], [1, 0]],
                    dtype=tf.int32, name='voigt')
                batch_size = cells.shape[0].value or energy.shape[0].value
                if batch_size is None:
                    raise ValueError("The batch size cannot be inferred.")
                voigt = tf.tile(
                    tf.reshape(voigt, [1, 6, 2]), (batch_size, 1, 1))
                indices = tf.tile(tf.reshape(
                    tf.range(batch_size), [batch_size, 1, 1]),
                    [1, 6, 1])
                voigt = tf.concat((indices, voigt), axis=2, name='indices')
            stress = tf.gather_nd(stress, voigt, name='stress')
            if verbose:
                log_tensor(stress)
            return stress

    def _get_reduced_total_pressure(self, energy, cells, verbose=True):
        """
        Return the Op to compute the reduced total pressure (eV).

            reduced_total_pressure = -0.5 * trace(dy/dC @ cells) / -3.0

        """
        with tf.name_scope("Pressure"):
            stress = self._get_reduced_full_stress_tensor(energy, cells)
            three = tf.constant(-3.0, dtype=tf.float64, name='three')
            total_pressure = tf.div(tf.trace(stress), three, 'pressure')
            if verbose:
                log_tensor(total_pressure)
            return total_pressure

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
                # if 'bias' in var.op.name:
                #     continue
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
                * 'reduced_stress' of shape `[batch_size, 6]` is required if
                  `self.stress == True`.
                * 'reduced_total_pressure' of shape `[batch_size, ]` is required
                  if `self.reduced_total_pressure == True`.
        labels : AttributeDict
            A dict of label tensors as the desired regression targets.
                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, N, 3]` is required if
                  `self.forces == True`.
                * 'reduced_stress' of shape `[batch_size, 6]` is required if
                  `self.reduced_stress and not self.reduced_total_pressure`.
                * 'reduced_total_pressure' of shape `[batch_size, ]` is required
                  if `self.reduced_total_pressure == True`.

        Returns
        -------
        loss : tf.Tensor
            A `float64` tensor as the total loss.
        losses : AttributeDict
            A dict. The loss tensor for energy, forces and reduced stress or
            reduced total pressure.

        """

        def _get_loss(source: str, tag: str, scope: str, add_eps=False):
            """
            Return the loss tensor for the `source`.
            """
            with tf.name_scope(scope):
                x = getattr(labels, source)
                y = getattr(predictions, source)
                mse = tf.reduce_mean(tf.squared_difference(x, y), name='mse')
                if add_eps:
                    # Add a very small 'eps' to the mean squared error to make
                    # sure `mse` is always greater than zero. Otherwise NaN may
                    # occur at `Sqrt_Grad`.
                    with tf.name_scope("safe_sqrt"):
                        eps = tf.constant(1e-14, dtype=tf.float64, name='eps')
                        mse = tf.add(mse, eps)
                loss = tf.sqrt(mse, name=f"{tag}_rmse")
                mae = tf.reduce_mean(tf.abs(x - y), name=f"{tag}_mae")

                tf.summary.scalar(loss.op.name + '/summary', loss,
                                  collections=[GraphKeys.TRAIN_SUMMARY, ])
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, mae)
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, loss)

                return loss

        with tf.name_scope("Loss"):

            losses = AttributeDict()
            losses.energy = _get_loss('energy', 'y', 'Energy')

            if self._forces:
                losses.forces = _get_loss('forces', 'f', 'Forces', add_eps=True)

            if self._total_pressure:
                losses.reduced_total_pressure = _get_loss(
                    'reduced_total_pressure', 'p', 'Pressure', add_eps=True)
            elif self._stress:
                losses.reduced_stress = _get_loss(
                    'reduced_stress', 's', 'Stress', add_eps=True)

            if self._l2_weight > 0.0:
                losses.l2 = self.add_l2_penalty()

        return tf.add_n(list(losses.values()), name='loss'), losses

    @staticmethod
    def add_grads_and_vars_summary(grads_and_vars, name, collection):
        """
        Add summary of the gradients.
        """
        list_of_ops = []
        for grad, var in grads_and_vars:
            if grad is not None:
                norm = tf.norm(grad, name=var.op.name + "/norm")
                list_of_ops.append(norm)
                tf.add_to_collection(collection, grad)
                with tf.name_scope("gradients/{}/".format(name)):
                    tf.summary.scalar(var.op.name + "/norm", norm,
                                      collections=[GraphKeys.TRAIN_SUMMARY, ])
        with tf.name_scope("gradients/{}/".format(name)):
            total_norm = tf.add_n(list_of_ops, name='sum')
            tf.summary.scalar('total', total_norm,
                              collections=[GraphKeys.TRAIN_SUMMARY, ])
            tf.add_to_collection(GraphKeys.TRAIN_METRICS, total_norm)

    @staticmethod
    def _add_gradients_cos_dist_summary(energy_grad_vars, grads_and_vars):
        """
        Compute the cosine distance of dL(energy)/dvars and dL(target)/dvars
        where `target` is `forces` or `stress`.
        """
        with tf.name_scope("CosDist"):
            eps = tf.constant(1e-14, dtype=tf.float64)
            for i, (grad, var) in enumerate(grads_and_vars):
                if grad is None:
                    continue
                energy_grad = energy_grad_vars[i][0]
                dot = tf.tensordot(
                    tf.reshape(grad, (-1,)),
                    tf.reshape(energy_grad, (-1,)), axes=1)
                norm = tf.norm(grad) * tf.norm(energy_grad) + eps
                cos_dist = tf.div(dot, norm, name=var.op.name + '/cos_dist')
                tf.summary.scalar(cos_dist.op.name + '/summary', cos_dist)

    def get_train_op(self, losses: AttributeDict, hparams: AttributeDict):
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

            with tf.control_dependencies(
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = get_optimizer(learning_rate, hparams.opt.method)

            collections = []

            with tf.name_scope("Energy"):
                grads_and_vars = optimizer.compute_gradients(losses.energy)
                self.add_grads_and_vars_summary(
                    grads_and_vars, 'energy', GraphKeys.ENERGY_GRADIENTS)
                collections.append(grads_and_vars)

            if self._forces:
                with tf.name_scope("Forces"):
                    grads_and_vars = optimizer.compute_gradients(losses.forces)
                    self._add_gradients_cos_dist_summary(
                        collections[0], grads_and_vars)
                    self.add_grads_and_vars_summary(
                        grads_and_vars, 'forces', GraphKeys.FORCES_GRADIENTS)
                    collections.append(grads_and_vars)

            if self._total_pressure:
                with tf.name_scope("Pressure"):
                    grads_and_vars = optimizer.compute_gradients(
                        losses.reduced_total_pressure)
                    self._add_gradients_cos_dist_summary(
                        collections[0], grads_and_vars)
                    self.add_grads_and_vars_summary(
                        grads_and_vars, 'pressure', GraphKeys.STRESS_GRADIENTS)
                    collections.append(grads_and_vars)

            elif self._stress:
                with tf.name_scope("Stress"):
                    grads_and_vars = optimizer.compute_gradients(
                        losses.reduced_stress)
                    self._add_gradients_cos_dist_summary(
                        collections[0], grads_and_vars)
                    self.add_grads_and_vars_summary(
                        grads_and_vars, 'stress', GraphKeys.STRESS_GRADIENTS)
                    collections.append(grads_and_vars)

            grads_and_vars = sum_of_grads_and_vars_collections(collections)
            apply_gradients_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)

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

    def get_eval_metrics_ops(self, predictions, labels, n_atoms=None):
        """
        Return a dict of Ops as the evaluation metrics.

        `predictions` and `labels` are `AttributeDict` with the following keys
        required:
            * 'energy' of shape `[batch_size, ]` is required.
            * 'forces' of shape `[batch_size, N, 3]` is required if
              `self.forces == True`.
            * 'reduced_stress' of shape `[batch_size, 6]` is required if
              `self.reduced_stress and not self.reduced_total_pressure`.
            * 'reduced_total_pressure' of shape `[batch_size, ]` is required if
              `self.reduced_total_pressure == True`.

        `n_atoms` is an optional `int32` tensor with shape `[batch_size, ]`,
        representing the number of atoms in each structure. If give, per-atom
        metrics will be evaluated.

        """

        def _per_atom_metric_fn(metric: tf.Tensor):
            return tf.div(x=metric,
                          y=tf.cast(n_atoms, tf.float64, name='cast'),
                          name=metric.op.name + '_atom')

        def _get_metric(source, tag, cast_to_per_atom=False):
            x = getattr(labels, source)
            y = getattr(predictions, source)
            if cast_to_per_atom:
                x = _per_atom_metric_fn(x)
                y = _per_atom_metric_fn(y)
                suffix = '_atom'
            else:
                suffix = ''
            return {
                f"{tag}_rmse{suffix}": tf.metrics.root_mean_squared_error(x, y),
                f"{tag}_mae{suffix}": tf.metrics.mean_absolute_error(x, y)
            }

        with tf.name_scope("Metrics"):

            metrics = _get_metric('energy', 'y', cast_to_per_atom=False)
            if n_atoms is not None:
                metrics.update(
                    _get_metric('energy', 'y', cast_to_per_atom=True))

            if self._forces:
                metrics.update(_get_metric(
                    'forces', 'f', cast_to_per_atom=False))

            if self._total_pressure:
                metrics.update(_get_metric(
                    'reduced_total_pressure', 'p', cast_to_per_atom=False))

            elif self._stress:
                metrics.update(_get_metric(
                    'reduced_stress', 's', cast_to_per_atom=False))

            return metrics

    @staticmethod
    def _get_1x1conv_nn(x, activation_fn, hidden_sizes, verbose=False,
                        **kwargs):
        """
        A helper function to construct a 1x1 convolutional neural network.

        Parameters
        ----------
        x : tf.Tensor
            The input of the convolutional neural network. `x` must be a tensor
            with rank 3 (conv1d), 4 (conv2d) or 5 (conv3d) of dtype `float64`.
        activation_fn : Callable
            The activation function.
        hidden_sizes : List[int]
            The size of the hidden layers.

        Returns
        -------
        y : tf.Tensor
            The output tensor. `x` and `y` has the same rank. The last dimension
            of `y` is 1.

        """
        kernel_initializer = kwargs.get(
            'kernel_initializer',
            xavier_initializer(seed=Defaults.seed, dtype=tf.float64))
        bias_initializer = kwargs.get(
            'bias_initializer',
            tf.zeros_initializer(dtype=tf.float64))

        rank = len(x.shape)
        if rank == 3:
            conv = tf.layers.conv1d
        elif rank == 4:
            conv = tf.layers.conv2d
        elif rank == 5:
            conv = tf.layers.conv3d
        else:
            raise ValueError(
                f"The rank of `x` should be 3, 4 or 5 but not {rank}")

        for j in range(len(hidden_sizes)):
            x = conv(
                inputs=x, filters=hidden_sizes[j],
                kernel_size=1, strides=1,
                activation=activation_fn,
                use_bias=True,
                reuse=tf.AUTO_REUSE,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f'{conv.__name__.capitalize()}{j + 1}')
            if verbose:
                log_tensor(x)
        y = conv(inputs=x, filters=1, kernel_size=1, strides=1, use_bias=False,
                 kernel_initializer=kernel_initializer, reuse=tf.AUTO_REUSE,
                 name='Output')
        if verbose:
            log_tensor(y)
        return y

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Build the neural network.
        """
        raise NotImplementedError("This method must be overridden!")

    def _check_keys(self, features: AttributeDict, labels: AttributeDict):
        """
        Check the keys of `features` and `labels`.
        """
        assert 'descriptors' in features
        assert 'positions' in features
        assert 'cells' in features
        assert 'mask' in features
        assert 'n_atoms' in features
        assert 'volume' in features

        assert 'energy' in labels
        if self._forces:
            assert 'forces' in labels
        if self._total_pressure:
            assert 'reduced_total_pressure' in labels
        if self._stress:
            assert 'reduced_stress' in labels

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
        outputs = self._build_nn(features, verbose)

        with tf.name_scope("Output"):

            predictions = AttributeDict()

            predictions.energy = self._get_energy(
                outputs, features, verbose=verbose)

            if self._forces:
                predictions.forces = self._get_forces(
                    predictions.energy, features.positions, verbose=verbose)

            if self._total_pressure:
                predictions.reduced_total_pressure = \
                    self._get_reduced_total_pressure(
                        predictions.energy, features.cells, verbose=verbose)
            elif self._stress:
                predictions.reduced_stress = self._get_reduced_stress(
                    predictions.energy, features.cells, verbose=verbose)

            return predictions

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
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.
        labels : AttributeDict
            A dict of reference tensors.
                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, N, 3]` is required if
                  `self.forces == True`.
                * 'reduced_stress' of shape `[batch_size, 6]` is required if
                  `self.reduced_stress and not self.reduced_total_pressure`.
                * 'reduced_total_pressure' of shape `[batch_size, ]` is required
                  if `self.reduced_total_pressure == True`.
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

        total_loss, losses = self.get_total_loss(predictions, labels)
        train_op = self.get_train_op(losses=losses, hparams=params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            training_hooks = self.get_training_hooks(hparams=params)
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                              train_op=train_op,
                                              training_hooks=training_hooks)

        eval_metrics_ops = self.get_eval_metrics_ops(
            predictions, labels, n_atoms=features.n_atoms)
        evaluation_hooks = self.get_evaluation_hooks(hparams=params)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=total_loss,
                                          eval_metric_ops=eval_metrics_ops,
                                          evaluation_hooks=evaluation_hooks)
