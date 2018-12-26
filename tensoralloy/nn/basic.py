# coding=utf-8
"""
This module defines the basic neural network for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from typing import List, Dict
from collections import namedtuple

from tensoralloy.nn.utils import GraphKeys
from tensoralloy.nn.utils import log_tensor
from tensoralloy.misc import safe_select, Defaults, AttributeDict

from .ops import get_train_op
from .losses import *
from .hooks import ExamplesPerSecondHook, LoggingTensorHook, ProfilerHook

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class _PropertyError(ValueError):
    """
    This error shall be raised if the given property is not valid.
    """
    tag = "valid"

    def __init__(self, name):
        super(_PropertyError, self).__init__()
        self.name = name

    def __str__(self):
        return f"'{self.name}' is not a '{self.tag}' property."


class MinimizablePropertyError(_PropertyError):
    """
    This error shall be raised if the given property cannot be minimized.
    """
    tag = "minimizable"


class ExportablePropertyError(_PropertyError):
    """
    This error shall be raised if the given property cannot be exported.
    """
    tag = "exportable"


# noinspection PyTypeChecker,PyArgumentList
class Property(namedtuple('Property', ('name', 'minimizable'))):
    """
    A property of a strucutre.
    """

    def __new__(cls, name: str, minimizable: bool):
        """
        Initialization method.

        Parameters
        ----------
        name : str
            The name of this property.
        minimizable : bool
            A boolean indicating whether this property can be minimized or not.

        """
        return super(Property, cls).__new__(cls, name, minimizable)

    def __eq__(self, other):
        if hasattr(other, "name"):
            return other.name == self.name
        else:
            return str(other) == self.name


exportable_properties = (
    Property('energy', True),
    Property('forces', True),
    Property('stress', True),
    Property('total_pressure', True),
    Property('hessian', False),
)

available_properties = tuple(
    prop for prop in exportable_properties if prop.minimizable
)


class BasicNN:
    """
    The base neural network class.
    """

    # The default collection for model variabls.
    default_collection = None

    def __init__(self, elements: List[str], hidden_sizes=None, activation=None,
                 loss_weights=None, minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces')):
        """
        Initialization method.

        Parameters
        ----------
        elements : List[str]
            A list of str as the ordered elements.
        hidden_sizes : int or List[int] or Dict[str, List[int]] or array_like
            A list of int or a dict of (str, list of int) or an int as the sizes
            of the hidden layers.
        activation : str
            The name of the activation function to use.
        minimize_properties : List[str]
            A list of str as the properties to minimize. Avaibale properties
            are: 'energy', 'forces', 'stress' and 'total_pressure'. For each
            property, its RMSE loss will be minimized.
        export_properties : List[str]
            A list of str as the properties to infer when exporting the model.
            'energy' will always be exported.
        loss_weights : AttributeDict or None
            The weights of the losses. Available keys are 'energy', 'forces',
            'stress', 'total_pressure' and 'l2'. If None, all will be set to 1.

        Notes
        -----
        At least one of `energy`, `forces`, `stress` or `total_pressure` must be
        True.

        """
        self._elements = elements
        self._hidden_sizes = self._get_hidden_sizes(
            safe_select(hidden_sizes, Defaults.hidden_sizes))
        self._activation = safe_select(activation, Defaults.activation)

        if len(minimize_properties) == 0:
            raise ValueError("At least one property should be minimized.")

        for prop in minimize_properties:
            if prop not in available_properties:
                raise MinimizablePropertyError(prop)

        for prop in export_properties:
            if prop not in exportable_properties:
                raise ExportablePropertyError(prop)

        if loss_weights is None:
            loss_weights = AttributeDict(
                {prop: 1.0 for prop in minimize_properties})
            loss_weights.l2 = 0.0
        else:
            for prop, val in loss_weights.items():
                if prop != 'l2' and prop not in available_properties:
                    raise MinimizablePropertyError(prop)
                loss_weights[prop] = max(val, 0.0)
            for prop in minimize_properties:
                if prop not in loss_weights:
                    loss_weights[prop] = 1.0

        self._loss_weights = loss_weights
        self._minimize_properties = list(minimize_properties)
        self._export_properties = list(export_properties)

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
    def minimize_properties(self) -> List[str]:
        """
        Return a list of str as the properties to minimize.
        """
        return self._minimize_properties

    @property
    def predict_properties(self) -> List[str]:
        """
        Return a list of str as the properties to predict.
        """
        return self._export_properties

    @property
    def loss_weights(self):
        """
        Return the weights of the loss terms.
        """
        return self._loss_weights

    def _get_hidden_sizes(self, hidden_sizes):
        """
        Convert `hidden_sizes` to a dict if needed.
        """
        results = {}
        for element in self._elements:
            if isinstance(hidden_sizes, dict):
                sizes = np.asarray(
                    hidden_sizes.get(element, Defaults.hidden_sizes),
                    dtype=np.int)
            else:
                sizes = np.atleast_1d(hidden_sizes).astype(np.int)
            assert (sizes > 0).all()
            results[element] = sizes.tolist()
        return results

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
            # Setup the splitting axis. `energy` may be a 1D vector or a scalar.
            axis = energy.shape.ndims
            # Please remember: f = -dE/dR
            forces = tf.negative(
                tf.split(dEdR, [1, -1], axis=axis, name='split')[1],
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
        with tf.name_scope("Full"):
            factor = tf.constant(-0.5, dtype=tf.float64, name='factor')
            dEdhT = tf.gradients(energy, cells)[0]
            # The cell tensor `h` in text books is column-major while in ASE
            # is row-major. So the Voigt indices and the matrix multiplication
            # below are transposed.
            if cells.shape.ndims == 2:
                stress = tf.matmul(cells, dEdhT)
            else:
                stress = tf.einsum('ijk,ikl->ijl', cells, dEdhT)
            stress = tf.multiply(factor, stress, name='full')
            return stress

    @staticmethod
    def _convert_to_voigt_stress(stress, batch_size, verbose=False):
        """
        Convert a 3x3 stress tensor or a Nx3x3 stress tensors to corresponding
        Voigt form(s).
        """
        ndims = stress.shape.ndims
        with tf.name_scope("Voigt"):
            voigt = tf.convert_to_tensor(
                [[0, 0], [1, 1], [2, 2], [1, 2], [2, 0], [1, 0]],
                dtype=tf.int32, name='voigt')
            if ndims == 3:
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

    def _get_reduced_stress(self, energy, cells, verbose=True):
        """
        Return the Op to compute the reduced stress (eV) in Voigt format.
        """
        with tf.name_scope("Stress"):
            stress = self._get_reduced_full_stress_tensor(energy, cells)
            ndims = stress.shape.ndims
            batch_size = cells.shape[0].value or energy.shape[0].value
            if ndims == 3 and batch_size is None:
                raise ValueError("The batch size cannot be inferred.")
            return self._convert_to_voigt_stress(
                stress, batch_size, verbose=verbose)

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

    @staticmethod
    def _get_hessian(energy: tf.Tensor, positions: tf.Tensor, verbose=True):
        """
        Return the Op to compute the Hessian matrix:

            hessian = d^2E / dR^2 = d(dE / dR) / dR

        """
        with tf.name_scope("Hessian"):
            hessian = tf.identity(tf.hessians(energy, positions)[0],
                                  name='hessian')
            if verbose:
                log_tensor(hessian)
            return hessian

    def get_total_loss(self, predictions, labels, n_atoms):
        """
        Get the total loss tensor.

        Parameters
        ----------
        predictions : AttributeDict
            A dict of tensors as the predictions.
                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is
                  required if 'forces' should be minimized.
                * 'stress' of shape `[batch_size, 6]` is required if
                  'stress' should be minimized.
                * 'total_pressure' of shape `[batch_size, ]` is required if
                  'total_pressure' should be minimized.
        labels : AttributeDict
            A dict of reference tensors.
                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is
                  required if 'forces' should be minimized.
                * 'stress' of shape `[batch_size, 6]` is required if
                  'stress' should be minimized.
                * 'total_pressure' of shape `[batch_size, ]` is required if
                  'total_pressure' should be minimized.
        n_atoms : tf.Tensor
            A `int64` tensor of shape `[batch_size, ]`.

        Returns
        -------
        total_loss : tf.Tensor
            A `float64` tensor as the total loss.
        losses : AttributeDict
            A dict. The loss tensor for energy, forces and stress or total
            pressure.

        """
        with tf.name_scope("Loss"):

            collections = [GraphKeys.TRAIN_METRICS]

            losses = AttributeDict()
            losses.energy = get_energy_loss(
                labels=labels.energy, predictions=predictions.energy,
                n_atoms=n_atoms, weight=self._loss_weights.energy,
                collections=collections)

            if 'forces' in self._minimize_properties:
                losses.forces = get_forces_loss(
                    labels=labels.forces, predictions=predictions.forces,
                    n_atoms=n_atoms, weight=self._loss_weights.forces,
                    collections=collections)

            if 'total_pressure' in self._minimize_properties:
                losses.total_pressure = get_total_pressure_loss(
                    labels=labels.total_pressure,
                    predictions=predictions.total_pressure,
                    weight=self._loss_weights.total_pressure,
                    collections=collections)

            elif 'stress' in self._minimize_properties:
                losses.stress = get_stress_loss(
                    labels=labels.stress,
                    predictions=predictions.stress,
                    weight=self._loss_weights.stress,
                    collections=collections)

            if self._loss_weights.l2 > 0.0:
                with tf.name_scope("L2"):
                    losses.l2 = tf.losses.get_regularization_loss()
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, losses.l2)

            for tensor in losses.values():
                tf.summary.scalar(tensor.op.name + '/summary', tensor)

        return tf.add_n(list(losses.values()), name='loss'), losses

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
                    summary_op=tf.summary.merge_all())

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
                    profiler_hook = ProfilerHook(
                        save_steps=hparams.train.profile_steps,
                        output_dir=f"{hparams.train.model_dir}-profile",
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

    def get_eval_metrics_ops(self, predictions, labels, n_atoms):
        """
        Return a dict of Ops as the evaluation metrics.

        `predictions` and `labels` are `AttributeDict` with the following keys
        required:
            * 'energy' of shape `[batch_size, ]` is required.
            * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is required
              if 'forces' should be minimized.
            * 'stress' of shape `[batch_size, 6]` is required if
              'stress' should be minimized.
            * 'total_pressure' of shape `[batch_size, ]` is required if
              'total_pressure' should be minimized.

        `n_atoms` is a `int64` tensor with shape `[batch_size, ]`, representing
        the number of atoms in each structure.

        """
        with tf.name_scope("Metrics"):

            metrics = {}

            n_atoms = tf.cast(n_atoms, labels.energy.dtype, name='n_atoms')

            with tf.name_scope("Energy"):
                x = labels.energy
                y = predictions.energy
                xn = labels.energy / n_atoms
                yn = predictions.energy / n_atoms
                ops_dict = {
                    'Energy/mae': tf.metrics.mean_absolute_error(x, y),
                    'Energy/mse': tf.metrics.mean_squared_error(x, y),
                    'Energy/mae/atom': tf.metrics.mean_absolute_error(xn, yn),
                }
                metrics.update(ops_dict)

            if 'forces' in self._minimize_properties:
                with tf.name_scope("Forces"):
                    with tf.name_scope("Split"):
                        x = tf.split(labels.forces, [1, -1], axis=1)[1]
                    y = predictions.forces
                    with tf.name_scope("Scale"):
                        n_max = tf.convert_to_tensor(
                            x.shape[1].value, dtype=x.dtype, name='n_max')
                        one = tf.constant(1.0, dtype=tf.float64, name='one')
                        weight = tf.div(one, tf.reduce_mean(n_atoms / n_max),
                                        name='weight')
                        x = tf.multiply(weight, x)
                        y = tf.multiply(weight, y)
                    ops_dict = {
                        'Forces/mae': tf.metrics.mean_absolute_error(x, y),
                        'Forces/mse': tf.metrics.mean_squared_error(x, y),
                    }
                    metrics.update(ops_dict)

            if 'total_pressure' in self._minimize_properties:
                with tf.name_scope("Pressure"):
                    x = labels.total_pressure
                    y = predictions.total_pressure
                    ops_dict = {
                        'Pressure/mae': tf.metrics.mean_absolute_error(x, y),
                        'Pressure/mse': tf.metrics.mean_squared_error(x, y)}
                    metrics.update(ops_dict)

            elif 'stress' in self._minimize_properties:
                with tf.name_scope("Stress"):
                    x = labels.stress
                    y = predictions.stress
                    ops_dict = {
                        'Stress/mae': tf.metrics.mean_absolute_error(x, y),
                        'Stress/mse': tf.metrics.mean_squared_error(x, y)}
                    metrics.update(ops_dict)

            return metrics

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

        for prop in self._minimize_properties:
            assert prop in labels

    def build(self, features: AttributeDict, mode=tf.estimator.ModeKeys.TRAIN,
              verbose=True):
        """
        Build the atomic neural network.

        Parameters
        ----------
        features : AttributeDict
            A dict of input tensors:
                * 'descriptors', a dict of (element, (value, mask)) where
                  `element` represents the symbol of an element, `value` is the
                  descriptors of `element` and `mask` is the mask of `value`.
                * 'positions' of shape `[batch_size, n_atoms_max + 1, 3]`.
                * 'cells' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, n_atoms_max + 1]`.
                * 'composition' of shape `[batch_size, n_elements]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        mode : tf.estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, the prediction tensors will be logged.

        Returns
        -------
        predictions : AttributeDict
            A dict of output tensors.

        """
        outputs = self._build_nn(features, verbose)

        if mode == tf.estimator.ModeKeys.PREDICT:
            properties = self._export_properties
        else:
            properties = self._minimize_properties

        with tf.name_scope("Output"):

            predictions = AttributeDict()

            predictions.energy = self._get_energy(
                outputs, features, verbose=verbose)

            if 'forces' in properties:
                predictions.forces = self._get_forces(
                    predictions.energy, features.positions, verbose=verbose)

            if 'total_pressure' in properties:
                predictions.total_pressure = \
                    self._get_reduced_total_pressure(
                        predictions.energy, features.cells, verbose=verbose)
            elif 'stress' in properties:
                predictions.stress = self._get_reduced_stress(
                    predictions.energy, features.cells, verbose=verbose)

            if 'hessian' in properties:
                predictions.hessian = self._get_hessian(
                    predictions.energy, features.positions, verbose=verbose)

            return predictions

    def model_fn(self, features: AttributeDict, labels: AttributeDict,
                 mode: tf.estimator.ModeKeys, params: AttributeDict):
        """
        Initialize a model function for `tf.estimator.Estimator`.

        Parameters
        ----------
        features : AttributeDict
            A dict of tensors:
                * 'descriptors', a dict of (element, (value, mask)) where
                  `element` represents the symbol of an element, `value` is the
                  descriptors of `element` and `mask` is the mask of `value`.
                * 'positions' of shape `[batch_size, n_atoms_max + 1, 3]`.
                * 'cells' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, n_atoms_max + 1]`.
                * 'composition' of shape `[batch_size, n_elements]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        labels : AttributeDict
            A dict of reference tensors.
                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is
                  required if 'forces' should be minimized.
                * 'stress' of shape `[batch_size, 6]` is required if
                  'stress' should be minimized.
                * 'total_pressure' of shape `[batch_size, ]` is required if
                  'total_pressure' should be minimized.
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
                                 mode=mode,
                                 verbose=(mode == tf.estimator.ModeKeys.TRAIN))

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        total_loss, losses = self.get_total_loss(predictions=predictions,
                                                 labels=labels,
                                                 n_atoms=features.n_atoms)
        train_op = get_train_op(losses=losses, hparams=params,
                                minimize_properties=self._minimize_properties)

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
