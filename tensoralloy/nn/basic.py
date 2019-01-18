# coding=utf-8
"""
This module defines the basic neural network for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json
import shutil

from typing import List, Dict
from collections import namedtuple
from os.path import join, dirname
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_io

from tensoralloy.utils import GraphKeys, AttributeDict, Defaults, safe_select
from tensoralloy.nn.utils import log_tensor
from tensoralloy.nn.ops import get_train_op
from tensoralloy.nn.hooks import RestoreEmaVariablesHook, ProfilerHook
from tensoralloy.nn.hooks import ExamplesPerSecondHook, LoggingTensorHook
from tensoralloy.nn.hooks import WarmStartFromVariablesHook, NanTensorHook
from tensoralloy.nn import losses as loss_ops
from tensoralloy.transformer.base import BaseTransformer
from tensoralloy.transformer.base import BatchDescriptorTransformer

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

    def __init__(self,
                 elements: List[str],
                 hidden_sizes=None,
                 activation=None,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces'),
                 positive_energy_mode=False):
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
        positive_energy_mode : bool
            A boolean flag. Defaults to False. If True, the true energies will
            be converted to positive values by multiplying with `-1` before
            computing the energy loss. Thus, the predicted energies of the NN
            will also be positive.

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

        self._minimize_properties = list(minimize_properties)
        self._export_properties = list(export_properties)
        self._positive_energy_mode = positive_energy_mode

        self._transformer: BaseTransformer = None

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
    def positive_energy_mode(self):
        """
        Return True if positive energy mode is enabled.
        """
        return self._positive_energy_mode

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

    def attach_transformer(self, clf: BaseTransformer):
        """
        Attach a descriptor transformer to this NN.
        """
        self._transformer = clf

    def _get_transformer(self):
        """
        Get the attached transformer.
        """
        return self._transformer

    transformer = property(_get_transformer, attach_transformer)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        raise NotImplementedError("This method must be overridden!")

    def _get_energy_op(self, outputs, features, name='energy', verbose=True):
        """
        Return the Op to compute total energy.
        """
        raise NotImplementedError("This method must be overridden!")

    @staticmethod
    def _get_forces_op(energy, positions, name='forces', verbose=True):
        """
        Return the Op to compute atomic forces (eV / Angstrom).
        """
        dEdR = tf.gradients(energy, positions, name='dEdR')[0]
        # Setup the splitting axis. `energy` may be a 1D vector or a scalar.
        axis = energy.shape.ndims
        # Please remember: f = -dE/dR
        forces = tf.negative(
            tf.split(dEdR, [1, -1], axis=axis, name='split')[1],
            name=name)
        if verbose:
            log_tensor(forces)
        return forces

    @staticmethod
    def _get_reduced_full_stress_tensor(energy: tf.Tensor, cells):
        """
        Return the Op to compute the reduced stress tensor `dE/dh @ h` where `h`
        is the cell tensor.
        """
        with tf.name_scope("Full"):
            factor = tf.constant(1.0, dtype=energy.dtype, name='factor')
            dEdh = tf.identity(tf.gradients(energy, cells)[0], name='dEdh')
            # The cell tensor `h` in text books is column-major while in ASE
            # is row-major. So the Voigt indices and the matrix multiplication
            # below are transposed.
            if cells.shape.ndims == 2:
                stress = tf.matmul(dEdh, cells)
            else:
                stress = tf.einsum('ijk,ikl->ijl', dEdh, cells)
            stress = tf.multiply(factor, stress, name='full')
            return stress

    @staticmethod
    def _convert_to_voigt_stress(stress, batch_size, name='stress',
                                 verbose=False):
        """
        Convert a 3x3 stress tensor or a Nx3x3 stress tensors to corresponding
        Voigt form(s).
        """
        ndims = stress.shape.ndims
        with tf.name_scope("Voigt"):
            voigt = tf.convert_to_tensor(
                [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]],
                dtype=tf.int32, name='voigt')
            if ndims == 3:
                voigt = tf.tile(
                    tf.reshape(voigt, [1, 6, 2]), (batch_size, 1, 1))
                indices = tf.tile(tf.reshape(
                    tf.range(batch_size), [batch_size, 1, 1]),
                    [1, 6, 1])
                voigt = tf.concat((indices, voigt), axis=2, name='indices')
            stress = tf.gather_nd(stress, voigt, name=name)
            if verbose:
                log_tensor(stress)
            return stress

    def _get_stress_op(self, energy: tf.Tensor, cells, name='stress',
                       verbose=True):
        """
        Return the Op to compute the reduced stress (eV) in Voigt format.
        """
        stress = self._get_reduced_full_stress_tensor(energy, cells)
        ndims = stress.shape.ndims
        batch_size = cells.shape[0].value or energy.shape[0].value
        if ndims == 3 and batch_size is None:
            raise ValueError("The batch size cannot be inferred.")
        return self._convert_to_voigt_stress(
            stress, batch_size, name=name, verbose=verbose)

    def _get_total_pressure_op(self, energy: tf.Tensor, cells, name='pressure',
                               verbose=True):
        """
        Return the Op to compute the reduced total pressure (eV).

            reduced_total_pressure = -0.5 * trace(dy/dC @ cells) / -3.0

        """
        stress = self._get_reduced_full_stress_tensor(energy, cells)
        three = tf.constant(-3.0, dtype=energy.dtype, name='three')
        total_pressure = tf.div(tf.trace(stress), three, name=name)
        if verbose:
            log_tensor(total_pressure)
        return total_pressure

    @staticmethod
    def _get_hessian_op(energy: tf.Tensor, positions: tf.Tensor, name='hessian',
                        verbose=True):
        """
        Return the Op to compute the Hessian matrix:

            hessian = d^2E / dR^2 = d(dE / dR) / dR

        """
        hessian = tf.identity(tf.hessians(energy, positions)[0], name=name)
        if verbose:
            log_tensor(hessian)
        return hessian

    @staticmethod
    def _check_loss_hparams(hparams: AttributeDict):
        """
        Check the hyper parameters and add missing but required parameters.
        """

        def _convert_to_attr_dict(adict):
            if not isinstance(adict, AttributeDict):
                return AttributeDict(adict)
            else:
                return adict

        defaults = AttributeDict(
            energy=AttributeDict(weight=1.0,
                                 per_atom_loss=False),
            forces=AttributeDict(weight=1.0),
            stress=AttributeDict(weight=1.0),
            total_pressure=AttributeDict(weight=1.0),
            l2=AttributeDict(weight=0.01))

        if hparams is None:
            hparams = AttributeDict(loss=defaults)
        else:
            hparams = _convert_to_attr_dict(hparams)
            if 'loss' not in hparams:
                hparams.loss = defaults
            else:
                hparams.loss = _convert_to_attr_dict(hparams.loss)

                def _check_section(section):
                    if section not in hparams.loss:
                        hparams.loss[section] = defaults[section]
                    else:
                        hparams.loss[section] = _convert_to_attr_dict(
                            hparams.loss[section])
                        for key, value in defaults[section].items():
                            if key not in hparams.loss[section]:
                                hparams.loss[section][key] = value

                _check_section('energy')
                _check_section('forces')
                _check_section('stress')
                _check_section('total_pressure')
                _check_section('l2')

        return hparams

    def get_total_loss(self, predictions, labels, n_atoms,
                       hparams: AttributeDict):
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
        hparams : AttributeDict
            A dict of hyper parameters for defining loss functions. Essential
            keypaths for this method are:
                - 'hparams.loss.energy.weight'
                - 'hparams.loss.energy.per_atom_loss'
                - 'hparams.loss.forces.weight'
                - 'hparams.loss.stress.weight'
                - 'hparams.loss.total_pressure.weight'
                - 'hparams.loss.l2.weight'

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
            hparams = self._check_loss_hparams(hparams)

            losses = AttributeDict()
            losses.energy = loss_ops.get_energy_loss(
                labels=labels.energy,
                predictions=predictions.energy,
                n_atoms=n_atoms,
                weight=hparams.loss.energy.weight,
                per_atom_loss=hparams.loss.energy.per_atom_loss,
                collections=collections)

            if 'forces' in self._minimize_properties:
                losses.forces = loss_ops.get_forces_loss(
                    labels=labels.forces,
                    predictions=predictions.forces,
                    n_atoms=n_atoms,
                    weight=hparams.loss.forces.weight,
                    collections=collections)

            if 'total_pressure' in self._minimize_properties:
                losses.total_pressure = loss_ops.get_total_pressure_loss(
                    labels=labels.total_pressure,
                    predictions=predictions.total_pressure,
                    weight=hparams.loss.total_pressure.weight,
                    collections=collections)

            elif 'stress' in self._minimize_properties:
                losses.stress = loss_ops.get_stress_loss(
                    labels=labels.stress,
                    predictions=predictions.stress,
                    weight=hparams.loss.stress.weight,
                    collections=collections)

            losses.l2 = loss_ops.get_l2_regularization_loss(
                weight=hparams.loss.l2.weight,
                collections=collections)

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

    def get_training_hooks(self,
                           losses: AttributeDict,
                           ema: tf.train.ExponentialMovingAverage,
                           hparams) -> List[tf.train.SessionRunHook]:
        """
        Return a list of `tf.train.SessionRunHook` objects for training.

        Parameters
        ----------
        losses : AttributeDict
            A dict. The loss tensor for energy, forces and stress or total
            pressure.
        ema : tf.train.ExponentialMovingAverage
            A function to obtain moving averaged variables.
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

            with tf.name_scope("Nan"):
                nan_tensor_hook = NanTensorHook(fail_on_nan_loss=True, **losses)

            hooks = [summary_saver_hook, examples_per_sec_hook, nan_tensor_hook]

            if len(tf.get_collection(GraphKeys.TRAIN_METRICS)) > 0:
                logging_tensor_hook = LoggingTensorHook(
                    tensors=self.get_logging_tensors(GraphKeys.TRAIN_METRICS),
                    every_n_iter=hparams.train.log_steps,
                    at_end=True)
                hooks.append(logging_tensor_hook)

            if hparams.train.profile_steps:
                with tf.name_scope("Profile"):
                    profiler_hook = ProfilerHook(
                        save_steps=hparams.train.profile_steps,
                        output_dir=f"{hparams.train.model_dir}-profile",
                        show_memory=True)
                hooks.append(profiler_hook)

            if hparams.train.previous_checkpoint is not None:
                with tf.name_scope("Restore"):
                    warm_start_hook = WarmStartFromVariablesHook(
                        previous_checkpoint=hparams.train.previous_checkpoint,
                        ema=ema)
                hooks.append(warm_start_hook)

        return hooks

    def get_evaluation_hooks(self,
                             ema: tf.train.ExponentialMovingAverage,
                             hparams: AttributeDict):
        """
        Return a list of `tf.train.SessionRunHook` objects for evaluation.
        """
        hooks = []

        with tf.name_scope("Hooks"):

            if len(tf.get_collection(GraphKeys.EVAL_METRICS)) > 0:
                with tf.name_scope("Accuracy"):
                    logging_tensor_hook = LoggingTensorHook(
                        tensors=self.get_logging_tensors(
                            GraphKeys.EVAL_METRICS),
                        every_n_iter=hparams.train.eval_steps,
                        at_end=True)
                hooks.append(logging_tensor_hook)

            with tf.name_scope("EMA"):
                restore_ema_hook = RestoreEmaVariablesHook(ema=ema)
                hooks.append(restore_ema_hook)

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
                xn = x / n_atoms
                yn = y / n_atoms
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
                        one = tf.constant(1.0, dtype=x.dtype, name='one')
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

    def _get_model_outputs(self,
                           features: AttributeDict,
                           descriptors: AttributeDict,
                           mode: tf.estimator.ModeKeys,
                           verbose=False):
        """
        Build the NN model and return raw outputs.

        Parameters
        ----------
        features : AttributeDict
            A dict of input raw property tensors:
                * 'positions' of shape `[batch_size, n_atoms_max + 1, 3]`.
                * 'cells' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, n_atoms_max + 1]`.
                * 'composition' of shape `[batch_size, n_elements]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        descriptors : AttributeDict
            A dict of Ops to get atomic descriptors. This should be produced by
            an overrided `BaseTransformer.get_descriptors()`.
        mode : tf.estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        raise NotImplementedError("This method must be overridden!")

    def _check_keys(self, features: AttributeDict, labels: AttributeDict):
        """
        Check the keys of `features` and `labels`.
        """
        assert 'positions' in features
        assert 'cells' in features
        assert 'mask' in features
        assert 'n_atoms' in features
        assert 'volume' in features
        assert isinstance(self._transformer, BaseTransformer)

        for prop in self._minimize_properties:
            assert prop in labels

    def build(self,
              features: AttributeDict,
              mode: tf.estimator.ModeKeys.TRAIN,
              verbose=True):
        """
        Build the atomic neural network.

        Parameters
        ----------
        features : AttributeDict
            A dict of input raw property tensors:
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

        # 'descriptors', a dict of (element, (value, mask)) where `element`
        # represents the symbol of an element, `value` is the descriptors of
        # `element` and `mask` is the mask of `value`.
        descriptors = self._transformer.get_descriptors(features)

        outputs = self._get_model_outputs(
            features=features,
            descriptors=descriptors,
            mode=mode,
            verbose=verbose)

        if mode == tf.estimator.ModeKeys.PREDICT:
            properties = self._export_properties
        else:
            properties = self._minimize_properties

        with tf.name_scope("Output"):

            predictions = AttributeDict()

            with tf.name_scope("Energy"):
                if self._positive_energy_mode:
                    energy = self._get_energy_op(
                        outputs, features, name='energy/positive',
                        verbose=verbose)
                    predictions.energy = tf.negative(energy, name='energy')
                else:
                    predictions.energy = self._get_energy_op(
                        outputs, features, name='energy', verbose=verbose)

            if 'forces' in properties:
                with tf.name_scope("Forces"):
                    predictions.forces = self._get_forces_op(
                        predictions.energy, features.positions, name='forces',
                        verbose=verbose)

            if 'total_pressure' in properties:
                with tf.name_scope("Pressure"):
                    predictions.total_pressure = \
                        self._get_total_pressure_op(
                            predictions.energy, features.cells, name='pressure',
                            verbose=verbose)
            elif 'stress' in properties:
                with tf.name_scope("Stress"):
                    predictions.stress = self._get_stress_op(
                        predictions.energy, features.cells, name='stress',
                        verbose=verbose)

            if 'hessian' in properties:
                with tf.name_scope("Hessian"):
                    predictions.hessian = self._get_hessian_op(
                        predictions.energy, features.positions, name='hessian',
                        verbose=verbose)

            return predictions

    def model_fn(self,
                 features: AttributeDict,
                 labels: AttributeDict,
                 mode: tf.estimator.ModeKeys,
                 params: AttributeDict):
        """
        Initialize a model function for `tf.estimator.Estimator`.

        In this method `features` are raw property (positions, cells, etc)
        tensors. Because `tf.estimator.Estimator` requires a `features` as the
        first arg of `model_fn`, we cannot change its name here.

        Parameters
        ----------
        features : AttributeDict
            A dict of raw property tensors:
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
            Hyperparameters for building and training a NN model.

        Returns
        -------
        spec : tf.estimator.EstimatorSpec
            Ops and objects returned from a `model_fn` and passed to an
            `Estimator`. `EstimatorSpec` fully defines the model to be run
            by an `Estimator`.

        """
        self._check_keys(features, labels)

        predictions = self.build(features=features,
                                 mode=mode,
                                 verbose=(mode == tf.estimator.ModeKeys.TRAIN))

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        total_loss, losses = self.get_total_loss(predictions=predictions,
                                                 labels=labels,
                                                 n_atoms=features.n_atoms,
                                                 hparams=params)
        ema, train_op = get_train_op(
            losses=losses,
            hparams=params,
            minimize_properties=self._minimize_properties)

        if mode == tf.estimator.ModeKeys.TRAIN:
            training_hooks = self.get_training_hooks(
                losses=losses,
                ema=ema,
                hparams=params)
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                              train_op=train_op,
                                              training_hooks=training_hooks)

        eval_metrics_ops = self.get_eval_metrics_ops(
            predictions=predictions,
            labels=labels,
            n_atoms=features.n_atoms)
        evaluation_hooks = self.get_evaluation_hooks(ema=ema, hparams=params)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=total_loss,
                                          eval_metric_ops=eval_metrics_ops,
                                          evaluation_hooks=evaluation_hooks)

    def export(self, output_graph_path: str, checkpoint=None,
               keep_tmp_files=True):
        """
        Freeze the graph and export the model to a pb file.

        Parameters
        ----------
        output_graph_path : str
            The name of the output graph file.
        checkpoint : str or None
            The tensorflow checkpoint file to restore or None.
        keep_tmp_files : bool
            If False, the intermediate files will be deleted.

        """

        graph = tf.Graph()

        logdir = join(dirname(output_graph_path), 'export')
        input_graph_name = 'input_graph.pb'
        saved_model_ckpt = join(logdir, 'saved_model')
        saved_model_meta = f"{saved_model_ckpt}.meta"

        with graph.as_default():

            if self._transformer is None:
                raise ValueError("A transformer must be attached before "
                                 "exporting to a pb file.")
            elif isinstance(self._transformer, BatchDescriptorTransformer):
                clf = self._transformer.as_descriptor_transformer()
            else:
                clf = self._transformer

            configs = self.as_dict()
            configs.pop('class')

            nn = self.__class__(**configs)
            nn.attach_transformer(clf)
            predictions = nn.build(clf.placeholders,
                                   mode=tf.estimator.ModeKeys.PREDICT,
                                   verbose=True)

            # Encode the JSON dict of the serialized transformer into the graph.
            with tf.name_scope("Transformer/"):
                transformer_params = tf.constant(
                    json.dumps(clf.as_dict()), name='params')

            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                # Restore the moving averaged variables
                ema = tf.train.ExponentialMovingAverage(
                    Defaults.variable_moving_average_decay)
                saver = tf.train.Saver(ema.variables_to_restore())
                if checkpoint is not None:
                    saver.restore(sess, checkpoint)

                # Create another saver to save the trainable variables
                saver = tf.train.Saver(var_list=tf.model_variables())
                checkpoint_path = saver.save(
                    sess, saved_model_ckpt, global_step=0)
                graph_io.write_graph(graph_or_graph_def=graph,
                                     logdir=logdir,
                                     name=input_graph_name)

            input_graph_path = join(logdir, input_graph_name)
            input_saver_def_path = ""
            input_binary = False
            restore_op_name = "save/restore_all"
            filename_tensor_name = "save/Const:0"
            clear_devices = True
            input_meta_graph = saved_model_meta

            output_node_names = [transformer_params.op.name]

            for tensor in predictions.values():
                output_node_names.append(tensor.op.name)

            for node in graph.as_graph_def().node:
                name = node.name
                if name.startswith('Placeholders/'):
                    output_node_names.append(name)

            freeze_graph.freeze_graph(
                input_graph_path, input_saver_def_path, input_binary,
                checkpoint_path, ",".join(output_node_names), restore_op_name,
                filename_tensor_name, output_graph_path, clear_devices, "", "",
                input_meta_graph)

        if not keep_tmp_files:
            shutil.rmtree(logdir, ignore_errors=True)

        tf.logging.info(f"Model exported to {output_graph_path}")
