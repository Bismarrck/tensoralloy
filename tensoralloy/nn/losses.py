# coding=utf-8
"""
This module defines various loss functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List, Union, Tuple
from enum import Enum

from tensoralloy.precision import get_float_dtype
from tensoralloy.nn.dataclasses import L2LossOptions, EnergyLossOptions
from tensoralloy.nn.dataclasses import ForcesLossOptions, StressLossOptions
from tensoralloy.nn.dataclasses import PressureLossOptions
from tensoralloy.nn.utils import is_first_replica
from tensoralloy.utils import GraphKeys


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


__all__ = ["get_energy_loss", "get_forces_loss", "get_stress_loss",
           "LossMethod"]


class LossMethod(Enum):
    """
    The methods for computing losses.

    rmse : the standard RMSE loss
    rrmse : the relative RMSE loss, rrmse = mean(norm(x - y, -1) / norm(y, -1)).
    logcosh : the Log-Cosh loss.

    """
    rmse = 0
    rrmse = 1
    logcosh = 2


def _logcosh(x, dtype=None, name=None):
    """
    The keras implemnetation of `logcosh`.
    """
    two = tf.convert_to_tensor(2.0, dtype=dtype, name='two')
    return tf.math.subtract(x + tf.nn.softplus(-two * x), tf.math.log(two),
                            name=name)


def _get_relative_rmse_loss(labels: tf.Tensor, predictions: tf.Tensor):
    """
    Return the relative RMSE as the loss of atomic forces.
    """
    with tf.name_scope("Relative"):
        diff = tf.subtract(labels, predictions)
        upper = tf.linalg.norm(diff, axis=1, keepdims=False, name='upper')
        lower = tf.linalg.norm(labels, axis=1, keepdims=False, name='lower')
        ratio = tf.math.truediv(upper, lower, name='ratio')
        loss = tf.reduce_mean(ratio, name='loss')
    return loss


def _get_rmse_loss(x: tf.Tensor, y: tf.Tensor,
                   is_per_atom_loss=False):
    """
    Return the RMSE loss tensor. The MAE loss will also be calculated.
    """
    if is_per_atom_loss:
        suffix = "/atom"
    else:
        suffix = ""
    mae = tf.reduce_mean(tf.abs(x - y), name='mae' + suffix)
    mse = tf.reduce_mean(tf.squared_difference(x, y), name='mse' + suffix)
    dtype = get_float_dtype()
    eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
    mse = tf.add(mse, eps, name='mse/safe')
    loss = tf.sqrt(mse, name='rmse' + suffix)
    return loss, mae


def _get_logcosh_loss(x: tf.Tensor, y: tf.Tensor, is_per_atom_loss=False):
    """
    Return the Log-Cosh loss tensor and the MAE loss tensor.
    """
    if is_per_atom_loss:
        suffix = "/atom"
    else:
        suffix = ""
    diff = tf.math.subtract(x, y, name='diff')
    mae = tf.reduce_mean(tf.abs(x - y), name='mae' + suffix)
    dtype = get_float_dtype()
    z = tf.reduce_mean(_logcosh(diff, dtype), name='logcosh' + suffix)
    return z, mae


def _get_weighted_loss(loss_weight: tf.Tensor,
                       raw_loss: tf.Tensor,
                       mae: Union[None, tf.Tensor], collections):
    """
    Return the final weighted loss tensor.
    """
    loss = tf.multiply(raw_loss, loss_weight, name='weighted/loss')
    if collections is not None and is_first_replica():
        if mae is not None:
            tf.add_to_collections(collections, mae)
        tf.add_to_collections(collections, raw_loss)
        tf.add_to_collections(collections, loss)
    return loss


def create_weight_tensor(weight: Union[float, Tuple[float, float]],
                         max_train_steps: Union[int, tf.Tensor],
                         dtype=tf.float32):
    """
    Create the static or dynamic loss weight tensor.
    """
    if isinstance(weight, float):
        return tf.convert_to_tensor(weight, name='weight')
    else:
        global_step = tf.train.get_or_create_global_step()
        global_step = tf.cast(global_step, dtype, name='global_step')
        max_train_steps = tf.convert_to_tensor(max_train_steps, dtype=dtype,
                                               name='max_steps')
        w0 = tf.convert_to_tensor(weight[0], dtype=dtype, name='w0')
        w1 = tf.convert_to_tensor(weight[1], dtype=dtype, name='w1')
        slope = tf.truediv(w1 - w0, max_train_steps, name='slope')
        weight = tf.add(w0, slope * global_step, name='weight')
        tf.add_to_collection(GraphKeys.TRAIN_METRICS, weight)
        return weight


def get_energy_loss(labels,
                    predictions,
                    n_atoms,
                    max_train_steps,
                    options: EnergyLossOptions,
                    collections=None,
                    name_scope="Energy"):
    """
    Return the loss tensor of the energy.

    Parameters
    ----------
    labels : tf.Tensor
        A float tensor of shape `[batch_size, ]` as the reference energy.
    predictions : tf.Tensor
        A float tensor of shape `[batch_size, ]` as the predicted energy.
    n_atoms : tf.Tensor
        A `int64` tensor of shape `[batch_size, ]` as the number of atoms of
        each structure.
    max_train_steps : Union[int, tf.Tensor]
        The maximum number of training steps.
    options : EnergyLossOptions
        Options for computing energy loss.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.
    name_scope : str
        The name scope for this function.

    Returns
    -------
    loss : tf.Tensor
        A float tensor as the total loss.

    """
    method = LossMethod[options.method]
    per_atom_loss = options.per_atom_loss
    assert method != LossMethod.rrmse
    print(options)

    with tf.name_scope(name_scope):
        assert labels.shape.ndims == 1
        assert predictions.shape.ndims == 1
        if per_atom_loss:
            n_atoms = tf.cast(n_atoms, labels.dtype, name='n_atoms')
            x = tf.math.truediv(labels, n_atoms, name='labels')
            y = tf.math.truediv(predictions, n_atoms, name='predictions')
        else:
            x = tf.identity(labels, name='labels')
            y = tf.identity(predictions, name='predictions')

        if method == LossMethod.rmse:
            raw_loss, mae = _get_rmse_loss(x, y, is_per_atom_loss=per_atom_loss)
        else:
            raw_loss, mae = _get_logcosh_loss(x, y,
                                              is_per_atom_loss=per_atom_loss)
        weight = create_weight_tensor(
            options.weight, max_train_steps, dtype=mae.dtype)
        return _get_weighted_loss(weight, raw_loss, mae, collections)


def _absolute_forces_loss(labels: tf.Tensor,
                          predictions: tf.Tensor,
                          atom_masks: tf.Tensor,
                          method=LossMethod.rmse):
    """
    Return the absolute loss of atomic forces. The `mask` tensor of shape
    `[batch_size, n_atoms_max + 1]` is required to eliminate the zero-padding
    effect.
    """
    assert method != LossMethod.rrmse

    with tf.name_scope("Absolute"):
        dtype = get_float_dtype()
        diff = tf.math.subtract(labels, predictions, name='diff')
        mask = tf.cast(tf.split(atom_masks, [1, -1], axis=1)[1], tf.bool,
                       name='mask')
        diff = tf.boolean_mask(diff, mask, axis=0, name='diff/mask')
        eps = tf.constant(dtype.eps, dtype=dtype, name='eps')

        if method == LossMethod.rmse:
            val = tf.math.square(diff, name='square')
        else:
            val = _logcosh(diff, dtype=dtype, name='logcosh')
        mae = tf.reduce_mean(tf.math.abs(diff), name='mae')

        if method == LossMethod.rmse:
            val = tf.reduce_mean(val, name='mse')
            val = tf.math.add(val, eps, name='mse/safe')
            loss = tf.sqrt(val, name='rmse')
        else:
            loss = tf.reduce_mean(val, name='logcosh/mean')

    return loss, mae


def get_forces_loss(labels,
                    predictions,
                    atom_masks,
                    max_train_steps: Union[int, tf.Tensor],
                    options: ForcesLossOptions,
                    collections=None):
    """
    Return the loss tensor of the atomic forces.

    Parameters
    ----------
    labels : tf.Tensor
        A float tensor of shape `[batch_size, n_atoms_max + 1, 3]` as the
        reference forces.
    predictions : tf.Tensor
        A float tensor of shape `[batch_size, n_atoms_max, 3]` as the predicted
        forces.
    atom_masks : tf.Tensor
        A float tensor of shape `[batch_size, ]` as the atom masks of each
        structure.
    max_train_steps : Union[int, tf.Tensor]
        The maximum number of training steps.
    options : ForcesLossOptions
        Options for computing force loss.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.

    Returns
    -------
    loss : tf.Tensor
        A float tensor as the total loss.

    """
    with tf.name_scope("Forces"):
        assert labels.shape.ndims == 3 and labels.shape[2].value == 3
        assert predictions.shape.ndims == 3 and predictions.shape[2].value == 3
        with tf.name_scope("Split"):
            labels = tf.split(
                labels, [1, -1], axis=1, name='split')[1]
        method = LossMethod[options.method]
        weight = create_weight_tensor(options.weight, max_train_steps,
                                      labels.dtype)
        raw_loss, mae = _absolute_forces_loss(labels, predictions,
                                              atom_masks=atom_masks,
                                              method=method)
        return _get_weighted_loss(weight, raw_loss, mae, collections)


def get_stress_loss(labels,
                    predictions,
                    max_train_steps: Union[int, tf.Tensor],
                    options: StressLossOptions,
                    collections=None):
    """
    Return the (relative) RMSE loss of the stress.

    Parameters
    ----------
    labels : tf.Tensor
        A float tensor of shape `[batch_size, 6]` as the reference stress.
    predictions : tf.Tensor
        A float tensor of shape `[batch_size, 6]` as the predicted stress.
    max_train_steps : Union[int, tf.Tensor]
        The maximum number of training steps.
    options : StressLossOptions
        Options for computing stress loss.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.

    Returns
    -------
    loss : tf.Tensor
        A float tensor as the total loss.

    """
    with tf.name_scope("Stress"):
        assert labels.shape.ndims == 2 and labels.shape[1].value == 6
        assert predictions.shape.ndims == 2 and predictions.shape[1].value == 6
        method = LossMethod[options.method]
        if method == LossMethod.rmse:
            raw_loss, mae = _get_rmse_loss(
                labels,
                predictions,
                is_per_atom_loss=False)
        elif method == LossMethod.rrmse:
            raw_loss = _get_relative_rmse_loss(
                labels,
                predictions)
            mae = None
        else:
            raw_loss, mae = _get_logcosh_loss(
                labels,
                predictions,
                is_per_atom_loss=False)
        weight = create_weight_tensor(options.weight, max_train_steps,
                                      dtype=mae.dtype)
        return _get_weighted_loss(weight, raw_loss, mae, collections)


def get_total_pressure_loss(labels,
                            predictions,
                            max_train_steps: Union[int, tf.Tensor],
                            options: PressureLossOptions,
                            collections=None):
    """
    Return the RMSE loss of the total pressure.

    Parameters
    ----------
    labels : tf.Tensor
        A float tensor of shape `[batch_size, ]` as the reference pressure.
        energies.
    predictions : tf.Tensor
        A float tensor of shape `[batch_size, ]` as the predicted pressure.
    max_train_steps : Union[int, tf.Tensor]
        The maximum number of training steps.
    options : StressLossOptions
        Options for computing pressure loss.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.

    Returns
    -------
    loss : tf.Tensor
        A float tensor as the total loss.

    """
    method = LossMethod[options.method]
    assert method != LossMethod.rrmse

    with tf.name_scope("Pressure"):
        assert labels.shape.ndims == 1
        assert predictions.shape.ndims == 1

        if method == LossMethod.rmse:
            raw_loss, mae = _get_rmse_loss(labels, predictions,
                                           is_per_atom_loss=False)
        else:
            raw_loss, mae = _get_logcosh_loss(labels, predictions,
                                              is_per_atom_loss=False)
        weight = create_weight_tensor(options.weight, max_train_steps,
                                      dtype=labels.dtype)
        return _get_weighted_loss(weight, raw_loss, mae, collections)


def get_l2_regularization_loss(options: L2LossOptions, collections=None):
    """
    Return the total L2 regularization loss.

    Parameters
    ----------
    options : L2LossOptions
        The options of L2 loss.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.

    Returns
    -------
    loss : tf.Tensor or None
        A float tensor as the total loss.

    """
    with tf.name_scope("L2"):
        if options.weight == 0.0:
            return None

        name = 'total_regularization_loss'
        losses = tf.compat.v1.losses.get_regularization_losses()
        if losses:
            l2 = tf.add_n(losses, name=name)
        else:
            l2 = tf.constant(0.0, dtype=get_float_dtype(), name=name)
        loss_weight = tf.convert_to_tensor(
            options.weight, l2.dtype, name='weight')
        if options.decayed:
            loss_weight = tf.compat.v1.train.exponential_decay(
                learning_rate=loss_weight,
                global_step=tf.compat.v1.train.get_global_step(),
                decay_steps=options.decay_steps,
                decay_rate=options.decay_rate,
                name='weight/decayed')
            if collections is not None and is_first_replica():
                tf.add_to_collections(collections, loss_weight)

        loss = tf.multiply(l2, loss_weight, name='weighted/loss')
        if collections is not None and is_first_replica():
            tf.add_to_collections(collections, l2)
            tf.add_to_collections(collections, loss)
        return loss
