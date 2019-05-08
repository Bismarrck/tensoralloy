# coding=utf-8
"""
This module defines various loss functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List, Union
from enum import Enum

from tensoralloy.dtypes import get_float_dtype


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


__all__ = ["get_energy_loss", "get_forces_loss", "get_stress_loss",
           "get_total_pressure_loss", "get_total_pressure_loss",
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


def _get_relative_rmse_loss(labels: tf.Tensor, predictions: tf.Tensor,
                            weights: Union[tf.Tensor, None]):
    """
    Return the relative RMSE as the loss of atomic forces.
    """
    with tf.name_scope("Relative"):
        diff = tf.subtract(labels, predictions)
        upper = tf.linalg.norm(diff, axis=1, keepdims=False, name='upper')
        lower = tf.linalg.norm(labels, axis=1, keepdims=False, name='lower')
        ratio = tf.math.truediv(upper, lower, name='ratio')
        if weights is not None:
            ratio = tf.math.multiply(ratio, weights, name='ration/conf')
        loss = tf.reduce_mean(ratio, name='loss')
    return loss


def _get_rmse_loss(x: tf.Tensor, y: tf.Tensor, weights=None,
                   is_per_atom_loss=False):
    """
    Return the RMSE loss tensor. The MAE loss will also be calculated.
    """
    if is_per_atom_loss:
        suffix = "/atom"
    else:
        suffix = ""
    mae = tf.reduce_mean(tf.abs(x - y), name='mae' + suffix)
    if weights is None:
        mse = tf.reduce_mean(tf.squared_difference(x, y), name='mse' + suffix)
    else:
        if x.shape.ndims == 2:
            weights = tf.reshape(weights, (-1, 1))
        mse = tf.reduce_mean(
            tf.math.multiply(weights,
                             tf.squared_difference(x, y),
                             name='conf'),
            name='mse' + suffix)
    dtype = get_float_dtype()
    eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
    mse = tf.add(mse, eps, name='mse/safe')
    loss = tf.sqrt(mse, name='rmse' + suffix)
    return loss, mae


def _get_logcosh_loss(x: tf.Tensor, y: tf.Tensor, weights=None,
                      is_per_atom_loss=False):
    """
    Return the Log-Cosh loss tensor and the MAE loss tensor.
    """
    if is_per_atom_loss:
        suffix = "/atom"
    else:
        suffix = ""
    diff = tf.math.subtract(x, y, name='diff')
    mae = tf.reduce_mean(tf.abs(x - y), name='mae' + suffix)
    if weights is not None:
        if x.shape.ndims == 2:
            weights = tf.reshape(weights, (-1, 1))
        diff = tf.multiply(diff, weights, name='diff/conf')
    dtype = get_float_dtype()
    eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
    cosh = tf.math.cosh(diff, name='cosh')
    cosh = tf.add(cosh, eps, name='cosh/safe')
    logcosh = tf.reduce_mean(tf.math.log(cosh), name='logcosh')
    return logcosh, mae


def get_energy_loss(labels, predictions, n_atoms, confidences=None, weight=1.0,
                    collections=None, per_atom_loss=False,
                    method=LossMethod.rmse):
    """
    Return the loss tensor of the energy.

    Parameters
    ----------
    labels : tf.Tensor
        A `float64` tensor of shape `[batch_size, ]` as the reference energy.
    predictions : tf.Tensor
        A `float64` tensor of shape `[batch_size, ]` as the predicted energy.
    confidences : tf.Tensor or None
        A `float64` tensor of shape `[batch_size, ]` as the energy confidences
        or None. If None, all energies are trusted equally.
    n_atoms : tf.Tensor
        A `int64` tensor of shape `[batch_size, ]` as the number of atoms of
        each structure.
    weight : float or tf.Tensor
        The weight of the loss.
    per_atom_loss : bool
        If True, return the per-atom loss; otherwise, return the per-structrue
        loss. Defaults to False.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.
    method : LossMethod
        The loss method to use.

    Returns
    -------
    loss : tf.Tensor
        A `float64` tensor as the RMSE loss.

    """
    assert method != LossMethod.rrmse

    with tf.name_scope("Energy"):
        assert labels.shape.ndims == 1
        assert predictions.shape.ndims == 1
        if per_atom_loss:
            n_atoms = tf.cast(n_atoms, labels.dtype, name='n_atoms')
            x = tf.math.truediv(labels, n_atoms, name='labels')
            y = tf.math.truediv(predictions, n_atoms, name='labels')
        else:
            x = tf.identity(labels, name='labels')
            y = tf.identity(predictions, name='predictions')
        raw_loss, mae = _get_rmse_loss(x, y, weights=confidences,
                                       is_per_atom_loss=per_atom_loss)
        weight = tf.convert_to_tensor(weight, raw_loss.dtype, name='weight')
        loss = tf.multiply(raw_loss, weight, name='weighted/loss')
        if collections is not None:
            tf.add_to_collections(collections, mae)
            tf.add_to_collections(collections, raw_loss)
            tf.add_to_collections(collections, loss)
        return loss


def _absolute_forces_loss(labels: tf.Tensor,
                          predictions: tf.Tensor,
                          n_atoms: tf.Tensor,
                          confidences: Union[tf.Tensor, None]):
    """
    Return the absolute RMSE as the loss of atomic forces.

        loss = sqrt(weight * MSE(labels, predictions))

    where `weight` is a scaling factor to eliminate the zero-padding effect.

    """
    with tf.name_scope("Absolute"):
        dtype = get_float_dtype()
        if confidences is None:
            mse = tf.reduce_mean(
                tf.squared_difference(labels, predictions), name='raw_mse')
        else:
            mse = tf.reduce_mean(
                tf.multiply(tf.reshape(confidences, (-1, 1, 1)),
                            tf.squared_difference(labels, predictions),
                            name='conf'),
                name='raw_mse')
        mae = tf.reduce_mean(tf.abs(labels - predictions), name='raw_mae')
        with tf.name_scope("Safe"):
            # Add a very small 'eps' to the mean squared error to make
            # sure `mse` is always greater than zero. Otherwise NaN may
            # occur at `Sqrt_Grad`.
            eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
            mse = tf.add(mse, eps)
        with tf.name_scope("Scale"):
            n_reals = tf.cast(n_atoms, dtype=labels.dtype)
            n_max = tf.convert_to_tensor(
                labels.shape[1].value, dtype=labels.dtype, name='n_max')
            one = tf.constant(1.0, dtype=dtype, name='one')
            weight = tf.math.truediv(one, tf.reduce_mean(n_reals / n_max),
                                     name='weight')
        mse = tf.multiply(mse, weight, name='mse')
    mae = tf.multiply(mae, weight, name='mae')
    loss = tf.sqrt(mse, name='rmse')
    return loss, mae


def get_forces_loss(labels, predictions, n_atoms, confidences=None, weight=1.0,
                    collections=None):
    """
    Return the loss tensor of the atomic forces.

    Parameters
    ----------
    labels : tf.Tensor
        A `float64` tensor of shape `[batch_size, n_atoms_max + 1, 3]` as the
        reference forces.
    predictions : tf.Tensor
        A `float64` tensor of shape `[batch_size, n_atoms_max, 3]` as the
        predicted forces.
    n_atoms : tf.Tensor
        A `int64` tensor of shape `[batch_size, ]` as the number of atoms of
        each structure.
    confidences : tf.Tensor or None
        A `float64` tensor of shape `[batch_size, ]` as the force confidences
        or None. If None, all forces are trusted equally.
    weight : float or tf.Tensor
        The weight of the loss.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.

    Returns
    -------
    loss : tf.Tensor
        A `float64` tensor as the RMSE loss.

    """
    with tf.name_scope("Forces"):
        assert labels.shape.ndims == 3 and labels.shape[2].value == 3
        assert predictions.shape.ndims == 3 and predictions.shape[2].value == 3
        with tf.name_scope("Split"):
            labels = tf.split(
                labels, [1, -1], axis=1, name='split')[1]
        raw_loss, mae = _absolute_forces_loss(labels, predictions, n_atoms,
                                              confidences=confidences)
        weight = tf.convert_to_tensor(weight, raw_loss.dtype, name='weight')
        loss = tf.multiply(raw_loss, weight, name='weighted/loss')
        if collections is not None:
            tf.add_to_collections(collections, mae)
            tf.add_to_collections(collections, raw_loss)
            tf.add_to_collections(collections, loss)
        return loss


def get_stress_loss(labels, predictions, weight=1.0, confidences=None,
                    collections=None, method=LossMethod.rmse):
    """
    Return the (relative) RMSE loss of the stress.

    Parameters
    ----------
    labels : tf.Tensor
        A `float64` tensor of shape `[batch_size, 6]` as the reference stress.
    predictions : tf.Tensor
        A `float64` tensor of shape `[batch_size, 6]` as the predicted stress.
    weight : float or tf.Tensor
        The weight of the loss.
    confidences : tf.Tensor or None
        A `float64` tensor of shape `[batch_size, ]` as the stress confidences
        or None. If None, all stress tensors are trusted equally.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.
    method : LossMethod
        The loss method to use.

    Returns
    -------
    loss : tf.Tensor
        A `float64` tensor as the RMSE loss.

    """
    with tf.name_scope("Stress"):
        assert labels.shape.ndims == 2 and labels.shape[1].value == 6
        assert predictions.shape.ndims == 2 and predictions.shape[1].value == 6

        if method == LossMethod.rmse:
            raw_loss, mae = _get_rmse_loss(
                labels,
                predictions,
                is_per_atom_loss=False,
                weights=confidences)
        elif method == LossMethod.rrmse:
            raw_loss = _get_relative_rmse_loss(
                labels,
                predictions,
                weights=confidences)
            mae = None
        else:
            raw_loss, mae = _get_logcosh_loss(
                labels,
                predictions,
                is_per_atom_loss=False,
                weights=confidences)
        weight = tf.convert_to_tensor(weight, raw_loss.dtype, name='weight')
        loss = tf.multiply(raw_loss, weight, name='weighted/loss')
        if collections is not None:
            tf.add_to_collections(collections, raw_loss)
            tf.add_to_collections(collections, loss)
            if mae is not None:
                tf.add_to_collections(collections, mae)
        return loss


def get_total_pressure_loss(labels, predictions, weight=1.0, confidences=None,
                            collections=None, method=LossMethod.rmse):
    """
    Return the RMSE loss of the total pressure.

    Parameters
    ----------
    labels : tf.Tensor
        A `float64` tensor of shape `[batch_size, ]` as the reference pressure.
        energies.
    predictions : tf.Tensor
        A `float64` tensor of shape `[batch_size, ]` as the predicted pressure.
    weight : float or tf.Tensor
        The weight of the loss.
    confidences : tf.Tensor or None
        A `float64` tensor of shape `[batch_size, ]` as the stress confidences
        or None. If None, all stress tensors are trusted equally.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.
    method : LossMethod
        The loss method to use.

    Returns
    -------
    loss : tf.Tensor
        A `float64` tensor as the RMSE loss.

    """
    assert method != LossMethod.rrmse

    with tf.name_scope("Pressure"):
        assert labels.shape.ndims == 1
        assert predictions.shape.ndims == 1

        if method == LossMethod.rmse:
            raw_loss, mae = _get_rmse_loss(labels, predictions,
                                           is_per_atom_loss=False,
                                           weights=confidences)
        else:
            raw_loss, mae = _get_logcosh_loss(labels, predictions,
                                              confidences, False)

        weight = tf.convert_to_tensor(weight, raw_loss.dtype, name='weight')
        loss = tf.multiply(raw_loss, weight, name='weighted/loss')
        if collections is not None:
            tf.add_to_collections(collections, mae)
            tf.add_to_collections(collections, raw_loss)
            tf.add_to_collections(collections, loss)
        return loss


def get_l2_regularization_loss(weight=1.0, collections=None):
    """
    Return the total L2 regularization loss.

    Parameters
    ----------
    weight : float or tf.Tensor
        The weight of the loss.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.

    Returns
    -------
    loss : tf.Tensor
        A `float64` tensor of the per-atom RMSE of `labels` and `predictions`.

    """
    with tf.name_scope("L2"):
        name = 'total_regularization_loss'
        losses = tf.losses.get_regularization_losses()
        if losses:
            l2 = tf.add_n(losses, name=name)
        else:
            l2 = tf.constant(0.0, dtype=get_float_dtype(), name=name)
        weight = tf.convert_to_tensor(weight, l2.dtype, name='weight')
        loss = tf.multiply(l2, weight, name='weighted/loss')
        if collections is not None:
            tf.add_to_collections(collections, l2)
            tf.add_to_collections(collections, loss)
        return loss
