# coding=utf-8
"""
This module defines various loss functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List

from tensoralloy.dtypes import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


__all__ = ["get_energy_loss", "get_forces_loss", "get_stress_loss",
           "get_total_pressure_loss"]


def _get_rmse_loss(x: tf.Tensor, y: tf.Tensor, is_per_atom_loss=False):
    """
    Return the RMSE loss tensor. The MAE loss will also be calculated.
    """
    if is_per_atom_loss:
        suffix = "/atom"
    else:
        suffix = ""
    mse = tf.reduce_mean(tf.squared_difference(x, y), name='mse' + suffix)
    mae = tf.reduce_mean(tf.abs(x - y), name='mae' + suffix)
    loss = tf.sqrt(mse, name='rmse' + suffix)
    return loss, mae


def get_energy_loss(labels, predictions, n_atoms, weight=1.0, collections=None,
                    per_atom_loss=False):
    """
    Return the loss tensor of the energy.

    Parameters
    ----------
    labels : tf.Tensor
        A `float64` tensor of shape `[batch_size, ]` as the reference energy.
    predictions : tf.Tensor
        A `float64` tensor of shape `[batch_size, ]` as the predicted energy.
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

    Returns
    -------
    loss : tf.Tensor
        A `float64` tensor as the RMSE loss.

    """
    with tf.name_scope("Energy"):
        assert labels.shape.ndims == 1
        assert predictions.shape.ndims == 1
        if per_atom_loss:
            n_atoms = tf.cast(n_atoms, labels.dtype, name='n_atoms')
            x = tf.div(labels, n_atoms, name='labels')
            y = tf.div(predictions, n_atoms, name='labels')
        else:
            x = tf.identity(labels, name='labels')
            y = tf.identity(predictions, name='predictions')
        raw_loss, mae = _get_rmse_loss(x, y, is_per_atom_loss=per_atom_loss)
        weight = tf.convert_to_tensor(weight, raw_loss.dtype, name='weight')
        loss = tf.multiply(raw_loss, weight, name='weighted/loss')
        if collections is not None:
            tf.add_to_collections(collections, mae)
            tf.add_to_collections(collections, raw_loss)
            tf.add_to_collections(collections, loss)
        return loss


def _absolute_forces_loss(labels: tf.Tensor, predictions: tf.Tensor,
                          n_atoms: tf.Tensor):
    """
    Return the absolute RMSE as the loss of atomic forces.

        loss = sqrt(weight * MSE(labels, predictions))

    where `weight` is a scaling factor to eliminate the zero-padding effect.

    """
    with tf.name_scope("Absolute"):
        dtype = get_float_dtype()
        mse = tf.reduce_mean(
            tf.squared_difference(labels, predictions), name='raw_mse')
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
            weight = tf.div(one, tf.reduce_mean(n_reals / n_max), name='weight')
        mse = tf.multiply(mse, weight, name='mse')
    mae = tf.multiply(mae, weight, name='mae')
    loss = tf.sqrt(mse, name='rmse')
    return loss, mae


def get_forces_loss(labels, predictions, n_atoms, weight=1.0, collections=None):
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
        raw_loss, mae = _absolute_forces_loss(labels, predictions, n_atoms)
        weight = tf.convert_to_tensor(weight, raw_loss.dtype, name='weight')
        loss = tf.multiply(raw_loss, weight, name='weighted/loss')
        if collections is not None:
            tf.add_to_collections(collections, mae)
            tf.add_to_collections(collections, raw_loss)
            tf.add_to_collections(collections, loss)
        return loss


def _get_relative_rmse_loss(labels: tf.Tensor, predictions: tf.Tensor,
                            use_mse=False):
    """
    Return the relative RMSE as the loss of atomic forces.
    """
    with tf.name_scope("Relative"):
        if use_mse:
            diff = tf.squared_difference(labels, predictions, name='diff')
            upper = tf.reduce_sum(diff, axis=1, keepdims=False, name='upper')
            ll = tf.square(labels, name='ll')
            lower = tf.reduce_sum(ll, axis=1, keepdims=False, name='lower')
        else:
            diff = tf.subtract(labels, predictions)
            upper = tf.linalg.norm(diff, axis=1, keepdims=False, name='upper')
            lower = tf.linalg.norm(labels, axis=1, keepdims=False, name='lower')
        ratio = tf.div(upper, lower, name='ratio')
        loss = tf.reduce_mean(ratio, name='loss')
    return loss


def get_stress_loss(labels, predictions, weight=1.0, collections=None):
    """
    Return the relative RMSE loss of the stress.

    Parameters
    ----------
    labels : tf.Tensor
        A `float64` tensor of shape `[batch_size, 6]` as the reference stress.
    predictions : tf.Tensor
        A `float64` tensor of shape `[batch_size, 6]` as the predicted stress.
    weight : float or tf.Tensor
        The weight of the loss.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.

    Returns
    -------
    loss : tf.Tensor
        A `float64` tensor as the RMSE loss.

    """
    with tf.name_scope("Stress"):
        assert labels.shape.ndims == 2 and labels.shape[1].value == 6
        assert predictions.shape.ndims == 2 and predictions.shape[1].value == 6
        raw_loss = _get_relative_rmse_loss(labels, predictions)
        weight = tf.convert_to_tensor(weight, raw_loss.dtype, name='weight')
        loss = tf.multiply(raw_loss, weight, name='weighted/loss')
        if collections is not None:
            tf.add_to_collections(collections, raw_loss)
            tf.add_to_collections(collections, loss)
        return loss


def get_total_pressure_loss(labels, predictions, weight=1.0, collections=None):
    """
    Return the relative RMSE loss of the total pressure.

    Parameters
    ----------
    labels : tf.Tensor
        A `float64` tensor of shape `[batch_size, ]` as the reference pressure.
        energies.
    predictions : tf.Tensor
        A `float64` tensor of shape `[batch_size, ]` as the predicted pressure.
    weight : float or tf.Tensor
        The weight of the loss.
    collections : List[str] or None
        A list of str as the collections where the loss tensors should be added.

    Returns
    -------
    loss : tf.Tensor
        A `float64` tensor as the RMSE loss.

    """
    with tf.name_scope("Pressure"):
        assert labels.shape.ndims == 1
        assert predictions.shape.ndims == 1
        raw_loss = _get_relative_rmse_loss(labels, predictions)
        weight = tf.convert_to_tensor(weight, raw_loss.dtype, name='weight')
        loss = tf.multiply(raw_loss, weight, name='weighted/loss')
        if collections is not None:
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
        l2 = tf.losses.get_regularization_loss()
        weight = tf.convert_to_tensor(weight, l2.dtype, name='weight')
        loss = tf.multiply(l2, weight, name='weighted/loss')
        if collections is not None:
            tf.add_to_collections(collections, l2)
            tf.add_to_collections(collections, loss)
        return loss
