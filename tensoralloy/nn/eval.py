#!coding=utf-8
"""
This module defines evaluation metrics and hooks for `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.utils import GraphKeys
from tensoralloy.nn.utils import get_tensors_dict_for_hook
from tensoralloy.nn.hooks import LoggingTensorHook, RestoreEmaVariablesHook
from tensoralloy.nn.dataclasses import TrainParameters

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_evaluation_hooks(ema: tf.train.ExponentialMovingAverage,
                         train_parameters: TrainParameters):
    """
    Return a list of `tf.train.SessionRunHook` objects for evaluation.
    """
    hooks = []

    with tf.name_scope("Hooks"):

        if len(tf.get_collection(GraphKeys.EVAL_METRICS)) > 0:
            with tf.name_scope("Accuracy"):
                logging_tensor_hook = LoggingTensorHook(
                    tensors=get_tensors_dict_for_hook(
                        GraphKeys.EVAL_METRICS),
                    mode=tf_estimator.ModeKeys.EVAL,
                    every_n_iter=train_parameters.eval_steps,
                    at_end=True)
            hooks.append(logging_tensor_hook)

        with tf.name_scope("EMA"):
            restore_ema_hook = RestoreEmaVariablesHook(ema=ema)
            hooks.append(restore_ema_hook)

    return hooks


def get_eval_metrics_ops(eval_properties, predictions, labels, n_atoms,
                         atom_masks):
    """
    Return a dict of Ops as the evaluation metrics.

    Always required:
        * 'energy' of shape `[batch_size, ]`.
        * 'energy_confidence' of shape `[batch_size, ]`

    Required if 'forces' should be minimized:
        * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is
          required if 'forces' should be minimized.
        * 'forces_confidence' of shape `[batch_size, ]` is required if
          'forces' should be minimized.
        * 'atom_masks' of shape `[batch_size, n_atoms_max + 1]`

    Required if 'stress' or 'total_pressure' should be minimized:
        * 'stress' of shape `[batch_size, 6]` is required if
          'stress' should be minimized.
        * 'pulay_stress' of shape `[batch_size, ]`
        * 'total_pressure' of shape `[batch_size, ]`
        * 'stress_confidence' of shape `[batch_size, ]`

    `n_atoms` is a `int64` tensor with shape `[batch_size, ]`, representing
    the number of atoms in each structure.

    """
    with tf.name_scope("Metrics"):

        metrics = {}
        n_atoms = tf.cast(n_atoms, labels["energy"].dtype, name='n_atoms')

        with tf.name_scope("Energy"):
            x = labels["energy"]
            y = predictions["energy"]
            xn = x / n_atoms
            yn = y / n_atoms
            ops_dict = {
                'Energy/mae': tf.metrics.mean_absolute_error(x, y),
                'Energy/mse': tf.metrics.mean_squared_error(x, y),
                'Energy/mae/atom': tf.metrics.mean_absolute_error(xn, yn),
            }
            metrics.update(ops_dict)

        if 'forces' in eval_properties:
            with tf.name_scope("Forces"):
                with tf.name_scope("Split"):
                    x = tf.split(labels["forces"], [1, -1], axis=1)[1]
                    mask = tf.cast(tf.split(atom_masks, [1, -1], axis=1)[1],
                                   tf.bool)
                x = tf.boolean_mask(x, mask, axis=0, name='x')
                y = tf.boolean_mask(
                    predictions["forces"], mask, axis=0, name='y')
                with tf.name_scope("Flatten"):
                    x = tf.reshape(x, (-1, ), name='x')
                    y = tf.reshape(y, (-1, ), name='y')
                ops_dict = {
                    'Forces/mae': tf.metrics.mean_absolute_error(x, y),
                    'Forces/mse': tf.metrics.mean_squared_error(x, y),
                }
                metrics.update(ops_dict)

        if 'total_pressure' in eval_properties:
            with tf.name_scope("Pressure"):
                x = labels.total_pressure
                y = predictions.total_pressure
                ops_dict = {
                    'Pressure/mae': tf.metrics.mean_absolute_error(x, y),
                    'Pressure/mse': tf.metrics.mean_squared_error(x, y)}
                metrics.update(ops_dict)

        if 'stress' in eval_properties:
            with tf.name_scope("Stress"):
                x = labels["stress"]
                y = predictions["stress"]
                ops_dict = {
                    'Stress/mae': tf.metrics.mean_absolute_error(x, y),
                    'Stress/mse': tf.metrics.mean_squared_error(x, y)}
                with tf.name_scope("rRMSE"):
                    upper = tf.linalg.norm(x - y, axis=-1)
                    lower = tf.linalg.norm(x, axis=-1)
                    ops_dict['Stress/relative'] = \
                        tf.metrics.mean(upper / lower)
                metrics.update(ops_dict)

        for tensor in tf.get_collection(GraphKeys.EVAL_METRICS):
            slist = tensor.op.name.split("/")
            if "Elastic" in tensor.op.name:
                istart = slist.index('Elastic')
                key = "/".join(slist[istart:])
                metrics[key] = (tensor, tf.no_op())
            elif "Rose" in tensor.op.name:
                istart = slist.index('Rose')
                istop = slist.index('EOS')
                key = "/".join(slist[istart: istop])
                metrics[key] = (tensor, tf.no_op())
            elif "Diff" in tensor.op.name:
                if 'pred' in tensor.op.name:
                    key = "/".join(slist[-3:])
                else:
                    istart = slist.index('Diff')
                    istop = slist.index('mae')
                    key = "/".join(slist[istart: istop - 1])
                metrics[key] = (tensor, tf.no_op())

        return metrics
