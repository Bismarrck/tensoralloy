#!coding=utf-8
"""
This module defines evaluation metrics and hooks for `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.utils import GraphKeys, ModeKeys
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
                    mode=ModeKeys.EVAL,
                    every_n_iter=train_parameters.eval_steps,
                    at_end=True)
            hooks.append(logging_tensor_hook)

        with tf.name_scope("EMA"):
            restore_ema_hook = RestoreEmaVariablesHook(ema=ema)
            hooks.append(restore_ema_hook)

    return hooks
