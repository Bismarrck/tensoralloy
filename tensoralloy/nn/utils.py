# coding=utf-8
"""
This module defines tensorflow-graph related utility functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import Iterable
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.opt import NadamOptimizer

from tensoralloy.misc import Defaults

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def sum_of_grads_and_vars_collections(grads_and_vars_collections):
    """
    Calculate the total gradient from `grad_and_vars` of different losses.

    Parameters
    ----------
    grads_and_vars_collections: Iterable
        A list of lists of (gradient, variable) tuples.

    Returns
    -------
    outputs: Sized[Tuple[tf.Tensor, tf.Variable]]
        List of pairs of (gradient, variable) as the total gradient.

    """
    # Merge gradients
    outputs = []

    for grads_and_vars in zip(*grads_and_vars_collections):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grads_and_vars:
            if g is None:
                continue

            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        v = grads_and_vars[0][1]

        # If the grads are all None, we just return a None grad.
        if len(grads) == 0:
            grad_and_var = (None, v)

        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_sum(grad, 0)

            # Keep in mind that the Variables are redundant because they are
            # shared across towers. So .. we will just return the first tower's
            # pointer to the Variable.
            grad_and_var = (grad, v)

        outputs.append(grad_and_var)
    return outputs


class GraphKeys:
    """
    Standard names for variable collections.
    """

    # Summary keys
    TRAIN_SUMMARY = 'train_summary'
    EVAL_SUMMARY = 'eval_summary'

    # Variable keys
    NORMALIZE_VARIABLES = 'normalize_vars'
    STATIC_ENERGY_VARIABLES = 'static_energy_vars'

    # Metrics Keys
    TRAIN_METRICS = 'train_metrics'
    EVAL_METRICS = 'eval_metrics'

    # Gradient Keys
    ENERGY_GRADIENTS = 'energy_gradients'
    FORCES_GRADIENTS = 'forces_gradients'
    STRESS_GRADIENTS = 'stress_gradients'


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
    tf.logging.info("{:<48s} : [{}]".format(tensor.op.name, dimensions))


def msra_initializer(dtype=tf.float64, seed=Defaults.seed):
    """
    Return the so-called `MSRA` initializer.

    See Also
    --------
    [Delving Deep into Rectifiers](http://arxiv.org/pdf/1502.01852v1.pdf)

    """
    return variance_scaling_initializer(dtype=dtype, seed=seed)
