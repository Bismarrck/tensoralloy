#!coding=utf-8
"""
This module defines tensorflow-graph related utility functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow.contrib.opt import NadamOptimizer
from typing import Dict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def is_first_replica():
    """
    Return True if the defined replica is the first replica.
    """
    replica_context = tf.distribute.get_replica_context()
    if isinstance(replica_context, tf.distribute.ReplicaContext):
        graph = tf.get_default_graph()
        # An ugly approach to obtain the replica id because
        # `replica_context.replica_id_in_sync_group` returns a tensor.
        result = replica_context.replica_id_in_sync_group
        if isinstance(result, int):
            value = result
        else:
            node = [n for n in graph.as_graph_def().node
                    if n.name == result.op.name][0]
            value = node.attr.get('value').tensor.int_val[0]
        if value != 0:
            return False
    return True


def get_activation_fn(fn_name="softplus"):
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
    elif fn_name.lower() == "elu":
        return tf.nn.elu
    else:
        raise ValueError(
            f"The activation function '{fn_name}' cannot be recognized!")


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
                decay_fn = tf.compat.v1.train.exponential_decay
            elif decay_function == 'inverse_time':
                decay_fn = tf.compat.v1.train.inverse_time_decay
            elif decay_function == 'natural_exp':
                decay_fn = tf.compat.v1.train.natural_exp_decay
            else:
                raise ValueError(
                    "'{}' is not supported!".format(decay_function))
            learning_rate = decay_fn(learning_rate,
                                     global_step=global_step,
                                     decay_rate=decay_rate,
                                     decay_steps=decay_steps,
                                     staircase=staircase,
                                     name="learning_rate")
        tf.compat.v1.summary.scalar('learning_rate_at_step', learning_rate)
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
            return tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=kwargs.get('beta1', 0.9))
        elif method.lower() == 'nadam':
            return NadamOptimizer(
                learning_rate=learning_rate, beta1=kwargs.get('beta1', 0.9))
        elif method.lower() == 'adadelta':
            return tf.compat.v1.train.AdadeltaOptimizer(
                learning_rate=learning_rate, rho=kwargs.get('rho', 0.95))
        elif method.lower() == 'rmsprop':
            return tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=learning_rate, decay=kwargs.get('decay', 0.9),
                momentum=kwargs.get('momentum', 0.0))
        elif method.lower() == 'sgd':
            return tf.compat.v1.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=kwargs.get('momentum', 0.9),
                use_nesterov=kwargs.get('use_nesterov', True))
        else:
            raise ValueError(
                "Supported SGD optimizers: adam, nadam, adadelta, rmsprop.")


def get_tensors_dict_for_hook(key) -> Dict[str, tf.Tensor]:
    """
    Return a dict of logging tensors for a `LoggingTensorHook`.
    """
    tensors = {}
    for tensor in tf.get_collection(key):
        tensors[tensor.op.name] = tensor
    return tensors


def log_tensor(tensor: tf.Tensor):
    """
    Print the name and shape of the input Tensor.
    """
    dimensions = ",".join(["{:5d}".format(dim if dim is not None else -1)
                           for dim in tensor.get_shape().as_list()])
    tf.logging.info("{:<60s} : [{}]".format(tensor.op.name, dimensions))


def logcosh(x, dtype=None, name=None):
    """
    The keras implemnetation of `logcosh`.
    """
    two = tf.convert_to_tensor(2.0, dtype=dtype, name='two')
    return tf.math.subtract(x + tf.nn.softplus(-two * x), tf.math.log(two),
                            name=name)
