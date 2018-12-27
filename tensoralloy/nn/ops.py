# coding=utf-8
"""
This module defines core Ops of `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List

from tensoralloy.misc import AttributeDict, Defaults
from tensoralloy.nn import utils

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def sum_of_grads_and_vars(list_of_grads_and_vars):
    """
    Calculate the total gradient from `grad_and_vars` of different losses.

    Parameters
    ----------
    list_of_grads_and_vars: Iterable
        A list of lists of (gradient, variable) tuples.

    Returns
    -------
    outputs: Sized[Tuple[tf.Tensor, tf.Variable]]
        List of pairs of (gradient, variable) as the total gradient.

    """
    with tf.name_scope("SumGrads"):

        if len(list_of_grads_and_vars) == 1:
            return list_of_grads_and_vars[0]

        # Merge gradients
        outputs = []

        for grads_and_vars in zip(*list_of_grads_and_vars):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for i, (g, _) in enumerate(grads_and_vars):
                if g is None:
                    continue

                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a tower dimension which we will average over below.
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
                # shared across towers. So we will just return the first tower's
                # pointer to the Variable.
                grad_and_var = (grad, v)

            outputs.append(grad_and_var)
        return outputs


def add_grads_and_vars_summary(grads_and_vars, name):
    """
    Add summary of the gradients.
    """
    list_of_ops = []
    for grad, var in grads_and_vars:
        if grad is not None:
            norm = tf.norm(grad, name=var.op.name + "/norm")
            list_of_ops.append(norm)
            with tf.name_scope("GradNorm/{}/".format(name)):
                tf.summary.scalar(var.op.name + "/norm", norm)
    with tf.name_scope("GradNorm/{}/".format(name)):
        total_norm = tf.add_n(list_of_ops, name='total')
        tf.summary.scalar('total', total_norm)
        tf.add_to_collection(utils.GraphKeys.TRAIN_METRICS, total_norm)
    return total_norm


def add_gradients_cos_dist_summary(energy_grad_vars, grads_and_vars):
    """
    Compute the cosine distance of dL(energy)/dvars and dL(target)/dvars
    where `target` is `forces` or `stress`.
    """
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


def get_train_op(losses: AttributeDict, hparams: AttributeDict,
                 minimize_properties: List[str]):
    """
    Return the Op for a training step.

    Parameters
    ----------
    losses : AttributeDict
        A dict of loss tensors.
    hparams : AttributeDict
        The hyper parameters.
    minimize_properties : List[str]
        A list of str as the structural properties to minimize.

    Returns
    -------
    ema : tf.train.ExponentialMovingAverage
        The ExponentialMovingAverage controller.
    train_op : tf.Operation
        The Op for a training step.

    """
    with tf.name_scope("Optimize"):

        global_step = tf.train.get_or_create_global_step()
        learning_rate = utils.get_learning_rate(
            global_step,
            learning_rate=hparams.opt.learning_rate,
            decay_function=hparams.opt.decay_function,
            decay_rate=hparams.opt.decay_rate,
            decay_steps=hparams.opt.decay_steps,
            staircase=hparams.opt.staircase
        )

        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = utils.get_optimizer(learning_rate, hparams.opt.method)

        grads_and_vars = {}
        total_norms = {}

        for prop in minimize_properties:
            with tf.name_scope(
                    "".join([word.capitalize() for word in prop.split('_')])):
                g = optimizer.compute_gradients(losses[prop])
                total_norm = add_grads_and_vars_summary(g, prop)
                grads_and_vars[prop] = g
                total_norms[prop] = total_norm

        with tf.name_scope("Histogram"):
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name + '/hist', var)

        grads_and_vars = sum_of_grads_and_vars(
            list_of_grads_and_vars=[grads_and_vars[prop]
                                    for prop in minimize_properties])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradients_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)

        with tf.control_dependencies([apply_gradients_op]):
            with tf.name_scope("Average"):
                ema = tf.train.ExponentialMovingAverage(
                    Defaults.variable_moving_average_decay)
                variable_averages_op = ema.apply(tf.trainable_variables())

    # Use an absolute name scope for cosine distances.
    with tf.name_scope("CosDist"):
        if 'energy' in minimize_properties:
            for prop in minimize_properties:
                if prop == 'energy':
                    continue
                with tf.name_scope(prop):
                    add_gradients_cos_dist_summary(
                        grads_and_vars['energy'], grads_and_vars[prop])

    return ema, variable_averages_op
