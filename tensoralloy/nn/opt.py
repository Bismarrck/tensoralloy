# coding=utf-8
"""
This module defines core Ops of `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List
from tensorflow.python.training.basic_session_run_hooks import NanTensorHook
from tensorflow.python.training.basic_session_run_hooks import ProfilerHook

from tensoralloy.utils import AttributeDict, Defaults, GraphKeys
from tensoralloy.nn.utils import get_optimizer, get_learning_rate
from tensoralloy.nn.utils import get_tensors_dict_for_hook
from tensoralloy.nn.dataclasses import OptParameters, TrainParameters
from tensoralloy.nn.hooks import LoggingTensorHook, ExamplesPerSecondHook
from tensoralloy.nn.hooks import WarmStartFromVariablesHook
from tensoralloy.dtypes import get_float_dtype

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
        tf.add_to_collection(GraphKeys.TRAIN_METRICS, total_norm)
    return total_norm


def add_gradients_cos_dist_summary(energy_grad_vars, grads_and_vars):
    """
    Compute the cosine distance of dL(energy)/dvars and dL(target)/dvars
    where `target` is `forces` or `stress`.
    """
    dtype = get_float_dtype()
    eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
    for i, (grad, var) in enumerate(grads_and_vars):
        if grad is None:
            continue
        energy_grad = energy_grad_vars[i][0]
        dot = tf.tensordot(
            tf.reshape(grad, (-1,)),
            tf.reshape(energy_grad, (-1,)), axes=1)
        norm = tf.norm(grad) * tf.norm(energy_grad) + eps
        cos_dist = tf.math.truediv(dot, norm, name=var.op.name + '/cos_dist')
        tf.summary.scalar(cos_dist.op.name + '/summary', cos_dist)


def get_train_op(losses: AttributeDict, opt_parameters: OptParameters,
                 minimize_properties: List[str]):
    """
    Return the Op for a training step.

    Parameters
    ----------
    losses : AttributeDict
        A dict of loss tensors.
    opt_parameters : OptParameters
        The hyper parameters for minimizing the total loss.
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
        learning_rate = get_learning_rate(
            global_step,
            learning_rate=opt_parameters.learning_rate,
            decay_function=opt_parameters.decay_function,
            decay_rate=opt_parameters.decay_rate,
            decay_steps=opt_parameters.decay_steps,
            staircase=opt_parameters.staircase
        )

        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = get_optimizer(learning_rate, opt_parameters.method)

        grads_and_vars = {}
        total_norms = {}

        for prop in minimize_properties:
            if prop not in losses:
                continue
            with tf.name_scope(
                    "".join([word.capitalize() for word in prop.split('_')])):
                g = optimizer.compute_gradients(losses[prop])
                total_norm = add_grads_and_vars_summary(g, prop)
                grads_and_vars[prop] = g
                total_norms[prop] = total_norm

        with tf.name_scope("Histogram"):
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name + '/hist', var)

        gradients = sum_of_grads_and_vars(
            list_of_grads_and_vars=[g for prop, g in grads_and_vars.items()])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradients_op = optimizer.apply_gradients(
                gradients, global_step=global_step)

        with tf.control_dependencies([apply_gradients_op]):
            with tf.name_scope("Average"):
                ema = tf.train.ExponentialMovingAverage(
                    Defaults.variable_moving_average_decay)
                variable_averages_op = ema.apply(tf.trainable_variables())

    # Use an absolute name scope for cosine distances.
    with tf.name_scope("CosDist"):
        if 'energy' in grads_and_vars:
            for prop, g in grads_and_vars.items():
                if prop == 'energy':
                    continue
                with tf.name_scope(prop):
                    add_gradients_cos_dist_summary(grads_and_vars['energy'], g)

    return ema, variable_averages_op


def get_training_hooks(losses: AttributeDict,
                       ema: tf.train.ExponentialMovingAverage,
                       train_parameters: TrainParameters):
    """
    Return a list of `tf.train.SessionRunHook` objects for training.

    Parameters
    ----------
    losses : AttributeDict
        A dict. The loss tensor for energy, forces and stress or total
        pressure.
    ema : tf.train.ExponentialMovingAverage
        A function to obtain moving averaged variables.
    train_parameters : TrainParameters
        Hyper parameters for this function.

    """
    with tf.name_scope("Hooks"):

        with tf.name_scope("Summary"):
            summary_saver_hook = tf.train.SummarySaverHook(
                save_steps=train_parameters.summary_steps,
                output_dir=train_parameters.model_dir,
                summary_op=tf.summary.merge_all())

        with tf.name_scope("Speed"):
            examples_per_sec_hook = ExamplesPerSecondHook(
                batch_size=train_parameters.batch_size,
                every_n_steps=train_parameters.log_steps)

        with tf.name_scope("Nan"):
            nan_tensor_hook = NanTensorHook(fail_on_nan_loss=True, **losses)

        hooks = [summary_saver_hook, examples_per_sec_hook, nan_tensor_hook]

        if len(tf.get_collection(GraphKeys.TRAIN_METRICS)) > 0:
            logging_tensor_hook = LoggingTensorHook(
                tensors=get_tensors_dict_for_hook(GraphKeys.TRAIN_METRICS),
                every_n_iter=train_parameters.log_steps,
                at_end=True)
            hooks.append(logging_tensor_hook)

        if train_parameters.profile_steps:
            with tf.name_scope("Profile"):
                profiler_hook = ProfilerHook(
                    save_steps=train_parameters.profile_steps,
                    output_dir=f"{train_parameters.model_dir}-profile",
                    show_memory=True)
            hooks.append(profiler_hook)

        if train_parameters.previous_checkpoint:
            with tf.name_scope("Restore"):
                ckpt = train_parameters.previous_checkpoint
                warm_start_hook = WarmStartFromVariablesHook(
                    previous_checkpoint=ckpt,
                    ema=ema,
                    restart=train_parameters.restart)
            hooks.append(warm_start_hook)

    return hooks
