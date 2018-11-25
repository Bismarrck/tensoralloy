# coding=utf-8
"""
This module defines tensorflow-graph related utility functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import Iterable

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
