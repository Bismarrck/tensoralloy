#!coding=utf-8
"""
The special temperature-dependent potential for Be.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import List
from tensoralloy.nn.utils import log_tensor, get_activation_fn
from tensoralloy.nn.atomic import TemperatureDependentAtomicNN
from tensoralloy.nn.convolutional import convolution1x1

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class BeNN(TemperatureDependentAtomicNN):
    """
    The special temperature-dependent potential for Be. MD trajectories were
    generated in PHYSICAL REVIEW B 99, 064102 (2019).
    """

    def _get_electron_entropy(self,
                              h: tf.Tensor,
                              t: tf.Tensor,
                              element: str,
                              collections: List[str],
                              verbose=True):
        """
        Model electron entropy S with the free electron model. The parameters
        a, b, c, d and dt are manually fitted.

        Parameters
        ----------
        h : tf.Tensor
            Input features.
        t : tf.Tensor
            The electron temperature tensor.
        element : str
            The target element.
        collections : List[str]
            A list of str as the collections where the variables should be
            added.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        with tf.variable_scope("S"):
            t2 = tf.square(t, name='T2')
            a = tf.constant(-0.5718444, dtype=h.dtype, name='a')
            b = tf.constant(0.83744317, dtype=h.dtype, name='b')
            c = tf.constant(-0.2110962, dtype=h.dtype, name='c')
            d = tf.constant(1.45, dtype=h.dtype, name='d')
            dt = tf.multiply(d, t, name='dt')
            one = tf.constant(1.0, dtype=h.dtype, name='one')
            ft = tf.square(tf.nn.relu(one - dt), name='ft')
            eentropy = tf.add(a * t2 * ft, b * t + c * (one - ft),
                              name='eentropy')
            deviation = convolution1x1(
                h,
                activation_fn=get_activation_fn(self._activation),
                hidden_sizes=self._hidden_sizes[element],
                num_out=1,
                l2_weight=1.0,
                collections=collections,
                output_bias=False,
                output_bias_mean=0.0,
                use_resnet_dt=self._use_resnet_dt,
                kernel_initializer=self._kernel_initializer,
                variable_scope=None,
                verbose=verbose)
            deviation = tf.nn.softplus(
                tf.squeeze(deviation, axis=2), name='deviation')
            eentropy = tf.multiply(eentropy, deviation, name='atomic')
            if verbose:
                log_tensor(eentropy)
            return eentropy
