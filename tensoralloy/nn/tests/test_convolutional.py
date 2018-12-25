# coding=utf-8
"""
This module defines tests of `tensoralloy.nn.convolutional`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_almost_equal

from ..convolutional import convolution1x1

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_convolution1x1():
    """
    Test the function `convolution1x1`.
    """
    with tf.Graph().as_default():

        x = tf.convert_to_tensor(np.random.rand(5, 100, 8), name='x')
        y_nn = convolution1x1(x, activation_fn=tf.nn.tanh,
                              hidden_sizes=[128, 64], l2_weight=0.01)
        y_true = tf.convert_to_tensor(np.random.rand(5, 100, 1), name='y_true')
        loss = tf.reduce_mean(tf.squared_difference(y_nn, y_true))
        tf.train.AdamOptimizer().minimize(loss)

        reg_loss = tf.losses.get_regularization_loss()

        l2_losses = []
        for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
            l2_losses.append(tf.nn.l2_loss(var))
        l2_loss = tf.add_n(l2_losses)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            l2_value, reg_value = sess.run([l2_loss, reg_loss])

            assert_almost_equal(l2_value * 0.01, reg_value, delta=1e-8)


def test_convolution1x1_without_l2_reg():
    """
    Test the function `convolution1x1` without l2 regularization.
    """
    with tf.Graph().as_default():

        x = tf.convert_to_tensor(np.random.rand(5, 100, 8), name='x')
        y_nn = convolution1x1(x, activation_fn=tf.nn.tanh,
                              hidden_sizes=[128, 64], l2_weight=0.0)
        y_true = tf.convert_to_tensor(np.random.rand(5, 100, 1), name='y_true')
        loss = tf.reduce_mean(tf.squared_difference(y_nn, y_true))
        tf.train.AdamOptimizer().minimize(loss)

        reg_loss = tf.losses.get_regularization_loss()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            assert_almost_equal(sess.run(reg_loss), 0.0, delta=1e-12)



if __name__ == "__main__":
    nose.main()
