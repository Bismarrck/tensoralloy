# coding=utf-8
"""
This module defines the model function.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from dataset import Dataset
from misc import Defaults
from ase.db import connect

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

dataset = Dataset(connect('datasets/qm7.db'), name='qm7', k_max=2)

if not dataset.load_tfrecords('datasets/qm7'):
    dataset.to_records('datasets/qm7', test_size=1165, verbose=True)

batch_size = 10
hidden_sizes = [64, 32]
graph = tf.Graph()

with graph.as_default():

    with tf.name_scope("Dataset"):
        examples = dataset.next_batch(batch_size=batch_size)

    splits = dataset.descriptor.get_computation_graph_from_batch(
        examples, batch_size)

    y_true = examples.y_true
    y_true.set_shape([batch_size, ])

    xavier = xavier_initializer(seed=Defaults.seed)

    with tf.name_scope("ANN"):
        outputs = []
        for i, element in enumerate(dataset.descriptor.elements):
            with tf.variable_scope(element):
                x = tf.cast(splits[i], tf.float32)
                for j in range(len(hidden_sizes)):
                    with tf.variable_scope('Hidden{}'.format(j + 1)):
                        x = tf.layers.conv1d(
                            inputs=x, filters=hidden_sizes[j], kernel_size=1,
                            strides=1, activation=tf.nn.leaky_relu,
                            use_bias=True, kernel_initializer=xavier,
                            name='1x1Conv{}'.format(i + 1))
                yi = tf.layers.conv1d(inputs=x, filters=1, kernel_size=1,
                                      strides=1, use_bias=False,
                                      kernel_initializer=xavier, name='Output')
                outputs.append(tf.squeeze(yi, axis=2, name='ae'))

    with tf.name_scope("Energy"):
        y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
        y_total = tf.reduce_sum(y_atomic, axis=1, keep_dims=False, name='y')

    with tf.name_scope("Loss"):
        rmse = tf.metrics.root_mean_squared_error(y_true, y_total, name='rmse')
