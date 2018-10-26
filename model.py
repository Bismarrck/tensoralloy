# coding=utf-8
"""
This module defines the model function.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import time
from os.path import join
from tensorflow.contrib.layers import xavier_initializer
from dataset import Dataset
from misc import Defaults, check_path, AttributeDict
from multiprocessing import cpu_count
from ase.db import connect

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

dataset = Dataset(connect('datasets/qm7.db'), name='qm7', k_max=2)

if not dataset.load_tfrecords('datasets/qm7'):
    dataset.to_records('datasets/qm7', test_size=1165, verbose=True)

batch_size = 300
hidden_sizes = [64, 32]
learning_rate = 0.001
graph = tf.Graph()

with graph.as_default():

    with tf.name_scope("Dataset"):
        examples = dataset.next_batch(batch_size=batch_size)

    splits = dataset.descriptor.get_computation_graph_from_batch(
        examples, batch_size)

    y_true = examples.y_true
    y_true.set_shape([batch_size, ])

    kernel_initializer = xavier_initializer(
        seed=Defaults.seed, dtype=tf.float64)
    bias_initializer = tf.zeros_initializer(dtype=tf.float64)

    with tf.variable_scope("ANN"):
        outputs = []
        for i, element in enumerate(dataset.descriptor.elements):
            with tf.variable_scope(element):
                x = tf.identity(splits[i], name='x')
                for j in range(len(hidden_sizes)):
                    with tf.variable_scope('Hidden{}'.format(j + 1)):
                        x = tf.layers.conv1d(
                            inputs=x, filters=hidden_sizes[j], kernel_size=1,
                            strides=1, activation=tf.nn.leaky_relu,
                            use_bias=True,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            name='1x1Conv{}'.format(i + 1))
                yi = tf.layers.conv1d(inputs=x, filters=1, kernel_size=1,
                                      strides=1, use_bias=False,
                                      kernel_initializer=kernel_initializer,
                                      name='Output')
                outputs.append(tf.squeeze(yi, axis=2, name='ae'))

    with tf.name_scope("Energy"):
        y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
        y_total = tf.reduce_sum(y_atomic, axis=1, keepdims=False, name='y')

    with tf.name_scope("Penalty"):
        for var in tf.trainable_variables():
            if 'bias' in var.op.name:
                continue
            l2 = tf.nn.l2_loss(var, name=var.op.name + "/l2")
            tf.add_to_collection('l2_losses', l2)
        l2_loss = tf.add_n(tf.get_collection('l2_losses'), name='l2_sum')
        alpha = tf.convert_to_tensor(0.001, dtype=tf.float64, name='alpha')
        l2 = tf.multiply(l2_loss, alpha, name='l2')

    with tf.name_scope("Loss"):
        mse = tf.reduce_mean(tf.squared_difference(y_true, y_total), name='mse')
        rmse = tf.sqrt(mse, name='rmse')
        loss = tf.add(rmse, l2, name='loss')
        mae = tf.reduce_mean(tf.abs(y_true - y_total), name='mae')

    with tf.name_scope("Optimization"):
        global_step = tf.train.get_or_create_global_step(graph=graph)
        minimize_op = tf.train.AdamOptimizer(
            learning_rate).minimize(loss, global_step)

    with tf.name_scope("Average"):
        variable_averages = tf.train.ExponentialMovingAverage(
            Defaults.variable_moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

    train_op = tf.group(minimize_op, variables_averages_op)

    with tf.name_scope("Summary"):
        summary = []
        for var in tf.trainable_variables():
            summary.append(tf.summary.histogram(var.op.name, var))
        summary_op = tf.summary.merge(summary)

    train_dir = check_path('datasets/qm7/train')
    saver = tf.train.Saver()

    with tf.Session(
            config=tf.ConfigProto(device_count={'cpu': cpu_count()})) as sess:

        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        tic = time.time()
        for _ in range(1000):
            step, step_result = sess.run([global_step,
                                          AttributeDict(train_op=train_op,
                                                        loss=loss,
                                                        mae=mae,
                                                        summary=summary_op)])
            if step and step % 100 == 0:
                speed = (step + 1) / (time.time() - tic)
                print('step: {:5d}, loss: {:8.5f}, mae: {:8.5f}, '
                      'speed: {:.1f}'.format(step, step_result.loss,
                                             step_result.mae, speed))
                summary_writer.add_summary(step_result.summary, step)
            if step and step % 1000 == 0:
                saver.save(sess, join(train_dir, dataset.name), global_step)
