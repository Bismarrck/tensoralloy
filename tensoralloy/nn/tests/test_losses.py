# coding=utf-8
"""
This module defines unit tests of the module `losses`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from nose.tools import assert_less, assert_equal
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensoralloy.nn.losses import get_energy_loss, get_forces_loss
from tensoralloy.nn.losses import get_stress_loss
from tensoralloy.dtypes import set_precision, get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_energy_loss():
    """
    Test the function `get_energy_loss`.
    """
    with set_precision('medium'):

        dtype = get_float_dtype()

        x = np.random.uniform(0.0, 3.0, size=(6, )).astype(dtype.as_numpy_dtype)
        y = np.random.uniform(0.0, 3.0, size=(6, )).astype(dtype.as_numpy_dtype)
        n = np.random.randint(1, 5, size=(6, )).astype(dtype.as_numpy_dtype)

        y_rmse = np.sqrt(mean_squared_error(x / n, y / n))
        y_mae = mean_absolute_error(x / n, y / n)

        with tf.Graph().as_default():

            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            n = tf.convert_to_tensor(n)

            rmse = get_energy_loss(x, y, weights=None,
                                   n_atoms=n, collections=['UnitTest'])
            assert_equal(len(tf.get_collection('UnitTest')), 3)

            mae = tf.get_default_graph().get_tensor_by_name('Energy/mae:0')

            with tf.Session() as sess:
                assert_less(y_rmse - sess.run(rmse), 1e-8)
                assert_less(y_mae - sess.run(mae), 1e-8)


def test_forces_loss():
    """
    Test the function `get_forces_loss`.
    """
    with set_precision('high'):

        x = np.random.uniform(0.0, 3.0, size=(6, 8, 3))
        y = np.random.uniform(0.0, 3.0, size=(6, 8, 3))
        n_atoms = np.asarray([5, 6, 4, 8, 5, 3])
        mask = np.zeros((6, 9))
        for i in range(len(n_atoms)):
            mask[i, 1: n_atoms[i] + 1] = 1.0
        x[5] = 0.0
        y[5] = 0.0
        weights = np.ones(6) * 0.75

        mae_values = []
        rmse_values = []

        for i in range(len(x)):
            for j in range(n_atoms[i]):
                mae_values.append(np.abs(x[i, j] - y[i, j]))
                rmse_values.append(np.square(x[i, j] - y[i, j]))

        y_mae = np.mean(mae_values)
        y_rmse = np.sqrt(np.mean(rmse_values))

        with tf.Graph().as_default():

            x = tf.convert_to_tensor(np.insert(x, 0, 0, axis=1))
            y = tf.convert_to_tensor(y)
            c = tf.convert_to_tensor(weights)
            mask = tf.convert_to_tensor(mask)

            rmse = get_forces_loss(x, y, mask=mask, loss_weight=10.0,
                                   collections=['UnitTest'],
                                   weights=c)
            mae = tf.get_default_graph().get_tensor_by_name(
                'Forces/Absolute/mae:0')

            with tf.Session() as sess:
                assert_less(y_mae - sess.run(mae), 1e-8)
                assert_less(y_rmse * 10.0 * 0.75 - sess.run(rmse), 1e-8)


def test_stress_loss():
    """
    Test the function `get_stress_loss`.
    """
    with set_precision('high'):

        x = np.random.uniform(0.1, 3.0, size=(8, 6))
        y = np.random.uniform(0.1, 3.0, size=(8, 6))
        confidences = np.ones(8) * 0.5

        y_rmse = np.mean(
            np.linalg.norm(x - y, axis=1) / np.linalg.norm(x, axis=1))

        with tf.Graph().as_default():
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            c = tf.convert_to_tensor(confidences)

            rmse = get_stress_loss(x, y, weights=c)

            with tf.Session() as sess:
                assert_less(y_rmse * 0.5 - sess.run(rmse), 1e-8)


if __name__ == "__main__":
    nose.run()
