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
from tensoralloy.dtypes import set_float_precision, get_float_dtype, Precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_energy_loss():
    """
    Test the function `get_energy_loss`.
    """
    set_float_precision(Precision.medium)

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

        rmse = get_energy_loss(x, y, n, collections=['UnitTest'])
        assert_equal(len(tf.get_collection('UnitTest')), 3)

        mae = tf.get_default_graph().get_tensor_by_name('Energy/mae:0')

        with tf.Session() as sess:
            assert_less(y_rmse - sess.run(rmse), 1e-8)
            assert_less(y_mae - sess.run(mae), 1e-8)

    set_float_precision(Precision.high)


def test_forces_loss():
    """
    Test the function `get_forces_loss`.
    """
    x = np.random.uniform(0.0, 3.0, size=(6, 8, 3))
    y = np.random.uniform(0.0, 3.0, size=(6, 8, 3))
    n_atoms = np.asarray([5, 6, 4, 8, 5, 3])
    x[5] = 0.0
    y[5] = 0.0

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
        n_atoms = tf.convert_to_tensor(n_atoms)

        rmse = get_forces_loss(x, y, n_atoms, weight=10.0,
                               collections=['UnitTest'])
        mae = tf.get_default_graph().get_tensor_by_name('Forces/mae:0')

        with tf.Session() as sess:
            assert_less(y_mae - sess.run(mae), 1e-8)
            assert_less(y_rmse * 10.0 - sess.run(rmse), 1e-8)


def test_stress_loss():
    """
    Test the function `get_stress_loss`.
    """
    x = np.random.uniform(0.1, 3.0, size=(6, 6))
    y = np.random.uniform(0.1, 3.0, size=(6, 6))

    y_rmse = np.mean(np.linalg.norm(x - y, axis=1) / np.linalg.norm(x, axis=1))

    with tf.Graph().as_default():
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        rmse = get_stress_loss(x, y)

        with tf.Session() as sess:
            assert_less(y_rmse - sess.run(rmse), 1e-8)


if __name__ == "__main__":
    nose.run()
