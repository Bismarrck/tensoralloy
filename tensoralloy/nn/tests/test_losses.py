# coding=utf-8
"""
This module defines unit tests of the module `losses`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_less
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..losses import norm_loss, rmse_loss, mae_loss

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_norm_loss():
    """
    Test the function `norm_loss`.
    """
    compositions = np.asarray([[5], [6], [4], [8], [8], [5]])
    x = np.random.rand(6, 8, 3)
    y = np.random.rand(6, 8, 3)

    for i in range(len(x)):
        x[i, compositions[i, 0]:] = 0.0
        y[i, compositions[i, 0]:] = 0.0
    x[5] = 0.0

    values = []
    for i in range(6):
        for j in range(compositions[i, 0]):
            a = np.linalg.norm(x[i, j])
            if a < 0.01:
                values.append(np.linalg.norm(y[i, j]))
            else:
                b = np.linalg.norm(x[i, j] - y[i, j])
                values.append(b / a)
    ref = np.mean(values)

    with tf.Graph().as_default():
        loss = norm_loss(x, y, compositions, name='loss')
        with tf.Session() as sess:
            pred = sess.run(loss)
            assert_less(ref - pred, 1e-8)


def test_rmse_and_mae_losses():
    """
    Test the functions `rmse_loss` and `mae_loss` with `compositions`.
    """
    x = np.random.uniform(0.0, 3.0, size=(6, 8, 3))
    x[5] = 0.0
    y = np.random.uniform(0.0, 3.0, size=(6, 8, 3))
    y[5] = 0.0
    compositions = np.asarray([[5], [6], [4], [8], [5], [3]])

    mae_values = []
    rmse_values = []

    for i in range(len(x)):
        n = compositions[i, 0]
        for j in range(n):
            mae_values.append(np.abs(x[i, j] - y[i, j]))
            rmse_values.append(np.square(x[i, j] - y[i, j]))

    y_mae = np.mean(mae_values)
    y_rmse = np.sqrt(np.mean(rmse_values))

    with tf.Graph().as_default():

        mae = mae_loss(x, y, compositions=compositions, name='mae')
        rmse = rmse_loss(x, y, compositions=compositions, name='rmse')

        with tf.Session() as sess:
            assert_less(y_mae - sess.run(mae), 1e-8)
            assert_less(y_rmse - sess.run(rmse), 1e-8)


def test_normal_rmse_and_mae_losses():
    """
    Test the functions `rmse_loss` and `mae_loss` without `compositions`.
    """
    x = np.random.uniform(0.0, 3.0, size=(6, 8))
    x[5] = 0.0
    y = np.random.uniform(0.0, 3.0, size=(6, 8))
    y[5] = 0.0

    y_mae = mean_absolute_error(x, y)
    y_rmse = np.sqrt(mean_squared_error(x, y))

    with tf.Graph().as_default():

        mae = mae_loss(x, y, name='mae')
        rmse = rmse_loss(x, y, name='rmse')

        with tf.Session() as sess:
            assert_less(y_mae - sess.run(mae), 1e-8)
            assert_less(y_rmse - sess.run(rmse), 1e-8)


if __name__ == "__main__":
    nose.run()
